# src/storage/vector_store.py
import chromadb
from chromadb.config import Settings
from .embedding_manager import EmbeddingManager
from .document_store import DocumentStore
from typing import List, Dict, Any, Optional
import random
import numpy as np
from sklearn.cluster import KMeans

class VectorStore:
    """
    Handles document chunking, embedding, deduplication, temporal scoring, topic clustering,
    and metadata management using ChromaDB as the backend.
    """
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_manager = EmbeddingManager()
        self.document_store = DocumentStore()
        # Ensure the 'documents' collection exists and is ready for use
        self.collection = self.client.get_or_create_collection("documents")
        # TODO: Initialize ChromaDB collections, etc.

    def chunk_document(self, document: str, chunk_size: int = 512) -> List[str]:
        """
        Split a document into chunks of approximately chunk_size words for embedding and storage.
        """
        words = document.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def deduplicate(self, chunks: List[str], embeddings: List[Any] = None) -> List[int]:
        """
        Detect and filter out duplicate or near-duplicate chunks.
        Returns indices of unique chunks. (Simple hash-based deduplication for now)
        """
        seen = set()
        unique_indices = []
        for idx, chunk in enumerate(chunks):
            chunk_hash = hash(chunk)
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_indices.append(idx)
        return unique_indices

    def add_document(self, document: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Chunk, embed, deduplicate, and add a document and its metadata to the vector store.
        Returns list of chunk IDs.
        """
        # 1. Chunk the document
        chunks = self.chunk_document(document)
        # 2. Deduplicate chunks (hash-based for now)
        unique_indices = self.deduplicate(chunks)
        unique_chunks = [chunks[i] for i in unique_indices]
        # 3. Generate embeddings for each unique chunk
        embeddings = self.embedding_manager.generate_embeddings(unique_chunks)
        # 4. Store unique_chunks, embeddings, and metadata in ChromaDB
        collection = self.client.get_or_create_collection("documents")
        chunk_ids = []
        for i, chunk in enumerate(unique_chunks):
            chunk_id = f"chunk_{hash(chunk)}"
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            # Store in ChromaDB
            collection.add(
                ids=[chunk_id],
                embeddings=[embeddings[i]],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            chunk_ids.append(chunk_id)
        # 5. Return list of chunk IDs
        return chunk_ids

    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]) -> None:
        """
        Update the content and/or metadata of an existing document.
        """
        # 1. Delete existing chunks for this document
        self.delete_document(doc_id)
        # 2. Add the new content as a new document
        self.add_document(new_content, new_metadata)

    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document and its chunks from the vector store.
        """
        collection = self.client.get_or_create_collection("documents")
        # Query to find all chunks with the document ID in metadata
        results = collection.query(
            query_embeddings=[[0.0]*384],  # Dummy embedding for metadata-only query
            n_results=1000,
            where={"doc_id": doc_id}
        )
        if results["ids"] and len(results["ids"][0]) > 0:
            # Delete all chunks for this document
            collection.delete(ids=results["ids"][0])

    def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional metadata filters. Returns ranked results.
        """
        # 1. Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query_text])[0]
        # 2. Search in ChromaDB
        collection = self.client.get_or_create_collection("documents")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        # 3. Format results
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": results["distances"][0][i] if "distances" in results else None
                }
                formatted_results.append(result)
        return formatted_results

    def temporal_score(self, metadata: Dict[str, Any]) -> float:
        """
        Compute a temporal relevance score based on document metadata (e.g., timestamp).
        Returns a score between 0 and 1, where 1 is most recent.
        """
        from datetime import datetime, timezone
        # Extract timestamp from metadata
        timestamp = metadata.get("timestamp")
        if not timestamp:
            return 0.5  # Default score for documents without timestamp
        
        # Convert to datetime if it's a string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                return 0.5
        
        # Calculate days since epoch
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        days_old = (now - timestamp).days
        
        # Simple exponential decay: newer documents get higher scores
        # Score = exp(-days_old / 30) where 30 is the half-life in days
        score = max(0.1, min(1.0, 2.0 ** (-days_old / 30)))
        return score

    def cluster_topics(self, embeddings: List[Any], n_clusters: int = 5) -> List[int]:
        """
        Cluster document chunks by topic using their embeddings.
        Returns cluster labels for each chunk using KMeans.
        """
        if not embeddings or len(embeddings) < n_clusters:
            # Not enough data to cluster, assign all to one cluster
            return [0 for _ in embeddings]
        # Convert embeddings to numpy array if needed
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels.tolist()
