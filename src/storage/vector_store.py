# src/storage/vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.core.utils import chunk_text, embed_chunks
import uuid
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
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        # Ensure the 'documents' collection exists and is ready for use
        self.collection = self.client.get_or_create_collection("newsletter_content")

    def deduplicate(self, chunks: List[str]) -> List[int]:
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

    def add_document(self, document: str, metadata: Dict[str, Any], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """
        Chunk, embed, deduplicate, and add a document and its metadata to the vector store.
        Returns list of chunk IDs.
        """
        # 1. Chunk the document using shared utility
        chunks = chunk_text(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # 2. Deduplicate chunks (hash-based for now)
        unique_indices = self.deduplicate(chunks)
        unique_chunks = [chunks[i] for i in unique_indices]
        # 3. Generate embeddings for each unique chunk using shared utility
        embeddings = embed_chunks(unique_chunks)
        # 4. Store unique_chunks, embeddings, and metadata in ChromaDB
        chunk_ids = []
        for i, chunk in enumerate(unique_chunks):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embeddings[i]],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]):
        self.delete_document(doc_id)
        self.add_document(new_content, new_metadata)

    def delete_document(self, doc_id: str):
        # Query to find all chunks with the document ID in metadata
        results = self.collection.query(
            query_embeddings=[[0.0]*384],  # Dummy embedding for metadata-only query
            n_results=1000,
            where={"doc_id": doc_id}
        )
        if results and results.get("ids") and results["ids"] and len(results["ids"][0]) > 0:
            self.collection.delete(ids=results["ids"][0])

    def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = embed_chunks([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        formatted_results = []
        if (
            results and results.get("ids") and results["ids"] and
            results.get("documents") and results["documents"] and
            results.get("metadatas") and results["metadatas"]
        ):
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": results["distances"][0][i] if "distances" in results and results["distances"] else None
                }
                formatted_results.append(result)
        return formatted_results

    def temporal_score(self, metadata: Dict[str, Any]) -> float:
        from datetime import datetime, timezone
        timestamp = metadata.get("timestamp")
        if not timestamp:
            return 0.5
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                return 0.5
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        days_old = (now - timestamp).days
        score = max(0.1, min(1.0, 2.0 ** (-days_old / 30)))
        return score

    def cluster_topics(self, embeddings: List[Any], n_clusters: int = 5) -> List[int]:
        if not embeddings or len(embeddings) < n_clusters:
            return [0 for _ in embeddings]
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels.tolist()