"""
ChromaDB Storage Provider Implementation

This module provides the ChromaDB-based storage provider implementation
for the unified storage system.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import shutil
from pathlib import Path
import time
from sklearn.cluster import KMeans

from .base import (
    StorageProvider, StorageConfig, DocumentMetadata, SearchResult, 
    StorageStats, StorageType, DataType
)
from src.core.utils import chunk_text, embed_chunks

logger = logging.getLogger(__name__)


class ChromaStorageProvider(StorageProvider):
    """ChromaDB-based storage provider implementing the unified interface."""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.client = None
        self.collection = None
        self.embedding_cache = {}
        self.query_cache = {}
        
    def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.config.db_path,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name
            )
            
            # Update stats based on existing data
            try:
                count = self.collection.count()
                self.stats.total_documents = count
                self.stats.total_chunks = count
            except:
                pass
                
            logger.info(f"Initialized ChromaDB storage at {self.config.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def add_document(self, content: str, metadata: DocumentMetadata) -> str:
        """Add a document to ChromaDB with chunking and embedding."""
        try:
            # Check cache first
            cache_key = f"doc_{metadata.doc_id}"
            if self._cache_get(cache_key):
                logger.warning(f"Document {metadata.doc_id} already exists in cache")
                return metadata.doc_id
            
            # Chunk the document
            chunks = chunk_text(
                content, 
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap
            )
            
            if not chunks:
                logger.warning("No chunks generated from document")
                return metadata.doc_id
            
            # Generate embeddings with caching
            embeddings = self._get_embeddings_with_cache(chunks)
            
            # Generate chunk IDs and metadata
            chunk_ids = [f"{metadata.doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "doc_id": metadata.doc_id,
                    "title": metadata.title,
                    "source": metadata.source,
                    "content_type": metadata.content_type.value,
                    "timestamp": metadata.timestamp.isoformat(),
                    "author": metadata.author,
                    "tags": json.dumps(metadata.tags),
                    "version": metadata.version,
                    "checksum": metadata.checksum,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=chunk_ids,
                metadatas=chunk_metadatas
            )
            
            # Update cache and stats
            self._cache_set(cache_key, metadata.doc_id)
            self._update_stats(documents_added=1, chunks_added=len(chunks))
            
            logger.info(f"Added document {metadata.doc_id} with {len(chunks)} chunks")
            return metadata.doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document {metadata.doc_id}: {e}")
            return metadata.doc_id
    
    def update_document(self, doc_id: str, content: str, metadata: DocumentMetadata) -> bool:
        """Update an existing document in ChromaDB."""
        try:
            # Delete existing chunks for this document
            self.collection.delete(where={"doc_id": doc_id})
            
            # Add the updated document
            new_doc_id = self.add_document(content, metadata)
            return new_doc_id == doc_id
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from ChromaDB."""
        try:
            # Delete all chunks for this document
            self.collection.delete(where={"doc_id": doc_id})
            
            # Remove from cache
            cache_key = f"doc_{doc_id}"
            if cache_key in self.embedding_cache:
                del self.embedding_cache[cache_key]
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
               top_k: int = 10) -> List[SearchResult]:
        """Search for documents in ChromaDB."""
        try:
            start_time = time.time()
            
            # Check query cache
            cache_key = f"query_{hash(query + str(filters) + str(top_k))}"
            cached_results = self._cache_get(cache_key)
            if cached_results:
                logger.info("Returning cached search results")
                return cached_results
            
            # Build where clause for filters
            where_clause = self._build_where_clause(filters)
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = self._process_search_results(results, query, top_k)
            
            # Cache results
            self._cache_set(cache_key, search_results)
            
            # Update query time stats
            query_time = (time.time() - start_time) * 1000
            self.stats.average_query_time_ms = (
                (self.stats.average_query_time_ms + query_time) / 2
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Tuple[str, DocumentMetadata]]:
        """Retrieve a specific document from ChromaDB."""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return None
            
            # Reconstruct document content
            chunks = results["documents"]
            content = " ".join(chunks)
            
            # Get metadata from first chunk
            first_metadata = results["metadatas"][0]
            metadata = DocumentMetadata(
                doc_id=first_metadata["doc_id"],
                title=first_metadata["title"],
                source=first_metadata["source"],
                content_type=DataType(first_metadata["content_type"]),
                timestamp=datetime.fromisoformat(first_metadata["timestamp"]),
                author=first_metadata.get("author"),
                tags=json.loads(first_metadata["tags"]) if first_metadata["tags"] else [],
                version=first_metadata.get("version", "1.0"),
                checksum=first_metadata.get("checksum")
            )
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        try:
            count = self.collection.count()
            self.stats.total_documents = count
            self.stats.total_chunks = count
            
            # Calculate cache hit rate
            total_cache_requests = len(self.embedding_cache) + len(self.query_cache)
            if total_cache_requests > 0:
                cache_hits = sum(1 for v in self.embedding_cache.values() if v is not None)
                cache_hits += sum(1 for v in self.query_cache.values() if v is not None)
                self.stats.cache_hit_rate = cache_hits / total_cache_requests
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        
        return self.stats
    
    def backup(self) -> bool:
        """Create a backup of the ChromaDB collection."""
        try:
            backup_path = f"{self.config.db_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(self.config.db_path, backup_path)
            logger.info(f"Backup created at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        try:
            if not Path(backup_path).exists():
                logger.error(f"Backup path {backup_path} does not exist")
                return False
            
            # Stop current client
            if self.client:
                self.client = None
                self.collection = None
            
            # Restore from backup
            shutil.rmtree(self.config.db_path, ignore_errors=True)
            shutil.copytree(backup_path, self.config.db_path)
            
            # Reinitialize
            return self.initialize()
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def migrate(self, target_provider: 'StorageProvider') -> bool:
        """Migrate data to another storage provider."""
        try:
            # Get all documents from current provider
            all_results = self.collection.get(include=["documents", "metadatas"])
            
            # Add to target provider
            for i, (doc, metadata) in enumerate(zip(all_results["documents"], all_results["metadatas"])):
                doc_metadata = DocumentMetadata(
                    doc_id=metadata["doc_id"],
                    title=metadata["title"],
                    source=metadata["source"],
                    content_type=DataType(metadata["content_type"]),
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    author=metadata.get("author"),
                    tags=json.loads(metadata["tags"]) if metadata["tags"] else [],
                    version=metadata.get("version", "1.0"),
                    checksum=metadata.get("checksum")
                )
                target_provider.add_document(doc, doc_metadata)
            
            logger.info(f"Migrated {len(all_results['documents'])} documents")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _get_embeddings_with_cache(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with caching."""
        embeddings = []
        
        for text in texts:
            cache_key = f"emb_{hash(text)}"
            cached_embedding = self._cache_get(cache_key)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                # Generate new embedding
                embedding = embed_chunks([text])[0]
                embeddings.append(embedding)
                self._cache_set(cache_key, embedding)
        
        return embeddings
    
    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        if not filters:
            return None
        
        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value
        
        return where_clause
    
    def _process_search_results(self, results: Dict[str, Any], query: str, top_k: int) -> List[SearchResult]:
        """Process raw ChromaDB search results into SearchResult objects."""
        search_results = []
        
        if not results["documents"]:
            return search_results
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        )):
            # Convert distance to similarity score
            similarity_score = 1.0 - distance
            
            # Create DocumentMetadata object
            doc_metadata = DocumentMetadata(
                doc_id=metadata["doc_id"],
                title=metadata["title"],
                source=metadata["source"],
                content_type=DataType(metadata["content_type"]),
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                author=metadata.get("author"),
                tags=json.loads(metadata["tags"]) if metadata["tags"] else [],
                version=metadata.get("version", "1.0"),
                checksum=metadata.get("checksum")
            )
            
            # Generate highlights
            highlights = self._generate_highlights(query, doc)
            
            # Create SearchResult
            search_result = SearchResult(
                doc_id=metadata["doc_id"],
                content=doc,
                metadata=doc_metadata,
                similarity_score=similarity_score,
                rank=i + 1,
                highlights=highlights
            )
            
            search_results.append(search_result)
        
        return search_results
    
    def _generate_highlights(self, query: str, content: str) -> List[str]:
        """Generate highlights for search results."""
        highlights = []
        query_terms = query.lower().split()
        
        # Simple highlight generation
        content_lower = content.lower()
        for term in query_terms:
            if term in content_lower:
                start = content_lower.find(term)
                end = start + len(term)
                highlight = content[max(0, start-20):min(len(content), end+20)]
                highlights.append(highlight)
        
        return highlights[:3]  # Limit to 3 highlights
    
    def deduplicate(self, chunks: List[str]) -> List[int]:
        """Remove duplicate chunks based on content similarity."""
        if len(chunks) <= 1:
            return list(range(len(chunks)))
        
        # Generate embeddings for all chunks
        embeddings = self._get_embeddings_with_cache(chunks)
        
        # Use clustering to find similar chunks
        n_clusters = min(len(chunks), 5)  # Limit clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Keep one representative from each cluster
        unique_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_indices:
                unique_indices.append(cluster_indices[0])
        
        return unique_indices
    
    def temporal_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate temporal relevance score."""
        try:
            timestamp_str = metadata.get("timestamp")
            if not timestamp_str:
                return 0.5
            
            doc_time = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            age_days = (now - doc_time).days
            
            # Exponential decay: newer documents get higher scores
            decay_factor = 0.95
            score = decay_factor ** age_days
            return max(0.1, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def cluster_topics(self, embeddings: List[Any], n_clusters: int = 5) -> List[int]:
        """Cluster embeddings to identify topics."""
        if len(embeddings) < n_clusters:
            return list(range(len(embeddings)))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels.tolist() 