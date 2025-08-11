"""
Memory Storage Provider

This module provides an in-memory storage provider for testing and development.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    DataType,
    DocumentMetadata,
    SearchResult,
    StorageConfig,
    StorageProvider,
    StorageStats,
)

logger = logging.getLogger(__name__)


class MemoryStorageProvider(StorageProvider):
    """In-memory storage provider for testing and development."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.documents: Dict[str, Tuple[str, DocumentMetadata]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the memory storage."""
        try:
            self.documents.clear()
            self.embeddings.clear()
            self.initialized = True
            logger.info("Initialized memory storage")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize memory storage: {e}")
            return False

    def add_document(self, content: str, metadata: DocumentMetadata) -> str:
        """Add a document to memory storage."""
        try:
            # Check cache first
            cache_key = f"doc_{metadata.doc_id}"
            if self._cache_get(cache_key):
                logger.warning(
                    f"Document {
                        metadata.doc_id} already exists in cache")
                return metadata.doc_id

            # Store document
            self.documents[metadata.doc_id] = (content, metadata)

            # Generate embedding (simplified - in practice you'd use the
            # embedding model)
            embedding = self._generate_simple_embedding(content)
            self.embeddings[metadata.doc_id] = embedding

            # Update cache and stats
            self._cache_set(cache_key, {
                "content": content,
                "metadata": metadata
            })
            self._update_stats(documents_added=1, chunks_added=1)

            logger.info(f"Added document {metadata.doc_id} to memory storage")
            return metadata.doc_id

        except Exception as e:
            logger.error(f"Failed to add document {metadata.doc_id}: {e}")
            raise

    def update_document(self, doc_id: str, content: str,
                        metadata: DocumentMetadata) -> bool:
        """Update an existing document."""
        try:
            if doc_id not in self.documents:
                return False

            # Update document
            self.documents[doc_id] = (content, metadata)

            # Update embedding
            embedding = self._generate_simple_embedding(content)
            self.embeddings[doc_id] = embedding

            # Update cache
            cache_key = f"doc_{doc_id}"
            self._cache_set(cache_key, {
                "content": content,
                "metadata": metadata
            })

            logger.info(f"Updated document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            if doc_id not in self.documents:
                return False

            # Remove document and embedding
            del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]

            # Remove from cache
            cache_key = f"doc_{doc_id}"
            if self._cache_get(cache_key):
                del self.cache[cache_key]

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
               top_k: int = 10) -> List[SearchResult]:
        """Search for documents using simple similarity."""
        try:
            start_time = time.time()

            # Check query cache
            cache_key = f"query_{hash(query + str(filters) + str(top_k))}"
            cached_results = self._cache_get(cache_key)
            if cached_results:
                self.stats.cache_hit_rate = min(
                    1.0, self.stats.cache_hit_rate + 0.1)
                return cached_results

            # Generate query embedding
            query_embedding = self._generate_simple_embedding(query)

            # Calculate similarities
            similarities = []
            for doc_id, (content, metadata) in self.documents.items():
                # Apply filters
                if filters and not self._matches_filters(metadata, filters):
                    continue

                # Calculate similarity (simplified cosine similarity)
                doc_embedding = self.embeddings.get(doc_id, [])
                similarity = self._calculate_similarity(
                    query_embedding, doc_embedding)

                # Generate highlights
                highlights = self._generate_highlights(query, content)

                search_result = SearchResult(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    similarity_score=similarity,
                    rank=len(similarities) + 1,
                    highlights=highlights
                )

                similarities.append(search_result)

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            results = similarities[:top_k]

            # Cache results
            self._cache_set(cache_key, results)

            # Update query time stats
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            self.stats.average_query_time_ms = (
                (self.stats.average_query_time_ms + query_time) / 2
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_document(
            self, doc_id: str) -> Optional[Tuple[str, DocumentMetadata]]:
        """Retrieve a specific document."""
        try:
            # Check cache first
            cache_key = f"doc_{doc_id}"
            cached = self._cache_get(cache_key)
            if cached:
                return cached["content"], cached["metadata"]

            # Get from storage
            if doc_id in self.documents:
                content, metadata = self.documents[doc_id]

                # Cache the result
                self._cache_set(
                    cache_key, {
                        "content": content, "metadata": metadata})

                return content, metadata

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        self.stats.total_documents = len(self.documents)
        # In memory, chunks = documents
        self.stats.total_chunks = len(self.documents)
        self.stats.storage_size_bytes = sum(
            len(content) + len(str(metadata))
            for content, metadata in self.documents.values()
        )
        return self.stats

    def backup(self) -> bool:
        """Create a backup (for memory storage, this is a no-op)."""
        try:
            # In memory storage, backup is just a log
            logger.info(f"Memory storage backup created with {
                        len(self.documents)} documents")
            self.stats.last_backup = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def restore(self, backup_path: str) -> bool:
        """Restore from backup (for memory storage, this is a no-op)."""
        try:
            logger.info(f"Memory storage restore from {backup_path} (no-op)")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def migrate(self, target_provider: 'StorageProvider') -> bool:
        """Migrate data to another storage provider."""
        try:
            migrated_count = 0
            for doc_id, (content, metadata) in self.documents.items():
                try:
                    target_provider.add_document(content, metadata)
                    migrated_count += 1
                except Exception as e:
                    logger.error(f"Failed to migrate document {doc_id}: {e}")

            logger.info(
                f"Migrated {migrated_count} documents to target provider")
            return migrated_count == len(self.documents)

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def clear_all(self):
        """Clear all data from memory storage."""
        self.documents.clear()
        self.embeddings.clear()
        if self.cache:
            self.cache.clear()
        self.stats = StorageStats(0, 0, 0)
        logger.info("Cleared all data from memory storage")

    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding for testing."""
        # This is a very simplified embedding - in practice you'd use a proper
        # model
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to 384-dimensional vector (matching typical embedding size)
        embedding = []
        for i in range(384):
            embedding.append(hash_bytes[i % 16] / 255.0)

        return embedding

    def _calculate_similarity(
            self,
            embedding1: List[float],
            embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0

        # Ensure same length
        min_len = min(len(embedding1), len(embedding2))
        if min_len == 0:
            return 0.0

        # Calculate dot product and magnitudes
        dot_product = sum(embedding1[i] * embedding2[i]
                          for i in range(min_len))
        mag1 = sum(x * x for x in embedding1[:min_len]) ** 0.5
        mag2 = sum(x * x for x in embedding2[:min_len]) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _matches_filters(self, metadata: DocumentMetadata,
                         filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key == "doc_id" and metadata.doc_id != value:
                return False
            elif key == "source" and metadata.source != value:
                return False
            elif key == "content_type" and metadata.content_type.value != value:
                return False
            elif key == "author" and metadata.author != value:
                return False
            elif key == "tags" and value not in metadata.tags:
                return False

        return True

    def _generate_highlights(self, query: str, content: str) -> List[str]:
        """Generate highlights for search results."""
        highlights = []
        query_words = query.lower().split()

        # Simple highlight generation
        content_lower = content.lower()
        for word in query_words:
            if len(word) > 3 and word in content_lower:
                # Find the word in original case
                start_idx = content_lower.find(word)
                if start_idx != -1:
                    end_idx = start_idx + len(word)
                    highlight = content[start_idx:end_idx]
                    if highlight not in highlights:
                        highlights.append(highlight)

        return highlights[:5]  # Limit to 5 highlights
