"""
Unified Storage Interface

This module provides a unified interface for all storage operations including
vector databases, document storage, and data management capabilities.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageType(Enum):
    """Types of storage backends."""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MEMORY = "memory"


class DataType(Enum):
    """Types of data that can be stored."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class StorageConfig:
    """Configuration for storage backends."""
    storage_type: StorageType
    db_path: str = "./data/storage"
    collection_name: str = "newsletter_content"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    enable_cache: bool = True
    cache_size: int = 1000
    enable_backup: bool = True
    backup_interval: int = 24  # hours


@dataclass
class DocumentMetadata:
    """Standardized document metadata."""
    doc_id: str
    title: str
    source: str
    content_type: DataType
    timestamp: datetime
    author: Optional[str] = None
    tags: List[str] = None
    version: str = "1.0"
    checksum: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.checksum is None:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate a simple checksum for the document."""
        import hashlib
        content = f"{self.title}{self.source}{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class SearchResult:
    """Standardized search result."""
    doc_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float
    rank: int
    highlights: List[str] = None

    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


@dataclass
class StorageStats:
    """Storage statistics."""
    total_documents: int
    total_chunks: int
    storage_size_bytes: int
    last_backup: Optional[datetime] = None
    cache_hit_rate: float = 0.0
    average_query_time_ms: float = 0.0


class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self.cache = {} if config.enable_cache else None
        self.stats = StorageStats(0, 0, 0)

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: DocumentMetadata) -> str:
        """Add a document to storage."""
        pass

    @abstractmethod
    def update_document(self, doc_id: str, content: str,
                        metadata: DocumentMetadata) -> bool:
        """Update an existing document."""
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
               top_k: int = 10) -> List[SearchResult]:
        """Search for documents."""
        pass

    @abstractmethod
    def get_document(
            self, doc_id: str) -> Optional[Tuple[str, DocumentMetadata]]:
        """Retrieve a specific document."""
        pass

    @abstractmethod
    def get_stats(self) -> StorageStats:
        """Get storage statistics."""
        pass

    @abstractmethod
    def backup(self) -> bool:
        """Create a backup of the storage."""
        pass

    @abstractmethod
    def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        pass

    @abstractmethod
    def migrate(self, target_provider: 'StorageProvider') -> bool:
        """Migrate data to another storage provider."""
        pass

    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())

    def _update_stats(self, documents_added: int = 0, chunks_added: int = 0):
        """Update storage statistics."""
        self.stats.total_documents += documents_added
        self.stats.total_chunks += chunks_added

    def _cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.cache is not None:
            return self.cache.get(key)
        return None

    def _cache_set(self, key: str, value: Any):
        """Set value in cache."""
        if self.cache is not None:
            if len(self.cache) >= self.config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value


class StorageManager:
    """Manages multiple storage providers and provides unified interface."""

    def __init__(self, primary_config: StorageConfig,
                 backup_configs: List[StorageConfig] = None):
        self.primary_config = primary_config
        self.backup_configs = backup_configs or []
        self.primary_provider = self._create_provider(primary_config)
        self.backup_providers = [self._create_provider(config)
                                 for config in self.backup_configs]
        self.providers = [self.primary_provider] + self.backup_providers

    def _create_provider(self, config: StorageConfig) -> StorageProvider:
        """Create a storage provider based on configuration."""
        if config.storage_type == StorageType.CHROMA:
            from .vector_store import ChromaStorageProvider
            return ChromaStorageProvider(config)
        elif config.storage_type == StorageType.MEMORY:
            from .memory_store import MemoryStorageProvider
            return MemoryStorageProvider(config)
        else:
            raise ValueError(
                f"Unsupported storage type: {
                    config.storage_type}")

    def initialize_all(self) -> bool:
        """Initialize all storage providers."""
        try:
            for provider in self.providers:
                if not provider.initialize():
                    return False
            return True
        except Exception as e:
            print(f"Error initializing storage providers: {e}")
            return False

    def add_document(self, content: str, metadata: DocumentMetadata) -> str:
        """Add document to primary storage and replicate to backups."""
        doc_id = self.primary_provider.add_document(content, metadata)

        # Replicate to backup providers
        for provider in self.backup_providers:
            try:
                provider.add_document(content, metadata)
            except Exception as e:
                print(f"Warning: Failed to replicate to backup provider: {e}")

        return doc_id

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None,
               top_k: int = 10) -> List[SearchResult]:
        """Search across all providers and merge results."""
        all_results = []

        for provider in self.providers:
            try:
                results = provider.search(query, filters, top_k)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Search failed for provider: {e}")

        # Merge and deduplicate results
        return self._merge_search_results(all_results, top_k)

    def _merge_search_results(
            self,
            results: List[SearchResult],
            top_k: int) -> List[SearchResult]:
        """Merge and deduplicate search results."""
        # Group by doc_id and take the best score
        doc_results = {}
        for result in results:
            if result.doc_id not in doc_results or result.similarity_score > doc_results[
                    result.doc_id].similarity_score:
                doc_results[result.doc_id] = result

        # Sort by similarity score and return top_k
        sorted_results = sorted(doc_results.values(),
                                key=lambda x: x.similarity_score, reverse=True)
        return sorted_results[:top_k]

    def backup_all(self) -> bool:
        """Create backups of all storage providers."""
        success = True
        for provider in self.providers:
            try:
                if not provider.backup():
                    success = False
            except Exception as e:
                print(f"Warning: Backup failed for provider: {e}")
                success = False
        return success

    def get_combined_stats(self) -> StorageStats:
        """Get combined statistics from all providers."""
        combined_stats = StorageStats(0, 0, 0)

        for provider in self.providers:
            try:
                stats = provider.get_stats()
                combined_stats.total_documents += stats.total_documents
                combined_stats.total_chunks += stats.total_chunks
                combined_stats.storage_size_bytes += stats.storage_size_bytes
                combined_stats.cache_hit_rate = max(
                    combined_stats.cache_hit_rate, stats.cache_hit_rate)
                combined_stats.average_query_time_ms = max(
                    combined_stats.average_query_time_ms, stats.average_query_time_ms)
            except Exception as e:
                print(f"Warning: Failed to get stats from provider: {e}")

        return combined_stats
