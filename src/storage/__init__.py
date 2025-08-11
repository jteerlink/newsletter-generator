"""
Unified storage system for newsletter generation.

This module provides a unified interface for storing and retrieving
newsletter content, supporting both vector databases and traditional storage.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .base import (
    DataType,
    DocumentMetadata,
    SearchResult,
    StorageConfig,
    StorageError,
    StorageProvider,
    StorageType,
)
from .data_manager import DataManager
from .memory_store import MemoryStorageProvider
from .vector_store import ChromaStorageProvider

logger = logging.getLogger(__name__)

# Global storage provider instance
_storage_provider: Optional[StorageProvider] = None


def get_storage_provider(
        config: Optional[StorageConfig] = None) -> StorageProvider:
    """Get or create the global storage provider instance."""
    global _storage_provider

    if _storage_provider is None:
        if config is None:
            config = StorageConfig(
                storage_type=StorageType.CHROMA,
                db_path="./data/chroma_db",
                collection_name="newsletter_content",
                chunk_size=1000,
                chunk_overlap=100
            )

        # Try ChromaDB first, fallback to memory store
        try:
            _storage_provider = ChromaStorageProvider(config)
            if not _storage_provider.initialize():
                logger.warning(
                    "ChromaDB initialization failed, falling back to memory store")
                config.storage_type = StorageType.MEMORY
                _storage_provider = MemoryStorageProvider(config)
                if not _storage_provider.initialize():
                    raise RuntimeError(
                        "Failed to initialize both ChromaDB and memory storage")
        except Exception as e:
            logger.warning(
                f"ChromaDB failed, falling back to memory store: {e}")
            config.storage_type = StorageType.MEMORY
            _storage_provider = MemoryStorageProvider(config)
            if not _storage_provider.initialize():
                raise RuntimeError(
                    "Failed to initialize memory storage provider")

    return _storage_provider


def add_document(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None
) -> str:
    """Add a document to the storage system."""
    provider = get_storage_provider()
    return provider.add_document(content, metadata, doc_id)


def search_documents(
    query: str,
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    """Search documents in the storage system."""
    provider = get_storage_provider()
    return provider.search_documents(query, n_results, filters)


def get_document(doc_id: str) -> Optional[DocumentMetadata]:
    """Get a document by ID."""
    provider = get_storage_provider()
    return provider.get_document(doc_id)


def delete_document(doc_id: str) -> bool:
    """Delete a document by ID."""
    provider = get_storage_provider()
    return provider.delete_document(doc_id)


def list_documents(limit: int = 100) -> List[DocumentMetadata]:
    """List documents in the storage system."""
    provider = get_storage_provider()
    return provider.list_documents(limit)


def clear_storage() -> bool:
    """Clear all documents from storage."""
    provider = get_storage_provider()
    return provider.clear_storage()


def get_storage_stats() -> Dict[str, Any]:
    """Get storage system statistics."""
    provider = get_storage_provider()
    return provider.get_stats()


def add_text_to_db(
        text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Add text to the database (alias for add_document)."""
    return add_document(text, metadata)


def get_db_collection():
    """Get the database collection (returns the storage provider)."""
    return get_storage_provider()


def StorageStats():
    """Get storage statistics (alias for get_storage_stats)."""
    return get_storage_stats()


# Export main classes for direct access
__all__ = [
    'StorageProvider',
    'StorageConfig',
    'DocumentMetadata',
    'DataType',
    'SearchResult',
    'StorageError',
    'ChromaStorageProvider',
    'MemoryStorageProvider',
    'DataManager',
    'get_storage_provider',
    'add_document',
    'search_documents',
    'get_document',
    'delete_document',
    'list_documents',
    'clear_storage',
    'get_storage_stats',
    'add_text_to_db',
    'get_db_collection',
    'StorageStats'
]
