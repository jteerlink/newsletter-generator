"""
Legacy wrapper functions for backward compatibility.

This module provides wrapper functions that maintain the same interface as the old
vector_db.py functions, allowing existing code to work with the new unified storage system.
"""

from typing import List, Dict, Any, Optional
import logging
from .base import StorageConfig, DocumentMetadata, DataType
from .vector_store import ChromaStorageProvider

logger = logging.getLogger(__name__)

# Global storage provider instance for legacy functions
_legacy_storage_provider: Optional[ChromaStorageProvider] = None

def _get_legacy_storage_provider() -> ChromaStorageProvider:
    """Get or create the legacy storage provider instance."""
    global _legacy_storage_provider
    
    if _legacy_storage_provider is None:
        config = StorageConfig(
            db_path="./data/chroma_db",
            collection_name="newsletter_content",
            chunk_size=1000,
            chunk_overlap=100
        )
        _legacy_storage_provider = ChromaStorageProvider(config)
        if not _legacy_storage_provider.initialize():
            raise RuntimeError("Failed to initialize legacy storage provider")
    
    return _legacy_storage_provider

def get_db_collection(path="./data/chroma_db", name="newsletter_content"):
    """
    Legacy function: Return a persistent ChromaDB collection object.
    
    This function is maintained for backward compatibility but now uses the
    unified storage system internally.
    
    Args:
        path (str): Path to the ChromaDB directory.
        name (str): Name of the collection.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    provider = _get_legacy_storage_provider()
    return provider.collection

def add_text_to_db(text: str, collection=None, chunk_size=1000, chunk_overlap=100):
    """
    Legacy function: Chunk the text, embed each chunk, and add to the ChromaDB collection.
    
    This function is maintained for backward compatibility but now uses the
    unified storage system internally.
    
    Args:
        text (str): The text to ingest.
        collection: The ChromaDB collection object (ignored, maintained for compatibility).
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    provider = _get_legacy_storage_provider()
    
    # Create metadata for the document
    from datetime import datetime
    import uuid
    
    metadata = DocumentMetadata(
        doc_id=str(uuid.uuid4()),
        title="Legacy Ingest",
        source="legacy_ingest",
        content_type=DataType.TEXT,
        timestamp=datetime.now(),
        author="system",
        tags=["legacy"]
    )
    
    # Add document using the unified interface
    doc_id = provider.add_document(text, metadata)
    logger.info(f"Added document {doc_id} to the vector database using legacy wrapper.")
    return doc_id

def search_vector_db(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Legacy function: Search the vector database.
    
    This function is maintained for backward compatibility but now uses the
    unified storage system internally.
    
    Args:
        query (str): The search query.
        n_results (int): Number of results to return.

    Returns:
        List[Dict[str, Any]]: List of search results with 'content', 'metadata', and 'score' keys.
    """
    provider = _get_legacy_storage_provider()
    
    # Search using the unified interface
    search_results = provider.search(query, top_k=n_results)
    
    # Convert to legacy format
    legacy_results = []
    for result in search_results:
        legacy_result = {
            'content': result.content,
            'metadata': result.metadata.to_dict() if result.metadata else {},
            'score': result.score
        }
        legacy_results.append(legacy_result)
    
    return legacy_results 