import tempfile

import pytest

from src.storage import ChromaStorageProvider
from src.storage.base import StorageConfig


def test_vector_store_add_and_query():
    text = "This is a test document. " * 50
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(
            db_path=tmpdir,
            collection_name="test_collection",
            chunk_size=50,
            chunk_overlap=10
        )
        store = ChromaStorageProvider(config)
        assert store.initialize()
        
        from datetime import datetime

        from src.storage.base import DataType, DocumentMetadata
        
        metadata = DocumentMetadata(
            doc_id="test123",
            title="Test Document",
            source="unit_test",
            content_type=DataType.TEXT,
            timestamp=datetime.now(),
            author="test",
            tags=["test"]
        )
        
        doc_id = store.add_document(text, metadata)
        assert isinstance(doc_id, str)
        
        # Query for a relevant chunk
        results = store.search("test document", top_k=3)
        assert isinstance(results, list)
        assert any("test document" in r.content for r in results)


def test_vector_store_deduplicate():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfig(
            db_path=tmpdir,
            collection_name="test_collection",
            chunk_size=50,
            chunk_overlap=10
        )
        store = ChromaStorageProvider(config)
        assert store.initialize()
        
        chunks = ["repeat chunk", "repeat chunk", "unique chunk"]
        unique_indices = store.deduplicate(chunks)
        assert set(unique_indices) == {0, 2} 