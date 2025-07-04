import pytest
from src.storage.vector_store import VectorStore
import tempfile

def test_vector_store_add_and_query():
    text = "This is a test document. " * 50
    metadata = {"source": "unit_test", "doc_id": "test123"}
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(db_path=tmpdir)
        chunk_ids = store.add_document(text, metadata, chunk_size=50, chunk_overlap=10)
        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) > 1
        # Query for a relevant chunk
        results = store.query("test document", filters={"source": "unit_test"}, top_k=3)
        assert isinstance(results, list)
        assert any("test document" in r["document"] for r in results)


def test_vector_store_deduplicate():
    store = VectorStore(db_path=tempfile.mkdtemp())
    chunks = ["repeat chunk", "repeat chunk", "unique chunk"]
    unique_indices = store.deduplicate(chunks)
    assert set(unique_indices) == {0, 2} 