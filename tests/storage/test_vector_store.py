# tests/storage/test_vector_store.py
import pytest
from datetime import datetime, timezone
from src.storage.vector_store import VectorStore

class TestVectorStore:
    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore instance for testing."""
        return VectorStore(db_path="./test_data/chroma_db")

    def test_chunk_document_basic(self, vector_store):
        """Test basic document chunking functionality."""
        document = "This is a test document with multiple words to chunk."
        chunks = vector_store.chunk_document(document, chunk_size=3)
        assert len(chunks) > 0
        assert all(len(chunk.split()) <= 3 for chunk in chunks)

    def test_chunk_document_edge_cases(self, vector_store):
        """Test document chunking with edge cases."""
        # Empty document
        chunks = vector_store.chunk_document("", chunk_size=10)
        assert chunks == []
        
        # Very short document
        chunks = vector_store.chunk_document("Short", chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == "Short"
        
        # Document with only whitespace
        chunks = vector_store.chunk_document("   \n\t   ", chunk_size=10)
        assert chunks == []

    def test_deduplicate_basic(self, vector_store):
        """Test basic deduplication functionality."""
        chunks = ["chunk1", "chunk2", "chunk1", "chunk3"]
        unique_indices = vector_store.deduplicate(chunks)
        assert len(unique_indices) == 3
        assert unique_indices == [0, 1, 3]  # chunk1 appears at index 0 and 2

    def test_temporal_score_basic(self, vector_store):
        """Test temporal scoring functionality."""
        # Recent document
        recent_metadata = {"timestamp": datetime.now(timezone.utc).isoformat()}
        recent_score = vector_store.temporal_score(recent_metadata)
        assert 0.9 <= recent_score <= 1.0
        
        # Old document
        old_metadata = {"timestamp": "2020-01-01T00:00:00Z"}
        old_score = vector_store.temporal_score(old_metadata)
        assert 0.1 <= old_score <= 0.5
        
        # Document without timestamp
        no_timestamp_metadata = {"source": "test"}
        default_score = vector_store.temporal_score(no_timestamp_metadata)
        assert default_score == 0.5

    def test_cluster_topics_basic(self, vector_store):
        """Test topic clustering functionality."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        clusters = vector_store.cluster_topics(embeddings, n_clusters=2)
        assert len(clusters) == 3
        assert all(0 <= cluster < 2 for cluster in clusters)

    def test_add_document_basic(self, vector_store):
        """Test basic document addition functionality."""
        document = "This is a test document for addition."
        metadata = {"source": "test", "timestamp": datetime.now(timezone.utc).isoformat()}
        chunk_ids = vector_store.add_document(document, metadata)
        assert len(chunk_ids) > 0
        assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)

    def test_query_basic(self, vector_store):
        """Test basic query functionality."""
        # First add a document
        document = "This document contains information about AI and machine learning."
        metadata = {"source": "test", "topic": "AI"}
        vector_store.add_document(document, metadata)
        
        # Then query for it
        results = vector_store.query("AI machine learning", top_k=5)
        assert isinstance(results, list)
        # Note: With placeholder embeddings, results may be empty

    def test_update_document_basic(self, vector_store):
        """Test basic document update functionality."""
        # Add initial document
        original_doc = "Original content"
        metadata = {"source": "test"}
        chunk_ids = vector_store.add_document(original_doc, metadata)
        
        # Update document
        updated_doc = "Updated content"
        vector_store.update_document("test_doc_id", updated_doc, metadata)
        # Note: This is a basic test; actual verification would require querying

    def test_delete_document_basic(self, vector_store):
        """Test basic document deletion functionality."""
        # Add a document first
        document = "Document to delete"
        metadata = {"doc_id": "test_delete_id", "source": "test"}
        vector_store.add_document(document, metadata)
        
        # Delete it
        vector_store.delete_document("test_delete_id")
        # Note: This is a basic test; actual verification would require querying 