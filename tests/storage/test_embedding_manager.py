# tests/storage/test_embedding_manager.py
import pytest
import numpy as np
from src.storage.embedding_manager import EmbeddingManager

class TestEmbeddingManager:
    @pytest.fixture
    def embedding_manager(self):
        """Create an EmbeddingManager instance for testing."""
        return EmbeddingManager()

    def test_generate_embeddings_basic(self, embedding_manager):
        """Test basic embedding generation functionality."""
        texts = ["Hello world", "Test document", "Another text"]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        # Check output structure
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 384 for emb in embeddings)  # Expected dimension

    def test_generate_embeddings_consistency(self, embedding_manager):
        """Test that identical inputs produce consistent embeddings."""
        text = "Consistent test text"
        embedding1 = embedding_manager.generate_embeddings([text])[0]
        embedding2 = embedding_manager.generate_embeddings([text])[0]
        
        # With placeholder embeddings, they should be identical
        assert embedding1 == embedding2

    def test_generate_embeddings_different_inputs(self, embedding_manager):
        """Test that different inputs produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"
        embedding1 = embedding_manager.generate_embeddings([text1])[0]
        embedding2 = embedding_manager.generate_embeddings([text2])[0]
        
        # With placeholder embeddings, they should be different
        assert embedding1 != embedding2

    def test_generate_embeddings_empty_input(self, embedding_manager):
        """Test embedding generation with empty input."""
        embeddings = embedding_manager.generate_embeddings([])
        assert embeddings == []

    def test_generate_embeddings_single_text(self, embedding_manager):
        """Test embedding generation with single text input."""
        text = "Single text input"
        embeddings = embedding_manager.generate_embeddings([text])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for val in embeddings[0])

    def test_generate_embeddings_large_input(self, embedding_manager):
        """Test embedding generation with large number of texts."""
        texts = [f"Text {i}" for i in range(100)]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)

    def test_generate_embeddings_special_characters(self, embedding_manager):
        """Test embedding generation with special characters."""
        texts = [
            "Text with émojis 🚀",
            "Text with numbers 123",
            "Text with symbols @#$%",
            "Text with newlines\nand tabs\t"
        ]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 384 for emb in embeddings) 