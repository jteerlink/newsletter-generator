# src/storage/embedding_manager.py

from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """
    Handles embedding generation and management for documents using sentence-transformers.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using the loaded sentence-transformers model.
        Returns a list of 384-dimensional vectors (default for all-MiniLM-L6-v2).
        """
        if not texts:
            return []
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}") 