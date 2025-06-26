# src/storage/retrieval_system.py

from .vector_store import VectorStore
from typing import List, Dict, Any, Optional

class RetrievalSystem:
    """
    Handles semantic search and retrieval from the vector store.
    Provides search, filter, and ranking capabilities.
    """
    def __init__(self):
        self.vector_store = VectorStore()

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional metadata filters and return ranked results.
        Args:
            query (str): The semantic search query.
            filters (dict, optional): Metadata filters for the search.
            top_k (int): Number of top results to return.
        Returns:
            List[Dict[str, Any]]: Ranked search results from the vector store.
        """
        return self.vector_store.query(query_text=query, filters=filters, top_k=top_k) 