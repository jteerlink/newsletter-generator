from .base_tool import BaseTool
from storage.vector_store import VectorStore

class VectorSearchTool(BaseTool):
    """Tool for searching the local vector database for relevant content."""
    name = "vector_search"
    description = "Search the local vector database for relevant content"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic search query"},
            "max_results": {"type": "integer", "default": 20, "description": "Maximum number of results to return"},
            "similarity_threshold": {"type": "number", "default": 0.7, "description": "Minimum similarity score (0.0-1.0)"},
            "filters": {
                "type": "object",
                "properties": {
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"}
                        }
                    },
                    "source_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by source types (rss, web, api)"},
                    "topics": {"type": "array", "items": {"type": "string"}, "description": "Filter by content topics"}
                }
            }
        },
        "required": ["query"]
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.vector_store = VectorStore()

    def run(self, query, max_results=20, similarity_threshold=0.7, filters=None):
        """Run the vector search tool using the VectorStore."""
        # Convert filters to ChromaDB format if needed
        chroma_filters = {}
        if filters:
            if 'date_range' in filters:
                date_range = filters['date_range']
                if date_range.get('start_date'):
                    chroma_filters['timestamp'] = {"$gte": date_range['start_date']}
                if date_range.get('end_date'):
                    chroma_filters['timestamp'] = {"$lte": date_range['end_date']}
            if 'source_types' in filters:
                chroma_filters['source_type'] = {"$in": filters['source_types']}
            if 'topics' in filters:
                chroma_filters['topics'] = {"$in": filters['topics']}
        results = self.vector_store.query(query, filters=chroma_filters, top_k=max_results)
        # Filter by similarity threshold if available
        filtered = [r for r in results if r.get('similarity', 1.0) >= similarity_threshold]
        return {
            "results": filtered,
            "metadata": {
                "search_time": None,  # Could add timing info
                "results_count": len(filtered)
            }
        }
