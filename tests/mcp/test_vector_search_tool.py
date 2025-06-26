import pytest
from unittest.mock import patch, MagicMock
from mcp.tools.vector_search_tool import VectorSearchTool

@pytest.fixture
def tool():
    with patch("mcp.tools.vector_search_tool.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [
            {"id": "1", "document": "Doc1", "metadata": {}, "similarity": 0.8},
            {"id": "2", "document": "Doc2", "metadata": {}, "similarity": 0.6}
        ]
        yield VectorSearchTool()

def test_similarity_threshold(tool):
    # Only results >= 0.7 should be returned
    result = tool.run("test", max_results=2, similarity_threshold=0.7)
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "1"
    assert result["results"][0]["similarity"] >= 0.7

def test_formatting(tool):
    result = tool.run("test", max_results=2, similarity_threshold=0.5)
    assert "results" in result
    assert "metadata" in result
    assert result["metadata"]["results_count"] == 2

def test_no_results(monkeypatch):
    with patch("mcp.tools.vector_search_tool.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = []
        tool = VectorSearchTool()
        result = tool.run("test", max_results=2, similarity_threshold=0.5)
        assert result["results"] == []

def test_all_below_threshold(monkeypatch):
    with patch("mcp.tools.vector_search_tool.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [
            {"id": "1", "document": "Doc1", "metadata": {}, "similarity": 0.4},
            {"id": "2", "document": "Doc2", "metadata": {}, "similarity": 0.3}
        ]
        tool = VectorSearchTool()
        result = tool.run("test", max_results=2, similarity_threshold=0.5)
        assert result["results"] == []

def test_missing_query_param(tool):
    with pytest.raises(TypeError):
        tool.run()

def test_filters_passed(monkeypatch):
    with patch("mcp.tools.vector_search_tool.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [
            {"id": "1", "document": "Doc1", "metadata": {}, "similarity": 0.8}
        ]
        tool = VectorSearchTool()
        filters = {"date_range": {"start_date": "2024-01-01", "end_date": "2024-01-31"}, "source_types": ["rss"], "topics": ["AI"]}
        result = tool.run("test", max_results=1, similarity_threshold=0.7, filters=filters)
        # Check that the result is as expected
        assert result["results"][0]["id"] == "1"
