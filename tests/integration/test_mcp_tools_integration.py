import pytest
from unittest.mock import patch, MagicMock
from mcp.tools.web_search_tool import WebSearchTool
from mcp.tools.vector_search_tool import VectorSearchTool
from mcp.server.mcp_server import MCPServer

@pytest.fixture
def mcp_server():
    server = MCPServer()
    server.register_tool('web_search', WebSearchTool())
    server.register_tool('vector_search', VectorSearchTool())
    return server

def test_web_search_tool_integration(monkeypatch, mcp_server):
    # Patch DDGS to return fake results
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def text(self, query, max_results=10): return [{"title": "Integration", "href": "http://int.com", "body": "IntegrationSnippet"}]
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", FakeDDGS)
    tool = mcp_server.tools['web_search']
    result = tool.run("integration test", max_results=1)
    assert result["results"]
    assert result["results"][0]["title"] == "Integration"
    assert result["results"][0]["url"] == "http://int.com"
    assert result["results"][0]["snippet"] == "IntegrationSnippet"
    assert result["results"][0]["source"] == "duckduckgo"

def test_vector_search_tool_integration(monkeypatch, mcp_server):
    # Patch VectorStore to return fake results
    with patch("mcp.tools.vector_search_tool.VectorStore") as MockStore:
        instance = MockStore.return_value
        instance.query.return_value = [
            {"id": "int1", "document": "DocInt1", "metadata": {}, "similarity": 0.9}
        ]
        tool = VectorSearchTool()
        mcp_server.register_tool('vector_search', tool)
        result = tool.run("integration", max_results=1, similarity_threshold=0.7)
        assert result["results"]
        assert result["results"][0]["id"] == "int1"
        assert result["results"][0]["similarity"] == 0.9 