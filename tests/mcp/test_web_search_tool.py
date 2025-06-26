import os
import pytest
from unittest.mock import patch, MagicMock
from mcp.tools.web_search_tool import WebSearchTool

@pytest.fixture
def tool():
    return WebSearchTool()

def test_duckduckgo_default(monkeypatch, tool):
    # Patch DDGS to return fake results
    fake_results = [{"title": "Test", "href": "http://example.com", "body": "Snippet"}]
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def text(self, query, max_results=10): return fake_results
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", FakeDDGS)
    result = tool.run("AI news", max_results=1)
    assert result["results"]
    assert result["results"][0]["title"] == "Test"
    assert result["results"][0]["url"] == "http://example.com"
    assert result["results"][0]["snippet"] == "Snippet"
    assert result["results"][0]["source"] == "duckduckgo"

def test_duckduckgo_no_results(monkeypatch, tool):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def text(self, query, max_results=10): return []
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", FakeDDGS)
    result = tool.run("noresults", max_results=1)
    assert result["results"] == []

def test_duckduckgo_import_missing(monkeypatch, tool):
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", None)
    result = tool.run("AI news", max_results=1)
    assert result["results"] == []

def test_google_error_fallback(monkeypatch, tool):
    os.environ["WEB_SEARCH_USE_GOOGLE"] = "true"
    os.environ["GOOGLE_SEARCH_API_KEY"] = "fake"
    os.environ["GOOGLE_SEARCH_CX"] = "fake"
    # Simulate Google API error
    def fake_requests_get(*a, **k):
        raise Exception("Google API error")
    monkeypatch.setattr("requests.get", fake_requests_get)
    # Patch DDGS to return fake results
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def text(self, query, max_results=10): return [{"title": "Duck", "href": "http://duck.com", "body": "DuckSnippet"}]
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", FakeDDGS)
    result = tool.run("AI news", max_results=1)
    assert result["results"]
    assert result["results"][0]["title"] == "Duck"
    assert result["results"][0]["source"] == "duckduckgo"
    del os.environ["WEB_SEARCH_USE_GOOGLE"]
    del os.environ["GOOGLE_SEARCH_API_KEY"]
    del os.environ["GOOGLE_SEARCH_CX"]

def test_missing_query_param(tool):
    with pytest.raises(TypeError):
        tool.run()

def test_google_only_if_enabled(monkeypatch, tool):
    # Patch requests.get to simulate Google API
    os.environ["WEB_SEARCH_USE_GOOGLE"] = "true"
    os.environ["GOOGLE_SEARCH_API_KEY"] = "fake"
    os.environ["GOOGLE_SEARCH_CX"] = "fake"
    fake_response = MagicMock()
    fake_response.json.return_value = {"items": [{"title": "GTest", "link": "http://g.com", "snippet": "GSnippet"}]}
    fake_response.raise_for_status = lambda: None
    monkeypatch.setattr("requests.get", lambda *a, **k: fake_response)
    # Patch DDGS to fail so Google is used
    monkeypatch.setattr("mcp.tools.web_search_tool.DDGS", None)
    result = tool.run("AI news", max_results=1)
    assert result["results"]
    assert result["results"][0]["title"] == "GTest"
    assert result["results"][0]["url"] == "http://g.com"
    assert result["results"][0]["snippet"] == "GSnippet"
    assert result["results"][0]["source"] == "google"
    # Clean up env
    del os.environ["WEB_SEARCH_USE_GOOGLE"]
    del os.environ["GOOGLE_SEARCH_API_KEY"]
    del os.environ["GOOGLE_SEARCH_CX"]
