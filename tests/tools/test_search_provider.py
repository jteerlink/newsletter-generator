"""
Tests for the unified search provider interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.search_provider import (
    SearchResult, SearchQuery, UnifiedSearchProvider,
    SerperSearchProvider, DuckDuckGoSearchProvider, KnowledgeBaseSearchProvider,
    get_unified_search_provider, search_web, search_web_with_alternatives
)


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="TestSource"
        )
        
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.source == "TestSource"
        assert result.published_date is None
        assert result.relevance_score is None
        assert result.metadata is None


class TestSearchQuery:
    """Test SearchQuery dataclass."""
    
    def test_search_query_creation(self):
        """Test creating a SearchQuery."""
        query = SearchQuery(
            query="test query",
            max_results=10,
            search_type="news",
            language="en",
            region="us"
        )
        
        assert query.query == "test query"
        assert query.max_results == 10
        assert query.search_type == "news"
        assert query.language == "en"
        assert query.region == "us"
        assert query.filters is None


class TestSerperSearchProvider:
    """Test SerperSearchProvider."""
    
    @patch('src.tools.search_provider.SerperDevTool')
    def test_serper_provider_initialization(self, mock_serper_tool):
        """Test SerperSearchProvider initialization."""
        mock_serper_tool.return_value = Mock()
        
        provider = SerperSearchProvider()
        
        assert provider.name == "Serper"
        assert provider.serper_tool is not None
        assert provider.is_available() is True
    
    @patch('src.tools.search_provider.SerperDevTool')
    @patch.dict('os.environ', {'SERPER_API_KEY': 'test-key'})
    def test_serper_search_success(self, mock_serper_tool):
        """Test successful Serper search."""
        mock_tool = Mock()
        mock_tool.run.return_value = {
            'organic': [
                {
                    'title': 'Test Result',
                    'link': 'https://example.com',
                    'snippet': 'Test snippet'
                }
            ]
        }
        mock_serper_tool.return_value = mock_tool
        
        provider = SerperSearchProvider()
        query = SearchQuery(query="test query", max_results=5)
        
        results = provider.search(query)
        
        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].snippet == "Test snippet"
        assert results[0].source == "Serper"
    
    @patch('src.tools.search_provider.SerperDevTool')
    def test_serper_search_no_api_key(self, mock_serper_tool):
        """Test Serper search without API key."""
        mock_serper_tool.return_value = Mock()
        
        with patch.dict('os.environ', {}, clear=True):
            provider = SerperSearchProvider()
            query = SearchQuery(query="test query")
            
            with pytest.raises(Exception):
                provider.search(query)
    
    @patch('src.tools.search_provider.SerperDevTool')
    def test_serper_search_error(self, mock_serper_tool):
        """Test Serper search with error."""
        mock_tool = Mock()
        mock_tool.run.side_effect = Exception("API error")
        mock_serper_tool.return_value = mock_tool
        
        provider = SerperSearchProvider()
        query = SearchQuery(query="test query")
        
        with pytest.raises(Exception):
            provider.search(query)


class TestDuckDuckGoSearchProvider:
    """Test DuckDuckGoSearchProvider."""
    
    @patch('src.tools.search_provider.requests.get')
    def test_duckduckgo_search_success(self, mock_get):
        """Test successful DuckDuckGo search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'AbstractText': 'Test abstract',
            'AbstractURL': 'https://example.com',
            'AbstractSource': 'Test Source',
            'RelatedTopics': [
                {'Text': 'Related topic 1', 'FirstURL': 'https://related1.com'},
                {'Text': 'Related topic 2', 'FirstURL': 'https://related2.com'}
            ]
        }
        mock_get.return_value = mock_response
        
        provider = DuckDuckGoSearchProvider()
        query = SearchQuery(query="test query", max_results=3)
        
        results = provider.search(query)
        
        assert len(results) == 3
        assert results[0].title == "Test abstract"
        assert results[0].url == "https://example.com"
        assert results[0].source == "DuckDuckGo"
    
    @patch('src.tools.search_provider.requests.get')
    def test_duckduckgo_search_error(self, mock_get):
        """Test DuckDuckGo search with error."""
        mock_get.side_effect = Exception("Request failed")
        
        provider = DuckDuckGoSearchProvider()
        query = SearchQuery(query="test query")
        
        with pytest.raises(Exception):
            provider.search(query)


class TestKnowledgeBaseSearchProvider:
    """Test KnowledgeBaseSearchProvider."""
    
    @patch('src.tools.search_provider.search_vector_db')
    def test_knowledge_base_search_success(self, mock_search_vector_db):
        """Test successful knowledge base search."""
        mock_search_vector_db.return_value = [
            {'content': 'Test content 1', 'metadata': {'source': 'doc1'}},
            {'content': 'Test content 2', 'metadata': {'source': 'doc2'}}
        ]
        
        provider = KnowledgeBaseSearchProvider()
        query = SearchQuery(query="test query", max_results=2)
        
        results = provider.search(query)
        
        assert len(results) == 2
        assert results[0].title == "Knowledge Base Result"
        assert results[0].content == "Test content 1"
        assert results[0].source == "KnowledgeBase"
    
    @patch('src.tools.search_provider.search_vector_db')
    def test_knowledge_base_search_error(self, mock_search_vector_db):
        """Test knowledge base search with error."""
        mock_search_vector_db.side_effect = Exception("Database error")
        
        provider = KnowledgeBaseSearchProvider()
        query = SearchQuery(query="test query")
        
        with pytest.raises(Exception):
            provider.search(query)


class TestUnifiedSearchProvider:
    """Test UnifiedSearchProvider."""
    
    def test_unified_provider_initialization(self):
        """Test UnifiedSearchProvider initialization."""
        provider = UnifiedSearchProvider()
        
        assert len(provider.providers) > 0
        assert provider.default_max_results == 5
        assert provider.default_timeout == 30
    
    @patch('src.tools.search_provider.SerperSearchProvider')
    def test_unified_search_success(self, mock_serper_provider_class):
        """Test successful unified search."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.search.return_value = [
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Test snippet",
                source="Serper"
            )
        ]
        mock_serper_provider_class.return_value = mock_provider
        
        provider = UnifiedSearchProvider()
        query = SearchQuery(query="test query")
        
        results = provider.search(query)
        
        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].source == "Serper"
    
    def test_unified_search_all_providers_fail(self):
        """Test unified search when all providers fail."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.search.side_effect = Exception("All providers failed")
        
        provider = UnifiedSearchProvider(providers=[mock_provider])
        query = SearchQuery(query="test query")
        
        results = provider.search(query)
        
        assert len(results) == 0
    
    def test_unified_search_with_alternatives(self):
        """Test unified search with alternative queries."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.search.side_effect = [
            [],  # First query returns no results
            [SearchResult(
                title="Alternative Result",
                url="https://example.com",
                snippet="Alternative snippet",
                source="Serper"
            )]  # Second query returns results
        ]
        
        provider = UnifiedSearchProvider(providers=[mock_provider])
        query = SearchQuery(query="test query")
        
        results = provider.search_with_alternatives(query, ["alternative query"])
        
        assert len(results) == 1
        assert results[0].title == "Alternative Result"
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        mock_provider1 = Mock()
        mock_provider1.name = "Provider1"
        mock_provider1.is_available.return_value = True
        
        mock_provider2 = Mock()
        mock_provider2.name = "Provider2"
        mock_provider2.is_available.return_value = False
        
        provider = UnifiedSearchProvider(providers=[mock_provider1, mock_provider2])
        
        available = provider.get_available_providers()
        
        assert "Provider1" in available
        assert "Provider2" not in available


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.tools.search_provider.get_unified_search_provider')
    def test_search_web_function(self, mock_get_provider):
        """Test search_web convenience function."""
        mock_provider = Mock()
        mock_provider.search.return_value = [
            SearchResult(
                title="Test Result",
                url="https://example.com",
                snippet="Test snippet",
                source="Serper"
            )
        ]
        mock_get_provider.return_value = mock_provider
        
        results = search_web("test query", max_results=3)
        
        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        mock_provider.search.assert_called_once()
    
    @patch('src.tools.search_provider.get_unified_search_provider')
    def test_search_web_with_alternatives_function(self, mock_get_provider):
        """Test search_web_with_alternatives convenience function."""
        mock_provider = Mock()
        mock_provider.search_with_alternatives.return_value = [
            SearchResult(
                title="Alternative Result",
                url="https://example.com",
                snippet="Alternative snippet",
                source="Serper"
            )
        ]
        mock_get_provider.return_value = mock_provider
        
        results = search_web_with_alternatives("test query", ["alternative"])
        
        assert len(results) == 1
        assert results[0].title == "Alternative Result"
        mock_provider.search_with_alternatives.assert_called_once()
    
    def test_get_unified_search_provider_singleton(self):
        """Test that get_unified_search_provider returns singleton."""
        provider1 = get_unified_search_provider()
        provider2 = get_unified_search_provider()
        
        assert provider1 is provider2


class TestErrorHandling:
    """Test error handling in search providers."""
    
    @patch('src.tools.search_provider.SerperDevTool')
    def test_serper_provider_import_error(self, mock_serper_tool):
        """Test SerperSearchProvider with import error."""
        mock_serper_tool.side_effect = ImportError("Module not found")
        
        provider = SerperSearchProvider()
        
        assert provider.serper_tool is None
        assert provider.is_available() is False
    
    @patch('src.tools.search_provider.search_vector_db')
    def test_knowledge_base_search_error(self, mock_search_vector_db):
        """Test knowledge base search with error."""
        mock_search_vector_db.side_effect = Exception("Database error")
        
        provider = KnowledgeBaseSearchProvider()
        query = SearchQuery(query="test query")
        
        with pytest.raises(Exception):
            provider.search(query) 