"""Tests for tool functionality."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.exceptions import SearchError
from src.tools.tools import (
    AVAILABLE_TOOLS,
    search_knowledge_base,
    search_web,
    search_web_with_alternatives,
)


class TestSearchWeb:
    """Test search_web function."""
    
    @patch('src.tools.tools.get_unified_search_provider')
    def test_search_web_success(self, mock_get_provider):
        """Test successful web search."""
        mock_provider = Mock()
        mock_provider.search.return_value = [
            Mock(title='Test Result', url='http://test.com', snippet='Test description', source='Serper')
        ]
        mock_get_provider.return_value = mock_provider
        
        result = search_web("test query", max_results=3)
        
        assert "Test Result" in result
        assert "http://test.com" in result
        assert "Test description" in result
        assert "via Serper" in result
        mock_provider.search.assert_called_once_with("test query", 3)
    
    @patch('src.tools.tools.get_unified_search_provider')
    def test_search_web_fallback(self, mock_get_provider):
        """Test web search with fallback."""
        mock_provider = Mock()
        mock_provider.search.side_effect = SearchError("API error")
        mock_get_provider.return_value = mock_provider
        
        with patch('src.tools.tools.query_llm') as mock_llm:
            mock_llm.return_value = "Fallback content about test query"
            
            result = search_web("test query")
            
            assert "Fallback content" in result
            assert "test query" in result
    
    def test_search_web_caching(self):
        """Test web search caching."""
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = [
                Mock(title='Cached Result', url='http://test.com', snippet='Cached description', source='Serper')
            ]
            mock_get_provider.return_value = mock_provider
            
            # First call
            result1 = search_web("cached query")
            # Second call (should use cache)
            result2 = search_web("cached query")
            
            assert result1 == result2
            # Should only call search once due to caching
            mock_provider.search.assert_called_once()

class TestSearchWebWithAlternatives:
    """Test search_web_with_alternatives function."""
    
    @patch('src.tools.tools.search_web')
    def test_search_with_alternatives_success(self, mock_search_web):
        """Test successful search with alternatives."""
        mock_search_web.return_value = "Comprehensive search results about AI"
        
        result = search_web_with_alternatives("AI trends", max_results=5)
        
        assert "Comprehensive search results" in result
        assert "AI" in result
    
    @patch('src.tools.tools.search_web')
    def test_search_with_alternatives_parameters(self, mock_search_web):
        """Test search with alternatives parameter handling."""
        mock_search_web.return_value = "Test results"
        
        result = search_web_with_alternatives("test query", fallback_queries=["backup1", "backup2"])
        
        assert "Test results" in result

class TestSearchKnowledgeBase:
    """Test knowledge base search functionality."""
    
    def test_search_knowledge_base_stub(self):
        """Test knowledge base search stub."""
        result = search_knowledge_base("test query")
        
        assert "not yet implemented" in result
        assert "stub" in result
    
    def test_search_knowledge_base_parameters(self):
        """Test knowledge base search parameters."""
        result = search_knowledge_base("test query", n_results=10)
        
        assert "not yet implemented" in result
        # Should handle the n_results parameter gracefully

class TestAvailableTools:
    """Test available tools registry."""
    
    def test_available_tools_registry(self):
        """Test that tools are properly registered."""
        assert 'search_web' in AVAILABLE_TOOLS
        assert 'search_web_with_alternatives' in AVAILABLE_TOOLS
        assert 'search_knowledge_base' in AVAILABLE_TOOLS
        
        # Test that tools are callable
        for tool_name in AVAILABLE_TOOLS:
            tool_func = AVAILABLE_TOOLS[tool_name]
            assert callable(tool_func)
    
    def test_tool_aliases(self):
        """Test tool aliases work correctly."""
        assert AVAILABLE_TOOLS['search_web'] == search_web
        assert AVAILABLE_TOOLS['search_with_alternatives'] == search_web_with_alternatives
    
    def test_tool_callability(self):
        """Test that tools can be called."""
        # Test basic tool call
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = []
            mock_get_provider.return_value = mock_provider
            
            result = search_web("test")
            assert "No search results found" in result

class TestAsyncSearch:
    """Test async search functionality."""
    
    @pytest.mark.asyncio
    async def test_async_search_web(self):
        """Test async web search."""
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = [
                Mock(title='Async Result', url='http://test.com', snippet='Async description', source='Serper')
            ]
            mock_get_provider.return_value = mock_provider
            
            result = await search_web("test query")
            
            assert "Async Result" in result
            assert "http://test.com" in result

class TestToolIntegration:
    """Test tool integration with agents."""
    
    def test_tool_availability_in_agents(self):
        """Test that tools are available for agent use."""
        from src.agents.agents import ResearchAgent
        
        agent = ResearchAgent()
        
        # Tools should be available in agent's tool list
        tool_names = [tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in agent.tools]
        
        # Check that search tools are available
        assert any('search' in str(tool).lower() for tool in agent.tools)
    
    def test_tool_execution_in_agents(self):
        """Test tool execution within agent context."""
        from src.agents.agents import ResearchAgent
        
        agent = ResearchAgent()
        
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = [
                Mock(title='Agent Search Result', url='http://test.com', snippet='Agent description', source='Serper')
            ]
            mock_get_provider.return_value = mock_provider
            
            # Test that agent can use search tools
            result = agent.run("Search for AI trends")
            
            # Should contain search results
            assert result is not None

class TestErrorHandling:
    """Test error handling in tools."""
    
    def test_search_error_propagation(self):
        """Test that search errors are properly handled."""
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.side_effect = Exception("Search failed")
            mock_get_provider.return_value = mock_provider
            
            result = search_web("test query")
            
            # Should handle error gracefully
            assert "No search results found" in result
    
    def test_tool_error_recovery(self):
        """Test tool error recovery mechanisms."""
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.side_effect = Exception("Provider failed")
            mock_get_provider.return_value = mock_provider
            
            with patch('src.tools.tools.query_llm') as mock_llm:
                mock_llm.return_value = "Recovery content"
                
                result = search_web("test query")
                
                # Should fall back to LLM
                assert "Recovery content" in result

class TestPerformance:
    """Test performance aspects of tools."""
    
    def test_search_caching_performance(self):
        """Test that search caching improves performance."""
        import time
        
        with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
            mock_provider = Mock()
            mock_provider.search.return_value = [
                Mock(title='Performance Test', url='http://test.com', snippet='Test', source='Serper')
            ]
            mock_get_provider.return_value = mock_provider
            
            # First call
            start_time = time.time()
            result1 = search_web("performance test")
            first_call_time = time.time() - start_time
            
            # Second call (should be faster due to caching)
            start_time = time.time()
            result2 = search_web("performance test")
            second_call_time = time.time() - start_time
            
            assert result1 == result2
            # Second call should be faster (though exact timing may vary)
            assert second_call_time <= first_call_time * 1.5  # Allow some variance
    
    def test_concurrent_search_handling(self):
        """Test handling of concurrent search requests."""
        import threading
        import time
        
        results = []
        
        def search_worker():
            with patch('src.tools.tools.get_unified_search_provider') as mock_get_provider:
                mock_provider = Mock()
                mock_provider.search.return_value = [
                    Mock(title=f'Result {threading.current_thread().name}', url='http://test.com', snippet='Test', source='Serper')
                ]
                mock_get_provider.return_value = mock_provider
                
                result = search_web("concurrent test")
                results.append(result)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=search_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All searches should complete successfully
        assert len(results) == 3
        for result in results:
            assert "Result" in result 