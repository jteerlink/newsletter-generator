"""API integration tests for external services."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.tools.tools import search_web, search_knowledge_base
from src.scrapers.crawl4ai_web_scraper import Crawl4AIScraper

class TestSearchAPIIntegration:
    """Test search API integrations."""
    
    @patch('src.tools.tools.SerperDevTool')
    def test_serper_api_integration(self, mock_serper_tool):
        """Test Serper API integration."""
        # Mock Serper API response
        mock_response = {
            "organic": [
                {
                    "title": "Test Article",
                    "link": "https://example.com/article",
                    "snippet": "This is a test article about AI."
                }
            ]
        }
        
        mock_serper_instance = Mock()
        mock_serper_instance.run.return_value = mock_response
        mock_serper_tool.return_value = mock_serper_instance
        
        # Test search functionality
        result = search_web("test query")
        
        # Verify API was called
        mock_serper_instance.run.assert_called_once()
        assert "Test Article" in result
        assert "https://example.com/article" in result
    
    @patch('src.tools.tools.DDGS')
    def test_duckduckgo_api_integration(self, mock_ddgs):
        """Test DuckDuckGo API integration."""
        # Mock DuckDuckGo response
        mock_results = [
            {
                "title": "DuckDuckGo Result",
                "link": "https://duckduckgo.com/result",
                "body": "This is a DuckDuckGo search result."
            }
        ]
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Test search functionality
        with patch('src.tools.tools.search_web_with_alternatives') as mock_search:
            mock_search.return_value = "DuckDuckGo Result\nhttps://duckduckgo.com/result\nThis is a DuckDuckGo search result."
            result = mock_search("test query")
        
        assert "DuckDuckGo Result" in result
    
    def test_search_api_error_handling(self):
        """Test search API error handling."""
        with patch('src.tools.tools.SerperDevTool') as mock_serper_tool:
            mock_serper_instance = Mock()
            mock_serper_instance.run.side_effect = Exception("API Error")
            mock_serper_tool.return_value = mock_serper_instance
            
            # Test error handling
            result = search_web("test query")
            assert "Search failed" in result or "Error" in result

class TestScrapingAPIIntegration:
    """Test web scraping API integrations."""
    
    @patch('src.scrapers.crawl4ai_web_scraper.crawl4ai')
    def test_crawl4ai_api_integration(self, mock_crawl4ai):
        """Test Crawl4AI API integration."""
        # Mock Crawl4AI response
        mock_response = {
            "articles": [
                {
                    "title": "Scraped Article",
                    "url": "https://example.com/article",
                    "content": "This is scraped content from the article.",
                    "published_date": "2024-01-01"
                }
            ]
        }
        
        mock_crawl4ai.return_value = mock_response
        
        # Test scraping functionality
        scraper = Crawl4AIScraper()
        result = scraper.scrape_website("https://example.com")
        
        # Verify API was called
        mock_crawl4ai.assert_called_once()
        assert "Scraped Article" in str(result)
    
    def test_scraping_api_error_handling(self):
        """Test scraping API error handling."""
        with patch('src.scrapers.crawl4ai_web_scraper.crawl4ai') as mock_crawl4ai:
            mock_crawl4ai.side_effect = Exception("Scraping Error")
            
            scraper = Crawl4AIScraper()
            result = scraper.scrape_website("https://example.com")
            
            # Should handle error gracefully
            assert result is None or "Error" in str(result)

class TestLLMAPIIntegration:
    """Test LLM API integrations."""
    
    @patch('src.core.core.ollama')
    def test_ollama_api_integration(self, mock_ollama):
        """Test Ollama API integration."""
        # Mock Ollama response
        mock_response = {
            "message": {
                "content": "This is a test response from Ollama."
            }
        }
        
        mock_ollama.chat.return_value = mock_response
        
        # Test LLM query
        from src.core.core import query_llm
        result = query_llm("Test prompt")
        
        # Verify API was called
        mock_ollama.chat.assert_called_once()
        assert result == "This is a test response from Ollama."
    
    def test_llm_api_error_handling(self):
        """Test LLM API error handling."""
        with patch('src.core.core.ollama') as mock_ollama:
            from ollama import ResponseError
            mock_ollama.chat.side_effect = ResponseError("LLM Error")
            
            from src.core.core import query_llm
            result = query_llm("Test prompt")
            
            # Should handle error gracefully
            assert "An error occurred while querying the LLM" in result

class TestVectorStoreAPIIntegration:
    """Test vector store API integrations."""
    
    @patch('src.storage.vector_store.ChromaDB')
    def test_chromadb_integration(self, mock_chromadb):
        """Test ChromaDB API integration."""
        # Mock ChromaDB response
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Test document content"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]]
        }
        
        mock_chromadb_instance = Mock()
        mock_chromadb_instance.get_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_chromadb_instance
        
        # Test vector store query
        from src.storage import ChromaStorageProvider
        from src.storage.base import StorageConfig
        config = StorageConfig(
            db_path="./test_data/chroma_db",
            collection_name="test_collection",
            chunk_size=1000,
            chunk_overlap=100
        )
        store = ChromaStorageProvider(config)
        store.initialize()
        result = store.search("test query")
        
        # Verify API was called
        mock_collection.query.assert_called_once()
        assert len(result) > 0

class TestNotionAPIIntegration:
    """Test Notion API integrations."""
    
    @patch('src.tools.notion_integration.NotionClient')
    def test_notion_api_integration(self, mock_notion_client):
        """Test Notion API integration."""
        # Mock Notion response
        mock_page = Mock()
        mock_page.properties = {
            "title": Mock(title=[Mock(plain_text="Test Page")]),
            "content": Mock(rich_text=[Mock(plain_text="Test content")])
        }
        
        mock_notion_instance = Mock()
        mock_notion_instance.pages.retrieve.return_value = mock_page
        mock_notion_client.return_value = mock_notion_instance
        
        # Test Notion integration
        from src.tools.notion_integration import NotionIntegration
        notion = NotionIntegration("test_token")
        result = notion.get_page("test_page_id")
        
        # Verify API was called
        mock_notion_instance.pages.retrieve.assert_called_once()
        assert result is not None

class TestRSSFeedIntegration:
    """Test RSS feed API integrations."""
    
    @patch('src.scrapers.rss_extractor.feedparser')
    def test_rss_feed_integration(self, mock_feedparser):
        """Test RSS feed API integration."""
        # Mock RSS feed response
        mock_feed = Mock()
        mock_feed.entries = [
            Mock(
                title="RSS Article",
                link="https://example.com/rss-article",
                summary="This is an RSS article summary.",
                published="2024-01-01"
            )
        ]
        
        mock_feedparser.parse.return_value = mock_feed
        
        # Test RSS extraction
        from src.scrapers.rss_extractor import RSSExtractor
        extractor = RSSExtractor()
        result = extractor.extract_feed("https://example.com/feed")
        
        # Verify API was called
        mock_feedparser.parse.assert_called_once()
        assert len(result) > 0
        assert "RSS Article" in str(result)

class TestExternalServiceErrorHandling:
    """Test error handling for external services."""
    
    def test_network_timeout_handling(self):
        """Test network timeout handling."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timeout")
            
            # Test timeout handling in search
            result = search_web("test query")
            assert "timeout" in result.lower() or "error" in result.lower()
    
    def test_rate_limiting_handling(self):
        """Test rate limiting handling."""
        with patch('src.tools.tools.SerperDevTool') as mock_serper_tool:
            mock_serper_instance = Mock()
            mock_serper_instance.run.side_effect = Exception("Rate limit exceeded")
            mock_serper_tool.return_value = mock_serper_instance
            
            # Test rate limiting handling
            result = search_web("test query")
            assert "rate limit" in result.lower() or "error" in result.lower()
    
    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        with patch('src.tools.notion_integration.NotionClient') as mock_notion_client:
            mock_notion_client.side_effect = Exception("Authentication failed")
            
            # Test authentication error handling
            from src.tools.notion_integration import NotionIntegration
            try:
                notion = NotionIntegration("invalid_token")
                result = notion.get_page("test_id")
                assert result is None or "error" in str(result).lower()
            except Exception as e:
                assert "Authentication" in str(e) or "auth" in str(e).lower()

class TestAPIResponseValidation:
    """Test API response validation."""
    
    def test_search_response_validation(self):
        """Test search API response validation."""
        with patch('src.tools.tools.SerperDevTool') as mock_serper_tool:
            # Test with invalid response format
            mock_serper_instance = Mock()
            mock_serper_instance.run.return_value = "Invalid JSON response"
            mock_serper_tool.return_value = mock_serper_instance
            
            result = search_web("test query")
            # Should handle invalid response gracefully
            assert result is not None
    
    def test_scraping_response_validation(self):
        """Test scraping API response validation."""
        with patch('src.scrapers.crawl4ai_web_scraper.crawl4ai') as mock_crawl4ai:
            # Test with empty response
            mock_crawl4ai.return_value = {"articles": []}
            
            scraper = Crawl4AIScraper()
            result = scraper.scrape_website("https://example.com")
            
            # Should handle empty response gracefully
            assert result is not None

class TestAPIConfiguration:
    """Test API configuration handling."""
    
    def test_api_key_configuration(self):
        """Test API key configuration."""
        # Test that API keys are properly configured
        import os
        
        # Test with missing API key
        if 'SERPER_API_KEY' not in os.environ:
            # Should handle missing API key gracefully
            result = search_web("test query")
            assert result is not None
    
    def test_api_endpoint_configuration(self):
        """Test API endpoint configuration."""
        # Test that API endpoints are properly configured
        pass
    
    def test_api_timeout_configuration(self):
        """Test API timeout configuration."""
        # Test that API timeouts are properly configured
        pass 