"""
Tests for the unified scraper interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from src.scrapers.scraper import (
    ScrapedContent, ScrapingRequest, ScrapingProvider,
    Crawl4AIScrapingProvider, RequestsScrapingProvider, SeleniumScrapingProvider,
    UnifiedScraper, get_unified_scraper, scrape_url, async_scrape_url
)


class TestScrapedContent:
    """Test ScrapedContent dataclass."""
    
    def test_scraped_content_creation(self):
        """Test creating a ScrapedContent."""
        content = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            source="TestSource"
        )
        
        assert content.title == "Test Title"
        assert content.url == "https://example.com"
        assert content.content == "Test content"
        assert content.source == "TestSource"
        assert content.published_date is None
        assert content.author is None
        assert content.metadata is None
        assert content.word_count is None
        assert content.language is None
        assert content.tags is None
    
    def test_scraped_content_with_metadata(self):
        """Test creating a ScrapedContent with metadata."""
        published_date = datetime.now()
        content = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            published_date=published_date,
            author="Test Author",
            metadata={"key": "value"},
            word_count=10,
            language="en",
            tags=["tag1", "tag2"]
        )
        
        assert content.published_date == published_date
        assert content.author == "Test Author"
        assert content.metadata["key"] == "value"
        assert content.word_count == 10
        assert content.language == "en"
        assert content.tags == ["tag1", "tag2"]


class TestScrapingRequest:
    """Test ScrapingRequest dataclass."""
    
    def test_scraping_request_creation(self):
        """Test creating a ScrapingRequest."""
        request = ScrapingRequest(
            url="https://example.com",
            timeout=60,
            max_retries=5,
            use_javascript=True,
            extract_metadata=False,
            follow_redirects=False,
            user_agent="Test Agent"
        )
        
        assert request.url == "https://example.com"
        assert request.timeout == 60
        assert request.max_retries == 5
        assert request.use_javascript is True
        assert request.extract_metadata is False
        assert request.follow_redirects is False
        assert request.user_agent == "Test Agent"
    
    def test_scraping_request_defaults(self):
        """Test ScrapingRequest with default values."""
        request = ScrapingRequest(url="https://example.com")
        
        assert request.url == "https://example.com"
        assert request.timeout == 30
        assert request.max_retries == 3
        assert request.use_javascript is False
        assert request.extract_metadata is True
        assert request.follow_redirects is True
        assert request.user_agent is None


class TestScrapingProvider:
    """Test base ScrapingProvider."""
    
    def test_scraping_provider_initialization(self):
        """Test ScrapingProvider initialization."""
        # Create a concrete implementation for testing
        class TestProvider(ScrapingProvider):
            def _perform_scraping(self, request: ScrapingRequest) -> ScrapedContent:
                return ScrapedContent(
                    title="Test",
                    url=request.url,
                    content="Test content",
                    source=self.name
                )
        
        provider = TestProvider("TestProvider")
        assert provider.name == "TestProvider"
    
    def test_scraping_provider_abstract_method(self):
        """Test that ScrapingProvider is abstract."""
        # Test that we can't instantiate the abstract class directly
        with pytest.raises(TypeError):
            ScrapingProvider("TestProvider")


class TestCrawl4AIScrapingProvider:
    """Test Crawl4AIScrapingProvider."""
    
    @patch('src.scrapers.scraper.AsyncWebCrawler')
    def test_crawl4ai_provider_initialization(self, mock_crawler_class):
        """Test Crawl4AIScrapingProvider initialization."""
        mock_crawler = Mock()
        mock_crawler_class.return_value = mock_crawler
        
        provider = Crawl4AIScrapingProvider()
        
        assert provider.name == "Crawl4AI"
        assert provider.crawler is not None
        assert provider.is_available() is True
    
    @patch('src.scrapers.scraper.AsyncWebCrawler')
    def test_crawl4ai_provider_import_error(self, mock_crawler_class):
        """Test Crawl4AIScrapingProvider with import error."""
        mock_crawler_class.side_effect = ImportError("Module not found")
        
        provider = Crawl4AIScrapingProvider()
        
        assert provider.crawler is None
        assert provider.is_available() is False
    
    @patch('src.scrapers.scraper.AsyncWebCrawler')
    @patch('src.scrapers.scraper.asyncio.run')
    def test_crawl4ai_scraping_success(self, mock_asyncio_run, mock_crawler_class):
        """Test successful Crawl4AI scraping."""
        mock_crawler = Mock()
        mock_crawler_class.return_value = mock_crawler
        
        mock_result = {
            'markdown': 'Test content',
            'title': 'Test Title',
            'metadata': {
                'author': 'Test Author',
                'published_date': '2023-01-01T00:00:00Z',
                'language': 'en',
                'tags': ['tag1', 'tag2']
            }
        }
        mock_asyncio_run.return_value = mock_result
        
        provider = Crawl4AIScrapingProvider()
        request = ScrapingRequest(url="https://example.com")
        
        content = provider.scrape(request)
        
        assert content.title == "Test Title"
        assert content.url == "https://example.com"
        assert content.content == "Test content"
        assert content.author == "Test Author"
        assert content.language == "en"
        assert content.tags == ["tag1", "tag2"]
        assert content.source == "Crawl4AI"


class TestRequestsScrapingProvider:
    """Test RequestsScrapingProvider."""
    
    @patch('src.scrapers.scraper.requests.Session')
    @patch('src.scrapers.scraper.BeautifulSoup')
    def test_requests_scraping_success(self, mock_bs4, mock_session_class):
        """Test successful requests scraping."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.content = b"<html><title>Test Title</title><body>Test content</body></html>"
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        mock_soup = Mock()
        mock_soup.get_text.return_value = "Test content"
        mock_bs4.return_value = mock_soup
        
        provider = RequestsScrapingProvider()
        request = ScrapingRequest(url="https://example.com")
        
        content = provider.scrape(request)
        
        assert content.title == "Test Title"
        assert content.url == "https://example.com"
        assert content.content == "Test content"
        assert content.source == "Requests"
    
    @patch('src.scrapers.scraper.requests.Session')
    def test_requests_scraping_error(self, mock_session_class):
        """Test requests scraping with error."""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Request failed")
        mock_session_class.return_value = mock_session
        
        provider = RequestsScrapingProvider()
        request = ScrapingRequest(url="https://example.com")
        
        with pytest.raises(Exception):
            provider.scrape(request)


class TestSeleniumScrapingProvider:
    """Test SeleniumScrapingProvider."""
    
    @patch('src.scrapers.scraper.webdriver.Chrome')
    @patch('src.scrapers.scraper.ChromeDriverManager')
    def test_selenium_provider_initialization(self, mock_driver_manager, mock_chrome):
        """Test SeleniumScrapingProvider initialization."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        
        provider = SeleniumScrapingProvider()
        
        assert provider.name == "Selenium"
        assert provider.driver is not None
        assert provider.is_available() is True
    
    @patch('src.scrapers.scraper.webdriver.Chrome')
    @patch('src.scrapers.scraper.ChromeDriverManager')
    @patch('src.scrapers.scraper.BeautifulSoup')
    def test_selenium_scraping_success(self, mock_bs4, mock_driver_manager, mock_chrome):
        """Test successful Selenium scraping."""
        mock_driver = Mock()
        mock_driver.page_source = "<html><title>Test Title</title><body>Test content</body></html>"
        mock_chrome.return_value = mock_driver
        
        mock_soup = Mock()
        mock_soup.get_text.return_value = "Test content"
        mock_bs4.return_value = mock_soup
        
        provider = SeleniumScrapingProvider()
        request = ScrapingRequest(url="https://example.com")
        
        content = provider.scrape(request)
        
        assert content.title == "Test Title"
        assert content.url == "https://example.com"
        assert content.content == "Test content"
        assert content.source == "Selenium"
    
    @patch('src.scrapers.scraper.webdriver.Chrome')
    def test_selenium_provider_import_error(self, mock_chrome):
        """Test SeleniumScrapingProvider with import error."""
        mock_chrome.side_effect = ImportError("Module not found")
        
        provider = SeleniumScrapingProvider()
        
        assert provider.driver is None
        assert provider.is_available() is False


class TestUnifiedScraper:
    """Test UnifiedScraper."""
    
    def test_unified_scraper_initialization(self):
        """Test UnifiedScraper initialization."""
        scraper = UnifiedScraper()
        
        assert len(scraper.providers) > 0
        assert scraper.default_timeout == 30
        assert scraper.default_max_retries == 3
    
    def test_unified_scraper_with_custom_providers(self):
        """Test UnifiedScraper with custom providers."""
        mock_provider = Mock()
        mock_provider.name = "TestProvider"
        mock_provider.is_available.return_value = True
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        assert len(scraper.providers) == 1
        assert scraper.providers[0].name == "TestProvider"
    
    def test_unified_scraping_success(self):
        """Test successful unified scraping."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.scrape.return_value = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            source="TestProvider"
        )
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        content = scraper.scrape("https://example.com")
        
        assert content.title == "Test Title"
        assert content.url == "https://example.com"
        assert content.content == "Test content"
        assert content.source == "TestProvider"
    
    def test_unified_scraping_all_providers_fail(self):
        """Test unified scraping when all providers fail."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.scrape.side_effect = Exception("All providers failed")
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        content = scraper.scrape("https://example.com")
        
        assert content.title == "Failed to scrape https://example.com"
        assert content.content == ""
        assert content.source == "unified_scraper"
    
    def test_unified_scraping_provider_not_available(self):
        """Test unified scraping when provider is not available."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = False
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        content = scraper.scrape("https://example.com")
        
        assert content.title == "Failed to scrape https://example.com"
        assert content.content == ""
        assert content.source == "unified_scraper"
    
    def test_unified_scraping_multiple_urls(self):
        """Test unified scraping with multiple URLs."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.scrape.return_value = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            source="TestProvider"
        )
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        urls = ["https://example1.com", "https://example2.com"]
        results = scraper.scrape_multiple(urls)
        
        assert len(results) == 2
        for result in results:
            assert result.title == "Test Title"
            assert result.content == "Test content"
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        mock_provider1 = Mock()
        mock_provider1.name = "Provider1"
        mock_provider1.is_available.return_value = True
        
        mock_provider2 = Mock()
        mock_provider2.name = "Provider2"
        mock_provider2.is_available.return_value = False
        
        scraper = UnifiedScraper(providers=[mock_provider1, mock_provider2])
        
        available = scraper.get_available_providers()
        
        assert "Provider1" in available
        assert "Provider2" not in available


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.scrapers.scraper.get_unified_scraper')
    def test_scrape_url_function(self, mock_get_scraper):
        """Test scrape_url convenience function."""
        mock_scraper = Mock()
        mock_scraper.scrape.return_value = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            source="TestProvider"
        )
        mock_get_scraper.return_value = mock_scraper
        
        result = scrape_url("https://example.com")
        
        assert result['title'] == "Test Title"
        assert result['url'] == "https://example.com"
        assert result['content'] == "Test content"
        assert result['source'] == "TestProvider"
    
    @patch('src.scrapers.scraper.get_unified_scraper')
    def test_async_scrape_url_function(self, mock_get_scraper):
        """Test async_scrape_url convenience function."""
        mock_scraper = Mock()
        # Create an async mock that returns a ScrapedContent
        async_mock = AsyncMock()
        async_mock.return_value = ScrapedContent(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            source="TestProvider"
        )
        mock_scraper.async_scrape = async_mock
        mock_get_scraper.return_value = mock_scraper
        
        import asyncio
        result = asyncio.run(async_scrape_url("https://example.com"))
        
        assert result['title'] == "Test Title"
        assert result['url'] == "https://example.com"
        assert result['content'] == "Test content"
        assert result['source'] == "TestProvider"
    
    def test_get_unified_scraper_singleton(self):
        """Test that get_unified_scraper returns singleton."""
        scraper1 = get_unified_scraper()
        scraper2 = get_unified_scraper()
        
        assert scraper1 is scraper2


class TestErrorHandling:
    """Test error handling in scrapers."""
    
    def test_scraping_provider_retry_logic(self):
        """Test scraping provider retry logic."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.scrape.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            ScrapedContent(
                title="Success",
                url="https://example.com",
                content="Success content",
                source="TestProvider"
            )
        ]
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        content = scraper.scrape("https://example.com")
        
        assert content.title == "Success"
        assert content.content == "Success content"
    
    def test_scraping_provider_all_retries_fail(self):
        """Test scraping provider when all retries fail."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.scrape.side_effect = Exception("All attempts failed")
        
        scraper = UnifiedScraper(providers=[mock_provider])
        
        content = scraper.scrape("https://example.com")
        
        assert content.title == "Failed to scrape https://example.com"
        assert content.content == ""
        assert content.source == "unified_scraper" 