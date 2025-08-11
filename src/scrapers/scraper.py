"""
Consolidated Scraper Interface

This module provides a unified interface for web scraping functionality,
consolidating different scraping implementations into a single, standardized interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Standardized scraped content structure."""
    title: str
    url: str
    content: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class ScrapingRequest:
    """Scraping request with configuration."""
    url: str
    timeout: int = 30
    max_retries: int = 3
    use_javascript: bool = False
    extract_metadata: bool = True
    follow_redirects: bool = True
    user_agent: Optional[str] = None


class ScrapingProvider(ABC):
    """Abstract base class for scraping providers."""

    def __init__(
            self,
            name: str,
            max_retries: int = 3,
            retry_delay: float = 1.0):
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0
        self.rate_limit_delay = 0.1

    @abstractmethod
    def _perform_scraping(self, request: ScrapingRequest) -> ScrapedContent:
        """Perform the actual scraping. To be implemented by subclasses."""
        pass

    def scrape(self, request: ScrapingRequest) -> ScrapedContent:
        """Perform scraping with retry logic and rate limiting."""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self._rate_limit()

                content = self._perform_scraping(request)
                logger.info(
                    f"Scraping successful with {
                        self.name}: {
                        content.title}")
                return content

            except Exception as e:
                logger.warning(
                    f"Scraping attempt {
                        attempt +
                        1} failed with {
                        self.name}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(
                        f"All scraping attempts failed with {
                            self.name}")
                    raise

    async def async_scrape(self, request: ScrapingRequest) -> ScrapedContent:
        """Async wrapper for scraping."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scrape, request)

    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def is_available(self) -> bool:
        """Check if the provider is available. Override in subclasses."""
        return True


class Crawl4AIScrapingProvider(ScrapingProvider):
    """Crawl4AI-based scraping provider."""

    def __init__(self, use_llm_extraction: bool = False,
                 llm_provider: str = "ollama/deepseek-r1"):
        super().__init__("Crawl4AI")
        self.use_llm_extraction = use_llm_extraction
        self.llm_provider = llm_provider
        self.crawler = None
        self._initialize_crawler()

    def _initialize_crawler(self):
        """Initialize Crawl4AI crawler."""
        try:
            from crawl4ai import AsyncWebCrawler
            self.crawler = AsyncWebCrawler()
            logger.info("Crawl4AI crawler initialized successfully")
        except ImportError:
            logger.warning(
                "Crawl4AI not available. Install with: pip install crawl4ai")
            self.crawler = None
        except Exception as e:
            logger.error(f"Failed to initialize Crawl4AI crawler: {e}")
            self.crawler = None

    def _perform_scraping(self, request: ScrapingRequest) -> ScrapedContent:
        """Perform scraping using Crawl4AI."""
        if not self.crawler:
            raise RuntimeError("Crawl4AI crawler not initialized")

        try:
            # Create run configuration
            run_config = {
                "url": request.url,
                "timeout": request.timeout,
                "follow_redirects": request.follow_redirects,
                "user_agent": request.user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            # Add LLM extraction if enabled
            if self.use_llm_extraction:
                run_config["llm_extraction_strategy"] = {
                    "provider": self.llm_provider,
                    "extract_metadata": request.extract_metadata
                }

            # Run crawler
            result = asyncio.run(self.crawler.arun(run_config))

            # Parse result
            return self._parse_crawl4ai_result(result, request)

        except Exception as e:
            logger.error(f"Crawl4AI scraping error: {e}")
            raise

    def _parse_crawl4ai_result(
            self, result: Dict[str, Any], request: ScrapingRequest) -> ScrapedContent:
        """Parse Crawl4AI result to standard format."""
        # Extract content
        content = result.get(
            'markdown',
            '') or result.get(
            'text',
            '') or result.get(
            'html',
            '')

        # Extract metadata
        metadata = result.get('metadata', {})
        title = metadata.get(
            'title',
            '') or result.get(
            'title',
            '') or f'Content from {
            request.url}'

        # Extract published date
        published_date = None
        if metadata.get('published_date'):
            try:
                published_date = datetime.fromisoformat(
                    metadata['published_date'].replace('Z', '+00:00'))
            except BaseException:
                pass

        # Extract author
        author = metadata.get('author') or metadata.get('byline')

        # Calculate word count
        word_count = len(content.split()) if content else 0

        return ScrapedContent(
            title=title,
            url=request.url,
            content=content,
            published_date=published_date,
            author=author,
            source=self.name,
            metadata=metadata,
            word_count=word_count,
            language=metadata.get('language'),
            tags=metadata.get('tags', [])
        )

    def is_available(self) -> bool:
        """Check if Crawl4AI provider is available."""
        return self.crawler is not None


class RequestsScrapingProvider(ScrapingProvider):
    """Simple requests-based scraping provider."""

    def __init__(self):
        super().__init__("Requests")

    def _perform_scraping(self, request: ScrapingRequest) -> ScrapedContent:
        """Perform scraping using requests and BeautifulSoup."""
        try:
            import requests
            from bs4 import BeautifulSoup

            # Setup session
            session = requests.Session()
            session.headers.update({
                'User-Agent': request.user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            })

            # Make request
            response = session.get(
                request.url,
                timeout=request.timeout,
                allow_redirects=request.follow_redirects
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract content
            content = self._extract_content(soup)

            # Extract metadata
            metadata = {}
            if request.extract_metadata:
                metadata = self._extract_metadata(soup)

            return ScrapedContent(
                title=metadata.get('title', f'Content from {request.url}'),
                url=request.url,
                content=content,
                published_date=metadata.get('published_date'),
                author=metadata.get('author'),
                source=self.name,
                metadata=metadata,
                word_count=len(content.split()) if content else 0,
                language=metadata.get('language'),
                tags=metadata.get('tags', [])
            )

        except Exception as e:
            logger.error(f"Requests scraping error: {e}")
            raise

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find main content areas
        content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            'main',
            '.main-content'
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)

        # Fallback to body text
        return soup.get_text(separator=' ', strip=True)

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from BeautifulSoup object."""
        metadata = {}

        # Extract title
        title_elem = soup.find('title')
        if title_elem:
            metadata['title'] = title_elem.get_text(strip=True)

        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')

            if name and content:
                if 'author' in name.lower():
                    metadata['author'] = content
                elif 'description' in name.lower():
                    metadata['description'] = content
                elif 'language' in name.lower():
                    metadata['language'] = content
                elif 'keywords' in name.lower():
                    metadata['tags'] = [tag.strip()
                                        for tag in content.split(',')]

        # Extract published date
        date_selectors = [
            'time[datetime]',
            '.published',
            '.date',
            '.timestamp'
        ]

        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_str = date_elem.get(
                    'datetime') or date_elem.get_text(strip=True)
                try:
                    metadata['published_date'] = datetime.fromisoformat(
                        date_str.replace('Z', '+00:00'))
                except BaseException:
                    pass
                break

        return metadata


class SeleniumScrapingProvider(ScrapingProvider):
    """Selenium-based scraping provider for JavaScript-heavy sites."""

    def __init__(self):
        super().__init__("Selenium")
        self.driver = None
        self._initialize_driver()

    def _initialize_driver(self):
        """Initialize Selenium WebDriver."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")

            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(
                    ChromeDriverManager().install()),
                options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")

        except ImportError:
            logger.warning(
                "Selenium not available. Install with: pip install selenium webdriver-manager")
            self.driver = None
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            self.driver = None

    def _perform_scraping(self, request: ScrapingRequest) -> ScrapedContent:
        """Perform scraping using Selenium."""
        if not self.driver:
            raise RuntimeError("Selenium WebDriver not initialized")

        try:
            # Set page load timeout
            self.driver.set_page_load_timeout(request.timeout)

            # Navigate to URL
            self.driver.get(request.url)

            # Wait for page to load
            time.sleep(2)

            # Get page source
            page_source = self.driver.page_source

            # Parse with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')

            # Extract content and metadata (similar to
            # RequestsScrapingProvider)
            content = self._extract_content(soup)
            metadata = self._extract_metadata(
                soup) if request.extract_metadata else {}

            return ScrapedContent(
                title=metadata.get('title', f'Content from {request.url}'),
                url=request.url,
                content=content,
                published_date=metadata.get('published_date'),
                author=metadata.get('author'),
                source=self.name,
                metadata=metadata,
                word_count=len(content.split()) if content else 0,
                language=metadata.get('language'),
                tags=metadata.get('tags', [])
            )

        except Exception as e:
            logger.error(f"Selenium scraping error: {e}")
            raise

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object."""
        # Same implementation as RequestsScrapingProvider
        for script in soup(["script", "style"]):
            script.decompose()

        content_selectors = [
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            'main',
            '.main-content'
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)

        return soup.get_text(separator=' ', strip=True)

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from BeautifulSoup object."""
        # Same implementation as RequestsScrapingProvider
        metadata = {}

        title_elem = soup.find('title')
        if title_elem:
            metadata['title'] = title_elem.get_text(strip=True)

        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')

            if name and content:
                if 'author' in name.lower():
                    metadata['author'] = content
                elif 'description' in name.lower():
                    metadata['description'] = content
                elif 'language' in name.lower():
                    metadata['language'] = content
                elif 'keywords' in name.lower():
                    metadata['tags'] = [tag.strip()
                                        for tag in content.split(',')]

        date_selectors = [
            'time[datetime]',
            '.published',
            '.date',
            '.timestamp'
        ]

        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_str = date_elem.get(
                    'datetime') or date_elem.get_text(strip=True)
                try:
                    metadata['published_date'] = datetime.fromisoformat(
                        date_str.replace('Z', '+00:00'))
                except BaseException:
                    pass
                break

        return metadata

    def cleanup(self):
        """Clean up Selenium WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except BaseException:
                pass
            self.driver = None

    def is_available(self) -> bool:
        """Check if Selenium provider is available."""
        return self.driver is not None


class UnifiedScraper:
    """Unified scraper that combines multiple scraping providers."""

    def __init__(self, providers: Optional[List[ScrapingProvider]] = None):
        self.providers = providers or self._get_default_providers()
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes

    def _get_default_providers(self) -> List[ScrapingProvider]:
        """Get default scraping providers in order of preference."""
        providers = []

        # Try Crawl4AI first (most advanced)
        crawl4ai_provider = Crawl4AIScrapingProvider()
        if crawl4ai_provider.is_available():
            providers.append(crawl4ai_provider)

        # Add Selenium for JavaScript-heavy sites
        selenium_provider = SeleniumScrapingProvider()
        if selenium_provider.is_available():
            providers.append(selenium_provider)

        # Add requests as fallback
        providers.append(RequestsScrapingProvider())

        return providers

    def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Perform unified scraping."""
        request = ScrapingRequest(url=url, **kwargs)

        # Try each provider in order
        for provider in self.providers:
            if not provider.is_available():
                continue

            try:
                content = provider.scrape(request)
                if content and content.content.strip():
                    logger.info(f"Scraping successful with {provider.name}")
                    return content
            except Exception as e:
                logger.warning(f"Scraping failed with {provider.name}: {e}")
                continue

        # If all providers fail, return empty content
        logger.error("All scraping providers failed")
        return ScrapedContent(
            title=f"Failed to scrape {url}",
            url=url,
            content="",
            source="unified_scraper"
        )

    async def async_scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Perform async unified scraping."""
        request = ScrapingRequest(url=url, **kwargs)

        # Try each provider in order
        for provider in self.providers:
            if not provider.is_available():
                continue

            try:
                content = await provider.async_scrape(request)
                if content and content.content.strip():
                    logger.info(
                        f"Async scraping successful with {
                            provider.name}")
                    return content
            except Exception as e:
                logger.warning(
                    f"Async scraping failed with {
                        provider.name}: {e}")
                continue

        # If all providers fail, return empty content
        logger.error("All async scraping providers failed")
        return ScrapedContent(
            title=f"Failed to scrape {url}",
            url=url,
            content="",
            source="unified_scraper"
        )

    def scrape_multiple(
            self,
            urls: List[str],
            **kwargs) -> List[ScrapedContent]:
        """Scrape multiple URLs."""
        results = []
        for url in urls:
            try:
                content = self.scrape(url, **kwargs)
                results.append(content)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append(ScrapedContent(
                    title=f"Failed to scrape {url}",
                    url=url,
                    content="",
                    source="unified_scraper"
                ))
        return results

    async def async_scrape_multiple(self, urls: List[str], **kwargs) -> List[ScrapedContent]:
        """Async scrape multiple URLs."""
        tasks = [self.async_scrape(url, **kwargs) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [
            provider.name for provider in self.providers if provider.is_available()]

    def cleanup(self):
        """Clean up resources."""
        for provider in self.providers:
            if hasattr(provider, 'cleanup'):
                provider.cleanup()


# Global instance for easy access
_unified_scraper = None


def get_unified_scraper() -> UnifiedScraper:
    """Get the global unified scraper instance."""
    global _unified_scraper
    if _unified_scraper is None:
        _unified_scraper = UnifiedScraper()
    return _unified_scraper


# Convenience functions for backward compatibility
def scrape_url(url: str, **kwargs) -> Dict[str, Any]:
    """Scrape URL and return dictionary format."""
    scraper = get_unified_scraper()
    content = scraper.scrape(url, **kwargs)

    return {
        'title': content.title,
        'url': content.url,
        'content': content.content,
        'published_date': content.published_date.isoformat() if content.published_date else None,
        'author': content.author,
        'source': content.source,
        'metadata': content.metadata,
        'word_count': content.word_count,
        'language': content.language,
        'tags': content.tags}


async def async_scrape_url(url: str, **kwargs) -> Dict[str, Any]:
    """Async scrape URL and return dictionary format."""
    scraper = get_unified_scraper()
    content = await scraper.async_scrape(url, **kwargs)

    return {
        'title': content.title,
        'url': content.url,
        'content': content.content,
        'published_date': content.published_date.isoformat() if content.published_date else None,
        'author': content.author,
        'source': content.source,
        'metadata': content.metadata,
        'word_count': content.word_count,
        'language': content.language,
        'tags': content.tags}
