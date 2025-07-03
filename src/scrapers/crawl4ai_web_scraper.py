"""
Crawl4AI-based web scraper for websites without RSS feeds
Enhanced with AI-powered content extraction and better JavaScript handling
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import time
from urllib.parse import urljoin, urlparse
import json
import re

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.cache_context import CacheMode

from src.scrapers.config_loader import SourceConfig
from src.scrapers.rss_extractor import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Crawl4AiWebScraper:
    """AI-powered web scraper using crawl4ai"""

    def __init__(
        self, 
        timeout: int = 30, 
        max_retries: int = 3, 
        headless: bool = True,
        use_llm_extraction: bool = False,
        llm_provider: str = "ollama/llama3",
        max_concurrent: int = 5
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.headless = headless
        self.use_llm_extraction = use_llm_extraction
        self.llm_provider = llm_provider
        self.max_concurrent = max_concurrent
        
        # Browser configuration
        self.browser_config = BrowserConfig(
            headless=self.headless,
            # Add additional browser configurations as needed
        )
        
        # Initialize crawler (will be done in async context)
        self.crawler = None

    async def _ensure_crawler(self):
        """Ensure crawler is initialized"""
        if self.crawler is None:
            self.crawler = AsyncWebCrawler(
                config=self.browser_config
            )
            await self.crawler.astart()

    async def extract_from_source(self, source: SourceConfig) -> List[Article]:
        """Extract articles from a single website source"""
        logger.info(f"Scraping website with crawl4ai: {source.name}")
        
        await self._ensure_crawler()
        
        for attempt in range(self.max_retries):
            try:
                articles = await self._crawl_and_extract(source)
                logger.info(f"Extracted {len(articles)} articles from {source.name}")
                return articles
                
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for {source.name}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        logger.error(
            f"Failed to scrape {source.name} after {self.max_retries} attempts"
        )
        return []

    async def _crawl_and_extract(self, source: SourceConfig) -> List[Article]:
        """Crawl a source and extract articles using crawl4ai"""
        try:
            # Configure the crawler run
            run_config = self._create_run_config(source)
            
            # Perform the crawl
            result = await self.crawler.arun(
                url=source.url,
                config=run_config
            )
            
            if not result.success:
                logger.error(f"Crawl failed for {source.name}: {result.error_message}")
                return []
            
            # Extract articles from the result
            articles = await self._extract_articles_from_result(result, source)
            
            # Remove duplicates based on URL
            return self._remove_duplicate_articles(articles)
            
        except Exception as e:
            logger.error(f"Error crawling {source.name}: {e}")
            return []

    def _create_run_config(self, source: SourceConfig) -> CrawlerRunConfig:
        """Create crawler run configuration based on source"""
        # Create markdown generator with content filtering
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter()
        )
        
        # Create extraction strategy
        extraction_strategy = None
        
        if self.use_llm_extraction:
            # Use LLM for intelligent article extraction
            extraction_strategy = self._create_llm_extraction_strategy()
        else:
            # Use CSS selectors for article extraction
            extraction_strategy = self._create_css_extraction_strategy(source)
        
        return CrawlerRunConfig(
            markdown_generator=markdown_generator,
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS,  # Always get fresh content
            wait_for="body",  # Wait for page to load
        )

    def _create_css_extraction_strategy(self, source: SourceConfig) -> JsonCssExtractionStrategy:
        """Create CSS-based extraction strategy"""
        # Enhanced selectors based on source configuration
        article_selectors = [
            source.selector if source.selector else "h1 a, h2 a, h3 a",
            "article h2 a, article h3 a",
            ".post-title a, .article-title a",
            ".entry-title a, .news-title a",
            ".headline a, .story-headline a",
            ".card-title a, .item-title a"
        ]
        
        # Schema for extracting article data
        schema = {
            "articles": [
                {
                    "title": {"selector": "text", "attr": "text"},
                    "url": {"selector": "attr", "attr": "href"},
                    "description": {
                        "selector": "parent::*",
                        "nested": {
                            "description": {"selector": "p, .excerpt, .summary", "attr": "text"}
                        }
                    },
                    "date": {
                        "selector": "parent::*",
                        "nested": {
                            "date": {"selector": "time, .date, .published", "attr": "text"}
                        }
                    },
                    "author": {
                        "selector": "parent::*",
                        "nested": {
                            "author": {"selector": ".author, .byline", "attr": "text"}
                        }
                    }
                }
            ]
        }
        
        # Create selector string combining all possibilities
        combined_selector = ", ".join(article_selectors)
        
        return JsonCssExtractionStrategy(schema={
            "articles": combined_selector
        })

    def _create_llm_extraction_strategy(self) -> LLMExtractionStrategy:
        """Create LLM-based extraction strategy for intelligent content extraction"""
        try:
            from crawl4ai.models import LLMConfig
            
            llm_config = LLMConfig(
                provider=self.llm_provider,
                # api_token can be set via environment variables if needed
            )
            
            instruction = """
            Extract all news articles, blog posts, or content items from this webpage.
            For each item, provide:
            - title: The article title
            - url: The link to the full article (make absolute URLs)
            - description: Brief description or excerpt if available
            - published_date: Publication date if found
            - author: Author name if available
            
            Return the results as a JSON array of objects.
            Only include substantial content items, not navigation links or ads.
            """
            
            return LLMExtractionStrategy(
                llm_config=llm_config,
                instruction=instruction
            )
            
        except ImportError:
            logger.warning("LLM extraction not available, falling back to CSS extraction")
            return None

    async def _extract_articles_from_result(self, result, source: SourceConfig) -> List[Article]:
        """Extract articles from crawl4ai result"""
        articles = []
        
        # First try structured extraction if available
        if result.extracted_content:
            try:
                extracted_data = json.loads(result.extracted_content)
                articles.extend(self._parse_extracted_data(extracted_data, source))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse extracted content: {e}")
        
        # Fallback to link extraction from markdown/html
        if not articles:
            articles = await self._extract_from_links(result, source)
        
        # Enhance articles with full content if needed
        articles = await self._enhance_articles_with_content(articles)
        
        return articles

    def _parse_extracted_data(self, data: Dict, source: SourceConfig) -> List[Article]:
        """Parse structured data extracted by crawl4ai"""
        articles = []
        
        # Handle different data structures
        article_list = []
        if isinstance(data, list):
            article_list = data
        elif isinstance(data, dict):
            if "articles" in data:
                article_list = data["articles"]
            else:
                article_list = [data]
        
        for item in article_list:
            article = self._create_article_from_data(item, source)
            if article:
                articles.append(article)
        
        return articles

    def _create_article_from_data(self, data: Dict, source: SourceConfig) -> Optional[Article]:
        """Create Article object from extracted data"""
        try:
            title = data.get("title", "").strip()
            if not title or len(title) < 10:
                return None
            
            url = data.get("url", "").strip()
            if not url:
                return None
            
            # Make URL absolute
            if not url.startswith(("http://", "https://")):
                url = urljoin(source.url, url)
            
            description = data.get("description", "").strip()
            
            # Parse date
            published = self._parse_date_string(data.get("published_date", ""))
            
            # Extract other metadata
            author = data.get("author", "").strip() or None
            
            return Article(
                title=title,
                url=url,
                description=description,
                published=published,
                source=source.name,
                category=source.category,
                author=author,
                source_type="website",
                fetch_status="success"
            )
            
        except Exception as e:
            logger.warning(f"Failed to create article from data: {e}")
            return None

    async def _extract_from_links(self, result, source: SourceConfig) -> List[Article]:
        """Extract articles from links found in the page"""
        articles = []
        
        # Use the links extracted by crawl4ai
        internal_links = result.links.internal if hasattr(result, 'links') else []
        
        for link in internal_links:
            if self._is_likely_article_link(link, source):
                article = Article(
                    title=link.text.strip() if link.text else "No Title",
                    url=link.href,
                    description="",
                    published=None,
                    source=source.name,
                    category=source.category,
                    source_type="website",
                    fetch_status="partial"
                )
                articles.append(article)
        
        return articles

    def _is_likely_article_link(self, link, source: SourceConfig) -> bool:
        """Determine if a link is likely to be an article"""
        if not link.text or len(link.text.strip()) < 10:
            return False
        
        # Skip common navigation links
        nav_keywords = {
            "home", "about", "contact", "privacy", "terms", "login", "register",
            "subscribe", "newsletter", "rss", "feed", "archive", "category",
            "tag", "search", "more", "next", "previous", "page"
        }
        
        text_lower = link.text.lower()
        return not any(keyword in text_lower for keyword in nav_keywords)

    async def _enhance_articles_with_content(self, articles: List[Article]) -> List[Article]:
        """Optionally enhance articles by fetching their full content"""
        # For now, return articles as-is
        # In the future, we could crawl individual article pages for full content
        return articles

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str:
            return None
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\w+ \d{1,2}, \d{4}', # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    from dateutil import parser
                    return parser.parse(match.group()).replace(tzinfo=timezone.utc)
                except:
                    continue
        
        return None

    def _remove_duplicate_articles(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on URL"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles

    async def extract_from_multiple_sources(self, sources: List[SourceConfig]) -> List[Article]:
        """Extract articles from multiple sources concurrently"""
        await self._ensure_crawler()
        
        # Process sources in batches to avoid overwhelming the system
        all_articles = []
        batch_size = min(self.max_concurrent, len(sources))
        
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.extract_from_source(source) for source in batch]
            
            try:
                # Execute batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for source, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process {source.name}: {result}")
                    else:
                        all_articles.extend(result)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
            
            # Add delay between batches
            if i + batch_size < len(sources):
                await asyncio.sleep(1)
        
        return all_articles

    async def cleanup(self):
        """Clean up resources"""
        if self.crawler:
            await self.crawler.aclose()
            self.crawler = None

    def __del__(self):
        """Cleanup on deletion"""
        if self.crawler:
            try:
                asyncio.create_task(self.cleanup())
            except:
                pass


class SmartCrawl4AiWebScraper(Crawl4AiWebScraper):
    """Enhanced version with AI-powered smart selectors and content understanding"""
    
    def __init__(self, *args, **kwargs):
        # Enable LLM extraction by default for smart scraper
        kwargs.setdefault('use_llm_extraction', True)
        super().__init__(*args, **kwargs)
    
    async def _crawl_and_extract(self, source: SourceConfig) -> List[Article]:
        """Enhanced crawling with smart content detection"""
        try:
            # First, try standard extraction
            articles = await super()._crawl_and_extract(source)
            
            # If we didn't get good results, try alternative strategies
            if len(articles) < 3:
                articles = await self._smart_extraction_fallback(source)
            
            return articles
            
        except Exception as e:
            logger.error(f"Smart extraction failed for {source.name}: {e}")
            return []
    
    async def _smart_extraction_fallback(self, source: SourceConfig) -> List[Article]:
        """Fallback extraction with different strategies"""
        logger.info(f"Trying smart extraction fallback for {source.name}")
        
        # Try with different wait conditions and JavaScript execution
        run_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            ),
            wait_for="body",
            js_code=[
                "window.scrollTo(0, 1000);",  # Scroll to load dynamic content
                "await new Promise(resolve => setTimeout(resolve, 2000));"  # Wait for content
            ],
            cache_mode=CacheMode.BYPASS
        )
        
        result = await self.crawler.arun(
            url=source.url,
            config=run_config
        )
        
        if result.success:
            return await self._extract_articles_from_result(result, source)
        
        return []


# Synchronous wrapper for backward compatibility
class WebScraperWrapper:
    """Synchronous wrapper for crawl4ai web scraper to maintain compatibility"""
    
    def __init__(self, *args, **kwargs):
        self.async_scraper = Crawl4AiWebScraper(*args, **kwargs)
        self._loop = None
    
    def _get_event_loop(self):
        """Get or create event loop for async operations"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def extract_from_source(self, source: SourceConfig) -> List[Article]:
        """Synchronous wrapper for extract_from_source"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_scraper.extract_from_source(source))
    
    def extract_from_multiple_sources(self, sources: List[SourceConfig]) -> List[Article]:
        """Synchronous wrapper for extract_from_multiple_sources"""
        loop = self._get_event_loop()
        return loop.run_until_complete(self.async_scraper.extract_from_multiple_sources(sources))
    
    def __del__(self):
        """Cleanup wrapper"""
        if self.async_scraper:
            try:
                loop = self._get_event_loop()
                loop.run_until_complete(self.async_scraper.cleanup())
            except:
                pass


# For backward compatibility, alias the wrapper as the main classes
Crawl4AiWebScraperSync = WebScraperWrapper


def main():
    """Test the crawl4ai web scraper"""
    import asyncio
    from src.scrapers.config_loader import ConfigLoader
    
    async def test_scraper():
        # Load configuration
        config = ConfigLoader("src/sources.yaml")
        website_sources = config.get_website_sources()[:2]  # Test first 2 sources
        
        # Create scraper
        scraper = Crawl4AiWebScraper(use_llm_extraction=False)
        
        try:
            # Test single source
            if website_sources:
                articles = await scraper.extract_from_source(website_sources[0])
                print(f"Extracted {len(articles)} articles from {website_sources[0].name}")
                
                for article in articles[:3]:  # Show first 3
                    print(f"- {article.title}")
                    print(f"  URL: {article.url}")
                    print(f"  Description: {article.description[:100]}...")
                    print()
        
        finally:
            await scraper.cleanup()
    
    # Run test
    asyncio.run(test_scraper())


if __name__ == "__main__":
    main() 