"""
Crawl4AI-based web scraper for websites without RSS feeds
Enhanced with AI-powered content extraction and better JavaScript handling
Updated for crawl4ai 0.6.3+ with improved extraction strategies
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import time
from urllib.parse import urljoin, urlparse
import json
import re
from dateutil import parser as date_parser

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.cache_context import CacheMode

# Handle imports for both direct execution and module import
try:
    from .config_loader import SourceConfig
    from .rss_extractor import Article
except ImportError:
    from config_loader import SourceConfig
    from rss_extractor import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Crawl4AiWebScraper:
    """AI-powered web scraper using crawl4ai 0.6.3+"""

    def __init__(
        self, 
        timeout: int = 30, 
        max_retries: int = 3, 
        headless: bool = True,
        use_llm_extraction: bool = False,
        llm_provider: str = "ollama/llama3",
        max_concurrent: int = 3,
        browser_type: str = "chromium"
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.headless = headless
        self.use_llm_extraction = use_llm_extraction
        self.llm_provider = llm_provider
        self.max_concurrent = max_concurrent
        self.browser_type = browser_type
        
        # Browser configuration for crawl4ai 0.6.3+
        self.browser_config = BrowserConfig(
            headless=self.headless,
            browser_type=self.browser_type,  # chromium, firefox, webkit
            # Additional optimizations
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        
        # Initialize crawler (will be done in async context)
        self.crawler = None

    async def _ensure_crawler(self):
        """Ensure crawler is initialized"""
        if self.crawler is None:
            self.crawler = AsyncWebCrawler(
                config=self.browser_config
            )
            try:
                await self.crawler.start()
                logger.info("Crawl4AI crawler initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize crawler: {e}")
                raise

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
            page_timeout=self.timeout * 1000,  # Page timeout in milliseconds
            delay_before_return_html=2.0,  # Wait for dynamic content
        )

    def _create_css_extraction_strategy(self, source: SourceConfig) -> JsonCssExtractionStrategy:
        """Create enhanced CSS-based extraction strategy"""
        # Enhanced selectors for better article detection
        title_selectors = [
            source.selector if source.selector else "h1 a, h2 a, h3 a",
            "article h1 a, article h2 a, article h3 a",
            ".post-title a, .article-title a, .entry-title a",
            ".news-title a, .headline a, .story-headline a",
            ".card-title a, .item-title a, .content-title a",
            "[data-testid*='headline'] a, [data-testid*='title'] a",
            ".title a, .heading a, .header a",
            # News-specific selectors
            ".news-item h2 a, .news-item h3 a",
            ".story h2 a, .story h3 a",
            ".article-item h2 a, .article-item h3 a",
        ]
        
        # Schema for extracting comprehensive article data
        # Use baseSelector to define the container elements for articles
        schema = {
            "baseSelector": "article, .post, .news-item, .story, .article-item, .card, .item, .entry",
            "fields": [
                {
                    "name": "title",
                    "selector": title_selectors,
                    "type": "text"
                },
                {
                    "name": "url",
                    "selector": title_selectors,
                    "type": "attribute",
                    "attribute": "href"
                },
                {
                    "name": "description",
                    "selector": "p:contains-text, .excerpt, .summary, .description, .lead, .intro, .snippet, .abstract, .article-excerpt, .post-excerpt",
                    "type": "text"
                },
                {
                    "name": "date",
                    "selector": "time, .date, .published, .timestamp, .article-date, .post-date, .news-date, [datetime], [data-date], [data-published], .meta-date, .publish-date, .creation-date",
                    "type": "text"
                },
                {
                    "name": "author",
                    "selector": ".author, .byline, .writer, .journalist, .article-author, .post-author, [data-author], [data-byline], .meta-author, .attribution",
                    "type": "text"
                },
                {
                    "name": "image",
                    "selector": "img[src], .featured-image img, .article-image img, .post-image img",
                    "type": "attribute",
                    "attribute": "src"
                }
            ]
        }
        
        return JsonCssExtractionStrategy(schema=schema)

    def _create_llm_extraction_strategy(self) -> LLMExtractionStrategy:
        """Create LLM-based extraction strategy for intelligent content extraction"""
        try:
            from crawl4ai.models import LLMConfig
            
            llm_config = LLMConfig(
                provider=self.llm_provider,
                # api_token can be set via environment variables if needed
                # model_kwargs for additional configuration
            )
            
            instruction = """
            Extract all news articles, blog posts, or content items from this webpage.
            For each item, provide:
            - title: The article title (required)
            - url: The link to the full article (make absolute URLs if relative)
            - description: Brief description, excerpt, or summary if available
            - author: Author name if mentioned
            - published_date: Publication date in ISO format if available
            - category: Content category if identifiable
            - tags: Relevant tags or keywords if available
            
            Focus on actual content articles, not navigation links or advertisements.
            Return the data as a JSON array of articles.
            """
            
            return LLMExtractionStrategy(
                llm_config=llm_config,
                instruction=instruction,
                schema={
                    "type": "object",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "description": {"type": "string"},
                                    "author": {"type": "string"},
                                    "published_date": {"type": "string"},
                                    "category": {"type": "string"},
                                    "tags": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["title", "url"]
                            }
                        }
                    }
                }
            )
            
        except ImportError:
            logger.warning("LLM extraction not available, falling back to CSS extraction")
            return None
        except Exception as e:
            logger.error(f"Error creating LLM extraction strategy: {e}")
            return None

    async def _extract_articles_from_result(self, result, source: SourceConfig) -> List[Article]:
        """Extract articles from crawl result"""
        articles = []
        
        try:
            # Try extraction strategy first (if available)
            if result.extracted_content:
                logger.info("Using extraction strategy results")
                extracted_data = json.loads(result.extracted_content)
                articles.extend(self._parse_extracted_data(extracted_data, source))
            
            # Fallback to link extraction from markdown/HTML
            if not articles:
                logger.info("Falling back to link extraction")
                articles.extend(await self._extract_from_links(result, source))
                
        except Exception as e:
            logger.error(f"Error extracting articles from result: {e}")
            # Final fallback
            articles.extend(await self._extract_from_links(result, source))
        
        return articles

    def _parse_extracted_data(self, data, source: SourceConfig) -> List[Article]:
        """Parse extracted data from JSON into Article objects"""
        articles = []
        
        # Handle different data structures
        if isinstance(data, list):
            # Data is directly a list of articles
            article_data = data
        elif isinstance(data, dict):
            # Data could be in various formats
            if 'articles' in data:
                article_data = data.get('articles', [])
            elif 'items' in data:
                article_data = data.get('items', [])
            else:
                # Treat the dict itself as a single article
                article_data = [data] if data else []
        else:
            logger.warning(f"Unexpected data type: {type(data)}")
            return articles
        
        for item in article_data:
            if isinstance(item, dict):
                article = self._create_article_from_data(item, source)
                if article:
                    articles.append(article)
        
        return articles

    def _create_article_from_data(self, data: Dict, source: SourceConfig) -> Optional[Article]:
        """Create Article object from extracted data"""
        try:
            title = data.get('title', '').strip()
            if not title or len(title) < 10:  # Skip very short titles
                return None
            
            url = data.get('url', '').strip()
            if not url:
                return None
                
            # Make URL absolute
            if not url.startswith(('http://', 'https://')):
                url = urljoin(source.url, url)
            
            # Parse description
            description = data.get('description', '').strip()
            
            # Parse date
            published = None
            date_str = data.get('published_date') or data.get('date')
            if date_str:
                published = self._parse_date_string(date_str)
            
            # Extract other metadata
            author = data.get('author', '').strip() or None
            tags = data.get('tags', []) or []
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            # Extract media URLs
            media_urls = []
            if 'image' in data and data['image']:
                image_url = data['image']
                if not image_url.startswith(('http://', 'https://')):
                    image_url = urljoin(source.url, image_url)
                media_urls.append(image_url)
            
            return Article(
                title=title,
                url=url,
                description=description,
                published=published,
                source=source.name,
                category=source.category,
                tags=tags,
                author=author,
                language=None,
                fetch_status="success",
                source_type="website",
                media_urls=media_urls,
                word_count=len(description.split()) if description else None
            )
            
        except Exception as e:
            logger.error(f"Error creating article from data: {e}")
            return None

    async def _extract_from_links(self, result, source: SourceConfig) -> List[Article]:
        """Extract articles from links in the crawl result"""
        articles = []
        
        try:
            # Use links from the result if available
            if hasattr(result, 'links') and result.links:
                # Handle both dict and object-style links
                if isinstance(result.links, dict):
                    all_links = result.links.get('internal', []) + result.links.get('external', [])
                else:
                    all_links = result.links.internal + result.links.external
                
                for link in all_links:
                    if self._is_likely_article_link(link, source):
                        # Handle both dict and object-style link data
                        if isinstance(link, dict):
                            title = link.get('text') or link.get('title') or "Untitled"
                            url = link.get('href') or link.get('url')
                        else:
                            title = link.text or link.title or "Untitled"
                            url = link.href
                        
                        if url:
                            article = Article(
                                title=title,
                                url=url,
                            description="",
                            source=source.name,
                            category=source.category,
                            fetch_status="success",
                            source_type="website"
                        )
                        articles.append(article)
            
        except Exception as e:
            logger.error(f"Error extracting from links: {e}")
        
        return articles

    def _is_likely_article_link(self, link, source: SourceConfig) -> bool:
        """Determine if a link is likely to be an article"""
        # Handle both dict and object-style links
        if isinstance(link, dict):
            href = link.get('href') or link.get('url')
            text = link.get('text') or link.get('title')
        else:
            href = link.href
            text = link.text
        
        if not href or not text:
            return False
        
        text = text.strip()
        href = href.lower()
        
        # Skip very short text
        if len(text) < 15:  # Increased minimum length
            return False
        
        # Skip common non-article patterns (expanded list)
        skip_patterns = [
            'home', 'about', 'contact', 'privacy', 'terms',
            'login', 'register', 'subscribe', 'newsletter',
            'more', 'read more', 'continue reading', 'next',
            'prev', 'previous', 'comments', 'share',
            'skip to main content', 'menu', 'navigation',
            'api platform', 'for business', 'pricing', 'support',
            'documentation', 'careers', 'research highlights',
            'publications', 'machine learning research',
            'back to top', 'footer', 'header', 'sidebar',
            'sign up', 'sign in', 'log out', 'my account'
        ]
        
        if any(pattern in text.lower() for pattern in skip_patterns):
            return False
        
        # Skip common non-article URL patterns
        if any(pattern in href for pattern in [
            '/tag/', '/category/', '/author/', '/page/',
            '/search/', '/archive/', '/feed/', '/rss/',
            '.css', '.js', '.pdf', '.jpg', '.png', '.gif'
        ]):
            return False
        
        return True

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats into datetime object"""
        if not date_str:
            return None
        
        try:
            # Use dateutil parser for flexible date parsing
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            
            # Ensure timezone awareness
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            
            return parsed_date
            
        except Exception:
            # Fallback patterns
            patterns = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%B %d, %Y',
                '%d %B %Y',
            ]
            
            for pattern in patterns:
                try:
                    return datetime.strptime(date_str, pattern).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date string: {date_str}")
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
        """Extract articles from multiple sources with concurrent processing"""
        logger.info(f"Processing {len(sources)} sources concurrently")
        
        await self._ensure_crawler()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def extract_with_semaphore(source):
            async with semaphore:
                return await self.extract_from_source(source)
        
        # Process sources concurrently
        tasks = [extract_with_semaphore(source) for source in sources]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in concurrent processing: {e}")
            results = []
        
        # Flatten results and handle exceptions
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Source {sources[i].name} failed: {result}")
            elif isinstance(result, list):
                all_articles.extend(result)
        
        logger.info(f"Total articles extracted: {len(all_articles)}")
        return all_articles

    async def cleanup(self):
        """Clean up crawler resources"""
        if self.crawler:
            try:
                await self.crawler.close()
                self.crawler = None
                logger.info("Crawler cleanup completed")
            except Exception as e:
                logger.error(f"Error during crawler cleanup: {e}")
                # Still set to None even if close fails
                self.crawler = None

    def __del__(self):
        """Cleanup on deletion - simplified to avoid hanging"""
        # During garbage collection, don't try to run async cleanup
        # as it can cause hanging. Just mark crawler as None.
        try:
            if hasattr(self, 'crawler'):
                self.crawler = None
        except Exception:
            pass  # Ignore cleanup errors during deletion


class SmartCrawl4AiWebScraper(Crawl4AiWebScraper):
    """Enhanced scraper with AI capabilities and smart fallbacks"""

    def __init__(self, *args, **kwargs):
        # Enable LLM extraction by default for smart scraper
        kwargs.setdefault('use_llm_extraction', True)
        super().__init__(*args, **kwargs)

    async def _crawl_and_extract(self, source: SourceConfig) -> List[Article]:
        """Enhanced crawling with smart fallbacks"""
        try:
            # Try primary extraction method
            articles = await super()._crawl_and_extract(source)
            
            # If no articles found, try alternative strategies
            if not articles:
                logger.info(f"No articles found for {source.name}, trying smart fallback")
                articles = await self._smart_extraction_fallback(source)
            
            return articles
            
        except Exception as e:
            logger.error(f"Smart extraction failed for {source.name}: {e}")
            return []

    async def _smart_extraction_fallback(self, source: SourceConfig) -> List[Article]:
        """Smart fallback extraction strategies"""
        articles = []
        
        try:
            # Try with different CSS strategies
            fallback_selectors = [
                "a[href*='article'], a[href*='post'], a[href*='story']",
                ".article a, .post a, .story a, .news a",
                "[class*='title'] a, [class*='headline'] a",
                "h1 a, h2 a, h3 a, h4 a"
            ]
            
            for selector in fallback_selectors:
                # Create temporary source config with different selector
                temp_source = SourceConfig(
                    name=source.name,
                    url=source.url,
                    type=source.type,
                    category=source.category,
                    active=source.active,
                    selector=selector
                )
                
                # Try extraction with this selector
                run_config = self._create_run_config(temp_source)
                result = await self.crawler.arun(url=source.url, config=run_config)
                
                if result.success:
                    temp_articles = await self._extract_articles_from_result(result, source)
                    if temp_articles:
                        articles.extend(temp_articles)
                        break  # Stop on first successful extraction
            
        except Exception as e:
            logger.error(f"Smart fallback failed: {e}")
        
        return articles


class WebScraperWrapper:
    """Synchronous wrapper for async crawl4ai scraper"""

    def __init__(self, scraper_class=None, *args, **kwargs):
        # Use the provided scraper class or default to Crawl4AiWebScraper
        if scraper_class is None:
            scraper_class = Crawl4AiWebScraper
        
        self.async_scraper = scraper_class(*args, **kwargs)
        self._loop = None

    def _get_event_loop(self):
        """Get or create event loop for async operations"""
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create new loop if none exists or current is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self._loop = loop
        return loop

    def extract_from_source(self, source: SourceConfig) -> List[Article]:
        """Synchronous wrapper for extract_from_source"""
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we reach here, there's already a running loop - use threading
            import concurrent.futures
            import threading
            
            result = []
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.async_scraper.extract_from_source(source))
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No running loop, we can use run_until_complete
            loop = self._get_event_loop()
            return loop.run_until_complete(self.async_scraper.extract_from_source(source))

    def extract_from_multiple_sources(self, sources: List[SourceConfig]) -> List[Article]:
        """Synchronous wrapper for extract_from_multiple_sources"""
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we reach here, there's already a running loop - use threading
            import concurrent.futures
            import threading
            
            result = []
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self.async_scraper.extract_from_multiple_sources(sources))
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No running loop, we can use run_until_complete
            loop = self._get_event_loop()
            return loop.run_until_complete(
                self.async_scraper.extract_from_multiple_sources(sources)
            )

    def cleanup(self):
        """Manual cleanup method - simplified to avoid hanging"""
        try:
            # Just set the async scraper to None to release resources
            # Don't try to run async cleanup as it can cause hanging
            if hasattr(self, 'async_scraper'):
                self.async_scraper = None
            if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                self._loop.close()
                self._loop = None
            logger.debug("Cleanup completed")
        except Exception:
            # If anything goes wrong, just continue - cleanup is best effort
            logger.debug("Cleanup skipped due to error")
            pass

    def __del__(self):
        """Cleanup wrapper - simplified to avoid hanging"""
        # During garbage collection, just close the loop if it exists
        # Don't try to run async cleanup as it can hang
        try:
            if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass  # Ignore cleanup errors during deletion


# For backward compatibility
SmartWebScraper = WebScraperWrapper


async def main():
    """Test the crawl4ai scraper"""
    
    async def test_scraper():
        # Load configuration
        try:
            from .config_loader import ConfigLoader
        except ImportError:
            from config_loader import ConfigLoader
        
        config = ConfigLoader("src/sources.yaml")
        sources = config.get_active_sources()[:2]  # Test with first 2 sources
        
        # Test standard scraper
        logger.info("Testing standard Crawl4AI scraper...")
        scraper = Crawl4AiWebScraper(use_llm_extraction=False)
        
        try:
            for source in sources:
                if source.type == "website":
                    articles = await scraper.extract_from_source(source)
                    logger.info(f"Extracted {len(articles)} articles from {source.name}")
                    
                    for article in articles[:3]:  # Show first 3
                        logger.info(f"  - {article.title}")
                        
        finally:
            await scraper.cleanup()
    
    await test_scraper()


if __name__ == "__main__":
    asyncio.run(main()) 