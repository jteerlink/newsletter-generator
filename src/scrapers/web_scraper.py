"""
Web scraper for websites without RSS feeds
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import time
from urllib.parse import urljoin, urlparse
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from src.scrapers.config_loader import SourceConfig
from src.scrapers.rss_extractor import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrape articles from websites"""

    def __init__(
        self, timeout: int = 30, max_retries: int = 3, use_selenium: bool = False
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_selenium = use_selenium
        self.ua = UserAgent()

        # Setup requests session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.ua.random,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        # Setup Selenium driver if needed
        self.driver = None
        if self.use_selenium:
            self._setup_selenium()

    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument(f"--user-agent={self.ua.random}")

            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(
                    ChromeDriverManager().install()
                ),
                options=chrome_options,
            )
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium WebDriver initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            self.use_selenium = False

    def extract_from_source(self, source: SourceConfig) -> List[Article]:
        """Extract articles from a single website source"""
        logger.info(f"Scraping website: {source.name}")

        for attempt in range(self.max_retries):
            try:
                if self.use_selenium and self.driver:
                    html_content = self._get_content_selenium(source.url)
                else:
                    html_content = self._get_content_requests(source.url)

                if not html_content:
                    continue

                articles = self._parse_html_content(html_content, source)
                logger.info(f"Extracted {len(articles)} articles from {source.name}")
                return articles

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for {source.name}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        logger.error(
            f"Failed to scrape {source.name} after {self.max_retries} attempts"
        )
        return []

    def _get_content_requests(self, url: str) -> Optional[str]:
        """Get webpage content using requests"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Requests failed for {url}: {e}")
            return None

    def _get_content_selenium(self, url: str) -> Optional[str]:
        """Get webpage content using Selenium"""
        try:
            self.driver.get(url)
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Selenium failed for {url}: {e}")
            return None

    def _parse_html_content(
        self, html_content: str, source: SourceConfig
    ) -> List[Article]:
        """Parse HTML content to extract articles"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            articles = []

            # Use custom selector or default selectors
            selector = (
                source.selector or "h1 a, h2 a, h3 a, .post-title a, .article-title a"
            )

            # Find article links
            article_links = soup.select(selector)

            for link in article_links:
                article = self._extract_article_from_link(link, source)
                if article:
                    articles.append(article)

            # Remove duplicates based on URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)

            return unique_articles

        except Exception as e:
            logger.error(f"Error parsing HTML content for {source.name}: {e}")
            return []

    def _extract_article_from_link(
        self, link_element, source: SourceConfig
    ) -> Optional[Article]:
        """Extract article information from a link element"""
        try:
            # Get title
            title = link_element.get_text(strip=True)
            # Titles shorter than 10 characters are skipped (likely not meaningful)
            if not title or len(title) < 10:  # Skip very short titles
                return None
            # Get URL
            url = link_element.get("href", "")
            if not url:
                return None
            # Make URL absolute
            if not url.startswith(("http://", "https://")):
                url = urljoin(source.url, url)
            # Try to extract description from nearby elements
            description = self._extract_description(link_element)
            # Try to extract publication date
            published = self._extract_date(link_element)
            # Try to extract raw HTML content (if available)
            raw_content = str(link_element)
            # Author (not always available)
            author = None
            parent = link_element.parent
            if parent:
                author_elem = parent.find(class_="author")
                if author_elem:
                    author = author_elem.get_text(strip=True)
            # Language (not always available)
            language = None
            html_elem = link_element.find_parent("html")
            if html_elem and html_elem.has_attr("lang"):
                language = html_elem["lang"]
            # Media URLs (look for images in parent)
            media_urls = []
            if parent:
                for img in parent.find_all("img"):
                    if img.has_attr("src"):
                        media_urls.append(img["src"])
            # Word count
            word_count = len(description.split()) if description else None
            # Canonical URL (not always available)
            canonical_url = url
            # Updated at (not available, fallback to published)
            updated_at = published
            return Article(
                title=title,
                url=url,
                description=description,
                published=published,
                source=source.name,
                category=source.category,
                tags=[],
                raw_content=raw_content,
                author=author,
                language=language,
                fetch_status="success",
                error_message=None,
                source_type="website",
                media_urls=media_urls,
                word_count=word_count,
                canonical_url=canonical_url,
                updated_at=updated_at,
            )
        except Exception as e:
            logger.error(f"Error extracting article: {e}")
            return None

    def _extract_description(self, link_element) -> str:
        """Try to extract description/summary near the link"""
        description = ""

        # Look for description in various places
        parent = link_element.parent
        if parent:
            # Look for sibling elements with description-like content
            for sibling in parent.find_next_siblings():
                if sibling.name in ["p", "div", "span"]:
                    text = sibling.get_text(strip=True)
                    # Only consider substantial text (longer than 50 chars)
                    if len(text) > 50:  # Only consider substantial text
                        description = text[:300]  # Limit length to 300 chars
                        break

            # Look in parent container
            if not description:
                parent_text = parent.get_text(strip=True)
                # Remove the title from parent text
                title = link_element.get_text(strip=True)
                parent_text = parent_text.replace(title, "").strip()
                if len(parent_text) > 50:
                    description = parent_text[:300]  # Limit length to 300 chars

        return description

    def _extract_date(self, link_element) -> Optional[datetime]:
        """Try to extract publication date near the link"""
        # Common date selectors
        date_selectors = [
            "time",
            ".date",
            ".published",
            ".post-date",
            "[datetime]",
            ".timestamp",
            ".article-date",
        ]

        # Look in parent and nearby elements
        parent = link_element.parent
        if parent:
            for selector in date_selectors:
                date_elem = parent.find(selector)
                if date_elem:
                    date_str = date_elem.get("datetime") or date_elem.get_text(
                        strip=True
                    )
                    if date_str:
                        return self._parse_date_string(date_str)

        return None

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        from dateutil.parser import parse

        try:
            return parse(date_str)
        except:
            return None

    def extract_from_multiple_sources(
        self, sources: List[SourceConfig]
    ) -> List[Article]:
        """Extract articles from multiple website sources"""
        all_articles = []

        for source in sources:
            if source.type == "website" and source.active:
                articles = self.extract_from_source(source)
                all_articles.extend(articles)

                # Add delay between requests to be respectful
                time.sleep(2)

        logger.info(
            f"Total articles extracted from website sources: {len(all_articles)}"
        )
        return all_articles

    def __del__(self):
        """Clean up Selenium driver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


class SmartWebScraper(WebScraper):
    """Enhanced web scraper with intelligent content extraction"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Common selectors for different site types
        self.site_selectors = {
            "blog": [
                "article h1 a",
                "article h2 a",
                ".post-title a",
                ".entry-title a",
                "h1.title a",
                "h2.title a",
            ],
            "news": [
                ".headline a",
                ".article-headline a",
                ".story-title a",
                "h1 a",
                "h2 a",
                ".title a",
            ],
            "company": [
                ".news-item a",
                ".press-release a",
                ".blog-post a",
                "article h2 a",
                ".post h2 a",
            ],
        }

    def _get_smart_selectors(self, source: SourceConfig) -> List[str]:
        """Get smart selectors based on source type and URL"""
        selectors = []

        # Use custom selector if provided
        if source.selector:
            selectors.append(source.selector)

        # Add category-based selectors
        if "blog" in source.name.lower() or "blog" in source.url.lower():
            selectors.extend(self.site_selectors["blog"])
        elif "news" in source.name.lower() or "news" in source.url.lower():
            selectors.extend(self.site_selectors["news"])
        elif source.category == "company-research":
            selectors.extend(self.site_selectors["company"])

        # Add default selectors
        selectors.extend(["h1 a", "h2 a", "h3 a"])

        return list(
            dict.fromkeys(selectors)
        )  # Remove duplicates while preserving order


def main():
    """Test the web scraper"""
    from src.scrapers.config_loader import ConfigLoader

    # Load configuration
    config = ConfigLoader()
    website_sources = config.get_website_sources()

    print(f"Found {len(website_sources)} website sources")

    # Test with first few sources (using regular scraper, not Selenium for speed)
    test_sources = website_sources[:2]  # Test with first 2 sources

    scraper = SmartWebScraper(use_selenium=False)
    articles = scraper.extract_from_multiple_sources(test_sources)

    print(f"\n=== Scraping Results ===")
    print(f"Total articles: {len(articles)}")

    # Show some examples
    for i, article in enumerate(articles[:5]):
        print(f"\n{i+1}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        if article.description:
            print(f"   Description: {article.description[:100]}...")


if __name__ == "__main__":
    main()
