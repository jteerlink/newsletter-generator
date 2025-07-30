"""
RSS feed extractor for news sources
"""

import feedparser
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from urllib.parse import urljoin, urlparse
import time
from fake_useragent import UserAgent
# Handle imports for both direct execution and module import
try:
    from .config_loader import SourceConfig
except ImportError:
    try:
        from config_loader import SourceConfig
    except ImportError:
        from src.scrapers.config_loader import SourceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Article:
    """Article data structure"""

    def __init__(
        self,
        title: str,
        url: str,
        description: str = "",
        published: Optional[datetime] = None,
        source: str = "",
        category: str = "",
        tags: List[str] = None,
        raw_content: str = None,
        author: str = None,
        language: str = None,
        fetch_status: str = None,
        error_message: str = None,
        source_type: str = None,
        media_urls: List[str] = None,
        word_count: int = None,
        canonical_url: str = None,
        updated_at: Optional[datetime] = None,
    ):
        self.title = title.strip()
        self.url = url
        self.description = description.strip()
        self.published = published or datetime.now(timezone.utc)
        self.source = source
        self.category = category
        self.tags = tags or []
        self.extracted_at = datetime.now(timezone.utc)
        self.raw_content = raw_content
        self.author = author
        self.language = language
        self.fetch_status = fetch_status
        self.error_message = error_message
        self.source_type = source_type
        self.media_urls = media_urls or []
        self.word_count = word_count
        self.canonical_url = canonical_url
        self.updated_at = updated_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "published": self.published.isoformat() if self.published else None,
            "source": self.source,
            "category": self.category,
            "tags": self.tags,
            "extracted_at": self.extracted_at.isoformat(),
            "raw_content": self.raw_content,
            "author": self.author,
            "language": self.language,
            "fetch_status": self.fetch_status,
            "error_message": self.error_message,
            "source_type": self.source_type,
            "media_urls": self.media_urls,
            "word_count": self.word_count,
            "canonical_url": self.canonical_url,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f"Article(title='{self.title[:50]}...', source='{self.source}')"


class RSSExtractor:
    """Extract articles from RSS feeds"""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.ua.random})

    def extract_from_source(self, source: SourceConfig) -> List[Article]:
        """Extract articles from a single RSS source"""
        if not source.rss_url:
            logger.warning(f"No RSS URL provided for {source.name}")
            return []

        logger.info(f"Extracting from RSS: {source.name}")

        for attempt in range(self.max_retries):
            try:
                # Parse the RSS feed
                feed = feedparser.parse(source.rss_url)

                # Check if feed was parsed successfully
                if feed.bozo and hasattr(feed, "bozo_exception"):
                    logger.warning(
                        f"RSS parsing warning for {source.name}: {feed.bozo_exception}"
                    )

                articles = []
                for entry in feed.entries:
                    article = self._parse_rss_entry(entry, source)
                    if article:
                        articles.append(article)

                logger.info(f"Extracted {len(articles)} articles from {source.name}")
                return articles

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for {source.name}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    # Exponential backoff: wait longer after each failed attempt (2^attempt seconds)
                    time.sleep(2**attempt)  # Exponential backoff

        logger.error(
            f"Failed to extract from {source.name} after {self.max_retries} attempts"
        )
        return []

    def _parse_rss_entry(self, entry: Any, source: SourceConfig) -> Optional[Article]:
        """Parse a single RSS entry into an Article"""
        try:
            title = getattr(entry, "title", "").strip()
            if not title:
                return None
            url = getattr(entry, "link", "").strip()
            if not url:
                return None
            if not url.startswith(("http://", "https://")):
                url = urljoin(source.url, url)
            description = ""
            raw_content = None
            if hasattr(entry, "summary"):
                description = entry.summary
            elif hasattr(entry, "description"):
                description = entry.description
            elif hasattr(entry, "content"):
                if isinstance(entry.content, list) and entry.content:
                    description = entry.content[0].get("value", "")
                    raw_content = entry.content[0].get("value", "")
                else:
                    description = str(entry.content)
                    raw_content = str(entry.content)
            # Clean up description (remove HTML tags)
            if description:
                from bs4 import BeautifulSoup

                description = (
                    BeautifulSoup(description, "html.parser").get_text().strip()
                )
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            tags = []
            if hasattr(entry, "tags"):
                tags = [tag.term for tag in entry.tags if hasattr(tag, "term")]
            author = getattr(entry, "author", None)
            language = getattr(entry, "language", None)
            media_urls = []
            if hasattr(entry, "media_content"):
                media_urls = [m.get("url") for m in entry.media_content if "url" in m]
            word_count = len(description.split()) if description else None
            canonical_url = (
                getattr(entry, "id", None) if getattr(entry, "id", None) else url
            )
            updated_at = published
            return Article(
                title=title,
                url=url,
                description=description,
                published=published,
                source=source.name,
                category=source.category,
                tags=tags,
                raw_content=raw_content,
                author=author,
                language=language,
                fetch_status="success",
                error_message=None,
                source_type="rss",
                media_urls=media_urls,
                word_count=word_count,
                canonical_url=canonical_url,
                updated_at=updated_at,
            )
        except Exception as e:
            logger.error(f"Error parsing RSS entry from {source.name}: {str(e)}")
            return Article(
                title=getattr(entry, "title", "ERROR"),
                url=getattr(entry, "link", ""),
                description="",
                published=None,
                source=source.name,
                category=source.category,
                tags=[],
                raw_content=None,
                author=None,
                language=None,
                fetch_status="error",
                error_message=str(e),
                source_type="rss",
                media_urls=[],
                word_count=None,
                canonical_url=None,
                updated_at=None,
            )

    def extract_from_multiple_sources(
        self, sources: List[SourceConfig]
    ) -> List[Article]:
        """Extract articles from multiple RSS sources"""
        all_articles = []

        for source in sources:
            if source.type == "rss" and source.active:
                articles = self.extract_from_source(source)
                all_articles.extend(articles)

                # Add delay between requests to be respectful
                time.sleep(1)

        logger.info(f"Total articles extracted from RSS sources: {len(all_articles)}")
        return all_articles

    def extract_recent_articles(
        self, sources: List[SourceConfig], hours_back: int = 24
    ) -> List[Article]:
        """Extract only recent articles within specified hours"""
        all_articles = self.extract_from_multiple_sources(sources)

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent_articles = [
            article
            for article in all_articles
            if article.published and article.published > cutoff_time
        ]

        logger.info(
            f"Found {len(recent_articles)} recent articles (last {hours_back} hours)"
        )
        return recent_articles


def main():
    """Test the RSS extractor"""
    try:
        from .config_loader import ConfigLoader
    except ImportError:
        try:
            from config_loader import ConfigLoader
        except ImportError:
            from src.scrapers.config_loader import ConfigLoader
    from datetime import timedelta

    # Load configuration
    config = ConfigLoader()
    rss_sources = config.get_rss_sources()

    print(f"Found {len(rss_sources)} RSS sources")

    # Test with first few sources
    test_sources = rss_sources[:3]  # Test with first 3 sources

    extractor = RSSExtractor()
    articles = extractor.extract_from_multiple_sources(test_sources)

    print(f"\n=== Extraction Results ===")
    print(f"Total articles: {len(articles)}")

    # Show some examples
    for i, article in enumerate(articles[:5]):
        print(f"\n{i+1}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        print(f"   Published: {article.published}")
        if article.description:
            print(f"   Description: {article.description[:100]}...")


if __name__ == "__main__":
    main()
