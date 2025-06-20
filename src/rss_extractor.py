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
from config_loader import SourceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Article:
    """Article data structure"""
    def __init__(self, title: str, url: str, description: str = "", 
                 published: Optional[datetime] = None, source: str = "",
                 category: str = "", tags: List[str] = None):
        self.title = title.strip()
        self.url = url
        self.description = description.strip()
        self.published = published or datetime.now(timezone.utc)
        self.source = source
        self.category = category
        self.tags = tags or []
        self.extracted_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary"""
        return {
            'title': self.title,
            'url': self.url,
            'description': self.description,
            'published': self.published.isoformat() if self.published else None,
            'source': self.source,
            'category': self.category,
            'tags': self.tags,
            'extracted_at': self.extracted_at.isoformat()
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
        self.session.headers.update({
            'User-Agent': self.ua.random
        })
    
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
                if feed.bozo and hasattr(feed, 'bozo_exception'):
                    logger.warning(f"RSS parsing warning for {source.name}: {feed.bozo_exception}")
                
                articles = []
                for entry in feed.entries:
                    article = self._parse_rss_entry(entry, source)
                    if article:
                        articles.append(article)
                
                logger.info(f"Extracted {len(articles)} articles from {source.name}")
                return articles
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {source.name}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"Failed to extract from {source.name} after {self.max_retries} attempts")
        return []
    
    def _parse_rss_entry(self, entry: Any, source: SourceConfig) -> Optional[Article]:
        """Parse a single RSS entry into an Article"""
        try:
            # Extract title
            title = getattr(entry, 'title', '').strip()
            if not title:
                return None
            
            # Extract URL
            url = getattr(entry, 'link', '').strip()
            if not url:
                return None
            
            # Make URL absolute if needed
            if not url.startswith(('http://', 'https://')):
                url = urljoin(source.url, url)
            
            # Extract description/summary
            description = ''
            if hasattr(entry, 'summary'):
                description = entry.summary
            elif hasattr(entry, 'description'):
                description = entry.description
            elif hasattr(entry, 'content'):
                if isinstance(entry.content, list) and entry.content:
                    description = entry.content[0].get('value', '')
                else:
                    description = str(entry.content)
            
            # Clean up description (remove HTML tags)
            if description:
                from bs4 import BeautifulSoup
                description = BeautifulSoup(description, 'html.parser').get_text().strip()
            
            # Extract published date
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            
            # Extract tags
            tags = []
            if hasattr(entry, 'tags'):
                tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
            
            return Article(
                title=title,
                url=url,
                description=description,
                published=published,
                source=source.name,
                category=source.category,
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry from {source.name}: {str(e)}")
            return None
    
    def extract_from_multiple_sources(self, sources: List[SourceConfig]) -> List[Article]:
        """Extract articles from multiple RSS sources"""
        all_articles = []
        
        for source in sources:
            if source.type == 'rss' and source.active:
                articles = self.extract_from_source(source)
                all_articles.extend(articles)
                
                # Add delay between requests to be respectful
                time.sleep(1)
        
        logger.info(f"Total articles extracted from RSS sources: {len(all_articles)}")
        return all_articles
    
    def extract_recent_articles(self, sources: List[SourceConfig], 
                              hours_back: int = 24) -> List[Article]:
        """Extract only recent articles within specified hours"""
        all_articles = self.extract_from_multiple_sources(sources)
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent_articles = [
            article for article in all_articles
            if article.published and article.published > cutoff_time
        ]
        
        logger.info(f"Found {len(recent_articles)} recent articles (last {hours_back} hours)")
        return recent_articles

def main():
    """Test the RSS extractor"""
    from config_loader import ConfigLoader
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
