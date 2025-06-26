import logging
from src.scrapers.data_processor import DataProcessor
from src.scrapers.rss_extractor import Article
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_metadata(metadata):
    # Convert None values to empty string or a default value
    return {k: (v if v is not None else "") for k, v in metadata.items()}

def migrate_sqlite_to_chromadb():
    processor = DataProcessor()
    # Fetch all articles from SQLite
    articles = processor.get_articles()
    logger.info(f"Found {len(articles)} articles in SQLite DB.")
    migrated = 0
    skipped = 0
    for art in articles:
        # Reconstruct Article object
        article = Article(
            title=art.get('title', ''),
            url=art.get('url', ''),
            description=art.get('description', ''),
            published=art.get('published'),
            source=art.get('source', ''),
            category=art.get('category', ''),
            tags=art.get('tags', []),
            raw_content=art.get('raw_content', None),
            author=art.get('author', None),
            language=art.get('language', None),
            fetch_status=art.get('fetch_status', None),
            error_message=art.get('error_message', None),
            source_type=art.get('source_type', None),
            media_urls=art.get('media_urls', []),
            word_count=art.get('word_count', None),
            canonical_url=art.get('canonical_url', None),
            updated_at=art.get('updated_at', None)
        )
        # Use DataProcessor's logic to add to ChromaDB
        try:
            # Use only the ChromaDB part of store_article_with_analysis
            content = article.raw_content or article.description or article.title
            metadata = {
                'url': article.url,
                'title': article.title,
                'source': article.source,
                'category': article.category,
                'tags': json.dumps(article.tags) if isinstance(article.tags, list) else article.tags,
                'published': str(article.published) if article.published else "",
                'author': article.author,
                'quality_score': art.get('quality_score', 0.0),
                'content_hash': art.get('content_hash', None)
            }
            metadata = sanitize_metadata(metadata)
            processor.vector_store.add_document(content, metadata)
            migrated += 1
        except Exception as e:
            logger.error(f"Failed to migrate article '{article.title}': {e}")
            skipped += 1
    logger.info(f"Migration complete. Migrated: {migrated}, Skipped: {skipped}")

if __name__ == "__main__":
    migrate_sqlite_to_chromadb() 