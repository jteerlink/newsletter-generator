"""
Data processing and storage utilities
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from urllib.parse import urlparse
import hashlib
# Handle imports for both direct execution and module import
try:
    from .rss_extractor import Article
    from .content_analyzer import ContentAnalyzer
except ImportError:
    from rss_extractor import Article
    from content_analyzer import ContentAnalyzer
# Handle storage import
try:
    from ..storage.vector_store import VectorStore
except ImportError:
    try:
        from src.storage.vector_store import VectorStore
    except ImportError:
        # Optional dependency - VectorStore functionality will be disabled
        VectorStore = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and store extracted articles with enhanced content analysis"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize content analyzer for Phase 1.2 enhancements
        self.content_analyzer = ContentAnalyzer()

        # VectorStore for ChromaDB storage
        self.vector_store = VectorStore()

        # Database setup - use the existing database location
        self.db_path = Path("src/data/raw/articles.db")
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with enhanced schema for Phase 1.2"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url_hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    description TEXT,
                    published TIMESTAMP,
                    source TEXT NOT NULL,
                    category TEXT,
                    tags TEXT,
                    extracted_at TIMESTAMP NOT NULL,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_content TEXT,
                    author TEXT,
                    language TEXT,
                    fetch_status TEXT,
                    error_message TEXT,
                    source_type TEXT,
                    media_urls TEXT,
                    word_count INTEGER,
                    canonical_url TEXT,
                    updated_at TIMESTAMP,
                    processed INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    readability_score REAL DEFAULT 0.0,
                    source_reliability REAL DEFAULT 0.5,
                    freshness_score REAL DEFAULT 0.5,
                    completeness_score REAL DEFAULT 0.0,
                    analyzed_at TIMESTAMP
                )
            """
            )

            # Add new columns for Phase 1.2 if they don't exist
            try:
                cursor.execute(
                    "ALTER TABLE articles ADD COLUMN quality_score REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute(
                    "ALTER TABLE articles ADD COLUMN readability_score REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute(
                    "ALTER TABLE articles ADD COLUMN source_reliability REAL DEFAULT 0.5"
                )
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute(
                    "ALTER TABLE articles ADD COLUMN freshness_score REAL DEFAULT 0.5"
                )
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute(
                    "ALTER TABLE articles ADD COLUMN completeness_score REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass

            try:
                cursor.execute("ALTER TABLE articles ADD COLUMN analyzed_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_source ON articles (source)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_category ON articles (category)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_published ON articles (published)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_extracted_at ON articles (extracted_at)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality_score ON articles (quality_score)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_content_hash ON articles (content_hash)
            """
            )

            conn.commit()

    def process_articles(self, articles: List[Article]) -> Dict[str, Any]:
        """Process a list of articles with enhanced content analysis"""
        results = {
            "total_articles": len(articles),
            "new_articles": 0,
            "duplicate_articles": 0,
            "high_quality_articles": 0,
            "errors": 0,
            "categories": {},
            "sources": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
        }

        # Get existing content hashes for duplicate detection
        existing_hashes = self._get_existing_content_hashes()

        for article in articles:
            try:
                # Enhanced processing with content analysis
                is_new, quality_info = self.store_article_with_analysis(
                    article, existing_hashes
                )

                if is_new:
                    results["new_articles"] += 1

                    # Track quality distribution
                    quality_score = quality_info.get("quality_score", 0)
                    if quality_score >= 0.7:
                        results["quality_distribution"]["high"] += 1
                        results["high_quality_articles"] += 1
                    elif quality_score >= 0.4:
                        results["quality_distribution"]["medium"] += 1
                    else:
                        results["quality_distribution"]["low"] += 1
                else:
                    results["duplicate_articles"] += 1

                # Update category counts
                category = quality_info.get("category", article.category)
                if category in results["categories"]:
                    results["categories"][category] += 1
                else:
                    results["categories"][category] = 1

                # Update source counts
                if article.source in results["sources"]:
                    results["sources"][article.source] += 1
                else:
                    results["sources"][article.source] = 1

            except Exception as e:
                logger.error(f"Error processing article '{article.title}': {e}")
                results["errors"] += 1

        return results

    def store_article_with_analysis(
        self, article: Article, existing_hashes: List[str]
    ) -> tuple[bool, Dict[str, Any]]:
        """Store article with enhanced content analysis, return (is_new, quality_info)"""
        # Prepare metadata for analysis
        metadata = {
            "url": article.url,
            "source": article.source,
            "title": article.title,
            "description": article.description,
            "published": article.published,
            "author": getattr(article, "author", None),
            "category": article.category,
            "tags": getattr(article, "tags", []),
        }

        # Get content for analysis
        content = (
            getattr(article, "raw_content", "") or article.description or article.title
        )

        # Analyze content
        analysis = self.content_analyzer.analyze_content(content, metadata)

        # Check for duplicates using content hash
        is_duplicate = self.content_analyzer.detect_duplicates(
            analysis["content_hash"], existing_hashes
        )

        if is_duplicate:
            logger.debug(f"Duplicate detected: {article.title} | Hash: {analysis['content_hash']}")
        else:
            logger.debug(f"New article: {article.title} | Hash: {analysis['content_hash']}")

        if is_duplicate:
            return False, analysis

        # Store article with enriched metadata
        url_hash = self._generate_url_hash(article.url)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if article already exists by URL
            cursor.execute("SELECT id FROM articles WHERE url_hash = ?", (url_hash,))
            if cursor.fetchone():
                return False, analysis  # Article already exists

            # Insert new article with enhanced metadata
            cursor.execute(
                """
                INSERT INTO articles (
                    url_hash, title, url, description, published, source, 
                    category, tags, extracted_at, content_hash, raw_content, author, language, 
                    fetch_status, error_message, source_type, media_urls, word_count, 
                    canonical_url, updated_at, processed, quality_score, readability_score,
                    source_reliability, freshness_score, completeness_score, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    url_hash,
                    article.title,
                    article.url,
                    article.description,
                    article.published,
                    article.source,
                    analysis["category"],  # Use analyzed category
                    json.dumps(analysis["tags"]),  # Use analyzed tags
                    article.extracted_at,
                    analysis["content_hash"],
                    getattr(article, "raw_content", None),
                    getattr(article, "author", None),
                    getattr(article, "language", None),
                    getattr(article, "fetch_status", None),
                    getattr(article, "error_message", None),
                    getattr(article, "source_type", None),
                    json.dumps(getattr(article, "media_urls", [])),
                    analysis["word_count"],
                    getattr(article, "canonical_url", None),
                    getattr(article, "updated_at", None),
                    1,  # Mark as processed
                    analysis["quality_score"],
                    analysis["readability_score"],
                    analysis["source_reliability"],
                    analysis["freshness_score"],
                    analysis["completeness_score"],
                    analysis["analyzed_at"],
                ),
            )

            conn.commit()
        # [NEW] Store article in ChromaDB (VectorStore)
        chroma_metadata = {
            "url": article.url,
            "title": article.title,
            "source": article.source,
            "category": analysis["category"],
            "tags": json.dumps(analysis["tags"]),  # Serialize tags as JSON string
            "published": str(article.published) if article.published else None,
            "quality_score": analysis["quality_score"],
            "content_hash": analysis["content_hash"],
        }
        logger.debug(f"Calling add_document with content length: {len(content)}, metadata: {chroma_metadata}")
        self.vector_store.add_document(content, chroma_metadata)
        return True, analysis

    def _get_existing_content_hashes(self) -> List[str]:
        """Get existing content hashes for duplicate detection"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_hash FROM articles WHERE content_hash IS NOT NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_high_quality_articles(
        self, min_quality: float = 0.7, limit: Optional[int] = None
    ) -> List[Dict]:
        """Get articles above a minimum quality threshold"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM articles WHERE quality_score >= ? ORDER BY quality_score DESC"
            params = [min_quality]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]

            articles = []
            for row in cursor.fetchall():
                article_dict = dict(zip(columns, row))
                # Parse JSON tags
                if article_dict["tags"]:
                    try:
                        article_dict["tags"] = json.loads(article_dict["tags"])
                    except:
                        article_dict["tags"] = []
                else:
                    article_dict["tags"] = []
                articles.append(article_dict)

            return articles

    def get_articles_by_category(
        self, category: str, min_quality: float = 0.0
    ) -> List[Dict]:
        """Get articles by category with optional quality filter"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM articles WHERE category = ? AND quality_score >= ? ORDER BY quality_score DESC"
            cursor.execute(query, (category, min_quality))

            columns = [description[0] for description in cursor.description]

            articles = []
            for row in cursor.fetchall():
                article_dict = dict(zip(columns, row))
                # Parse JSON tags
                if article_dict["tags"]:
                    try:
                        article_dict["tags"] = json.loads(article_dict["tags"])
                    except:
                        article_dict["tags"] = []
                else:
                    article_dict["tags"] = []
                articles.append(article_dict)

            return articles

    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics for all articles"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get quality score statistics
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    AVG(quality_score) as avg_quality,
                    MIN(quality_score) as min_quality,
                    MAX(quality_score) as max_quality,
                    COUNT(CASE WHEN quality_score >= 0.7 THEN 1 END) as high_quality,
                    COUNT(CASE WHEN quality_score >= 0.4 AND quality_score < 0.7 THEN 1 END) as medium_quality,
                    COUNT(CASE WHEN quality_score < 0.4 THEN 1 END) as low_quality
                FROM articles
            """
            )

            row = cursor.fetchone()
            stats = {
                "total_articles": row[0],
                "average_quality": round(row[1], 3) if row[1] else 0,
                "min_quality": row[2] if row[2] else 0,
                "max_quality": row[3] if row[3] else 0,
                "high_quality_count": row[4],
                "medium_quality_count": row[5],
                "low_quality_count": row[6],
            }

            return stats

    def store_article(self, article: Article) -> bool:
        """Store article in database, return True if new article"""
        url_hash = self._generate_url_hash(article.url)
        content_hash = self._generate_content_hash(article.title + article.description)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if article already exists
            cursor.execute("SELECT id FROM articles WHERE url_hash = ?", (url_hash,))
            if cursor.fetchone():
                return False  # Article already exists

            # Insert new article
            cursor.execute(
                """
                INSERT INTO articles (
                    url_hash, title, url, description, published, source, 
                    category, tags, extracted_at, content_hash, raw_content, author, language, fetch_status, error_message, source_type, media_urls, word_count, canonical_url, updated_at, processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    url_hash,
                    article.title,
                    article.url,
                    article.description,
                    article.published,
                    article.source,
                    article.category,
                    json.dumps(getattr(article, "tags", [])),
                    article.extracted_at,
                    content_hash,
                    getattr(article, "raw_content", None),
                    getattr(article, "author", None),
                    getattr(article, "language", None),
                    getattr(article, "fetch_status", None),
                    getattr(article, "error_message", None),
                    getattr(article, "source_type", None),
                    json.dumps(getattr(article, "media_urls", [])),
                    getattr(article, "word_count", None),
                    getattr(article, "canonical_url", None),
                    getattr(article, "updated_at", None),
                    0,  # processed flag default
                ),
            )

            conn.commit()
            return True  # New article stored

    def _generate_url_hash(self, url: str) -> str:
        """Generate hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()

    def get_articles(
        self,
        limit: Optional[int] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """Retrieve articles from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM articles WHERE 1=1"
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)

            if category:
                query += " AND category = ?"
                params.append(category)

            if since:
                query += " AND published >= ?"
                params.append(since)

            query += " ORDER BY published DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]

            articles = []
            for row in cursor.fetchall():
                article_dict = dict(zip(columns, row))
                # Parse JSON tags
                if article_dict["tags"]:
                    try:
                        article_dict["tags"] = json.loads(article_dict["tags"])
                    except:
                        article_dict["tags"] = []
                else:
                    article_dict["tags"] = []
                articles.append(article_dict)

            return articles

    def export_to_json(self, filename: Optional[str] = None, **filters) -> str:
        """Export articles to JSON file"""
        articles = self.get_articles(**filters)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(articles)} articles to {filepath}")
        return str(filepath)

    def export_to_csv(self, filename: Optional[str] = None, **filters) -> str:
        """Export articles to CSV file"""
        articles = self.get_articles(**filters)

        if not articles:
            logger.warning("No articles to export")
            return ""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Convert to pandas DataFrame for easier CSV export
        df = pd.DataFrame(articles)

        # Convert tags list to string
        df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if x else "")

        df.to_csv(filepath, index=False, encoding="utf-8")

        logger.info(f"Exported {len(articles)} articles to {filepath}")
        return str(filepath)

    def export_to_excel(self, filename: Optional[str] = None, **filters) -> str:
        """Export articles to Excel file"""
        articles = self.get_articles(**filters)

        if not articles:
            logger.warning("No articles to export")
            return ""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.xlsx"

        filepath = self.output_dir / filename

        # Convert to pandas DataFrame
        df = pd.DataFrame(articles)

        # Convert tags list to string
        df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if x else "")

        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Main articles sheet
            df.to_excel(writer, sheet_name="Articles", index=False)

            # Summary by source
            source_summary = df.groupby("source").size().reset_index(name="count")
            source_summary.to_excel(writer, sheet_name="By Source", index=False)

            # Summary by category
            category_summary = df.groupby("category").size().reset_index(name="count")
            category_summary.to_excel(writer, sheet_name="By Category", index=False)

        logger.info(f"Exported {len(articles)} articles to {filepath}")
        return str(filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total articles
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]

            # Articles by source
            cursor.execute(
                "SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY COUNT(*) DESC"
            )
            by_source = dict(cursor.fetchall())

            # Articles by category
            cursor.execute(
                "SELECT category, COUNT(*) FROM articles GROUP BY category ORDER BY COUNT(*) DESC"
            )
            by_category = dict(cursor.fetchall())

            # Recent articles (last 24 hours)
            cursor.execute(
                """
                SELECT COUNT(*) FROM articles 
                WHERE published >= datetime('now', '-1 day')
            """
            )
            recent_articles = cursor.fetchone()[0]

            # Date range
            cursor.execute(
                "SELECT MIN(published), MAX(published) FROM articles WHERE published IS NOT NULL"
            )
            date_range = cursor.fetchone()

            return {
                "total_articles": total_articles,
                "recent_articles_24h": recent_articles,
                "date_range": {"earliest": date_range[0], "latest": date_range[1]},
                "by_source": by_source,
                "by_category": by_category,
            }

    def deduplicate_articles(self) -> int:
        """Remove duplicate articles based on content hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Find duplicates by content hash
            cursor.execute(
                """
                SELECT content_hash, COUNT(*) as count, MIN(id) as keep_id
                FROM articles 
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash 
                HAVING COUNT(*) > 1
            """
            )

            duplicates = cursor.fetchall()
            deleted_count = 0

            for content_hash, count, keep_id in duplicates:
                # Delete all but the first occurrence
                cursor.execute(
                    """
                    DELETE FROM articles 
                    WHERE content_hash = ? AND id != ?
                """,
                    (content_hash, keep_id),
                )

                deleted_count += count - 1

            conn.commit()

            if deleted_count > 0:
                logger.info(f"Removed {deleted_count} duplicate articles")

            return deleted_count

    def cleanup_old_articles(self, days_to_keep: int = 30) -> int:
        """Remove articles older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM articles 
                WHERE published < datetime('now', '-{} days')
            """.format(
                    days_to_keep
                )
            )

            deleted_count = cursor.rowcount
            conn.commit()

            if deleted_count > 0:
                logger.info(
                    f"Removed {deleted_count} articles older than {days_to_keep} days"
                )

            return deleted_count


class ReportGenerator:
    """Generate reports from processed data"""

    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor

    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate daily summary report"""
        if not date:
            date = datetime.now(timezone.utc)

        # Get articles from the specified date
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date.replace(hour=23, minute=59, second=59)

        articles = self.data_processor.get_articles(since=start_date)
        daily_articles = [
            a
            for a in articles
            if a["published"]
            and start_date
            <= datetime.fromisoformat(a["published"].replace("Z", "+00:00"))
            <= end_date
        ]

        # Generate statistics
        stats = {
            "date": date.strftime("%Y-%m-%d"),
            "total_articles": len(daily_articles),
            "by_source": {},
            "by_category": {},
            "top_articles": [],
        }

        # Count by source and category
        for article in daily_articles:
            source = article["source"]
            category = article["category"]

            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        # Sort by count
        stats["by_source"] = dict(
            sorted(stats["by_source"].items(), key=lambda x: x[1], reverse=True)
        )
        stats["by_category"] = dict(
            sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True)
        )

        # Top articles (by recency and relevance)
        stats["top_articles"] = [
            {
                "title": a["title"],
                "source": a["source"],
                "url": a["url"],
                "published": a["published"],
            }
            for a in daily_articles[:10]  # Top 10
        ]

        return stats

    def save_report(
        self, report: Dict[str, Any], filename: Optional[str] = None
    ) -> str:
        """Save report to file"""
        if not filename:
            date_str = report.get("date", datetime.now().strftime("%Y-%m-%d"))
            filename = f"daily_report_{date_str}.json"

        filepath = self.data_processor.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Report saved to {filepath}")
        return str(filepath)


def main():
    """Test the data processor"""
    try:
        from .rss_extractor import Article
    except ImportError:
        from rss_extractor import Article
    from datetime import datetime, timezone

    # Create sample articles
    sample_articles = [
        Article(
            title="Sample Article 1",
            url="https://example.com/article1",
            description="This is a sample article description",
            published=datetime.now(timezone.utc),
            source="Test Source",
            category="test",
        ),
        Article(
            title="Sample Article 2",
            url="https://example.com/article2",
            description="Another sample article",
            published=datetime.now(timezone.utc),
            source="Test Source 2",
            category="test",
        ),
    ]

    # Process articles
    processor = DataProcessor()
    results = processor.process_articles(sample_articles)

    print("Processing Results:")
    print(json.dumps(results, indent=2))

    # Get statistics
    stats = processor.get_statistics()
    print("\nDatabase Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    # Generate report
    report_gen = ReportGenerator(processor)
    daily_report = report_gen.generate_daily_report()
    print("\nDaily Report:")
    print(json.dumps(daily_report, indent=2, default=str))


if __name__ == "__main__":
    main()
