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
from rss_extractor import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and store extracted articles"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Database setup
        self.db_path = self.output_dir / "articles.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_source ON articles (source)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON articles (category)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_published ON articles (published)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_extracted_at ON articles (extracted_at)
            ''')
            
            conn.commit()
    
    def process_articles(self, articles: List[Article]) -> Dict[str, Any]:
        """Process a list of articles"""
        results = {
            'total_articles': len(articles),
            'new_articles': 0,
            'duplicate_articles': 0,
            'errors': 0,
            'categories': {},
            'sources': {}
        }
        
        for article in articles:
            try:
                is_new = self.store_article(article)
                if is_new:
                    results['new_articles'] += 1
                else:
                    results['duplicate_articles'] += 1
                
                # Update category counts
                if article.category in results['categories']:
                    results['categories'][article.category] += 1
                else:
                    results['categories'][article.category] = 1
                
                # Update source counts
                if article.source in results['sources']:
                    results['sources'][article.source] += 1
                else:
                    results['sources'][article.source] = 1
                    
            except Exception as e:
                logger.error(f"Error processing article '{article.title}': {e}")
                results['errors'] += 1
        
        return results
    
    def store_article(self, article: Article) -> bool:
        """Store article in database, return True if new article"""
        url_hash = self._generate_url_hash(article.url)
        content_hash = self._generate_content_hash(article.title + article.description)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if article already exists
            cursor.execute('SELECT id FROM articles WHERE url_hash = ?', (url_hash,))
            if cursor.fetchone():
                return False  # Article already exists
            
            # Insert new article
            cursor.execute('''
                INSERT INTO articles (
                    url_hash, title, url, description, published, source, 
                    category, tags, extracted_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                url_hash,
                article.title,
                article.url,
                article.description,
                article.published,
                article.source,
                article.category,
                json.dumps(article.tags),
                article.extracted_at,
                content_hash
            ))
            
            conn.commit()
            return True  # New article stored
    
    def _generate_url_hash(self, url: str) -> str:
        """Generate hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_articles(self, limit: Optional[int] = None, source: Optional[str] = None,
                    category: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict]:
        """Retrieve articles from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM articles WHERE 1=1'
            params = []
            
            if source:
                query += ' AND source = ?'
                params.append(source)
            
            if category:
                query += ' AND category = ?'
                params.append(category)
            
            if since:
                query += ' AND published >= ?'
                params.append(since)
            
            query += ' ORDER BY published DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            articles = []
            for row in cursor.fetchall():
                article_dict = dict(zip(columns, row))
                # Parse JSON tags
                if article_dict['tags']:
                    try:
                        article_dict['tags'] = json.loads(article_dict['tags'])
                    except:
                        article_dict['tags'] = []
                else:
                    article_dict['tags'] = []
                articles.append(article_dict)
            
            return articles
    
    def export_to_json(self, filename: Optional[str] = None, **filters) -> str:
        """Export articles to JSON file"""
        articles = self.get_articles(**filters)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
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
        df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else '')
        
        df.to_csv(filepath, index=False, encoding='utf-8')
        
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
        df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if x else '')
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main articles sheet
            df.to_excel(writer, sheet_name='Articles', index=False)
            
            # Summary by source
            source_summary = df.groupby('source').size().reset_index(name='count')
            source_summary.to_excel(writer, sheet_name='By Source', index=False)
            
            # Summary by category
            category_summary = df.groupby('category').size().reset_index(name='count')
            category_summary.to_excel(writer, sheet_name='By Category', index=False)
        
        logger.info(f"Exported {len(articles)} articles to {filepath}")
        return str(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total articles
            cursor.execute('SELECT COUNT(*) FROM articles')
            total_articles = cursor.fetchone()[0]
            
            # Articles by source
            cursor.execute('SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY COUNT(*) DESC')
            by_source = dict(cursor.fetchall())
            
            # Articles by category
            cursor.execute('SELECT category, COUNT(*) FROM articles GROUP BY category ORDER BY COUNT(*) DESC')
            by_category = dict(cursor.fetchall())
            
            # Recent articles (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM articles 
                WHERE published >= datetime('now', '-1 day')
            ''')
            recent_articles = cursor.fetchone()[0]
            
            # Date range
            cursor.execute('SELECT MIN(published), MAX(published) FROM articles WHERE published IS NOT NULL')
            date_range = cursor.fetchone()
            
            return {
                'total_articles': total_articles,
                'recent_articles_24h': recent_articles,
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                },
                'by_source': by_source,
                'by_category': by_category
            }
    
    def deduplicate_articles(self) -> int:
        """Remove duplicate articles based on content hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find duplicates by content hash
            cursor.execute('''
                SELECT content_hash, COUNT(*) as count, MIN(id) as keep_id
                FROM articles 
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash 
                HAVING COUNT(*) > 1
            ''')
            
            duplicates = cursor.fetchall()
            deleted_count = 0
            
            for content_hash, count, keep_id in duplicates:
                # Delete all but the first occurrence
                cursor.execute('''
                    DELETE FROM articles 
                    WHERE content_hash = ? AND id != ?
                ''', (content_hash, keep_id))
                
                deleted_count += count - 1
            
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Removed {deleted_count} duplicate articles")
            
            return deleted_count
    
    def cleanup_old_articles(self, days_to_keep: int = 30) -> int:
        """Remove articles older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM articles 
                WHERE published < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Removed {deleted_count} articles older than {days_to_keep} days")
            
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
            a for a in articles 
            if a['published'] and 
            start_date <= datetime.fromisoformat(a['published'].replace('Z', '+00:00')) <= end_date
        ]
        
        # Generate statistics
        stats = {
            'date': date.strftime('%Y-%m-%d'),
            'total_articles': len(daily_articles),
            'by_source': {},
            'by_category': {},
            'top_articles': []
        }
        
        # Count by source and category
        for article in daily_articles:
            source = article['source']
            category = article['category']
            
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        # Sort by count
        stats['by_source'] = dict(sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True))
        stats['by_category'] = dict(sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True))
        
        # Top articles (by recency and relevance)
        stats['top_articles'] = [
            {
                'title': a['title'],
                'source': a['source'],
                'url': a['url'],
                'published': a['published']
            }
            for a in daily_articles[:10]  # Top 10
        ]
        
        return stats
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save report to file"""
        if not filename:
            date_str = report.get('date', datetime.now().strftime('%Y-%m-%d'))
            filename = f"daily_report_{date_str}.json"
        
        filepath = self.data_processor.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)

def main():
    """Test the data processor"""
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
            category="test"
        ),
        Article(
            title="Sample Article 2",
            url="https://example.com/article2",
            description="Another sample article",
            published=datetime.now(timezone.utc),
            source="Test Source 2",
            category="test"
        )
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
        