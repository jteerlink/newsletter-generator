"""
Main extraction orchestrator - combines RSS and web scraping
"""
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import argparse
import json
from pathlib import Path

from config_loader import ConfigLoader, SourceConfig
from rss_extractor import RSSExtractor
from web_scraper import SmartWebScraper
from data_processor import DataProcessor, ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsExtractor:
    """Main news extraction orchestrator"""
    
    def __init__(self, config_path: str = 'sources.yaml', output_dir: str = 'output',
                 use_selenium: bool = False, max_articles_per_source: int = 50):
        self.config = ConfigLoader(config_path)
        self.data_processor = DataProcessor(output_dir)
        self.report_generator = ReportGenerator(self.data_processor)
        
        # Initialize extractors
        self.rss_extractor = RSSExtractor()
        self.web_scraper = SmartWebScraper(use_selenium=use_selenium)
        
        self.max_articles_per_source = max_articles_per_source
        self.extraction_stats = {
            'start_time': None,
            'end_time': None,
            'total_sources': 0,
            'successful_sources': 0,
            'failed_sources': 0,
            'total_articles': 0,
            'rss_articles': 0,
            'web_articles': 0,
            'errors': []
        }
    
    def extract_from_all_sources(self, categories: List[str] = None,
                               source_types: List[str] = None) -> Dict[str, Any]:
        """Extract articles from all configured sources"""
        logger.info("Starting extraction from all sources")
        self.extraction_stats['start_time'] = datetime.now(timezone.utc)
        
        # Get sources to process
        sources = self._get_sources_to_process(categories, source_types)
        self.extraction_stats['total_sources'] = len(sources)
        
        logger.info(f"Processing {len(sources)} sources")
        
        all_articles = []
        
        # Process RSS sources
        rss_sources = [s for s in sources if s.type == 'rss']
        if rss_sources:
            logger.info(f"Processing {len(rss_sources)} RSS sources")
            rss_articles = self._extract_rss_articles(rss_sources)
            all_articles.extend(rss_articles)
            self.extraction_stats['rss_articles'] = len(rss_articles)
        
        # Process website sources
        web_sources = [s for s in sources if s.type == 'website']
        if web_sources:
            logger.info(f"Processing {len(web_sources)} website sources")
            web_articles = self._extract_web_articles(web_sources)
            all_articles.extend(web_articles)
            self.extraction_stats['web_articles'] = len(web_articles)
        
        # Process and store articles
        logger.info(f"Processing {len(all_articles)} total articles")
        processing_results = self.data_processor.process_articles(all_articles)
        
        # Update statistics
        self.extraction_stats['end_time'] = datetime.now(timezone.utc)
        self.extraction_stats['total_articles'] = len(all_articles)
        
        # Combine results
        results = {
            'extraction_stats': self.extraction_stats,
            'processing_results': processing_results,
            'articles': [article.to_dict() for article in all_articles[:100]]  # First 100 for preview
        }
        
        logger.info("Extraction completed successfully")
        return results
    
    def _get_sources_to_process(self, categories: List[str] = None,
                              source_types: List[str] = None) -> List[SourceConfig]:
        """Get filtered list of sources to process"""
        sources = self.config.get_active_sources()
        
        if categories:
            sources = [s for s in sources if s.category in categories]
        
        if source_types:
            sources = [s for s in sources if s.type in source_types]
        
        return sources
    
    def _extract_rss_articles(self, sources: List[SourceConfig]) -> List:
        """Extract articles from RSS sources"""
        articles = []
        
        for source in sources:
            try:
                logger.info(f"Extracting from RSS: {source.name}")
                source_articles = self.rss_extractor.extract_from_source(source)
                
                # Limit articles per source
                if len(source_articles) > self.max_articles_per_source:
                    source_articles = source_articles[:self.max_articles_per_source]
                    logger.info(f"Limited to {self.max_articles_per_source} articles from {source.name}")
                
                articles.extend(source_articles)
                self.extraction_stats['successful_sources'] += 1
                
            except Exception as e:
                logger.error(f"Failed to extract from RSS source {source.name}: {e}")
                self.extraction_stats['failed_sources'] += 1
                self.extraction_stats['errors'].append({
                    'source': source.name,
                    'type': 'rss',
                    'error': str(e)
                })
            
            # Add delay between requests
            time.sleep(1)
        
        return articles
    
    def _extract_web_articles(self, sources: List[SourceConfig]) -> List:
        """Extract articles from website sources"""
        articles = []
        
        for source in sources:
            try:
                logger.info(f"Scraping website: {source.name}")
                source_articles = self.web_scraper.extract_from_source(source)
                
                # Limit articles per source
                if len(source_articles) > self.max_articles_per_source:
                    source_articles = source_articles[:self.max_articles_per_source]
                    logger.info(f"Limited to {self.max_articles_per_source} articles from {source.name}")
                
                articles.extend(source_articles)
                self.extraction_stats['successful_sources'] += 1
                
            except Exception as e:
                logger.error(f"Failed to scrape website {source.name}: {e}")
                self.extraction_stats['failed_sources'] += 1
                self.extraction_stats['errors'].append({
                    'source': source.name,
                    'type': 'website',
                    'error': str(e)
                })
            
            # Add longer delay between website scraping
            time.sleep(2)
        
        return articles
    
    def extract_from_category(self, category: str) -> Dict[str, Any]:
        """Extract articles from sources in a specific category"""
        return self.extract_from_all_sources(categories=[category])
    
    def extract_rss_only(self) -> Dict[str, Any]:
        """Extract articles from RSS sources only"""
        return self.extract_from_all_sources(source_types=['rss'])
    
    def extract_websites_only(self) -> Dict[str, Any]:
        """Extract articles from website sources only"""
        return self.extract_from_all_sources(source_types=['website'])
    
    def generate_report(self, format: str = 'json') -> str:
        """Generate and save extraction report"""
        stats = self.data_processor.get_statistics()
        daily_report = self.report_generator.generate_daily_report()
        
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'database_stats': stats,
            'daily_summary': daily_report,
            'extraction_stats': self.extraction_stats
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'json':
            filename = f"extraction_report_{timestamp}.json"
            filepath = self.data_processor.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def export_articles(self, format: str = 'csv', **filters) -> str:
        """Export articles in specified format"""
        if format.lower() == 'json':
            return self.data_processor.export_to_json(**filters)
        elif format.lower() == 'csv':
            return self.data_processor.export_to_csv(**filters)
        elif format.lower() == 'excel':
            return self.data_processor.export_to_excel(**filters)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_data(self, remove_duplicates: bool = True, 
                    remove_old_articles: bool = False, days_to_keep: int = 30):
        """Clean up stored data"""
        if remove_duplicates:
            duplicates_removed = self.data_processor.deduplicate_articles()
            logger.info(f"Removed {duplicates_removed} duplicate articles")
        
        if remove_old_articles:
            old_removed = self.data_processor.cleanup_old_articles(days_to_keep)
            logger.info(f"Removed {old_removed} old articles")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='AI News Extractor')
    parser.add_argument('--config', default='sources.yaml', help='Configuration file path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--selenium', action='store_true', help='Use Selenium for web scraping')
    parser.add_argument('--max-articles', type=int, default=50, help='Max articles per source')
    
    # Extraction modes
    parser.add_argument('--rss-only', action='store_true', help='Extract from RSS sources only')
    parser.add_argument('--web-only', action='store_true', help='Extract from websites only')
    parser.add_argument('--category', help='Extract from specific category only')
    
    # Export options
    parser.add_argument('--export', choices=['json', 'csv', 'excel'], help='Export articles')
    parser.add_argument('--export-recent', type=int, help='Export articles from last N hours')
    
    # Maintenance
    parser.add_argument('--cleanup', action='store_true', help='Remove duplicates and old articles')
    parser.add_argument('--report', action='store_true', help='Generate extraction report')
    
    # Info
    parser.add_argument('--list-sources', action='store_true', help='List all configured sources')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = NewsExtractor(
        config_path=args.config,
        output_dir=args.output,
        use_selenium=args.selenium,
        max_articles_per_source=args.max_articles
    )
    
    try:
        # Handle info commands
        if args.list_sources:
            extractor.config.print_summary()
            return
        
        if args.stats:
            stats = extractor.data_processor.get_statistics()
            print(json.dumps(stats, indent=2, default=str))
            return
        
        # Handle maintenance
        if args.cleanup:
            extractor.cleanup_data()
            return
        
        # Handle exports
        if args.export:
            filters = {}
            if args.export_recent:
                from datetime import timedelta
                since = datetime.now(timezone.utc) - timedelta(hours=args.export_recent)
                filters['since'] = since
            
            filepath = extractor.export_articles(format=args.export, **filters)
            print(f"Articles exported to: {filepath}")
            return
        
        # Handle extraction
        if args.rss_only:
            results = extractor.extract_rss_only()
        elif args.web_only:
            results = extractor.extract_websites_only()
        elif args.category:
            results = extractor.extract_from_category(args.category)
        else:
            results = extractor.extract_from_all_sources()
        
        # Print summary
        print("\n=== Extraction Summary ===")
        print(f"Total articles extracted: {results['extraction_stats']['total_articles']}")
        print(f"New articles stored: {results['processing_results']['new_articles']}")
        print(f"Duplicate articles: {results['processing_results']['duplicate_articles']}")
        print(f"Successful sources: {results['extraction_stats']['successful_sources']}")
        print(f"Failed sources: {results['extraction_stats']['failed_sources']}")
        
        if results['extraction_stats']['errors']:
            print(f"\nErrors: {len(results['extraction_stats']['errors'])}")
            for error in results['extraction_stats']['errors'][:5]:  # Show first 5 errors
                print(f"  - {error['source']} ({error['type']}): {error['error']}")
        
        # Generate report if requested
        if args.report:
            report_path = extractor.generate_report()
            print(f"Report saved to: {report_path}")
    
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()
