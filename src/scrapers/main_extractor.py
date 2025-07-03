"""
Main extraction orchestrator - combines RSS and web scraping
Enhanced with crawl4ai integration
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import argparse
import json
from pathlib import Path
import sys

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when imported as module)
    from .config_loader import ConfigLoader, SourceConfig
    from .rss_extractor import RSSExtractor
    from .crawl4ai_web_scraper import Crawl4AiWebScraper, SmartCrawl4AiWebScraper, WebScraperWrapper
    from .data_processor import DataProcessor, ReportGenerator
    # For fallback compatibility, keep the old web scraper as backup
    try:
        from .web_scraper import SmartWebScraper as LegacyWebScraper
    except ImportError:
        LegacyWebScraper = None
except ImportError:
    # If relative imports fail, try absolute imports (when run directly)
    from config_loader import ConfigLoader, SourceConfig
    from rss_extractor import RSSExtractor
    from crawl4ai_web_scraper import Crawl4AiWebScraper, SmartCrawl4AiWebScraper, WebScraperWrapper
    from data_processor import DataProcessor, ReportGenerator
    # For fallback compatibility, keep the old web scraper as backup
    try:
        from web_scraper import SmartWebScraper as LegacyWebScraper
    except ImportError:
        LegacyWebScraper = None

# Ensure logs directory exists at repo root
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "extraction.log"

# Remove all handlers associated with the root logger object (to avoid duplicate logs)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up file and stream handlers explicitly
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
logger = logging.getLogger(__name__)


class NewsExtractor:
    """Main news extraction orchestrator with crawl4ai integration"""

    def __init__(
        self,
        config_path: str = "src/sources.yaml",
        output_dir: str = "output",
        use_crawl4ai: bool = True,
        use_smart_scraper: bool = True,
        use_llm_extraction: bool = False,
        use_selenium_fallback: bool = False,
        max_articles_per_source: int = 50,
        browser_type: str = "chromium",
        max_concurrent: int = 3,
        timeout: int = 30,
    ):
        self.config = ConfigLoader(config_path)
        self.data_processor = DataProcessor(output_dir)
        self.report_generator = ReportGenerator(self.data_processor)

        # Initialize extractors
        self.rss_extractor = RSSExtractor()
        
        # Web scraper configuration
        self.use_crawl4ai = use_crawl4ai
        self.use_smart_scraper = use_smart_scraper
        self.use_llm_extraction = use_llm_extraction
        self.use_selenium_fallback = use_selenium_fallback
        self.browser_type = browser_type
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # Initialize web scraper based on configuration
        self.web_scraper = self._initialize_web_scraper()

        self.max_articles_per_source = max_articles_per_source
        self.extraction_stats = {
            "start_time": None,
            "end_time": None,
            "total_sources": 0,
            "successful_sources": 0,
            "failed_sources": 0,
            "total_articles": 0,
            "rss_articles": 0,
            "web_articles": 0,
            "scraper_type": self._get_scraper_type(),
            "scraper_config": self._get_scraper_config(),
            "errors": [],
        }

    def _initialize_web_scraper(self):
        """Initialize the appropriate web scraper based on configuration"""
        if self.use_crawl4ai:
            try:
                logger.info("Initializing crawl4ai web scraper")
                
                scraper_kwargs = {
                    'timeout': self.timeout,
                    'headless': True,
                    'use_llm_extraction': self.use_llm_extraction,
                    'max_concurrent': self.max_concurrent,
                    'browser_type': self.browser_type,
                }
                
                if self.use_smart_scraper:
                    # Use smart scraper with advanced features
                    logger.info("Using SmartCrawl4AiWebScraper with enhanced AI capabilities")
                    return WebScraperWrapper(
                        scraper_class=SmartCrawl4AiWebScraper,
                        **scraper_kwargs
                    )
                else:
                    # Use standard crawl4ai scraper
                    logger.info("Using standard Crawl4AiWebScraper")
                    return WebScraperWrapper(
                        scraper_class=Crawl4AiWebScraper,
                        **scraper_kwargs
                    )
                    
            except Exception as e:
                logger.error(f"Failed to initialize crawl4ai scraper: {e}")
                if LegacyWebScraper and self.use_selenium_fallback:
                    logger.info("Falling back to legacy web scraper")
                    return LegacyWebScraper(use_selenium=self.use_selenium_fallback)
                else:
                    raise
        else:
            # Use legacy scraper if explicitly requested
            if LegacyWebScraper:
                logger.info("Using legacy web scraper")
                return LegacyWebScraper(use_selenium=self.use_selenium_fallback)
            else:
                raise ImportError("Legacy web scraper not available and crawl4ai disabled")

    def _get_scraper_type(self) -> str:
        """Get the type of scraper being used"""
        if self.use_crawl4ai:
            if self.use_smart_scraper:
                return "crawl4ai_smart"
            else:
                return "crawl4ai_standard"
        else:
            return "legacy_selenium" if self.use_selenium_fallback else "legacy_requests"

    def _get_scraper_config(self) -> Dict[str, Any]:
        """Get scraper configuration details"""
        return {
            "use_crawl4ai": self.use_crawl4ai,
            "use_smart_scraper": self.use_smart_scraper,
            "use_llm_extraction": self.use_llm_extraction,
            "browser_type": self.browser_type,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "use_selenium_fallback": self.use_selenium_fallback,
        }

    def extract_from_all_sources(
        self, categories: List[str] = None, source_types: List[str] = None
    ) -> Dict[str, Any]:
        """Extract articles from all configured sources"""
        logger.info("Starting extraction from all sources")
        self.extraction_stats["start_time"] = datetime.now(timezone.utc)

        # Get sources to process
        sources = self._get_sources_to_process(categories, source_types)
        self.extraction_stats["total_sources"] = len(sources)

        logger.info(f"Processing {len(sources)} sources with {self.extraction_stats['scraper_type']} scraper")

        all_articles = []

        # Process RSS sources
        rss_sources = [s for s in sources if s.type == "rss"]
        if rss_sources:
            logger.info(f"Processing {len(rss_sources)} RSS sources")
            rss_articles = self._extract_rss_articles(rss_sources)
            all_articles.extend(rss_articles)
            self.extraction_stats["rss_articles"] = len(rss_articles)

        # Process website sources
        web_sources = [s for s in sources if s.type == "website"]
        if web_sources:
            logger.info(f"Processing {len(web_sources)} website sources")
            web_articles = self._extract_web_articles(web_sources)
            all_articles.extend(web_articles)
            self.extraction_stats["web_articles"] = len(web_articles)

        # Process and store articles
        logger.info(f"Processing {len(all_articles)} total articles")
        processing_results = self.data_processor.process_articles(all_articles)

        # Update statistics
        self.extraction_stats["end_time"] = datetime.now(timezone.utc)
        self.extraction_stats["total_articles"] = len(all_articles)
        self.extraction_stats["successful_sources"] = len([s for s in sources if any(
            a.source == s.name for a in all_articles
        )])
        self.extraction_stats["failed_sources"] = self.extraction_stats["total_sources"] - self.extraction_stats["successful_sources"]

        # Combine results
        results = {
            "extraction_stats": self.extraction_stats,
            "processing_results": processing_results,
            "articles": [
                article.to_dict() for article in all_articles[:100]
            ],  # First 100 for preview
        }

        # Log final statistics
        duration = (self.extraction_stats["end_time"] - self.extraction_stats["start_time"]).total_seconds()
        logger.info(f"Extraction completed in {duration:.2f} seconds")
        logger.info(f"Successfully processed {self.extraction_stats['successful_sources']}/{self.extraction_stats['total_sources']} sources")
        logger.info(f"Total articles: {self.extraction_stats['total_articles']} (RSS: {self.extraction_stats['rss_articles']}, Web: {self.extraction_stats['web_articles']})")

        return results

    def _get_sources_to_process(
        self, categories: List[str] = None, source_types: List[str] = None
    ) -> List[SourceConfig]:
        """Get filtered list of sources to process"""
        sources = self.config.get_active_sources()

        if categories:
            sources = [s for s in sources if s.category in categories]

        if source_types:
            sources = [s for s in sources if s.type in source_types]

        return sources

    def _extract_rss_articles(self, sources: List[SourceConfig]) -> List:
        """Extract articles from RSS sources"""
        all_articles = []
        
        for source in sources:
            try:
                articles = self.rss_extractor.extract_from_source(source)
                
                # Limit articles per source
                if self.max_articles_per_source and len(articles) > self.max_articles_per_source:
                    articles = articles[:self.max_articles_per_source]
                    logger.info(f"Limited {source.name} to {self.max_articles_per_source} articles")
                
                all_articles.extend(articles)
                logger.info(f"Successfully extracted {len(articles)} articles from RSS: {source.name}")
                
            except Exception as e:
                error_msg = f"Failed to extract from RSS source {source.name}: {str(e)}"
                logger.error(error_msg)
                self.extraction_stats["errors"].append({
                    "source": source.name,
                    "type": "rss",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        return all_articles

    def _extract_web_articles(self, sources: List[SourceConfig]) -> List:
        """Extract articles from website sources"""
        all_articles = []
        
        try:
            # Use batch processing for efficiency with crawl4ai
            if hasattr(self.web_scraper, 'extract_from_multiple_sources'):
                logger.info("Using batch extraction for website sources")
                articles = self.web_scraper.extract_from_multiple_sources(sources)
                
                # Apply per-source limits
                source_article_count = {}
                filtered_articles = []
                
                for article in articles:
                    source_name = article.source
                    if source_name not in source_article_count:
                        source_article_count[source_name] = 0
                    
                    if source_article_count[source_name] < self.max_articles_per_source:
                        filtered_articles.append(article)
                        source_article_count[source_name] += 1
                
                all_articles.extend(filtered_articles)
                
                # Log per-source results
                for source in sources:
                    count = source_article_count.get(source.name, 0)
                    logger.info(f"Extracted {count} articles from website: {source.name}")
                
            else:
                # Fallback to individual source processing
                logger.info("Using individual extraction for website sources")
                for source in sources:
                    try:
                        articles = self.web_scraper.extract_from_source(source)
                        
                        # Limit articles per source
                        if self.max_articles_per_source and len(articles) > self.max_articles_per_source:
                            articles = articles[:self.max_articles_per_source]
                            logger.info(f"Limited {source.name} to {self.max_articles_per_source} articles")
                        
                        all_articles.extend(articles)
                        logger.info(f"Successfully extracted {len(articles)} articles from website: {source.name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to extract from website source {source.name}: {str(e)}"
                        logger.error(error_msg)
                        self.extraction_stats["errors"].append({
                            "source": source.name,
                            "type": "website", 
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })

        except Exception as e:
            error_msg = f"Failed to extract from website sources: {str(e)}"
            logger.error(error_msg)
            self.extraction_stats["errors"].append({
                "source": "all_websites",
                "type": "website_batch",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return all_articles

    def extract_from_category(self, category: str) -> Dict[str, Any]:
        """Extract articles from sources in a specific category"""
        return self.extract_from_all_sources(categories=[category])

    def extract_rss_only(self) -> Dict[str, Any]:
        """Extract articles from RSS sources only"""
        return self.extract_from_all_sources(source_types=["rss"])

    def extract_websites_only(self) -> Dict[str, Any]:
        """Extract articles from website sources only"""
        return self.extract_from_all_sources(source_types=["website"])

    def generate_report(self, format: str = "json") -> str:
        """Generate a report of extracted data"""
        if format == "json":
            return self.report_generator.generate_json_report()
        elif format == "html":
            return self.report_generator.generate_html_report()
        elif format == "csv":
            return self.report_generator.generate_csv_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_articles(self, format: str = "csv", **filters) -> str:
        """Export articles in specified format with optional filters"""
        return self.data_processor.export_articles(format=format, **filters)

    def cleanup_data(
        self,
        remove_duplicates: bool = True,
        remove_old_articles: bool = False,
        days_to_keep: int = 30,
    ):
        """Clean up stored data"""
        return self.data_processor.cleanup_data(
            remove_duplicates=remove_duplicates,
            remove_old_articles=remove_old_articles,
            days_to_keep=days_to_keep,
        )

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self.web_scraper, 'cleanup') and callable(self.web_scraper.cleanup):
            try:
                self.web_scraper.cleanup()
            except Exception:
                pass  # Ignore cleanup errors during deletion


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="News Extraction Tool with Crawl4AI")
    parser.add_argument(
        "--config", 
        default="src/sources.yaml", 
        help="Path to sources configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        default="output", 
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        help="Specific categories to extract"
    )
    parser.add_argument(
        "--source-types", 
        nargs="+", 
        choices=["rss", "website"], 
        help="Types of sources to extract from"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "csv", "html"], 
        default="json", 
        help="Output format for reports"
    )
    parser.add_argument(
        "--max-articles", 
        type=int, 
        default=50, 
        help="Maximum articles per source"
    )
    parser.add_argument(
        "--no-crawl4ai", 
        action="store_true", 
        help="Disable crawl4ai and use legacy scraper"
    )
    parser.add_argument(
        "--smart-scraper", 
        action="store_true", 
        default=True,
        help="Use smart crawl4ai scraper with AI capabilities"
    )
    parser.add_argument(
        "--llm-extraction", 
        action="store_true", 
        help="Use LLM for content extraction (requires LLM setup)"
    )
    parser.add_argument(
        "--browser-type", 
        choices=["chromium", "firefox", "webkit"], 
        default="chromium",
        help="Browser type for crawl4ai"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=3, 
        help="Maximum concurrent web scraping tasks"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30, 
        help="Timeout for web requests in seconds"
    )
    parser.add_argument(
        "--selenium-fallback", 
        action="store_true", 
        help="Enable Selenium fallback for legacy scraper"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    def logprint(*a, **k):
        print(*a, **k)
        logger.info(" ".join(str(x) for x in a))

    try:
        # Initialize extractor with CLI arguments
        extractor = NewsExtractor(
            config_path=args.config,
            output_dir=args.output_dir,
            use_crawl4ai=not args.no_crawl4ai,
            use_smart_scraper=args.smart_scraper and not args.no_crawl4ai,
            use_llm_extraction=args.llm_extraction,
            use_selenium_fallback=args.selenium_fallback,
            max_articles_per_source=args.max_articles,
            browser_type=args.browser_type,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
        )

        logprint(f"Initialized extractor with {extractor.extraction_stats['scraper_type']} scraper")
        logprint(f"Scraper configuration: {extractor.extraction_stats['scraper_config']}")

        # Perform extraction
        results = extractor.extract_from_all_sources(
            categories=args.categories, 
            source_types=args.source_types
        )

        # Generate report
        report_path = extractor.generate_report(format=args.format)
        logprint(f"Report generated: {report_path}")

        # Print summary
        stats = results["extraction_stats"]
        logprint(f"\n=== EXTRACTION SUMMARY ===")
        logprint(f"Scraper Type: {stats['scraper_type']}")
        logprint(f"Total Sources: {stats['total_sources']}")
        logprint(f"Successful Sources: {stats['successful_sources']}")
        logprint(f"Failed Sources: {stats['failed_sources']}")
        logprint(f"Total Articles: {stats['total_articles']}")
        logprint(f"RSS Articles: {stats['rss_articles']}")
        logprint(f"Web Articles: {stats['web_articles']}")
        
        if stats['errors']:
            logprint(f"Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                logprint(f"  - {error['source']} ({error['type']}): {error['error']}")

        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        logprint(f"Duration: {duration:.2f} seconds")

    except KeyboardInterrupt:
        logprint("Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logprint(f"Error during extraction: {e}")
        logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
