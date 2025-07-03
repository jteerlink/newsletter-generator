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

from src.scrapers.config_loader import ConfigLoader, SourceConfig
from src.scrapers.rss_extractor import RSSExtractor
from src.scrapers.crawl4ai_web_scraper import Crawl4AiWebScraper, SmartCrawl4AiWebScraper, WebScraperWrapper
from src.scrapers.data_processor import DataProcessor, ReportGenerator

# For fallback compatibility, keep the old web scraper as backup
try:
    from src.scrapers.web_scraper import SmartWebScraper as LegacyWebScraper
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
            "errors": [],
        }

    def _initialize_web_scraper(self):
        """Initialize the appropriate web scraper based on configuration"""
        if self.use_crawl4ai:
            try:
                logger.info("Initializing crawl4ai web scraper")
                
                if self.use_smart_scraper:
                    # Use smart scraper with advanced features
                    return WebScraperWrapper(
                        use_llm_extraction=self.use_llm_extraction,
                        headless=True,
                        max_concurrent=3  # Conservative concurrent limit
                    )
                else:
                    # Use standard crawl4ai scraper
                    return WebScraperWrapper(
                        use_llm_extraction=False,
                        headless=True,
                        max_concurrent=3
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

        # Combine results
        results = {
            "extraction_stats": self.extraction_stats,
            "processing_results": processing_results,
            "articles": [
                article.to_dict() for article in all_articles[:100]
            ],  # First 100 for preview
        }

        logger.info("Extraction completed successfully")
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
        articles = []

        for source in sources:
            try:
                logger.info(f"Extracting from RSS: {source.name}")
                source_articles = self.rss_extractor.extract_from_source(source)

                # Limit articles per source
                if len(source_articles) > self.max_articles_per_source:
                    source_articles = source_articles[: self.max_articles_per_source]
                    logger.info(
                        f"Limited to {self.max_articles_per_source} articles from {source.name}"
                    )

                articles.extend(source_articles)
                self.extraction_stats["successful_sources"] += 1

            except Exception as e:
                logger.error(f"Failed to extract from RSS source {source.name}: {e}")
                self.extraction_stats["failed_sources"] += 1
                self.extraction_stats["errors"].append(
                    {"source": source.name, "type": "rss", "error": str(e)}
                )

            # Add delay between requests
            time.sleep(1)

        return articles

    def _extract_web_articles(self, sources: List[SourceConfig]) -> List:
        """Extract articles from website sources using crawl4ai or legacy scraper"""
        articles = []

        for source in sources:
            try:
                logger.info(f"Scraping website: {source.name} (using {self.extraction_stats['scraper_type']})")
                source_articles = self.web_scraper.extract_from_source(source)

                # Limit articles per source
                if len(source_articles) > self.max_articles_per_source:
                    source_articles = source_articles[: self.max_articles_per_source]
                    logger.info(
                        f"Limited to {self.max_articles_per_source} articles from {source.name}"
                    )

                articles.extend(source_articles)
                self.extraction_stats["successful_sources"] += 1
                
                # Log success with scraper type
                logger.info(f"Successfully extracted {len(source_articles)} articles from {source.name}")

            except Exception as e:
                logger.error(f"Failed to scrape website {source.name}: {e}")
                self.extraction_stats["failed_sources"] += 1
                self.extraction_stats["errors"].append(
                    {"source": source.name, "type": "website", "error": str(e)}
                )

            # Add longer delay between website scraping
            time.sleep(2)

        return articles

    def extract_from_category(self, category: str) -> Dict[str, Any]:
        """Extract articles from sources in a specific category"""
        return self.extract_from_all_sources(categories=[category])

    def extract_rss_only(self) -> Dict[str, Any]:
        """Extract articles only from RSS sources"""
        return self.extract_from_all_sources(source_types=["rss"])

    def extract_websites_only(self) -> Dict[str, Any]:
        """Extract articles only from website sources"""
        return self.extract_from_all_sources(source_types=["website"])

    def generate_report(self, format: str = "json") -> str:
        """Generate extraction report"""
        # Get latest extraction data
        stats = self.extraction_stats
        
        if format == "json":
            return json.dumps(stats, indent=2, default=str)
        elif format == "summary":
            duration = None
            if stats["start_time"] and stats["end_time"]:
                duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            
            summary = f"""
=== Newsletter Extraction Report ===
Scraper Type: {stats['scraper_type']}
Total Sources: {stats['total_sources']}
Successful: {stats['successful_sources']}
Failed: {stats['failed_sources']}
Total Articles: {stats['total_articles']}
RSS Articles: {stats['rss_articles']}
Web Articles: {stats['web_articles']}
Duration: {duration:.2f}s
Errors: {len(stats['errors'])}
"""
            return summary
        else:
            return self.report_generator.generate_report(format)

    def export_articles(self, format: str = "csv", **filters) -> str:
        """Export articles in specified format"""
        return self.data_processor.export_articles(format, **filters)

    def cleanup_data(
        self,
        remove_duplicates: bool = True,
        remove_old_articles: bool = False,
        days_to_keep: int = 30,
    ):
        """Clean up old and duplicate data"""
        self.data_processor.cleanup_data(
            remove_duplicates=remove_duplicates,
            remove_old_articles=remove_old_articles,
            days_to_keep=days_to_keep,
        )

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self.web_scraper, '__del__'):
                self.web_scraper.__del__()
        except:
            pass


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Newsletter Content Extractor")
    parser.add_argument(
        "--config", 
        default="src/sources.yaml", 
        help="Path to sources configuration file"
    )
    parser.add_argument(
        "--output", 
        default="output", 
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--categories", 
        nargs="+", 
        help="Extract only from specific categories"
    )
    parser.add_argument(
        "--source-types", 
        nargs="+", 
        choices=["rss", "website"], 
        help="Extract only from specific source types"
    )
    parser.add_argument(
        "--rss-only", 
        action="store_true", 
        help="Extract only from RSS sources"
    )
    parser.add_argument(
        "--websites-only", 
        action="store_true", 
        help="Extract only from website sources"
    )
    parser.add_argument(
        "--use-legacy-scraper", 
        action="store_true", 
        help="Use legacy web scraper instead of crawl4ai"
    )
    parser.add_argument(
        "--use-llm-extraction", 
        action="store_true", 
        help="Enable LLM-based content extraction (requires LLM setup)"
    )
    parser.add_argument(
        "--use-selenium-fallback", 
        action="store_true", 
        help="Enable Selenium fallback for legacy scraper"
    )
    parser.add_argument(
        "--max-articles", 
        type=int, 
        default=50, 
        help="Maximum articles per source"
    )
    parser.add_argument(
        "--report-format", 
        choices=["json", "summary", "csv"], 
        default="summary", 
        help="Report format"
    )

    args = parser.parse_args()

    def logprint(*a, **k):
        print(*a, **k)
        logger.info(" ".join(str(x) for x in a))

    try:
        # Initialize extractor with enhanced configuration
        extractor = NewsExtractor(
            config_path=args.config,
            output_dir=args.output,
            use_crawl4ai=not args.use_legacy_scraper,
            use_smart_scraper=True,
            use_llm_extraction=args.use_llm_extraction,
            use_selenium_fallback=args.use_selenium_fallback,
            max_articles_per_source=args.max_articles,
        )

        logprint(f"Starting newsletter extraction with {extractor.extraction_stats['scraper_type']} scraper...")

        # Determine extraction method
        if args.rss_only:
            results = extractor.extract_rss_only()
        elif args.websites_only:
            results = extractor.extract_websites_only()
        else:
            results = extractor.extract_from_all_sources(
                categories=args.categories,
                source_types=args.source_types,
            )

        # Generate and display report
        report = extractor.generate_report(format=args.report_format)
        logprint(report)

        logprint("Newsletter extraction completed successfully!")

    except KeyboardInterrupt:
        logprint("Extraction interrupted by user")
    except Exception as e:
        logprint(f"Extraction failed: {e}")
        logger.exception("Detailed error information:")
        raise


if __name__ == "__main__":
    main()
