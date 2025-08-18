"""
Main extraction orchestrator - combines RSS and web scraping
Enhanced with crawl4ai integration
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when imported as module)
    from .config_loader import ConfigLoader, SourceConfig
    from .crawl4ai_web_scraper import (
        Crawl4AiWebScraper,
        SmartCrawl4AiWebScraper,
        WebScraperWrapper,
    )
    from .data_processor import DataProcessor, ReportGenerator
    from .rss_extractor import RSSExtractor
except ImportError:
    # If relative imports fail, try absolute imports (when run directly)
    try:
        from config_loader import ConfigLoader, SourceConfig
        from crawl4ai_web_scraper import (
            Crawl4AiWebScraper,
            SmartCrawl4AiWebScraper,
            WebScraperWrapper,
        )
        from data_processor import DataProcessor, ReportGenerator
        from rss_extractor import RSSExtractor
    except ImportError:
        from scrapers.config_loader import ConfigLoader, SourceConfig
        from scrapers.crawl4ai_web_scraper import (
            Crawl4AiWebScraper,
            SmartCrawl4AiWebScraper,
            WebScraperWrapper,
        )
        from scrapers.data_processor import DataProcessor, ReportGenerator
        from scrapers.rss_extractor import RSSExtractor

# Ensure logs directory exists at repo root
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "extraction.log"

# Remove all handlers associated with the root logger object (to avoid
# duplicate logs)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up file and stream handlers explicitly
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(__name__)


class NewsExtractor:
    """
    Main news extraction orchestrator that combines RSS and web scraping.
    Uses crawl4ai for enhanced web content extraction.
    """

    def __init__(self, config_path: str = "sources.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.rss_extractor = RSSExtractor()
        self.web_scraper = SmartCrawl4AiWebScraper()
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator()

    def extract_news(self,
                     sources: List[str] = None,
                     max_articles: int = 50) -> Dict[str,
                                                     Any]:
        """
        Extract news from configured sources.

        Args:
            sources: List of source names to extract from (None for all)
            max_articles: Maximum number of articles to extract per source

        Returns:
            Dictionary containing extracted articles and metadata
        """
        logger.info("Starting news extraction process")

        # Load configuration
        config = self.config_loader.load_config()

        if sources:
            # Filter to specified sources
            config.sources = {
                name: source for name,
                source in config.sources.items() if name in sources}

        all_articles = []

        for source_name, source_config in config.sources.items():
            logger.info(f"Processing source: {source_name}")

            try:
                if source_config.type == "rss":
                    articles = self._extract_rss(source_config, max_articles)
                elif source_config.type == "web":
                    articles = self._extract_web(source_config, max_articles)
                else:
                    logger.warning(
                        f"Unknown source type: {
                            source_config.type}")
                    continue

                all_articles.extend(articles)
                logger.info(
                    f"Extracted {
                        len(articles)} articles from {source_name}")

            except Exception as e:
                logger.error(f"Error processing source {source_name}: {e}")
                continue

        # Process and analyze articles
        processed_articles = self.data_processor.process_articles(all_articles)

        # Generate report
        report = self.report_generator.generate_report(processed_articles)

        return {
            'articles': processed_articles,
            'report': report,
            'total_articles': len(processed_articles),
            'sources_processed': len(config.sources)
        }

    def _extract_rss(self, source_config: SourceConfig,
                     max_articles: int) -> List[Dict[str, Any]]:
        """Extract articles from RSS feed."""
        try:
            articles = self.rss_extractor.extract_articles(
                source_config.url,
                max_articles=max_articles,
                include_content=source_config.include_content
            )
            return articles
        except Exception as e:
            logger.error(f"RSS extraction error: {e}")
            return []

    def _extract_web(self, source_config: SourceConfig,
                     max_articles: int) -> List[Dict[str, Any]]:
        """Extract articles from web scraping."""
        try:
            articles = self.web_scraper.extract_articles(
                source_config.url,
                max_articles=max_articles,
                selectors=source_config.selectors
            )
            return articles
        except Exception as e:
            logger.error(f"Web extraction error: {e}")
            return []


def main():
    """Main entry point for news extraction."""
    parser = argparse.ArgumentParser(
        description="Extract news from configured sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Specific sources to extract from"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum articles per source"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extraction_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sources.yaml",
        help="Configuration file path"
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = NewsExtractor(args.config)

    # Extract news
    results = extractor.extract_news(
        sources=args.sources,
        max_articles=args.max_articles
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Extraction complete. Results saved to {output_path}")
    logger.info(
        f"Extracted {
            results['total_articles']} articles from {
            results['sources_processed']} sources")


if __name__ == "__main__":
    main()
