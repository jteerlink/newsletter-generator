"""
Configuration loader for news sources
"""
import yaml
from typing import Dict, List, Any
from pathlib import Path
import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceConfig:
    """Individual source configuration"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.name = config_dict.get('name', '')
        self.url = config_dict.get('url', '')
        self.rss_url = config_dict.get('rss_url')
        self.type = config_dict.get('type', 'website')
        self.category = config_dict.get('category', 'general')
        self.active = config_dict.get('active', True)
        self.scrape_frequency = config_dict.get('scrape_frequency', 'daily')
        self.selector = config_dict.get('selector', 'h1 a, h2 a, h3 a')
        self.description = config_dict.get('description', '')
        self.feed = config_dict.get('feed')
        
    def __repr__(self):
        return f"SourceConfig(name='{self.name}', type='{self.type}', category='{self.category}')"

class ConfigLoader:
    """Load and manage source configurations"""
    
    def __init__(self, config_path: str = 'config/sources.yaml'):
        # Try the provided path first
        path = Path(config_path)
        if not path.exists():
            # Try relative to this file's parent (src/scrapers/../..)
            project_root = Path(__file__).resolve().parent.parent.parent
            alt_path = project_root / 'config' / 'sources.yaml'
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(f"Could not find sources.yaml at '{config_path}' or '{alt_path}'. Current working directory: {Path.cwd()}")
        self.config_path = path
        self.sources = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            raw_sources = config.get('sources', [])
            self.sources = [SourceConfig(source) for source in raw_sources]
            
            logger.info(f"Loaded {len(self.sources)} sources from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def get_sources_by_type(self, source_type: str) -> List[SourceConfig]:
        """Get sources filtered by type (rss, website)"""
        return [source for source in self.sources if source.type == source_type and source.active]
    
    def get_sources_by_category(self, category: str) -> List[SourceConfig]:
        """Get sources filtered by category"""
        return [source for source in self.sources if source.category == category and source.active]
    
    def get_active_sources(self) -> List[SourceConfig]:
        """Get all active sources"""
        return [source for source in self.sources if source.active]
    
    def get_rss_sources(self) -> List[SourceConfig]:
        """Get all RSS sources"""
        return self.get_sources_by_type('rss')
    
    def get_website_sources(self) -> List[SourceConfig]:
        """Get all website sources that need scraping"""
        return self.get_sources_by_type('website')
    
    def print_summary(self):
        """Print a summary of loaded sources"""
        total = len(self.sources)
        active = len(self.get_active_sources())
        rss = len(self.get_rss_sources())
        websites = len(self.get_website_sources())
        
        print(f"\n=== Source Configuration Summary ===")
        print(f"Total sources: {total}")
        print(f"Active sources: {active}")
        print(f"RSS sources: {rss}")
        print(f"Website sources: {websites}")
        
        # Category breakdown
        categories = {}
        for source in self.get_active_sources():
            categories[source.category] = categories.get(source.category, 0) + 1
        
        print(f"\nBy category:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")

if __name__ == "__main__":
    # Test the config loader
    config = ConfigLoader()
    config.print_summary()
    
    # Show some examples
    print(f"\nFirst 3 RSS sources:")
    for source in config.get_rss_sources()[:3]:
        print(f"  - {source.name}: {source.rss_url}")
        
    print(f"\nFirst 3 website sources:")
    for source in config.get_website_sources()[:3]:
        print(f"  - {source.name}: {source.url}")
