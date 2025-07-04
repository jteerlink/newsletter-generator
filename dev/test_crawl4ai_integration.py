#!/usr/bin/env python3
"""
Comprehensive test script for crawl4ai integration
Tests the complete pipeline from basic scraping to full newsletter extraction
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    print("=== Checking Dependencies ===")
    
    dependencies = {
        'crawl4ai': 'crawl4ai',
        'playwright': 'playwright',
        'asyncio': 'asyncio',
        'dateutil': 'python-dateutil',
        'yaml': 'PyYAML',
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("All dependencies are available!")
    return True

def check_crawl4ai_setup():
    """Check if crawl4ai is properly set up"""
    print("\n=== Checking Crawl4AI Setup ===")
    
    try:
        from crawl4ai import AsyncWebCrawler
        print("✓ crawl4ai imports successfully")
        
        # Try to check if playwright browsers are installed
        try:
            import subprocess
            result = subprocess.run(['playwright', 'install', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✓ Playwright CLI is available")
            else:
                print("⚠ Playwright CLI not found, browser installation may be needed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠ Playwright CLI not found, but crawl4ai may still work")
        
        return True
        
    except ImportError as e:
        print(f"✗ crawl4ai import failed: {e}")
        return False

async def test_basic_crawl4ai():
    """Test basic crawl4ai functionality"""
    print("\n=== Testing Basic Crawl4AI ===")
    
    try:
        from crawl4ai import AsyncWebCrawler
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            # Test with a simple, reliable page
            test_url = "https://httpbin.org/html"
            print(f"Testing crawl4ai with: {test_url}")
            
            result = await crawler.arun(url=test_url)
            
            if result.success:
                print(f"✓ Basic crawl successful")
                print(f"  - Status: {result.status_code}")
                print(f"  - Content length: {len(result.cleaned_html)}")
                print(f"  - Markdown length: {len(result.markdown)}")
                return True
            else:
                print(f"✗ Basic crawl failed: {result.error_message}")
                return False
                
    except Exception as e:
        print(f"✗ Basic crawl4ai test failed: {e}")
        traceback.print_exc()
        return False

def test_scraper_imports():
    """Test importing our custom scraper classes"""
    print("\n=== Testing Scraper Imports ===")
    
    try:
        from src.scrapers.crawl4ai_web_scraper import (
            Crawl4AiWebScraper, 
            SmartCrawl4AiWebScraper, 
            WebScraperWrapper
        )
        print("✓ Crawl4AI scraper classes imported successfully")
        
        from src.scrapers.rss_extractor import Article
        print("✓ Article data structure imported")
        
        from src.scrapers.config_loader import ConfigLoader, SourceConfig
        print("✓ Configuration classes imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Scraper import failed: {e}")
        traceback.print_exc()
        return False

async def test_async_scraper():
    """Test the async crawl4ai scraper"""
    print("\n=== Testing Async Scraper ===")
    
    try:
        from src.scrapers.crawl4ai_web_scraper import Crawl4AiWebScraper
        from src.scrapers.config_loader import SourceConfig
        
        # Create a test source
        test_source = SourceConfig({
            'name': 'HTTPBin Test',
            'url': 'https://httpbin.org/html',
            'type': 'website',
            'category': 'test',
            'active': True,
            'selectors': {
                'article_links': 'a',
                'title': 'h1',
                'content': 'p',
                'published_date': 'time'
            }
        })
        
        scraper = Crawl4AiWebScraper(
            timeout=30,
            headless=True,
            use_llm_extraction=False
        )
        
        articles = await scraper.extract_from_source(test_source)
        
        print(f"✓ Async scraper test completed")
        print(f"  - Articles found: {len(articles)}")
        
        if articles:
            article = articles[0]
            print(f"  - Sample article: {article.title[:50]}...")
            print(f"  - Source: {article.source}")
            print(f"  - URL: {article.url}")
        
        await scraper.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Async scraper test failed: {e}")
        traceback.print_exc()
        return False

def test_sync_wrapper():
    """Test the synchronous wrapper"""
    print("\n=== Testing Sync Wrapper ===")
    
    try:
        from src.scrapers.crawl4ai_web_scraper import WebScraperWrapper, Crawl4AiWebScraper
        from src.scrapers.config_loader import SourceConfig
        
        # Create a test source
        test_source = SourceConfig({
            'name': 'HTTPBin Test Sync',
            'url': 'https://httpbin.org/html',
            'type': 'website',
            'category': 'test',
            'active': True
        })
        
        scraper = WebScraperWrapper(
            scraper_class=Crawl4AiWebScraper,
            timeout=30,
            headless=True,
            use_llm_extraction=False
        )
        
        articles = scraper.extract_from_source(test_source)
        
        print(f"✓ Sync wrapper test completed")
        print(f"  - Articles found: {len(articles)}")
        
        if articles:
            article = articles[0]
            print(f"  - Sample article: {article.title[:50]}...")
        
        scraper.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Sync wrapper test failed: {e}")
        traceback.print_exc()
        return False

def test_config_file_integration():
    """Test integration with actual config file"""
    print("\n=== Testing Config File Integration ===")
    
    try:
        from src.scrapers.config_loader import ConfigLoader
        
        # Check if config file exists
        config_path = Path("src/sources.yaml")
        if not config_path.exists():
            print(f"⚠ Config file not found at {config_path}")
            print("Creating a minimal test config...")
            
            test_config = """
sources:
  - name: "Test HTTP Source"
    url: "https://httpbin.org/html"
    type: "website"
    category: "test"
    active: true
    
  - name: "Test RSS Source"
    url: "https://httpbin.org/xml"
    type: "rss"
    category: "test"
    active: true
"""
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(test_config)
            print(f"✓ Created test config at {config_path}")
        
        config = ConfigLoader(str(config_path))
        sources = config.get_active_sources()
        
        print(f"✓ Config loaded successfully")
        print(f"  - Total sources: {len(sources)}")
        print(f"  - Active sources: {len([s for s in sources if s.active])}")
        
        # Show first few sources
        for i, source in enumerate(sources[:3]):
            print(f"  - Source {i+1}: {source.name} ({source.type})")
        
        return True
        
    except Exception as e:
        print(f"✗ Config file test failed: {e}")
        traceback.print_exc()
        return False

def test_main_extractor():
    """Test the main extractor with crawl4ai"""
    print("\n=== Testing Main Extractor ===")
    
    try:
        from src.scrapers.main_extractor import NewsExtractor
        
        # Initialize with test configuration
        extractor = NewsExtractor(
            config_path="src/sources.yaml",
            output_dir="test_output",
            use_crawl4ai=True,
            use_smart_scraper=False,  # Use basic scraper for testing
            use_llm_extraction=False,
            max_articles_per_source=2,  # Limit for testing
            timeout=30,
            max_concurrent=1  # Conservative for testing
        )
        
        print(f"✓ Extractor initialized with {extractor.extraction_stats['scraper_type']} scraper")
        
        # Test extraction from a single category or source type
        results = extractor.extract_from_all_sources(source_types=["website"])
        
        stats = results["extraction_stats"]
        print(f"✓ Extraction completed:")
        print(f"  - Total sources: {stats['total_sources']}")
        print(f"  - Successful sources: {stats['successful_sources']}")
        print(f"  - Total articles: {stats['total_articles']}")
        print(f"  - Errors: {len(stats['errors'])}")
        
        if stats['errors']:
            print("  - Error details:")
            for error in stats['errors'][:2]:  # Show first 2 errors
                print(f"    * {error['source']}: {error['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Main extractor test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_test():
    """Run a simple performance comparison"""
    print("\n=== Performance Test ===")
    
    try:
        from src.scrapers.crawl4ai_web_scraper import WebScraperWrapper, Crawl4AiWebScraper
        from src.scrapers.config_loader import SourceConfig
        
        # Test URLs
        test_urls = [
            "https://httpbin.org/html",
            "https://example.com",
            "https://httpbin.org/delay/1"
        ]
        
        test_sources = [
            SourceConfig({
                'name': f'Test Source {i}',
                'url': url,
                'type': 'website',
                'category': 'test',
                'active': True
            })
            for i, url in enumerate(test_urls)
        ]
        
        scraper = WebScraperWrapper(
            scraper_class=Crawl4AiWebScraper,
            timeout=10,
            headless=True,
            max_concurrent=2
        )
        
        start_time = time.time()
        
        # Test batch processing if available
        if hasattr(scraper, 'extract_from_multiple_sources'):
            print("Testing batch extraction...")
            articles = scraper.extract_from_multiple_sources(test_sources)
        else:
            print("Testing individual extraction...")
            articles = []
            for source in test_sources:
                try:
                    source_articles = scraper.extract_from_source(source)
                    articles.extend(source_articles)
                except Exception as e:
                    print(f"  - Error with {source.name}: {e}")
        
        duration = time.time() - start_time
        
        print(f"✓ Performance test completed:")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Total articles: {len(articles)}")
        print(f"  - Sources: {len(test_sources)}")
        print(f"  - Average time per source: {duration/len(test_sources):.2f}s")
        
        scraper.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files created during testing"""
    print("\n=== Cleaning Up Test Files ===")
    
    test_dirs = ["test_output", "logs"]
    test_files = ["src/sources.yaml"]
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists() and test_path.is_dir():
            try:
                import shutil
                shutil.rmtree(test_path)
                print(f"✓ Removed test directory: {test_dir}")
            except Exception as e:
                print(f"⚠ Could not remove {test_dir}: {e}")
    
    print("Test cleanup completed")

async def main():
    """Run all tests"""
    print("🚀 Starting Crawl4AI Integration Tests")
    print("=" * 50)
    
    # Track test results
    test_results = {}
    
    # Dependency checks
    test_results['dependencies'] = check_dependencies()
    if not test_results['dependencies']:
        print("\n❌ Dependencies missing. Please install required packages.")
        return False
    
    test_results['crawl4ai_setup'] = check_crawl4ai_setup()
    if not test_results['crawl4ai_setup']:
        print("\n❌ Crawl4AI setup issues detected.")
        return False
    
    # Basic functionality tests
    test_results['basic_crawl4ai'] = await test_basic_crawl4ai()
    test_results['scraper_imports'] = test_scraper_imports()
    
    if not all([test_results['basic_crawl4ai'], test_results['scraper_imports']]):
        print("\n❌ Basic functionality tests failed.")
        return False
    
    # Advanced tests
    test_results['async_scraper'] = await test_async_scraper()
    test_results['sync_wrapper'] = test_sync_wrapper()
    test_results['config_integration'] = test_config_file_integration()
    test_results['main_extractor'] = test_main_extractor()
    test_results['performance'] = run_performance_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Crawl4AI integration is working correctly.")
        success = True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        success = False
    
    # Optional cleanup
    try:
        response = input("\nClean up test files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_test_files()
    except (KeyboardInterrupt, EOFError):
        print("\nSkipping cleanup.")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 