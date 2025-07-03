#!/usr/bin/env python3
"""
Test script for crawl4ai web scraper integration
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.scrapers.config_loader import ConfigLoader, SourceConfig
from src.scrapers.crawl4ai_web_scraper import Crawl4AiWebScraper, WebScraperWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_async_scraper():
    """Test the async crawl4ai scraper directly"""
    print("=== Testing Async Crawl4AI Scraper ===")
    
    # Create a test source
    test_source = SourceConfig({
        "name": "OpenAI Blog Test",
        "url": "https://openai.com/blog",
        "type": "website",
        "category": "test",
        "active": True,
        "selector": "article h2 a, .post-title a"
    })
    
    scraper = Crawl4AiWebScraper(
        use_llm_extraction=False,
        headless=True,
        max_retries=2
    )
    
    try:
        print(f"Testing extraction from: {test_source.url}")
        articles = await scraper.extract_from_source(test_source)
        
        print(f"✅ Successfully extracted {len(articles)} articles")
        
        # Show first few articles
        for i, article in enumerate(articles[:3]):
            print(f"\nArticle {i+1}:")
            print(f"  Title: {article.title}")
            print(f"  URL: {article.url}")
            print(f"  Description: {article.description[:100]}...")
            print(f"  Source: {article.source}")
            print(f"  Published: {article.published}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        logger.exception("Detailed error:")
        return False
        
    finally:
        await scraper.cleanup()


def test_sync_wrapper():
    """Test the synchronous wrapper"""
    print("\n=== Testing Synchronous Wrapper ===")
    
    # Create a test source
    test_source = SourceConfig({
        "name": "Google DeepMind Test", 
        "url": "https://deepmind.google/discover/blog/",
        "type": "website",
        "category": "test",
        "active": True,
        "selector": "article h2 a, .post-title a"
    })
    
    try:
        scraper = WebScraperWrapper(
            use_llm_extraction=False,
            headless=True,
            max_retries=2
        )
        
        print(f"Testing extraction from: {test_source.url}")
        articles = scraper.extract_from_source(test_source)
        
        print(f"✅ Successfully extracted {len(articles)} articles")
        
        # Show first few articles
        for i, article in enumerate(articles[:3]):
            print(f"\nArticle {i+1}:")
            print(f"  Title: {article.title}")
            print(f"  URL: {article.url}")
            print(f"  Description: {article.description[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        logger.exception("Detailed error:")
        return False


def test_with_config_file():
    """Test with actual configuration file"""
    print("\n=== Testing with Configuration File ===")
    
    try:
        # Load configuration
        config = ConfigLoader("src/sources.yaml")
        website_sources = config.get_website_sources()
        
        if not website_sources:
            print("⚠️  No website sources found in configuration")
            return True
        
        # Test first website source
        test_source = website_sources[0]
        print(f"Testing configured source: {test_source.name}")
        
        scraper = WebScraperWrapper(
            use_llm_extraction=False,
            headless=True,
            max_retries=1
        )
        
        articles = scraper.extract_from_source(test_source)
        print(f"✅ Successfully extracted {len(articles)} articles from {test_source.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        logger.exception("Detailed error:")
        return False


def test_main_extractor_integration():
    """Test integration with main extractor"""
    print("\n=== Testing Main Extractor Integration ===")
    
    try:
        from src.scrapers.main_extractor import NewsExtractor
        
        # Create extractor with crawl4ai enabled
        extractor = NewsExtractor(
            config_path="src/sources.yaml",
            output_dir="test_output", 
            use_crawl4ai=True,
            use_smart_scraper=False,
            use_llm_extraction=False,
            max_articles_per_source=5  # Limit for testing
        )
        
        print(f"Using scraper type: {extractor.extraction_stats['scraper_type']}")
        
        # Test website extraction only (faster)
        results = extractor.extract_websites_only()
        
        stats = results["extraction_stats"]
        print(f"✅ Main extractor test completed:")
        print(f"  Sources processed: {stats['total_sources']}")
        print(f"  Successful: {stats['successful_sources']}")
        print(f"  Failed: {stats['failed_sources']}")
        print(f"  Articles extracted: {stats['web_articles']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main extractor test failed: {e}")
        logger.exception("Detailed error:")
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    print("=== Checking Dependencies ===")
    
    missing_deps = []
    
    try:
        import crawl4ai
        print("✅ crawl4ai imported successfully")
    except ImportError:
        missing_deps.append("crawl4ai")
        print("❌ crawl4ai not available")
    
    try:
        from dateutil import parser
        print("✅ python-dateutil imported successfully")
    except ImportError:
        missing_deps.append("python-dateutil")
        print("❌ python-dateutil not available")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies available")
    return True


async def main():
    """Run all tests"""
    print("🚀 Testing Crawl4AI Web Scraper Integration\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Cannot proceed without required dependencies")
        return False
    
    # Run tests
    tests = [
        ("Async Scraper", test_async_scraper()),
        ("Sync Wrapper", lambda: test_sync_wrapper()),
        ("Config File", lambda: test_with_config_file()),
        ("Main Extractor", lambda: test_main_extractor_integration())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            if asyncio.iscoroutine(test_func):
                result = await test_func
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Crawl4AI integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == len(results)


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        sys.exit(1) 