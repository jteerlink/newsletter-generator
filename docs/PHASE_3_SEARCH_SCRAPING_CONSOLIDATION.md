# Phase 3: Search & Scraping Consolidation

## Overview

Phase 3 consolidates the search and scraping functionality into unified, standardized interfaces that provide better performance, reliability, and maintainability. This phase introduces a unified search provider, consolidated scraper interface, and comprehensive caching system.

## Architecture

### Unified Search Provider

The unified search provider abstracts different search engines and APIs behind a consistent interface, providing:

- **Standardized Results**: All search providers return `SearchResult` objects with consistent structure
- **Provider Abstraction**: Easy switching between Serper, DuckDuckGo, and knowledge base search
- **Fallback Logic**: Automatic fallback to alternative providers when primary fails
- **Async Support**: Both synchronous and asynchronous search operations
- **Caching Integration**: Built-in caching for improved performance

### Consolidated Scraper Interface

The consolidated scraper interface unifies different web scraping approaches:

- **Multiple Providers**: Support for Crawl4AI, Requests, and Selenium scraping
- **Standardized Content**: All scrapers return `ScrapedContent` objects
- **Retry Logic**: Automatic retry with exponential backoff
- **Rate Limiting**: Built-in rate limiting to respect server policies
- **Metadata Extraction**: Consistent metadata extraction across providers

### Cache Manager

The cache manager provides a unified caching system with:

- **Dual Storage**: Memory cache for speed, file cache for persistence
- **TTL Support**: Configurable time-to-live for cache entries
- **LRU Eviction**: Least recently used eviction for memory cache
- **Statistics**: Comprehensive cache statistics and monitoring
- **Decorator Support**: Easy caching with `@cached` decorator

## Implementation Details

### Search Provider Interface

```python
from src.tools.search_provider import get_unified_search_provider, SearchQuery

# Create a search query
query = SearchQuery(
    query="artificial intelligence trends",
    max_results=10,
    search_type="news",
    language="en"
)

# Get unified search provider
provider = get_unified_search_provider()

# Perform search
results = provider.search(query)

# Search with fallback queries
results = provider.search_with_alternatives(
    "primary query",
    fallback_queries=["alternative query 1", "alternative query 2"]
)
```

### Scraper Interface

```python
from src.scrapers.scraper import get_unified_scraper, ScrapingRequest

# Create a scraping request
request = ScrapingRequest(
    url="https://example.com/article",
    timeout=60,
    use_javascript=True,
    extract_metadata=True
)

# Get unified scraper
scraper = get_unified_scraper()

# Scrape single URL
content = scraper.scrape("https://example.com")

# Scrape multiple URLs
contents = scraper.scrape_multiple([
    "https://example1.com",
    "https://example2.com"
])
```

### Cache Manager

```python
from src.tools.cache_manager import get_cache_manager, cached

# Get cache manager
cache_manager = get_cache_manager()

# Manual caching
cache_manager.set("search", results, ttl=300, query="ai trends")
cached_results = cache_manager.get("search", "ai trends")

# Decorator-based caching
@cached("expensive_operation", ttl=600)
def expensive_function(param1, param2):
    # ... expensive computation
    return result

# Convenience functions
from src.tools.cache_manager import cache_search_results, get_cached_search_results

cache_search_results("query", results, ttl=300)
cached_results = get_cached_search_results("query")
```

## Key Features

### 1. Unified Search Provider

#### SearchResult Structure
```python
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    relevance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### Available Providers
- **SerperSearchProvider**: Google search via Serper API
- **DuckDuckGoSearchProvider**: DuckDuckGo search API
- **KnowledgeBaseSearchProvider**: Vector database search

#### Provider Selection Logic
1. Try Serper API (if configured)
2. Fallback to DuckDuckGo
3. Fallback to knowledge base
4. Return empty results if all fail

### 2. Consolidated Scraper

#### ScrapedContent Structure
```python
@dataclass
class ScrapedContent:
    title: str
    url: str
    content: str
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
```

#### Available Providers
- **Crawl4AIScrapingProvider**: Advanced web crawling with metadata extraction
- **RequestsScrapingProvider**: Simple HTTP requests with BeautifulSoup
- **SeleniumScrapingProvider**: JavaScript-enabled scraping

#### Provider Selection Logic
1. Try Crawl4AI (if available)
2. Fallback to Requests
3. Fallback to Selenium (for JavaScript-heavy sites)
4. Return error content if all fail

### 3. Cache Manager

#### Cache Levels
- **Memory Cache**: Fast access, limited size, LRU eviction
- **File Cache**: Persistent storage, larger capacity, TTL-based cleanup

#### Cache Features
- **Automatic Key Generation**: MD5 hash of operation and arguments
- **TTL Support**: Configurable expiration times
- **Statistics**: Hit rates, entry counts, cleanup metrics
- **Decorator Support**: Easy function caching

## Configuration

### Environment Variables

```bash
# Search Configuration
SERPER_API_KEY=your-serper-api-key-here
DUCKDUCKGO_API_KEY=your-duckduckgo-api-key-here

# Cache Configuration
CACHE_DIR=.cache
MEMORY_CACHE_SIZE=1000
MEMORY_CACHE_TTL=300
FILE_CACHE_TTL=3600

# Scraping Configuration
CRAWL4AI_API_KEY=your-crawl4ai-api-key-here
SELENIUM_DRIVER_PATH=/path/to/chromedriver
```

### Cache Manager Configuration

```python
from src.tools.cache_manager import CacheManager

# Custom cache configuration
cache_manager = CacheManager(
    memory_cache_size=500,
    memory_cache_ttl=150,
    file_cache_ttl=1800,
    cache_dir=".custom_cache"
)

# Enable/disable caching
cache_manager.disable()  # Disable all caching
cache_manager.enable()   # Re-enable caching
```

## Performance Optimizations

### 1. Caching Strategy
- **Memory Cache**: 5-minute TTL for frequently accessed data
- **File Cache**: 1-hour TTL for persistent storage
- **LRU Eviction**: Automatic cleanup of least used entries

### 2. Provider Fallback
- **Fast Failover**: Quick detection of provider failures
- **Parallel Requests**: Concurrent requests to multiple providers
- **Result Deduplication**: Remove duplicate results across providers

### 3. Rate Limiting
- **Request Throttling**: Respect API rate limits
- **Exponential Backoff**: Intelligent retry logic
- **Provider Rotation**: Distribute load across providers

## Testing

### Test Categories

1. **Unit Tests**: Individual component testing
   - `tests/tools/test_search_provider.py`
   - `tests/scrapers/test_scraper.py`
   - `tests/tools/test_cache_manager.py`

2. **Integration Tests**: End-to-end workflow testing
   - `tests/integration/test_api_integration.py`

3. **Performance Tests**: Performance benchmarking
   - `tests/performance/test_performance.py`

### Running Tests

```bash
# Run all Phase 3 tests
python scripts/run_phase3_tests.py

# Run specific test categories
pytest tests/tools/test_search_provider.py -v
pytest tests/scrapers/test_scraper.py -v
pytest tests/tools/test_cache_manager.py -v

# Run with coverage
pytest tests/ --cov=src.tools.search_provider --cov=src.tools.cache_manager --cov=src.scrapers.scraper --cov-report=term-missing
```

## Migration Guide

### From Old Search Functions

**Before:**
```python
from src.tools.tools import search_web, SerperSearchTool

# Direct Serper usage
serper_tool = SerperSearchTool()
results = serper_tool.search("query")

# Old search function
results = search_web("query")
```

**After:**
```python
from src.tools.search_provider import get_unified_search_provider

# Unified interface
provider = get_unified_search_provider()
results = provider.search("query", max_results=10)

# Still works with old function (now uses unified provider)
from src.tools.tools import search_web
results = search_web("query")
```

### From Old Scraping Functions

**Before:**
```python
from src.scrapers.crawl4ai_web_scraper import Crawl4AIScraper

scraper = Crawl4AIScraper()
content = scraper.scrape("https://example.com")
```

**After:**
```python
from src.scrapers.scraper import get_unified_scraper

scraper = get_unified_scraper()
content = scraper.scrape("https://example.com")
```

## Monitoring and Debugging

### Cache Statistics

```python
from src.tools.cache_manager import get_cache_manager

cache_manager = get_cache_manager()
stats = cache_manager.get_stats()

print(f"Memory cache entries: {stats['memory_cache']['total_entries']}")
print(f"File cache files: {stats['file_cache']['total_files']}")
print(f"Cache enabled: {stats['enabled']}")
```

### Provider Availability

```python
from src.tools.search_provider import get_unified_search_provider
from src.scrapers.scraper import get_unified_scraper

# Check available search providers
search_provider = get_unified_search_provider()
available_search = search_provider.get_available_providers()

# Check available scraping providers
scraper = get_unified_scraper()
available_scrapers = scraper.get_available_providers()
```

### Logging

```python
import logging

# Enable debug logging for search and scraping
logging.getLogger('src.tools.search_provider').setLevel(logging.DEBUG)
logging.getLogger('src.scrapers.scraper').setLevel(logging.DEBUG)
logging.getLogger('src.tools.cache_manager').setLevel(logging.DEBUG)
```

## Best Practices

### 1. Search Usage
- Use specific search types (news, web, images) when appropriate
- Implement fallback queries for better result coverage
- Cache search results to reduce API calls

### 2. Scraping Usage
- Use appropriate scraping provider for the content type
- Implement proper error handling for failed scrapes
- Respect robots.txt and rate limits

### 3. Caching Usage
- Set appropriate TTL values based on data freshness requirements
- Use the `@cached` decorator for expensive operations
- Monitor cache statistics for optimization opportunities

### 4. Error Handling
- Always handle provider failures gracefully
- Implement fallback strategies for critical operations
- Log errors for debugging and monitoring

## Future Enhancements

### Planned Features
1. **Advanced Caching**: Redis integration for distributed caching
2. **Provider Plugins**: Plugin system for custom search/scraping providers
3. **Result Ranking**: AI-powered result ranking and relevance scoring
4. **Content Analysis**: Automatic content analysis and categorization
5. **Performance Metrics**: Detailed performance monitoring and alerting

### Extension Points
- Custom search providers can implement the `SearchProvider` protocol
- Custom scrapers can implement the `ScrapingProvider` protocol
- Cache backends can be extended with custom storage implementations

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

2. **API Key Issues**
   - Verify environment variables are set correctly
   - Check API key validity and permissions
   - Ensure proper API quota management

3. **Cache Issues**
   - Check cache directory permissions
   - Verify cache configuration
   - Monitor cache statistics for issues

4. **Performance Issues**
   - Review cache hit rates
   - Check provider availability and response times
   - Monitor memory usage and cleanup

### Debug Commands

```bash
# Test imports
python -c "from src.tools.search_provider import get_unified_search_provider; print('Search provider OK')"

# Test cache
python -c "from src.tools.cache_manager import get_cache_manager; print(get_cache_manager().get_stats())"

# Test scraper
python -c "from src.scrapers.scraper import get_unified_scraper; print(get_unified_scraper().get_available_providers())"
```

## Conclusion

Phase 3 successfully consolidates search and scraping functionality into unified, robust interfaces. The new architecture provides:

- **Better Performance**: Caching and optimized provider selection
- **Improved Reliability**: Fallback logic and error handling
- **Enhanced Maintainability**: Standardized interfaces and comprehensive testing
- **Future Extensibility**: Plugin architecture and configuration options

The unified interfaces maintain backward compatibility while providing significant improvements in functionality, performance, and reliability. 