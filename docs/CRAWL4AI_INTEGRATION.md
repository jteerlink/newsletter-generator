# Crawl4AI Integration Documentation

## Overview

The newsletter generator has been enhanced with [Crawl4AI](https://github.com/unclecode/crawl4ai) integration, providing significantly improved web scraping capabilities with better JavaScript handling, content extraction, and performance.

## 🚀 Key Features

### Enhanced Web Scraping
- **JavaScript Rendering**: Full support for modern JavaScript-heavy websites
- **Multiple Browser Engines**: Support for Chromium, Firefox, and WebKit
- **Async Processing**: High-performance concurrent scraping
- **Smart Content Detection**: Advanced CSS selectors and optional LLM extraction
- **Content Cleaning**: Built-in HTML cleaning and markdown generation

### Advanced Extraction Modes
- **Standard Mode**: CSS selector-based extraction with enhanced article detection
- **Smart Mode**: AI-enhanced extraction with better content understanding
- **LLM Mode**: Optional LLM-powered content extraction (requires LLM setup)

### Performance Improvements
- **Batch Processing**: Process multiple sources simultaneously
- **Concurrent Execution**: Configurable concurrent request limits
- **Intelligent Caching**: Built-in caching for repeated requests
- **Timeout Management**: Configurable timeouts for different scenarios

## 📦 Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install crawl4ai==0.6.3 python-dateutil==2.8.2

# Install Playwright browsers (required for crawl4ai)
playwright install chromium
```

### 2. Verify Installation

Run the comprehensive test suite:

```bash
python test_crawl4ai_integration.py
```

## 🔧 Configuration

### Command Line Usage

The enhanced main extractor supports extensive configuration options:

```bash
# Basic usage with crawl4ai (default)
python src/scrapers/main_extractor.py

# Use smart scraper with AI capabilities
python src/scrapers/main_extractor.py --smart-scraper

# Enable LLM extraction (requires LLM setup)
python src/scrapers/main_extractor.py --llm-extraction

# Use different browser engines
python src/scrapers/main_extractor.py --browser-type firefox

# Adjust performance settings
python src/scrapers/main_extractor.py --max-concurrent 5 --timeout 60

# Fallback to legacy scraper
python src/scrapers/main_extractor.py --no-crawl4ai --selenium-fallback
```

### Full CLI Options

```bash
Options:
  --config PATH              Sources configuration file (default: src/sources.yaml)
  --output-dir PATH         Output directory (default: output)
  --categories [CATEGORIES] Specific categories to extract
  --source-types {rss,website} Types of sources to extract from
  --format {json,csv,html}  Output format for reports (default: json)
  --max-articles INT        Maximum articles per source (default: 50)
  --no-crawl4ai            Disable crawl4ai and use legacy scraper
  --smart-scraper          Use smart crawl4ai scraper with AI capabilities
  --llm-extraction         Use LLM for content extraction
  --browser-type {chromium,firefox,webkit} Browser type (default: chromium)
  --max-concurrent INT     Maximum concurrent tasks (default: 3)
  --timeout INT            Request timeout in seconds (default: 30)
  --selenium-fallback      Enable Selenium fallback for legacy scraper
  --verbose, -v            Enable verbose logging
```

### Programmatic Usage

```python
from src.scrapers.main_extractor import NewsExtractor

# Initialize with crawl4ai (recommended)
extractor = NewsExtractor(
    config_path="src/sources.yaml",
    use_crawl4ai=True,
    use_smart_scraper=True,
    browser_type="chromium",
    max_concurrent=3,
    timeout=30
)

# Extract from all sources
results = extractor.extract_from_all_sources()

# Extract from specific categories
tech_articles = extractor.extract_from_category("technology")

# Extract only from websites (skip RSS)
web_articles = extractor.extract_websites_only()
```

## 🎯 Migration from Legacy Scraper

### Automatic Fallback
The system automatically falls back to the legacy scraper if crawl4ai fails:

```python
extractor = NewsExtractor(
    use_crawl4ai=True,
    use_selenium_fallback=True  # Enable legacy fallback
)
```

### Manual Legacy Mode
Force legacy scraper usage:

```bash
python src/scrapers/main_extractor.py --no-crawl4ai --selenium-fallback
```

### Comparison

| Feature | Legacy Scraper | Crawl4AI |
|---------|---------------|----------|
| JavaScript Support | Limited (Selenium only) | Full (Playwright) |
| Performance | Slower, sequential | Fast, concurrent |
| Content Extraction | Basic CSS selectors | Advanced + AI options |
| Browser Support | Chrome via Selenium | Chromium, Firefox, WebKit |
| Memory Usage | Higher | Optimized |
| Reliability | Moderate | High |

## 🔍 Advanced Features

### Smart Content Detection

The SmartCrawl4AiWebScraper includes enhanced article detection:

```python
from src.scrapers.crawl4ai_web_scraper import SmartCrawl4AiWebScraper

scraper = SmartCrawl4AiWebScraper(
    use_llm_extraction=True,  # Optional LLM enhancement
    timeout=60,
    headless=True
)
```

### Custom Extraction Strategies

```python
# CSS-based extraction with fallbacks
css_strategy = ExtractionStrategy(
    extraction_type="css_extractor",
    css_extractor={
        "tags": ["article", "div.content", ".post-content"],
        "attributes": ["href", "title", "data-published"]
    }
)

# LLM-based extraction (requires setup)
llm_strategy = ExtractionStrategy(
    extraction_type="llm_extractor",
    llm_extractor={
        "provider": "openai",
        "api_token": "your-token",
        "instruction": "Extract article title, content, and publication date"
    }
)
```

### Batch Processing

```python
# Process multiple sources efficiently
scraper = WebScraperWrapper(
    scraper_class=Crawl4AiWebScraper,
    max_concurrent=5
)

# Batch extraction (if supported)
articles = scraper.extract_from_multiple_sources(sources)
```

## 🛠 Troubleshooting

### Common Issues

1. **Playwright Browser Installation**
   ```bash
   # Install browsers
   playwright install chromium
   
   # Or install all browsers
   playwright install
   ```

2. **Memory Issues**
   ```python
   # Reduce concurrent tasks
   extractor = NewsExtractor(max_concurrent=1)
   ```

3. **Timeout Errors**
   ```python
   # Increase timeout
   extractor = NewsExtractor(timeout=120)
   ```

4. **JavaScript-Heavy Sites**
   ```python
   # Use smart scraper with longer timeout
   extractor = NewsExtractor(
       use_smart_scraper=True,
       timeout=60,
       browser_type="chromium"
   )
   ```

### Debugging

Enable verbose logging:

```bash
python src/scrapers/main_extractor.py --verbose
```

Or programmatically:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Performance Tuning

```python
# High-performance configuration
extractor = NewsExtractor(
    use_crawl4ai=True,
    use_smart_scraper=True,
    max_concurrent=5,        # Increase for faster processing
    timeout=30,              # Reduce for faster timeouts
    browser_type="chromium", # Generally fastest
    max_articles_per_source=20  # Limit articles for speed
)
```

## 📊 Monitoring and Analytics

### Extraction Statistics

The enhanced extractor provides detailed statistics:

```python
results = extractor.extract_from_all_sources()
stats = results["extraction_stats"]

print(f"Scraper Type: {stats['scraper_type']}")
print(f"Total Sources: {stats['total_sources']}")
print(f"Success Rate: {stats['successful_sources']}/{stats['total_sources']}")
print(f"Total Articles: {stats['total_articles']}")
print(f"Duration: {duration} seconds")
```

### Error Tracking

```python
# Access detailed error information
for error in stats['errors']:
    print(f"Source: {error['source']}")
    print(f"Error: {error['error']}")
    print(f"Timestamp: {error['timestamp']}")
```

## 🔮 Future Enhancements

- **AI-Powered Source Discovery**: Automatically discover new article sources
- **Content Quality Scoring**: Rate article relevance and quality
- **Real-time Monitoring**: Live dashboards for extraction performance
- **Advanced Filtering**: ML-based content filtering and categorization
- **Integration APIs**: REST API for external integrations

## 📚 Additional Resources

- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [Playwright Documentation](https://playwright.dev/python/)
- [Test Suite](test_crawl4ai_integration.py) - Comprehensive testing examples
- [Legacy Migration Guide](#migration-from-legacy-scraper)

## 🤝 Support

For issues and questions:

1. Run the test suite: `python test_crawl4ai_integration.py`
2. Check logs in `logs/extraction.log`
3. Enable verbose mode for detailed debugging
4. Try fallback mode if crawl4ai issues persist

---

**Note**: This integration maintains full backward compatibility. Existing configurations and workflows will continue to work with enhanced performance and reliability. 