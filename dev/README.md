# Development & Testing Files

This directory contains development utilities, testing scripts, and benchmarking tools:

## Testing Scripts
- **test_crawl4ai_integration.py** - Integration tests for Crawl4AI web scraping functionality
- **test_prompts_ab.py** - A/B testing scripts for prompt optimization
- **test_tool_calls.py** - Tests for tool calling functionality and integrations

## Benchmarking & Performance
- **benchmark.py** - Performance benchmarking script for various system components
- **benchmark_results.csv** - Results from performance benchmarking runs

## Usage
These files are used for:
- Testing new features and integrations
- Performance analysis and optimization
- Development debugging and validation
- Quality assurance during development

## Running Tests
Most test files can be run directly with Python:
```bash
python dev/test_crawl4ai_integration.py
python dev/benchmark.py
```

Note: Some tests may require specific environment setup or API keys to be configured. 