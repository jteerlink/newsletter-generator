#!/usr/bin/env python3
"""
Test runner script for Phase 3: Search & Scraping Consolidation.

This script runs tests for the new unified search and scraping interfaces,
including the cache manager and consolidated implementations.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def run_search_provider_tests():
    """Run tests for the unified search provider."""
    return run_command(
        "pytest tests/tools/test_search_provider.py -v",
        "Unified Search Provider Tests"
    )

def run_scraper_tests():
    """Run tests for the unified scraper."""
    return run_command(
        "pytest tests/scrapers/test_scraper.py -v",
        "Unified Scraper Tests"
    )

def run_cache_manager_tests():
    """Run tests for the cache manager."""
    return run_command(
        "pytest tests/tools/test_cache_manager.py -v",
        "Cache Manager Tests"
    )

def run_updated_tools_tests():
    """Run tests for updated tools that use the new interfaces."""
    return run_command(
        "pytest tests/tools/test_tools.py -v",
        "Updated Tools Tests"
    )

def run_integration_tests():
    """Run integration tests for search and scraping."""
    return run_command(
        "pytest tests/integration/test_api_integration.py -v",
        "Search & Scraping Integration Tests"
    )

def run_performance_tests():
    """Run performance tests for the new interfaces."""
    return run_command(
        "pytest tests/performance/test_performance.py -v",
        "Performance Tests"
    )

def run_coverage_analysis():
    """Run coverage analysis for the new modules."""
    return run_command(
        "pytest tests/ --cov=src.tools.search_provider --cov=src.tools.cache_manager --cov=src.scrapers.scraper --cov-report=term-missing --tb=no -q",
        "Coverage Analysis for Phase 3 Modules"
    )

def run_linting_checks():
    """Run linting checks for the new code."""
    commands = [
        ("flake8 src/tools/search_provider.py src/tools/cache_manager.py src/scrapers/scraper.py --max-line-length=100", "Flake8 Linting"),
        ("black --check src/tools/search_provider.py src/tools/cache_manager.py src/scrapers/scraper.py", "Black Formatting Check"),
        ("mypy src/tools/search_provider.py src/tools/cache_manager.py src/scrapers/scraper.py --ignore-missing-imports", "MyPy Type Checking")
    ]
    
    results = []
    for command, description in commands:
        results.append(run_command(command, description))
    
    return all(results)

def run_import_tests():
    """Test that the new modules can be imported correctly."""
    test_script = """
import sys
sys.path.insert(0, '.')

try:
    from src.tools.search_provider import get_unified_search_provider, SearchResult, SearchQuery
    print("✅ Search provider imports successful")
except Exception as e:
    print(f"❌ Search provider import failed: {e}")
    sys.exit(1)

try:
    from src.tools.cache_manager import get_cache_manager, cached
    print("✅ Cache manager imports successful")
except Exception as e:
    print(f"❌ Cache manager import failed: {e}")
    sys.exit(1)

try:
    from src.scrapers.scraper import get_unified_scraper, ScrapedContent, ScrapingRequest
    print("✅ Scraper imports successful")
except Exception as e:
    print(f"❌ Scraper import failed: {e}")
    sys.exit(1)

print("✅ All Phase 3 imports successful")
"""
    
    with open("temp_import_test.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python temp_import_test.py", "Import Tests")
    
    # Clean up
    if os.path.exists("temp_import_test.py"):
        os.remove("temp_import_test.py")
    
    return success

def main():
    """Main test runner function."""
    print("🚀 Phase 3 Test Runner: Search & Scraping Consolidation")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Run import tests first
    print("\n📦 Testing imports...")
    results['imports'] = run_import_tests()
    
    # Run unit tests
    print("\n🧪 Running unit tests...")
    results['search_provider'] = run_search_provider_tests()
    results['scraper'] = run_scraper_tests()
    results['cache_manager'] = run_cache_manager_tests()
    results['updated_tools'] = run_updated_tools_tests()
    
    # Run integration tests
    print("\n🔗 Running integration tests...")
    results['integration'] = run_integration_tests()
    
    # Run performance tests
    print("\n⚡ Running performance tests...")
    results['performance'] = run_performance_tests()
    
    # Run coverage analysis
    print("\n📊 Running coverage analysis...")
    results['coverage'] = run_coverage_analysis()
    
    # Run linting checks
    print("\n🔍 Running linting checks...")
    results['linting'] = run_linting_checks()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test categories passed")
    
    if passed_tests == total_tests:
        print("🎉 All Phase 3 tests passed!")
        print("\nPhase 3 deliverables:")
        print("✅ Unified search provider interface")
        print("✅ Consolidated scraper interface")
        print("✅ Cache manager with memory and file caching")
        print("✅ Updated tools to use new interfaces")
        print("✅ Comprehensive test coverage")
        print("✅ Performance optimizations")
        print("✅ Code quality checks")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 