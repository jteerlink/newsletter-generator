#!/usr/bin/env python3
"""
Test script to verify configuration continuity
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "scrapers"))

from config_loader import ConfigLoader
from data_processor import DataProcessor
import json

def test_config_loader():
    """Test configuration loading"""
    print("=== Testing Configuration Loader ===")
    try:
        config = ConfigLoader()
        config.print_summary()
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_data_processor():
    """Test database connection"""
    print("\n=== Testing Database Connection ===")
    try:
        processor = DataProcessor()
        stats = processor.get_statistics()
        print(f"✓ Database connected successfully")
        print(f"  - Total articles: {stats.get('total_articles', 0)}")
        print(f"  - Sources: {stats.get('sources', {})}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_scheduler_config():
    """Test scheduler configuration"""
    print("\n=== Testing Scheduler Configuration ===")
    try:
        with open('config/scheduler_config.json', 'r') as f:
            scheduler_config = json.load(f)
        
        categories = scheduler_config.get('extraction', {}).get('categories', {})
        print(f"✓ Scheduler config loaded successfully")
        print(f"  - Configured categories: {list(categories.keys())}")
        return True
    except Exception as e:
        print(f"✗ Scheduler config loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Configuration Continuity Test")
    print("=" * 40)
    
    tests = [
        test_config_loader,
        test_data_processor,
        test_scheduler_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    if all(results):
        print("✓ All tests passed! Configuration is consistent.")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main()) 