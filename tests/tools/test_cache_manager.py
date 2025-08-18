"""
Tests for the cache manager.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.tools.cache_manager import (
    CacheEntry,
    CacheManager,
    FileCache,
    MemoryCache,
    cache_scraped_content,
    cache_search_results,
    cached,
    clear_scraping_cache,
    clear_search_cache,
    get_cache_manager,
    get_cached_scraped_content,
    get_cached_search_results,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now()
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.metadata is None
    
    def test_cache_entry_with_metadata(self):
        """Test creating a CacheEntry with metadata."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_at,
            expires_at=expires_at,
            access_count=5,
            metadata={"source": "test"}
        )
        
        assert entry.expires_at == expires_at
        assert entry.access_count == 5
        assert entry.metadata["source"] == "test"
    
    def test_cache_entry_is_expired(self):
        """Test cache entry expiration check."""
        created_at = datetime.now()
        expires_at = created_at - timedelta(hours=1)  # Expired
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_at,
            expires_at=expires_at
        )
        
        assert entry.is_expired() is True
    
    def test_cache_entry_not_expired(self):
        """Test cache entry not expired."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)  # Not expired
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_at,
            expires_at=expires_at
        )
        
        assert entry.is_expired() is False
    
    def test_cache_entry_no_expiration(self):
        """Test cache entry with no expiration."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now()
        )
        
        assert entry.is_expired() is False
    
    def test_cache_entry_access(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now()
        )
        
        initial_count = entry.access_count
        initial_last_accessed = entry.last_accessed
        
        entry.access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_last_accessed
    
    def test_cache_entry_to_dict(self):
        """Test cache entry serialization."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_at,
            expires_at=expires_at,
            access_count=3,
            metadata={"source": "test"}
        )
        
        data = entry.to_dict()
        
        assert data['key'] == "test_key"
        assert data['value'] == "test_value"
        assert data['access_count'] == 3
        assert data['metadata']['source'] == "test"
        assert 'created_at' in data
        assert 'expires_at' in data
        assert 'last_accessed' in data
    
    def test_cache_entry_from_dict(self):
        """Test cache entry deserialization."""
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=1)
        
        data = {
            'key': 'test_key',
            'value': 'test_value',
            'created_at': created_at.isoformat(),
            'expires_at': expires_at.isoformat(),
            'access_count': 3,
            'last_accessed': created_at.isoformat(),
            'metadata': {'source': 'test'}
        }
        
        entry = CacheEntry.from_dict(data)
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 3
        assert entry.metadata['source'] == "test"


class TestMemoryCache:
    """Test MemoryCache."""
    
    def test_memory_cache_initialization(self):
        """Test MemoryCache initialization."""
        cache = MemoryCache(max_size=100, default_ttl=600)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 600
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_memory_cache_set_and_get(self):
        """Test setting and getting values from memory cache."""
        cache = MemoryCache()
        
        cache.set("test_key", "test_value", ttl=300)
        
        result = cache.get("test_key")
        assert result == "test_value"
    
    def test_memory_cache_get_nonexistent(self):
        """Test getting non-existent key from memory cache."""
        cache = MemoryCache()
        
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_memory_cache_expiration(self):
        """Test memory cache expiration."""
        cache = MemoryCache(default_ttl=1)  # 1 second TTL
        
        cache.set("test_key", "test_value")
        
        # Value should be available immediately
        result = cache.get("test_key")
        assert result == "test_value"
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Value should be expired
        result = cache.get("test_key")
        assert result is None
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction in memory cache."""
        cache = MemoryCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_memory_cache_delete(self):
        """Test deleting from memory cache."""
        cache = MemoryCache()
        
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        deleted = cache.delete("test_key")
        assert deleted is True
        assert cache.get("test_key") is None
    
    def test_memory_cache_delete_nonexistent(self):
        """Test deleting non-existent key from memory cache."""
        cache = MemoryCache()
        
        deleted = cache.delete("nonexistent_key")
        assert deleted is False
    
    def test_memory_cache_clear(self):
        """Test clearing memory cache."""
        cache = MemoryCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert len(cache.cache) == 2
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert len(cache.access_order) == 0
    
    def test_memory_cache_cleanup_expired(self):
        """Test cleaning up expired entries from memory cache."""
        cache = MemoryCache(default_ttl=1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        assert len(cache.cache) == 0
    
    def test_memory_cache_get_stats(self):
        """Test getting memory cache statistics."""
        cache = MemoryCache(max_size=100)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['total_entries'] == 2
        assert stats['active_entries'] == 2
        assert stats['expired_entries'] == 0
        assert stats['max_size'] == 100


class TestFileCache:
    """Test FileCache."""
    
    def test_file_cache_initialization(self, tmp_path):
        """Test FileCache initialization."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir), default_ttl=3600)
        
        assert cache.cache_dir == cache_dir
        assert cache.default_ttl == 3600
        assert cache_dir.exists()
    
    def test_file_cache_set_and_get(self, tmp_path):
        """Test setting and getting values from file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        cache.set("test_key", "test_value", ttl=300)
        
        result = cache.get("test_key")
        assert result == "test_value"
    
    def test_file_cache_get_nonexistent(self, tmp_path):
        """Test getting non-existent key from file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        result = cache.get("nonexistent_key")
        assert result is None
    
    def test_file_cache_expiration(self, tmp_path):
        """Test file cache expiration."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir), default_ttl=1)
        
        cache.set("test_key", "test_value")
        
        # Value should be available immediately
        result = cache.get("test_key")
        assert result == "test_value"
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Value should be expired
        result = cache.get("test_key")
        assert result is None
    
    def test_file_cache_delete(self, tmp_path):
        """Test deleting from file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        deleted = cache.delete("test_key")
        assert deleted is True
        assert cache.get("test_key") is None
    
    def test_file_cache_delete_nonexistent(self, tmp_path):
        """Test deleting non-existent key from file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        deleted = cache.delete("nonexistent_key")
        assert deleted is False
    
    def test_file_cache_clear(self, tmp_path):
        """Test clearing file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        # Check that cache files are removed
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 0
    
    def test_file_cache_cleanup_expired(self, tmp_path):
        """Test cleaning up expired entries from file cache."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir), default_ttl=1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        cleaned = cache.cleanup_expired()
        assert cleaned == 2
        
        # Check that cache files are removed
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 0
    
    def test_file_cache_get_stats(self, tmp_path):
        """Test getting file cache statistics."""
        cache_dir = tmp_path / "cache"
        cache = FileCache(cache_dir=str(cache_dir))
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        
        assert stats['total_files'] == 2
        assert stats['active_files'] == 2
        assert stats['expired_files'] == 0
        assert stats['cache_dir'] == str(cache_dir)


class TestCacheManager:
    """Test CacheManager."""
    
    def setup_method(self):
        """Reset the global cache manager before each test."""
        global _cache_manager
        _cache_manager = None
    
    def test_cache_manager_initialization(self, tmp_path):
        """Test CacheManager initialization."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        
        assert manager.enabled is True
        assert manager.memory_cache is not None
        assert manager.file_cache is not None
        assert str(cache_dir) in str(manager.file_cache.cache_dir)
    
    def test_cache_manager_set_and_get(self, tmp_path):
        """Test set and get operations for CacheManager."""
        manager = CacheManager(cache_dir=str(tmp_path))
        manager.set('test_operation', 'test_value', ttl=10, test_key='test_key')
        assert manager.get('test_operation', test_key='test_key') == 'test_value'
        
        # Test with different parameters
        manager2 = CacheManager(memory_cache_size=2, cache_dir=str(tmp_path))
        manager2.set("test_operation2", "test_value2", ttl=10, test_key="test_key2")
        assert manager2.get("test_operation2", test_key="test_key2") == "test_value2"

    def test_cache_manager_get_nonexistent(self, tmp_path):
        """Test getting non-existent operation from cache manager."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        
        result = manager.get("nonexistent_operation", arg1="value1")
        assert result is None
    
    def test_cache_manager_disabled(self, tmp_path):
        """Test cache manager when disabled."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        manager.disable()
        
        manager.set("test_operation", "test_value", arg1="value1")
        result = manager.get("test_operation", arg1="value1")
        
        assert result is None
    
    def test_cache_manager_delete(self, tmp_path):
        """Test delete operation for CacheManager."""
        manager = CacheManager(memory_cache_size=2, cache_dir=str(tmp_path))
        manager.set("test_operation", "test_value", test_key="test_key")
        manager.delete("test_operation", test_key="test_key")
        assert manager.get("test_operation", test_key="test_key") is None
    
    def test_cache_manager_expiration(self, tmp_path):
        manager = CacheManager(memory_cache_size=10, cache_dir=str(tmp_path))
        manager.set('expiring_operation', 'expiring_value', ttl=1, key='expiring_key')
        import time
        time.sleep(2)
        assert manager.get('expiring_operation', key='expiring_key') is None
    
    def test_cache_manager_clear(self, tmp_path):
        manager = CacheManager(cache_dir=str(tmp_path))
        manager.set("operation1", "value1", ttl=10, key="key1")
        manager.set("operation2", "value2", ttl=10, key="key2")
        manager.clear()
        assert manager.get("operation1", key="key1") is None
        assert manager.get("operation2", key="key2") is None
    
    def test_cache_manager_cleanup(self, tmp_path):
        """Test cleaning up cache manager."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        
        # Set some values
        manager.set("operation1", "value1", arg1="val1")
        manager.set("operation2", "value2", arg1="val2")
        
        # Cleanup should return counts
        result = manager.cleanup()
        assert 'memory_cleaned' in result
        assert 'file_cleaned' in result
    
    def test_cache_manager_get_stats(self, tmp_path):
        """Test getting cache manager statistics."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        
        manager.set("operation1", "value1", arg1="val1")
        
        stats = manager.get_stats()
        
        assert 'memory_cache' in stats
        assert 'file_cache' in stats
        assert stats['enabled'] is True
    
    def test_cache_manager_enable_disable(self, tmp_path):
        """Test enabling and disabling cache manager."""
        cache_dir = tmp_path / "cache"
        manager = CacheManager(cache_dir=str(cache_dir))
        
        assert manager.enabled is True
        
        manager.disable()
        assert manager.enabled is False
        
        manager.enable()
        assert manager.enabled is True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Reset the global cache manager before each test."""
        global _cache_manager
        _cache_manager = None
    
    def test_get_cache_manager_singleton(self):
        """Test that get_cache_manager returns singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2
    
    def test_cached_decorator(self, tmp_path):
        """Test cached decorator."""
        cache_dir = tmp_path / "cache"
        
        @cached("test_operation", ttl=300)
        def test_function(arg1, arg2):
            return f"result_{arg1}_{arg2}"
        
        # First call should execute function
        result1 = test_function("a", "b")
        assert result1 == "result_a_b"
        
        # Second call should return cached result
        result2 = test_function("a", "b")
        assert result2 == "result_a_b"
    
    def test_cache_search_results(self, tmp_path):
        """Test cache_search_results function."""
        cache_dir = tmp_path / "cache"
        manager = get_cache_manager()
        
        cache_search_results("test query", ["result1", "result2"], ttl=300)
        
        cached_result = get_cached_search_results("test query")
        assert cached_result == ["result1", "result2"]
    
    def test_cache_scraped_content(self, tmp_path):
        """Test cache_scraped_content function."""
        cache_dir = tmp_path / "cache"
        manager = get_cache_manager()
        
        content = {"title": "Test", "content": "Test content"}
        cache_scraped_content("https://example.com", content, ttl=3600)
        
        cached_result = get_cached_scraped_content("https://example.com")
        assert cached_result == content
    
    def test_clear_search_cache(self):
        """Test clearing the search cache convenience function."""
        clear_search_cache()
        # No assertion needed, just ensure no error


    def test_clear_scraping_cache(self):
        """Test clearing the scraping cache convenience function."""
        clear_scraping_cache()
        # No assertion needed, just ensure no error 