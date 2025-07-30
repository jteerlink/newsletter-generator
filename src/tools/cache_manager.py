"""
Cache Manager for Search and Scraping Operations

This module provides a unified caching system for search and scraping operations,
improving performance and reducing redundant requests.
"""

from __future__ import annotations

import logging
import time
import json
import hashlib
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def access(self):
        """Mark the entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            key=data['key'],
            value=data['value'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            access_count=data['access_count'],
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            metadata=data.get('metadata')
        )


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a string representation of the arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            self.delete(key)
            return None
        
        # Update access metadata
        entry.access()
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        # If key already exists, remove from access order
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
        
        # Add to cache
        self.cache[key] = entry
        self.access_order.append(key)
        
        # Evict if cache is full
        if len(self.cache) > self.max_size:
            self._evict_lru()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.access_order:
            lru_key = self.access_order[0]
            self.delete(lru_key)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())
        active_entries = total_entries - expired_entries
        
        total_access_count = sum(entry.access_count for entry in self.cache.values())
        avg_access_count = total_access_count / total_entries if total_entries > 0 else 0
        
        return {
            'total_entries': total_entries,
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'total_access_count': total_access_count,
            'avg_access_count': avg_access_count,
            'max_size': self.max_size
        }


class FileCache:
    """File-based cache implementation."""
    
    def __init__(self, cache_dir: str = ".cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            
            # Check if expired
            if entry.is_expired():
                self.delete(key)
                return None
            
            # Update access metadata
            entry.access()
            
            # Save updated entry
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f)
            
            return entry.value
            
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in file cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        cache_file = self._get_cache_file(key)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from file cache."""
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            try:
                cache_file.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        return False
    
    def clear(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Error clearing cache directory: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files and return count of removed files."""
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry.from_dict(data)
                
                if entry.is_expired():
                    cache_file.unlink()
                    removed_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking cache file {cache_file}: {e}")
                # Remove corrupted cache files
                try:
                    cache_file.unlink()
                    removed_count += 1
                except:
                    pass
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = len(list(self.cache_dir.glob("*.json")))
        expired_files = 0
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                total_size += cache_file.stat().st_size
                
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry.from_dict(data)
                if entry.is_expired():
                    expired_files += 1
                    
            except Exception:
                expired_files += 1
        
        return {
            'total_files': total_files,
            'expired_files': expired_files,
            'active_files': total_files - expired_files,
            'total_size_bytes': total_size,
            'cache_dir': str(self.cache_dir)
        }


class CacheManager:
    """Unified cache manager for search and scraping operations."""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 memory_cache_ttl: int = 300,
                 file_cache_ttl: int = 3600,
                 cache_dir: str = ".cache"):
        self.memory_cache = MemoryCache(max_size=memory_cache_size, default_ttl=memory_cache_ttl)
        self.file_cache = FileCache(cache_dir=cache_dir, default_ttl=file_cache_ttl)
        self.enabled = True
    
    def _generate_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key for operation."""
        key_data = f"{operation}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, *args, **kwargs) -> Optional[Any]:
        """Get cached result for operation."""
        if not self.enabled:
            return None
        
        key = self._generate_key(operation, *args, **kwargs)
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit (memory): {operation}")
            return result
        
        # Try file cache
        result = self.file_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit (file): {operation}")
            # Store in memory cache for faster access
            self.memory_cache.set(key, result)
            return result
        
        logger.debug(f"Cache miss: {operation}")
        return None
    
    def set(self, operation: str, value: Any, ttl: Optional[int] = None, *args, **kwargs) -> None:
        """Set cached result for operation."""
        if not self.enabled:
            return
        
        key = self._generate_key(operation, *args, **kwargs)
        
        # Store in both caches
        self.memory_cache.set(key, value, ttl)
        self.file_cache.set(key, value, ttl)
        
        logger.debug(f"Cached result: {operation}")
    
    def delete(self, operation: str, *args, **kwargs) -> bool:
        """Delete cached result for operation."""
        key = self._generate_key(operation, *args, **kwargs)
        
        memory_deleted = self.memory_cache.delete(key)
        file_deleted = self.file_cache.delete(key)
        
        return memory_deleted or file_deleted
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.file_cache.clear()
        logger.info("All caches cleared")
    
    def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries from all caches."""
        memory_cleaned = self.memory_cache.cleanup_expired()
        file_cleaned = self.file_cache.cleanup_expired()
        
        logger.info(f"Cache cleanup: {memory_cleaned} memory entries, {file_cleaned} file entries removed")
        
        return {
            'memory_cleaned': memory_cleaned,
            'file_cleaned': file_cleaned
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'file_cache': self.file_cache.get_stats(),
            'enabled': self.enabled
        }
    
    def enable(self) -> None:
        """Enable caching."""
        self.enabled = True
        logger.info("Caching enabled")
    
    def disable(self) -> None:
        """Disable caching."""
        self.enabled = False
        logger.info("Caching disabled")


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(operation: str, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Try to get from cache
            cached_result = cache_manager.get(operation, *args, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(operation, result, ttl, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


# Convenience functions for common operations
def cache_search_results(query: str, results: Any, ttl: int = 300) -> None:
    """Cache search results."""
    cache_manager = get_cache_manager()
    cache_manager.set("search", results, ttl, query)

def get_cached_search_results(query: str) -> Optional[Any]:
    """Get cached search results."""
    cache_manager = get_cache_manager()
    return cache_manager.get("search", query)

def cache_scraped_content(url: str, content: Any, ttl: int = 3600) -> None:
    """Cache scraped content."""
    cache_manager = get_cache_manager()
    cache_manager.set("scrape", content, ttl, url)

def get_cached_scraped_content(url: str) -> Optional[Any]:
    """Get cached scraped content."""
    cache_manager = get_cache_manager()
    return cache_manager.get("scrape", url)

def clear_search_cache() -> None:
    """Clear search cache."""
    cache_manager = get_cache_manager()
    # This is a simplified approach - in a real implementation,
    # you might want to track keys by operation type
    cache_manager.clear()

def clear_scraping_cache() -> None:
    """Clear scraping cache."""
    cache_manager = get_cache_manager()
    # This is a simplified approach - in a real implementation,
    # you might want to track keys by operation type
    cache_manager.clear() 