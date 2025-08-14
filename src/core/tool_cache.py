"""
Tool Result Caching System

Intelligent caching for tool results across agents as specified in PRD Week 6.
Provides cross-agent coordination, result sharing, and intelligent cache
management with TTL and invalidation strategies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cached data."""
    SEARCH_RESULTS = "search_results"
    VECTOR_QUERY = "vector_query"
    CLAIM_VALIDATION = "claim_validation"
    INFORMATION_ENRICHMENT = "information_enrichment"
    AGENT_COORDINATION = "agent_coordination"
    WEB_SCRAPING = "web_scraping"
    ANALYSIS_RESULTS = "analysis_results"


class CachePolicy(Enum):
    """Cache invalidation policies."""
    TIME_BASED = "time_based"          # TTL-based expiration
    USAGE_BASED = "usage_based"        # LRU eviction
    CONTENT_BASED = "content_based"    # Content change detection
    DEPENDENCY_BASED = "dependency_based"  # Invalidate based on dependencies


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    data: Any
    cache_type: CacheType
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    dependencies: Set[str] = None
    metadata: Dict[str, Any] = None
    agent_source: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    average_ttl_seconds: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class ToolResultCache:
    """Intelligent caching for tool results across agents."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self.lock = Lock()
        
        # TTL configurations by cache type
        self.ttl_config = {
            CacheType.SEARCH_RESULTS: 1800,        # 30 minutes
            CacheType.VECTOR_QUERY: 3600,          # 1 hour
            CacheType.CLAIM_VALIDATION: 7200,      # 2 hours
            CacheType.INFORMATION_ENRICHMENT: 900,  # 15 minutes
            CacheType.AGENT_COORDINATION: 300,     # 5 minutes
            CacheType.WEB_SCRAPING: 1800,          # 30 minutes
            CacheType.ANALYSIS_RESULTS: 3600,      # 1 hour
        }
        
        # Agent coordination channels
        self.coordination_channels: Dict[str, Dict[str, Any]] = {}
        
    def cache_search_results(self, query: str, results: List[Any], 
                           provider: str = "unknown", ttl: Optional[int] = None,
                           agent_name: Optional[str] = None,
                           session_id: Optional[str] = None,
                           workflow_id: Optional[str] = None) -> str:
        """Cache search results with time-to-live."""
        cache_key = self._generate_search_key(query, provider)
        
        if ttl is None:
            ttl = self.ttl_config.get(CacheType.SEARCH_RESULTS, self.default_ttl)
        
        metadata = {
            "query": query,
            "provider": provider,
            "result_count": len(results),
            "result_types": [type(r).__name__ for r in results[:5]]
        }
        
        return self._store_entry(
            key=cache_key,
            data=results,
            cache_type=CacheType.SEARCH_RESULTS,
            ttl=ttl,
            metadata=metadata,
            agent_source=agent_name,
            session_id=session_id,
            workflow_id=workflow_id
        )
    
    def get_cached_results(self, query: str, provider: str = "unknown") -> Optional[List[Any]]:
        """Retrieve cached results if available and fresh."""
        cache_key = self._generate_search_key(query, provider)
        return self._retrieve_entry(cache_key)
    
    def cache_vector_query(self, query: str, results: List[Any], top_k: int = 5,
                          agent_name: Optional[str] = None,
                          session_id: Optional[str] = None,
                          workflow_id: Optional[str] = None) -> str:
        """Cache vector database query results."""
        cache_key = self._generate_vector_key(query, top_k)
        
        metadata = {
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
            "relevance_scores": [getattr(r, 'score', 0.0) for r in results[:3]]
        }
        
        return self._store_entry(
            key=cache_key,
            data=results,
            cache_type=CacheType.VECTOR_QUERY,
            ttl=self.ttl_config[CacheType.VECTOR_QUERY],
            metadata=metadata,
            agent_source=agent_name,
            session_id=session_id,
            workflow_id=workflow_id
        )
    
    def get_cached_vector_results(self, query: str, top_k: int = 5) -> Optional[List[Any]]:
        """Retrieve cached vector query results."""
        cache_key = self._generate_vector_key(query, top_k)
        return self._retrieve_entry(cache_key)
    
    def cache_claim_validation(self, claim_text: str, validation_result: Any,
                             agent_name: Optional[str] = None,
                             session_id: Optional[str] = None,
                             workflow_id: Optional[str] = None) -> str:
        """Cache claim validation results."""
        cache_key = self._generate_claim_key(claim_text)
        
        metadata = {
            "claim_text": claim_text[:100],
            "validation_status": getattr(validation_result, 'validation_status', 'unknown'),
            "confidence": getattr(validation_result, 'confidence', 0.0),
            "source_count": len(getattr(validation_result, 'sources', []))
        }
        
        return self._store_entry(
            key=cache_key,
            data=validation_result,
            cache_type=CacheType.CLAIM_VALIDATION,
            ttl=self.ttl_config[CacheType.CLAIM_VALIDATION],
            metadata=metadata,
            agent_source=agent_name,
            session_id=session_id,
            workflow_id=workflow_id
        )
    
    def get_cached_claim_validation(self, claim_text: str) -> Optional[Any]:
        """Retrieve cached claim validation results."""
        cache_key = self._generate_claim_key(claim_text)
        return self._retrieve_entry(cache_key)
    
    def share_between_agents(self, from_agent: str, to_agent: str, 
                           data: Dict[str, Any], message_type: str = "coordination",
                           session_id: Optional[str] = None,
                           workflow_id: Optional[str] = None) -> str:
        """Share tool results between agents in workflow."""
        coordination_key = f"{from_agent}->{to_agent}:{message_type}"
        
        metadata = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "data_keys": list(data.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        coordination_data = {
            "message": data,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "created_at": datetime.now(),
            "session_id": session_id,
            "workflow_id": workflow_id
        }
        
        # Store in coordination channels
        if to_agent not in self.coordination_channels:
            self.coordination_channels[to_agent] = {}
        
        self.coordination_channels[to_agent][coordination_key] = coordination_data
        
        # Also cache for longer-term storage
        return self._store_entry(
            key=coordination_key,
            data=coordination_data,
            cache_type=CacheType.AGENT_COORDINATION,
            ttl=self.ttl_config[CacheType.AGENT_COORDINATION],
            metadata=metadata,
            agent_source=from_agent,
            session_id=session_id,
            workflow_id=workflow_id
        )
    
    def get_agent_messages(self, agent_name: str, clear_after_read: bool = True) -> List[Dict[str, Any]]:
        """Get pending messages for an agent."""
        messages = []
        
        if agent_name in self.coordination_channels:
            for key, data in list(self.coordination_channels[agent_name].items()):
                messages.append(data)
                
                if clear_after_read:
                    del self.coordination_channels[agent_name][key]
        
        return messages
    
    def cache_analysis_results(self, analysis_key: str, results: Any,
                             dependencies: Optional[Set[str]] = None,
                             agent_name: Optional[str] = None,
                             session_id: Optional[str] = None,
                             workflow_id: Optional[str] = None) -> str:
        """Cache analysis results with dependency tracking."""
        cache_key = f"analysis:{analysis_key}"
        
        metadata = {
            "analysis_type": analysis_key,
            "result_type": type(results).__name__,
            "dependency_count": len(dependencies) if dependencies else 0
        }
        
        return self._store_entry(
            key=cache_key,
            data=results,
            cache_type=CacheType.ANALYSIS_RESULTS,
            ttl=self.ttl_config[CacheType.ANALYSIS_RESULTS],
            metadata=metadata,
            dependencies=dependencies,
            agent_source=agent_name,
            session_id=session_id,
            workflow_id=workflow_id
        )
    
    def get_cached_analysis(self, analysis_key: str) -> Optional[Any]:
        """Retrieve cached analysis results."""
        cache_key = f"analysis:{analysis_key}"
        return self._retrieve_entry(cache_key)
    
    def invalidate_by_dependency(self, dependency_key: str) -> int:
        """Invalidate all cache entries that depend on a given key."""
        invalidated_count = 0
        
        with self.lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if dependency_key in entry.dependencies:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                invalidated_count += 1
                self.stats.evictions += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries due to dependency: {dependency_key}")
        return invalidated_count
    
    def invalidate_by_session(self, session_id: str) -> int:
        """Invalidate all cache entries for a specific session."""
        invalidated_count = 0
        
        with self.lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.session_id == session_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                invalidated_count += 1
                self.stats.evictions += 1
        
        logger.info(f"Invalidated {invalidated_count} cache entries for session: {session_id}")
        return invalidated_count
    
    def cleanup_expired_entries(self) -> int:
        """Remove expired entries from cache."""
        current_time = datetime.now()
        expired_count = 0
        
        with self.lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if current_time > entry.expires_at:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                expired_count += 1
                self.stats.evictions += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired cache entries")
        
        return expired_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            self.stats.total_entries = len(self.cache)
            
            # Calculate total size (approximate)
            total_size = 0
            ttl_sum = 0
            type_counts = {}
            
            for entry in self.cache.values():
                try:
                    total_size += len(pickle.dumps(entry.data))
                except:
                    total_size += 1000  # Rough estimate for unpicklable objects
                
                ttl_sum += (entry.expires_at - entry.created_at).total_seconds()
                
                cache_type = entry.cache_type.value
                type_counts[cache_type] = type_counts.get(cache_type, 0) + 1
            
            self.stats.total_size_bytes = total_size
            if len(self.cache) > 0:
                self.stats.average_ttl_seconds = ttl_sum / len(self.cache)
            
            return {
                "performance": asdict(self.stats),
                "cache_contents": {
                    "total_entries": len(self.cache),
                    "types": type_counts,
                    "coordination_channels": len(self.coordination_channels),
                    "pending_messages": sum(len(messages) for messages in self.coordination_channels.values())
                },
                "memory_usage": {
                    "total_size_bytes": total_size,
                    "average_entry_size": total_size / len(self.cache) if len(self.cache) > 0 else 0,
                    "utilization": len(self.cache) / self.max_size
                }
            }
    
    def get_cache_entries_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get cache entries created by a specific agent."""
        entries = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry.agent_source == agent_name:
                    entries.append({
                        "key": key,
                        "cache_type": entry.cache_type.value,
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat(),
                        "access_count": entry.access_count,
                        "metadata": entry.metadata
                    })
        
        return entries
    
    def clear_cache(self, cache_type: Optional[CacheType] = None) -> int:
        """Clear cache entries, optionally filtered by type."""
        cleared_count = 0
        
        with self.lock:
            if cache_type is None:
                cleared_count = len(self.cache)
                self.cache.clear()
                self.coordination_channels.clear()
            else:
                keys_to_remove = []
                for key, entry in self.cache.items():
                    if entry.cache_type == cache_type:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.cache[key]
                    cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} cache entries" + 
                   (f" of type {cache_type.value}" if cache_type else ""))
        return cleared_count
    
    def _store_entry(self, key: str, data: Any, cache_type: CacheType, 
                    ttl: int, metadata: Optional[Dict[str, Any]] = None,
                    dependencies: Optional[Set[str]] = None,
                    agent_source: Optional[str] = None,
                    session_id: Optional[str] = None,
                    workflow_id: Optional[str] = None) -> str:
        """Store an entry in the cache."""
        current_time = datetime.now()
        expires_at = current_time + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            key=key,
            data=data,
            cache_type=cache_type,
            created_at=current_time,
            expires_at=expires_at,
            dependencies=dependencies or set(),
            metadata=metadata or {},
            agent_source=agent_source,
            session_id=session_id,
            workflow_id=workflow_id
        )
        
        with self.lock:
            # Check if cache is full and evict if necessary
            if len(self.cache) >= self.max_size:
                self._evict_entries(1)
            
            self.cache[key] = entry
            logger.debug(f"Cached entry: {key} (type: {cache_type.value}, TTL: {ttl}s)")
        
        return key
    
    def _retrieve_entry(self, key: str) -> Optional[Any]:
        """Retrieve an entry from the cache."""
        with self.lock:
            self.stats.total_requests += 1
            
            if key not in self.cache:
                self.stats.cache_misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if entry has expired
            if datetime.now() > entry.expires_at:
                del self.cache[key]
                self.stats.cache_misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self.stats.cache_hits += 1
            
            logger.debug(f"Cache hit: {key} (access count: {entry.access_count})")
            return entry.data
    
    def _evict_entries(self, count: int = 1) -> int:
        """Evict entries using LRU policy."""
        if not self.cache:
            return 0
        
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        evicted_count = 0
        for i in range(min(count, len(sorted_entries))):
            key = sorted_entries[i][0]
            del self.cache[key]
            evicted_count += 1
            self.stats.evictions += 1
        
        logger.debug(f"Evicted {evicted_count} cache entries")
        return evicted_count
    
    def _generate_search_key(self, query: str, provider: str) -> str:
        """Generate cache key for search results."""
        key_data = f"search:{provider}:{query}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_vector_key(self, query: str, top_k: int) -> str:
        """Generate cache key for vector query results."""
        key_data = f"vector:{query}:{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_claim_key(self, claim_text: str) -> str:
        """Generate cache key for claim validation results."""
        key_data = f"claim:{claim_text}"
        return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instance
_tool_cache = None
_cache_lock = Lock()


def get_tool_cache() -> ToolResultCache:
    """Get the global tool cache instance."""
    global _tool_cache
    if _tool_cache is None:
        with _cache_lock:
            if _tool_cache is None:
                _tool_cache = ToolResultCache()
    return _tool_cache