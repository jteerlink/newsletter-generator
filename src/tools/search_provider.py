"""
Unified search provider that combines multiple search sources.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

class SearchQuery:
    """Represents a search query."""
    
    def __init__(self, query: str, max_results: int = 5):
        self.query = query
        self.max_results = max_results
        self.timestamp = datetime.now()

class SearchResult:
    """Represents a search result."""
    
    def __init__(self, title: str, url: str, snippet: str, source: str = "unknown"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.timestamp = datetime.now()

class SerperSearchProvider:
    """
    Serper API search provider for Google search results.
    """
    
    def __init__(self):
        self.name = "serper_search"
        self.serper_tool = None
        self._initialize_serper()
    
    def _initialize_serper(self):
        """Initialize SerperDev tool with comprehensive error handling."""
        try:
            # Check if we're in a compatible Python environment
            import sys
            if sys.version_info < (3, 10):
                logger.warning("Python 3.9 detected - crewai_tools may have compatibility issues")
            
            # Try multiple import paths for SerperDevTool
            import_paths = [
                ('crewai_tools', 'SerperDevTool'),
                ('serper_dev', 'SerperDevTool'),
                ('crewai.tools', 'SerperDevTool')
            ]
            
            for module_name, class_name in import_paths:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    SerperDevTool = getattr(module, class_name)
                    self.serper_tool = SerperDevTool()
                    logger.info(f"Serper API tool initialized successfully via {module_name}")
                    return
                except (ImportError, AttributeError, TypeError, SyntaxError) as e:
                    logger.debug(f"Failed to import {class_name} from {module_name}: {e}")
                    continue
            
            # If all imports fail, log a clear message
            logger.warning("SerperDevTool not available. Install with: pip install crewai-tools")
            self.serper_tool = None
            
        except Exception as e:
            logger.error(f"Failed to initialize Serper tool: {e}")
            self.serper_tool = None


class DuckDuckGoSearchProvider:
    """DuckDuckGo search provider as fallback."""
    
    def __init__(self):
        self.name = "duckduckgo_search"
    
    def _perform_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search using DuckDuckGo."""
        try:
            import requests
            
            # Use DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query.query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Add instant answer if available
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('Heading', 'DuckDuckGo Result'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', ''),
                    source=self.name,
                    metadata={'type': 'instant_answer'}
                ))
            
            # Add related topics
            for topic in data.get('RelatedTopics', [])[:query.max_results]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(SearchResult(
                        title=topic.get('Text', '').split(' - ')[0] if ' - ' in topic.get('Text', '') else topic.get('Text', ''),
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        source=self.name,
                        metadata={'type': 'related_topic'}
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise


class KnowledgeBaseSearchProvider:
    """Local knowledge base search provider."""
    
    def __init__(self):
        self.name = "knowledge_base_search"
    
    def _perform_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search using local knowledge base."""
        try:
            # Try to import vector database functions
            try:
                from src.storage import search_vector_db
            except ImportError:
                try:
                    from src.storage import search_vector_db
                except ImportError:
                    logger.warning("Vector database not available")
                    return []
            
            # Search vector database
            results = search_vector_db(query.query, query.max_results)
            
            formatted_results = []
            for result in results:
                formatted_results.append(SearchResult(
                    title=result.get('title', 'Knowledge Base Result'),
                    url=result.get('url', ''),
                    snippet=result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                    source=self.name,
                    metadata={'score': result.get('score'), 'type': 'vector_search'}
                ))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Knowledge base search error: {e}")
            return []


class UnifiedSearchProvider:
    """Unified search provider that combines multiple search sources."""
    
    def __init__(self, providers: List | None = None):
        self.providers = providers or self._get_default_providers()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _get_default_providers(self) -> List:
        """Get default search providers in order of preference."""
        providers = []
        
        # Try Serper first (most reliable)
        serper_provider = SerperSearchProvider()
        if serper_provider.is_available():
            providers.append(serper_provider)
        
        # Add DuckDuckGo as fallback
        providers.append(DuckDuckGoSearchProvider())
        
        # Add knowledge base as local fallback
        providers.append(KnowledgeBaseSearchProvider())
        
        return providers
    
    @lru_cache(maxsize=100)
    def _cached_search(self, query_str: str, max_results: int) -> List[SearchResult]:
        """Cached search implementation."""
        query = SearchQuery(query=query_str, max_results=max_results)
        
        # Try each provider in order
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            try:
                results = provider.search(query)
                if results:
                    logger.info(f"Search successful with {provider.name}: {len(results)} results")
                    return results
            except Exception as e:
                logger.warning(f"Search failed with {provider.name}: {e}")
                continue
        
        # If all providers fail, return empty results
        logger.error("All search providers failed")
        return []
    
    def search(self, query: str, max_results: int = 5, search_type: str = "web") -> List[SearchResult]:
        """Perform unified search."""
        return self._cached_search(query, max_results)
    
    async def async_search(self, query: str, max_results: int = 5, search_type: str = "web") -> List[SearchResult]:
        """Perform async unified search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, max_results, search_type)
    
    def search_with_alternatives(self, primary_query: str, fallback_queries: List[str] | None = None, max_results: int = 5) -> List[SearchResult]:
        """Search with fallback queries."""
        # Try primary query first
        results = self.search(primary_query, max_results)
        if results:
            return results
        
        # Try fallback queries
        if fallback_queries:
            for fallback_query in fallback_queries:
                results = self.search(fallback_query, max_results)
                if results:
                    return results
        
        return []
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [provider.name for provider in self.providers if provider.is_available()]


# Global instance for easy access
_unified_search_provider = None

def get_unified_search_provider() -> UnifiedSearchProvider:
    """Get the global unified search provider instance."""
    global _unified_search_provider
    if _unified_search_provider is None:
        try:
            _unified_search_provider = UnifiedSearchProvider()
        except Exception as e:
            logger.warning(f"Failed to initialize unified search provider: {e}")
            # Create a minimal fallback provider
            _unified_search_provider = UnifiedSearchProvider(providers=[])
    return _unified_search_provider


# Convenience functions for backward compatibility
def search_web(query: str, max_results: int = 5) -> str:
    """Search web and return formatted string results."""
    provider = get_unified_search_provider()
    results = provider.search(query, max_results)
    
    if not results:
        return "No search results found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result.title}\n   URL: {result.url}\n   {result.snippet}")
    
    return "\n\n".join(formatted_results)


def search_web_with_alternatives(primary_query: str, fallback_queries: List[str] | None = None, max_results: int = 5) -> str:
    """Search with fallback queries and return formatted string results."""
    provider = get_unified_search_provider()
    results = provider.search_with_alternatives(primary_query, fallback_queries, max_results)
    
    if not results:
        return "No search results found with any query."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result.title}\n   URL: {result.url}\n   {result.snippet}")
    
    return "\n\n".join(formatted_results)


async def async_search_web(query: str, max_results: int = 5) -> str:
    """Async search web and return formatted string results."""
    provider = get_unified_search_provider()
    results = await provider.async_search(query, max_results)
    
    if not results:
        return "No search results found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result.title}\n   URL: {result.url}\n   {result.snippet}")
    
    return "\n\n".join(formatted_results) 