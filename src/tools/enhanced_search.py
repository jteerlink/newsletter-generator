"""
Enhanced Search Tools

This module provides enhanced search capabilities with confidence scoring,
multi-provider support, and intelligent result ranking for the newsletter
generation system.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with enhanced metadata."""
    title: str
    url: str
    snippet: str
    source: str
    confidence_score: float
    relevance_score: float
    freshness_score: float
    authority_score: float
    overall_score: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class SearchQuery:
    """Represents a search query with context."""
    query: str
    context: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 10
    freshness_days: Optional[int] = None


class EnhancedSearchTool:
    """
    Enhanced search tool with confidence scoring and multi-provider support.

    This class provides intelligent search capabilities with:
    - Multi-provider search (DuckDuckGo, Google, Bing)
    - Confidence scoring for results
    - Intelligent ranking and filtering
    - Context-aware query expansion
    """

    def __init__(self):
        """Initialize the enhanced search tool."""
        self.search_providers = {
            'duckduckgo': self._search_duckduckgo,
            'google': self._search_google,
            'bing': self._search_bing
        }
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache

    def search_with_confidence(self, query: str, context: Optional[str] = None,
                               max_results: int = 10) -> List[SearchResult]:
        """
        Search with confidence scoring.

        Args:
            query: Search query string
            context: Optional context for query expansion
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects with confidence scores
        """
        try:
            logger.info(f"Starting enhanced search for query: {query}")

            # Check cache first
            cache_key = self._generate_cache_key(query, context, max_results)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info("Returning cached search results")
                    return cached_result['results']

            # Expand query with context
            expanded_query = self._expand_query(query, context)

            # Search across multiple providers
            all_results = []
            for provider_name, provider_func in self.search_providers.items():
                try:
                    provider_results = provider_func(
                        expanded_query, max_results)
                    all_results.extend(provider_results)
                    logger.info(
                        f"Retrieved {
                            len(provider_results)} results from {provider_name}")
                except Exception as e:
                    logger.warning(
                        f"Error searching with {provider_name}: {e}")

            # Score and rank results
            scored_results = self._score_results(all_results, query, context)
            ranked_results = self._rank_results(scored_results)

            # Deduplicate results
            final_results = self._deduplicate_results(ranked_results)

            # Cache results
            self.cache[cache_key] = {
                'results': final_results,
                'timestamp': time.time()
            }

            logger.info(
                f"Enhanced search completed. Found {
                    len(final_results)} results")
            return final_results[:max_results]

        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []

    def _generate_cache_key(
            self,
            query: str,
            context: Optional[str],
            max_results: int) -> str:
        """Generate cache key for search results."""
        key_data = f"{query}:{context}:{max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _expand_query(self, query: str, context: Optional[str]) -> str:
        """Expand search query with context and synonyms."""
        expanded_query = query

        # Add context if provided
        if context:
            # Extract key terms from context
            context_terms = self._extract_key_terms(context)
            if context_terms:
                expanded_query = f"{query} {' '.join(context_terms[:3])}"

        # Add common synonyms for newsletter-related terms
        synonyms = {
            'news': 'latest updates breaking',
            'technology': 'tech innovation digital',
            'business': 'corporate industry market',
            'politics': 'government policy election',
            'health': 'medical wellness healthcare',
            'science': 'research study discovery'
        }

        for term, synonym_list in synonyms.items():
            if term.lower() in query.lower():
                expanded_query += f" {synonym_list}"

        return expanded_query.strip()

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for query expansion."""
        # Simple key term extraction (in production, use NLP libraries)
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {
            'the',
            'a',
            'an',
            'and',
            'or',
            'but',
            'in',
            'on',
            'at',
            'to',
            'for',
            'of',
            'with',
            'by'}
        key_terms = [
            word for word in words if word not in stop_words and len(word) > 3]
        return key_terms[:5]  # Return top 5 terms

    def _search_duckduckgo(
            self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo API."""
        try:
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            # Add abstract if available
            if 'AbstractText' in data and data['AbstractText']:
                results.append({
                    'title': data.get('AbstractSource', 'DuckDuckGo'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data['AbstractText'],
                    'source': 'duckduckgo'
                })

            # Add related topics
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:max_results - 1]:
                    if 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1] or 'Related Topic',
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic['Text'],
                            'source': 'duckduckgo'
                        })

            return results

        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []

    def _search_google(
            self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google (simulated - would need API key in production)."""
        # This is a simplified implementation
        # In production, you would use Google Custom Search API
        logger.info("Google search not implemented (requires API key)")
        return []

    def _search_bing(self, query: str,
                     max_results: int) -> List[Dict[str, Any]]:
        """Search using Bing (simulated - would need API key in production)."""
        # This is a simplified implementation
        # In production, you would use Bing Search API
        logger.info("Bing search not implemented (requires API key)")
        return []

    def _score_results(self,
                       results: List[Dict[str,
                                          Any]],
                       original_query: str,
                       context: Optional[str]) -> List[SearchResult]:
        """Score search results based on multiple factors."""
        scored_results = []

        for result in results:
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                result, original_query)

            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                result, original_query, context)

            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(result)

            # Calculate authority score
            authority_score = self._calculate_authority_score(result)

            # Calculate overall score
            overall_score = (confidence_score + relevance_score +
                             freshness_score + authority_score) / 4

            # Create SearchResult object
            search_result = SearchResult(
                title=result.get('title', ''),
                url=result.get('url', ''),
                snippet=result.get('snippet', ''),
                source=result.get('source', 'unknown'),
                confidence_score=confidence_score,
                relevance_score=relevance_score,
                freshness_score=freshness_score,
                authority_score=authority_score,
                overall_score=overall_score,
                metadata=result.get('metadata', {}),
                timestamp=datetime.now()
            )

            scored_results.append(search_result)

        return scored_results

    def _calculate_confidence_score(
            self, result: Dict[str, Any], query: str) -> float:
        """Calculate confidence score for a search result."""
        score = 0.5  # Base score

        # Check if query terms appear in title
        query_terms = query.lower().split()
        title_lower = result.get('title', '').lower()
        snippet_lower = result.get('snippet', '').lower()

        title_matches = sum(1 for term in query_terms if term in title_lower)
        snippet_matches = sum(
            1 for term in query_terms if term in snippet_lower)

        # Boost score for title matches
        if title_matches > 0:
            score += 0.3 * (title_matches / len(query_terms))

        # Boost score for snippet matches
        if snippet_matches > 0:
            score += 0.2 * (snippet_matches / len(query_terms))

        # Boost score for exact phrase matches
        if query.lower() in title_lower or query.lower() in snippet_lower:
            score += 0.2

        return min(1.0, score)

    def _calculate_relevance_score(self, result: Dict[str, Any], query: str,
                                   context: Optional[str]) -> float:
        """Calculate relevance score for a search result."""
        score = 0.5  # Base score

        # Check semantic similarity (simplified)
        query_terms = set(query.lower().split())
        content_terms = set(
            (result.get(
                'title',
                '') +
                ' ' +
                result.get(
                'snippet',
                '')).lower().split())

        # Calculate Jaccard similarity
        intersection = len(query_terms.intersection(content_terms))
        union = len(query_terms.union(content_terms))

        if union > 0:
            similarity = intersection / union
            score += 0.3 * similarity

        # Context relevance
        if context:
            context_terms = set(context.lower().split())
            context_intersection = len(
                context_terms.intersection(content_terms))
            if len(context_terms) > 0:
                context_similarity = context_intersection / len(context_terms)
                score += 0.2 * context_similarity

        return min(1.0, score)

    def _calculate_freshness_score(self, result: Dict[str, Any]) -> float:
        """Calculate freshness score for a search result."""
        # Simplified freshness calculation
        # In production, you would extract dates from content or metadata
        return 0.7  # Default score

    def _calculate_authority_score(self, result: Dict[str, Any]) -> float:
        """Calculate authority score for a search result."""
        score = 0.5  # Base score

        url = result.get('url', '')
        if url:
            domain = urlparse(url).netloc.lower()

            # Boost score for known authoritative domains
            authoritative_domains = {
                'reuters.com', 'ap.org', 'bbc.com', 'cnn.com', 'nytimes.com',
                'washingtonpost.com', 'wsj.com', 'techcrunch.com', 'wired.com',
                'arstechnica.com', 'github.com', 'stackoverflow.com'
            }

            if domain in authoritative_domains:
                score += 0.3

            # Boost score for .edu and .gov domains
            if domain.endswith('.edu') or domain.endswith('.gov'):
                score += 0.2

        return min(1.0, score)

    def _rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank search results by overall score."""
        return sorted(results, key=lambda x: x.overall_score, reverse=True)

    def _deduplicate_results(
            self,
            results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    def search_with_filters(
            self, query: str, filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Search with specific filters.

        Args:
            query: Search query
            filters: Dictionary of filters to apply

        Returns:
            Filtered search results
        """
        results = self.search_with_confidence(query)

        # Apply filters
        filtered_results = []
        for result in results:
            if self._apply_filters(result, filters):
                filtered_results.append(result)

        return filtered_results

    def _apply_filters(self, result: SearchResult,
                       filters: Dict[str, Any]) -> bool:
        """Apply filters to a search result."""
        # Date filter
        if 'date_after' in filters:
            if result.timestamp < filters['date_after']:
                return False

        # Domain filter
        if 'allowed_domains' in filters:
            domain = urlparse(result.url).netloc
            if domain not in filters['allowed_domains']:
                return False

        # Score filter
        if 'min_score' in filters:
            if result.overall_score < filters['min_score']:
                return False

        return True

    def get_search_analytics(self, query: str) -> Dict[str, Any]:
        """
        Get analytics for a search query.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with search analytics
        """
        results = self.search_with_confidence(query)

        if not results:
            return {'error': 'No results found'}

        # Calculate analytics
        avg_confidence = sum(
            r.confidence_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_freshness = sum(r.freshness_score for r in results) / len(results)
        avg_authority = sum(r.authority_score for r in results) / len(results)

        # Source distribution
        source_counts = {}
        for result in results:
            source_counts[result.source] = source_counts.get(
                result.source, 0) + 1

        return {
            'query': query,
            'total_results': len(results),
            'average_scores': {
                'confidence': round(avg_confidence, 3),
                'relevance': round(avg_relevance, 3),
                'freshness': round(avg_freshness, 3),
                'authority': round(avg_authority, 3)
            },
            'source_distribution': source_counts,
            'top_results': [
                {
                    'title': r.title,
                    'url': r.url,
                    'score': round(r.overall_score, 3)
                }
                for r in results[:5]
            ]
        }
