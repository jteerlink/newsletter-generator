"""
Enhanced Search Tools

This module provides enhanced search capabilities with confidence scoring,
multi-provider support, and intelligent result ranking for the newsletter
generation system. Includes ArXiv, GitHub, and NewsAPI integration.
"""

import hashlib
import json
import logging
import re
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


class ArxivSearchProvider:
    """ArXiv search provider for academic papers and research."""

    def __init__(self):
        self.name = "arxiv_search"
        self.base_url = "http://export.arxiv.org/api/query"

    def is_available(self) -> bool:
        """Check if ArXiv API is available."""
        try:
            response = requests.get(f"{self.base_url}?search_query=test&max_results=1", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv for academic papers."""
        try:
            search_query = self._build_arxiv_query(query)
            
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': min(max_results, 10),
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    def _build_arxiv_query(self, query: str) -> str:
        """Build ArXiv-compatible query string."""
        cleaned_query = re.sub(r'[^\w\s-]', '', query)
        return f"ti:{cleaned_query} OR abs:{cleaned_query}"

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response to result dictionaries."""
        results = []
        
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespace):
                title_elem = entry.find('atom:title', namespace)
                summary_elem = entry.find('atom:summary', namespace)
                id_elem = entry.find('atom:id', namespace)
                
                authors = []
                for author in entry.findall('atom:author', namespace):
                    name_elem = author.find('atom:name', namespace)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                if title_elem is not None and summary_elem is not None and id_elem is not None:
                    title = title_elem.text.strip()
                    summary = summary_elem.text.strip()
                    
                    if authors:
                        author_text = f"Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}"
                        summary = f"{author_text}\n\n{summary[:400]}..."
                    
                    results.append({
                        'title': f"[ArXiv] {title}",
                        'url': id_elem.text,
                        'snippet': summary,
                        'source': self.name
                    })
            
        except Exception as e:
            logger.error(f"Failed to parse ArXiv response: {e}")
        
        return results


class GitHubSearchProvider:
    """GitHub search provider for code examples and repositories."""

    def __init__(self):
        self.name = "github_search"
        self.api_base = "https://api.github.com/search"
        self.github_token = self._get_github_token()

    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment."""
        import os
        return os.getenv('GITHUB_TOKEN')

    def is_available(self) -> bool:
        """Check if GitHub API is available."""
        try:
            headers = {}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            response = requests.get(
                f"{self.api_base}/repositories?q=test&sort=stars&per_page=1",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub for repositories and code."""
        results = []
        
        try:
            repo_results = self._search_repositories(query, max_results)
            results.extend(repo_results)
            
            if self.github_token and len(results) < max_results:
                code_results = self._search_code(query, max_results - len(results))
                results.extend(code_results)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []

    def _search_repositories(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search GitHub repositories."""
        try:
            headers = {}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': min(max_results, 5)
            }
            
            response = requests.get(
                f"{self.api_base}/repositories",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for repo in data.get('items', []):
                title = f"[GitHub Repo] {repo['full_name']}"
                description = repo.get('description', 'No description available')
                
                snippet = f"⭐ {repo['stargazers_count']} stars | "
                snippet += f"Language: {repo.get('language', 'Unknown')} | "
                snippet += description[:200]
                
                results.append({
                    'title': title,
                    'url': repo['html_url'],
                    'snippet': snippet,
                    'source': self.name
                })
            
            return results
            
        except Exception as e:
            logger.error(f"GitHub repository search failed: {e}")
            return []

    def _search_code(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search GitHub code files."""
        try:
            headers = {'Authorization': f'token {self.github_token}'}
            
            params = {
                'q': query,
                'sort': 'indexed',
                'per_page': min(max_results, 3)
            }
            
            response = requests.get(
                f"{self.api_base}/code",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                repo_name = item['repository']['full_name']
                file_name = item['name']
                title = f"[GitHub Code] {file_name} in {repo_name}"
                
                snippet = f"File: {item['path']} | "
                snippet += f"Repository: {repo_name} | "
                snippet += item.get('text_matches', [{}])[0].get('fragment', 'Code example')[:200]
                
                results.append({
                    'title': title,
                    'url': item['html_url'],
                    'snippet': snippet,
                    'source': self.name
                })
            
            return results
            
        except Exception as e:
            logger.error(f"GitHub code search failed: {e}")
            return []


class NewsAPIProvider:
    """NewsAPI provider for recent news and developments."""

    def __init__(self):
        self.name = "news_api"
        self.api_key = self._get_news_api_key()
        self.base_url = "https://newsapi.org/v2"

    def _get_news_api_key(self) -> Optional[str]:
        """Get NewsAPI key from environment."""
        import os
        return os.getenv('NEWS_API_KEY')

    def is_available(self) -> bool:
        """Check if NewsAPI is available."""
        if not self.api_key:
            return False
        
        try:
            headers = {'X-API-Key': self.api_key}
            response = requests.get(
                f"{self.base_url}/sources",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for recent news articles."""
        if not self.is_available():
            logger.warning("NewsAPI not available - missing API key")
            return []
        
        try:
            headers = {'X-API-Key': self.api_key}
            
            params = {
                'q': query,
                'sortBy': 'relevancy',
                'pageSize': min(max_results, 10),
                'language': 'en'
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get('status') == 'error':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            results = []
            
            for article in data.get('articles', []):
                if article.get('title') and article.get('url'):
                    title = f"[News] {article['title']}"
                    description = article.get('description', '')
                    source_name = article.get('source', {}).get('name', 'Unknown')
                    published_at = article.get('publishedAt', '')
                    
                    snippet = f"Source: {source_name} | "
                    if published_at:
                        try:
                            date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            snippet += f"Published: {date_obj.strftime('%Y-%m-%d')} | "
                        except:
                            pass
                    snippet += description[:300] if description else 'No description available'
                    
                    results.append({
                        'title': title,
                        'url': article['url'],
                        'snippet': snippet,
                        'source': self.name
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return []


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
        # Initialize new provider instances
        self.arxiv_provider = ArxivSearchProvider()
        self.github_provider = GitHubSearchProvider()
        self.news_provider = NewsAPIProvider()
        
        self.search_providers = {
            'duckduckgo': self._search_duckduckgo,
            'google': self._search_google,
            'bing': self._search_bing,
            'arxiv': self._search_arxiv,
            'github': self._search_github,
            'news': self._search_news
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

    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Wrapper for ArXiv search provider."""
        if self.arxiv_provider.is_available():
            return self.arxiv_provider.search(query, max_results)
        return []

    def _search_github(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Wrapper for GitHub search provider."""
        if self.github_provider.is_available():
            return self.github_provider.search(query, max_results)
        return []

    def _search_news(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Wrapper for NewsAPI search provider."""
        if self.news_provider.is_available():
            return self.news_provider.search(query, max_results)
        return []

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

    def intelligent_search(self, query: str, context_hints: List[str] = None, max_results: int = 10) -> List[SearchResult]:
        """
        Perform intelligent search with provider selection based on query context.
        
        Args:
            query: Search query string
            context_hints: List of context hints to guide provider selection
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        context_hints = context_hints or []
        
        # Determine best providers for this query
        best_providers = self._select_best_providers(query, context_hints)
        
        # Execute search with selected providers
        all_results = []
        
        for provider_name in best_providers:
            if provider_name in self.search_providers:
                try:
                    provider_results = self.search_providers[provider_name](
                        query, max_results // len(best_providers) + 2)
                    all_results.extend(provider_results)
                    logger.info(f"Retrieved {len(provider_results)} results from {provider_name}")
                    
                    # Early termination if we have enough results
                    if len(all_results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Intelligent search failed with {provider_name}: {e}")
                    continue
        
        # Score and rank results
        scored_results = self._score_results(all_results, query, ' '.join(context_hints))
        ranked_results = self._rank_results(scored_results)
        
        # Deduplicate results
        final_results = self._deduplicate_results(ranked_results)
        
        return final_results[:max_results]

    def _select_best_providers(self, query: str, context_hints: List[str]) -> List[str]:
        """Select the best providers for a given query and context."""
        query_lower = query.lower()
        context_lower = ' '.join(context_hints).lower()
        
        selected_providers = []
        
        # Academic/research queries -> ArXiv first
        if any(term in query_lower or term in context_lower for term in 
               ['research', 'study', 'paper', 'academic', 'scientific', 'arxiv']):
            if self.arxiv_provider.is_available():
                selected_providers.append('arxiv')
        
        # Code/programming queries -> GitHub first
        if any(term in query_lower or term in context_lower for term in 
               ['code', 'programming', 'implementation', 'github', 'repository', 'api']):
            if self.github_provider.is_available():
                selected_providers.append('github')
        
        # News/recent development queries -> NewsAPI first
        if any(term in query_lower or term in context_lower for term in 
               ['news', 'recent', 'latest', 'development', 'breaking', 'today']):
            if self.news_provider.is_available():
                selected_providers.append('news')
        
        # Always include DuckDuckGo as a fallback for general web search
        selected_providers.append('duckduckgo')
        
        # Add Google if implemented
        if 'google' not in selected_providers:
            selected_providers.append('google')
        
        # Return unique providers, limited to top 3 for performance
        unique_providers = []
        for provider in selected_providers:
            if provider not in unique_providers:
                unique_providers.append(provider)
        
        return unique_providers[:3]

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


@dataclass
class ScoredResult(SearchResult):
    """SearchResult with additional scoring metadata."""
    provider_confidence: float
    fusion_score: float
    duplicate_count: int = 0


class MultiProviderSearchEngine:
    """Coordinated search across multiple providers with result fusion."""
    
    def __init__(self):
        self.enhanced_search = EnhancedSearchTool()
        self.provider_weights = {
            'arxiv': 0.95,      # Highest authority for research
            'github': 0.85,     # High authority for code
            'news': 0.70,       # Good for recent developments
            'duckduckgo': 0.65, # General web search
            'google': 0.80,     # High authority when available
            'bing': 0.75        # Good authority when available
        }
        
    def intelligent_search(self, query: str, providers: List[str] = None, 
                         max_results: int = 10) -> List[ScoredResult]:
        """Execute search across multiple providers with result fusion."""
        if providers is None:
            providers = ['arxiv', 'github', 'news', 'duckduckgo']
        
        # Collect results from all providers
        provider_results = {}
        successful_providers = []
        
        for provider in providers:
            if provider in self.enhanced_search.search_providers:
                try:
                    results = self.enhanced_search.search_providers[provider](
                        query, max_results)
                    
                    # Convert to SearchResult objects
                    search_results = self.enhanced_search._score_results(
                        results, query, None)
                    
                    provider_results[provider] = search_results
                    successful_providers.append(provider)
                    logger.info(f"Retrieved {len(search_results)} results from {provider}")
                    
                except Exception as e:
                    logger.warning(f"Provider {provider} search failed: {e}")
                    provider_results[provider] = []
        
        # If no providers succeeded, try fallback search with unified search provider
        if not successful_providers:
            logger.warning("All primary providers failed, trying fallback search")
            try:
                from .search_provider import get_unified_search_provider
                fallback_provider = get_unified_search_provider()
                fallback_results = fallback_provider.search(query, max_results)
                
                if fallback_results:
                    # Convert to ScoredResult format
                    scored_fallback = []
                    for result in fallback_results:
                        scored_result = ScoredResult(
                            title=result.title,
                            url=result.url,
                            snippet=result.snippet,
                            source=result.source,
                            confidence_score=0.7,  # Default confidence
                            fusion_score=0.7,
                            provider_scores={'fallback': 0.7}
                        )
                        scored_fallback.append(scored_result)
                    logger.info(f"Retrieved {len(scored_fallback)} results from fallback search")
                    return scored_fallback[:max_results]
            except Exception as e:
                logger.error(f"Fallback search also failed: {e}")
        
        # Fuse results with confidence scoring
        fused_results = self.confidence_scoring(provider_results, query)
        
        # Deduplicate with fusion scoring
        deduplicated = self.deduplicate_results(fused_results)
        
        # Sort by fusion score and return top results
        return sorted(deduplicated, key=lambda x: x.fusion_score, reverse=True)[:max_results]
    
    def confidence_scoring(self, provider_results: Dict[str, List[SearchResult]], 
                          query: str) -> List[ScoredResult]:
        """Apply confidence scoring based on source authority."""
        scored_results = []
        
        for provider, results in provider_results.items():
            provider_weight = self.provider_weights.get(provider, 0.5)
            
            for result in results:
                # Calculate provider-specific confidence
                provider_confidence = result.confidence_score * provider_weight
                
                # Calculate fusion score combining multiple factors
                fusion_score = self._calculate_fusion_score(
                    result, provider_confidence, provider_weight)
                
                scored_result = ScoredResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    source=f"{provider}:{result.source}",
                    confidence_score=result.confidence_score,
                    relevance_score=result.relevance_score,
                    freshness_score=result.freshness_score,
                    authority_score=result.authority_score,
                    overall_score=result.overall_score,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    provider_confidence=provider_confidence,
                    fusion_score=fusion_score
                )
                
                scored_results.append(scored_result)
        
        return scored_results
    
    def deduplicate_results(self, results: List[ScoredResult]) -> List[ScoredResult]:
        """Remove duplicate information across providers."""
        # Group by URL first
        url_groups = {}
        for result in results:
            if result.url in url_groups:
                url_groups[result.url].append(result)
            else:
                url_groups[result.url] = [result]
        
        deduplicated = []
        
        for url, group in url_groups.items():
            if len(group) == 1:
                # Single result, no duplication
                deduplicated.append(group[0])
            else:
                # Multiple results for same URL, merge them
                merged_result = self._merge_duplicate_results(group)
                deduplicated.append(merged_result)
        
        # Also check for content similarity across different URLs
        final_results = self._remove_content_duplicates(deduplicated)
        
        return final_results
    
    def _calculate_fusion_score(self, result: SearchResult, 
                               provider_confidence: float, 
                               provider_weight: float) -> float:
        """Calculate fusion score combining multiple factors."""
        # Base fusion score from original overall score
        fusion_score = result.overall_score * 0.4
        
        # Add provider confidence
        fusion_score += provider_confidence * 0.3
        
        # Add authority and relevance boost
        fusion_score += result.authority_score * 0.2
        fusion_score += result.relevance_score * 0.1
        
        # Normalize to 0-1 range
        return min(1.0, fusion_score)
    
    def _merge_duplicate_results(self, duplicates: List[ScoredResult]) -> ScoredResult:
        """Merge multiple results for the same URL."""
        # Sort by fusion score to get the best version
        best_result = max(duplicates, key=lambda x: x.fusion_score)
        
        # Combine information from all duplicates
        all_sources = [r.source for r in duplicates]
        combined_snippets = []
        
        for dup in duplicates:
            if dup.snippet and dup.snippet not in combined_snippets:
                combined_snippets.append(dup.snippet[:200])
        
        # Create merged result
        merged = ScoredResult(
            title=best_result.title,
            url=best_result.url,
            snippet=" | ".join(combined_snippets[:3]),  # Limit combined snippets
            source=f"merged:{','.join(set(all_sources))}",
            confidence_score=max(r.confidence_score for r in duplicates),
            relevance_score=max(r.relevance_score for r in duplicates),
            freshness_score=max(r.freshness_score for r in duplicates),
            authority_score=max(r.authority_score for r in duplicates),
            overall_score=max(r.overall_score for r in duplicates),
            metadata=best_result.metadata,
            timestamp=best_result.timestamp,
            provider_confidence=max(r.provider_confidence for r in duplicates),
            fusion_score=max(r.fusion_score for r in duplicates),
            duplicate_count=len(duplicates) - 1
        )
        
        return merged
    
    def _remove_content_duplicates(self, results: List[ScoredResult]) -> List[ScoredResult]:
        """Remove results with very similar content."""
        final_results = []
        
        for result in results:
            is_duplicate = False
            
            for existing in final_results:
                similarity = self._calculate_content_similarity(result, existing)
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    # Keep the higher-scored result
                    if result.fusion_score > existing.fusion_score:
                        final_results.remove(existing)
                        final_results.append(result)
                    break
            
            if not is_duplicate:
                final_results.append(result)
        
        return final_results
    
    def _calculate_content_similarity(self, result1: ScoredResult, 
                                    result2: ScoredResult) -> float:
        """Calculate content similarity between two results."""
        # Compare titles and snippets
        title1_words = set(result1.title.lower().split())
        title2_words = set(result2.title.lower().split())
        
        snippet1_words = set(result1.snippet.lower().split())
        snippet2_words = set(result2.snippet.lower().split())
        
        # Calculate Jaccard similarity for titles
        title_intersection = len(title1_words.intersection(title2_words))
        title_union = len(title1_words.union(title2_words))
        title_similarity = title_intersection / title_union if title_union > 0 else 0
        
        # Calculate Jaccard similarity for snippets
        snippet_intersection = len(snippet1_words.intersection(snippet2_words))
        snippet_union = len(snippet1_words.union(snippet2_words))
        snippet_similarity = snippet_intersection / snippet_union if snippet_union > 0 else 0
        
        # Weighted average (titles matter more)
        overall_similarity = (title_similarity * 0.7) + (snippet_similarity * 0.3)
        
        return overall_similarity
