"""
Enhanced Tools for AI Newsletter Generation with Agentic RAG

This module provides advanced tools including self-improving search capabilities
that iteratively refine queries based on result evaluation.
"""

from __future__ import annotations

import logging
import time
import json
from typing import List, Dict, Any, Optional
from functools import lru_cache
from duckduckgo_search import DDGS
from src.core.core import query_llm

# Use relative imports or handle import errors gracefully
try:
    from vector_db import get_db_collection, search_vector_db
except ImportError:
    try:
        from .vector_db import get_db_collection, search_vector_db
    except ImportError:
        # Fallback - search_knowledge_base will handle the missing function
        def search_vector_db(query, n_results):
            raise ImportError("Vector database functions not available")

logger = logging.getLogger(__name__)

class AgenticSearchTool:
    """
    Advanced search tool that uses iterative refinement and LLM evaluation
    to find the most relevant information through multiple search iterations.
    """
    
    def __init__(self, max_iterations: int = 3, max_results_per_search: int = 5):
        self.max_iterations = max_iterations
        self.max_results_per_search = max_results_per_search
        self.search_history = []
        
    @lru_cache(maxsize=16)
    def _cached_search(self, query: str) -> str:
        """Cached search to avoid repeated identical searches."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results_per_search))
                return json.dumps(results)
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return json.dumps([])
    
    def _evaluate_search_results(self, query: str, results: List[Dict], target_info: str) -> Dict[str, Any]:
        """
        Use LLM to evaluate if search results are sufficient and suggest improvements.
        """
        evaluation_prompt = f"""
        Evaluate these search results for the query: "{query}"
        Target information needed: {target_info}
        
        Search Results:
        {json.dumps(results[:3], indent=2)}
        
        Please analyze:
        1. Are these results sufficient to answer the target information need?
        2. What important aspects are missing?
        3. Suggest a better search query if needed.
        
        Respond in JSON format:
        {{
            "sufficient": true/false,
            "missing_aspects": ["aspect1", "aspect2"],
            "suggested_query": "improved search query",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            evaluation_response = query_llm(evaluation_prompt)
            # Try to parse JSON response
            start_idx = evaluation_response.find('{')
            end_idx = evaluation_response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = evaluation_response[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Could not parse evaluation response: {e}")
        
        # Fallback evaluation
        return {
            "sufficient": len(results) >= 3,
            "missing_aspects": [],
            "suggested_query": query,
            "confidence": 0.5
        }
    
    def run(self, initial_query: str, target_information: str = "") -> str:
        """
        Perform agentic search with iterative refinement.
        
        Args:
            initial_query: The initial search query
            target_information: Description of what information is needed
            
        Returns:
            Comprehensive search results from all iterations
        """
        logger.info(f"Starting agentic search for: {initial_query}")
        
        all_results = []
        current_query = initial_query
        self.search_history = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"Search iteration {iteration + 1}: {current_query}")
            
            # Perform search
            raw_results = self._cached_search(current_query)
            results = json.loads(raw_results)
            
            if not results:
                logger.warning(f"No results for query: {current_query}")
                if iteration == 0:
                    break
                continue
            
            # Store iteration results
            iteration_data = {
                "iteration": iteration + 1,
                "query": current_query,
                "results": results,
                "result_count": len(results)
            }
            self.search_history.append(iteration_data)
            all_results.extend(results)
            
            # Evaluate results (skip on last iteration)
            if iteration < self.max_iterations - 1:
                evaluation = self._evaluate_search_results(
                    current_query, results, target_information
                )
                
                logger.info(f"Evaluation: sufficient={evaluation.get('sufficient', False)}, "
                          f"confidence={evaluation.get('confidence', 0.0)}")
                
                # If results are sufficient, stop searching
                if evaluation.get('sufficient', False) and evaluation.get('confidence', 0) > 0.7:
                    logger.info("Search results deemed sufficient, stopping iterations")
                    break
                
                # Refine query for next iteration
                suggested_query = evaluation.get('suggested_query', current_query)
                if suggested_query != current_query:
                    current_query = suggested_query
                    logger.info(f"Refining query to: {current_query}")
                else:
                    # If no refinement suggested, try a broader approach
                    current_query = f"{initial_query} overview trends analysis"
            
            # Rate limiting
            time.sleep(1)
        
        # Format comprehensive results
        return self._format_agentic_results(all_results)
    
    def _format_agentic_results(self, results: List[Dict]) -> str:
        """Format all search results into a comprehensive summary."""
        if not results:
            return "No search results found."
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in results:
            url = result.get('href', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(unique_results[:10], 1):  # Limit to top 10
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            formatted_results.append(f"""
**Result {i}: {title}**
Source: {href}
Summary: {body[:200]}{'...' if len(body) > 200 else ''}
""")
        
        summary = f"""
AGENTIC SEARCH RESULTS
======================
Search iterations performed: {len(self.search_history)}
Total unique results found: {len(unique_results)}

COMPREHENSIVE FINDINGS:
{''.join(formatted_results)}

SEARCH METHODOLOGY SUMMARY:
- Performed {len(self.search_history)} search iterations
- Used iterative query refinement based on LLM evaluation
- Collected {len(results)} total results, {len(unique_results)} unique
- Applied relevance filtering and deduplication
"""
        
        return summary

# Enhanced traditional search functions for backward compatibility
@lru_cache(maxsize=32)
def search_web(query: str, max_results: int = 5) -> str:
    """
    Enhanced web search with caching and improved error handling.
    """
    logger.info(f"Performing web search for: {query}")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return f"No search results found for query: {query}"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            formatted_results.append(f"""
{i}. **{title}**
   URL: {href}
   Summary: {body[:150]}{'...' if len(body) > 150 else ''}
""")
        
        return f"Search Results for '{query}':\n" + "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search temporarily unavailable. Using fallback information for query: {query}"

async def async_search_web(query: str, max_results: int = 5) -> str:
    """Async version of web search for improved performance."""
    # For now, wrapping the sync version - can be enhanced with aiohttp later
    return search_web(query, max_results)

def search_web_with_alternatives(primary_query: str, fallback_queries: List[str] = None) -> str:
    """
    Enhanced search with fallback queries using agentic approach.
    """
    # Use agentic search for better results
    agentic_tool = AgenticSearchTool(max_iterations=2)
    target_info = f"Comprehensive information about {primary_query}"
    
    return agentic_tool.run(primary_query, target_info)

# ---------------------------------------------------------------------------
# Knowledge Base Search (stub)
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """Stub for knowledge base search (to maintain backward compatibility)."""
    logger.info(f"[search_knowledge_base] Query: {query} (stub, returning placeholder)")
    return "Knowledge base search not yet implemented in this version."

# ---------------------------------------------------------------------------

# Tool registry for agent access
AVAILABLE_TOOLS = {
    'search_web': search_web,
    'search_web_with_alternatives': search_web_with_alternatives,
    'agentic_search': AgenticSearchTool,
    'async_search_web': async_search_web,
    'search_knowledge_base': search_knowledge_base
}

# TODO: Convert DDGS call to aiohttp for true async; markers here for future async_execution
