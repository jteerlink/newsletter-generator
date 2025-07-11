"""
Enhanced Tools for AI Newsletter Generation with Agentic RAG

This module provides advanced tools including self-improving search capabilities
that iteratively refine queries based on result evaluation.
"""

from __future__ import annotations

import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Replace DuckDuckGo with Serper API
try:
    from crewai_tools import SerperDevTool
    SERPER_AVAILABLE = True
except ImportError:
    SERPER_AVAILABLE = False
    SerperDevTool = None

from core.core import query_llm

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

class SerperSearchTool:
    """
    Serper API search tool for reliable Google search results.
    Replaces DuckDuckGo search with Google via Serper API.
    """
    
    def __init__(self, max_results_per_search: int = 5):
        self.max_results_per_search = max_results_per_search
        self.serper_tool = None
        
        # Initialize Serper tool
        if SERPER_AVAILABLE:
            try:
                self.serper_tool = SerperDevTool()
                logger.info("Serper API tool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Serper tool: {e}")
                self.serper_tool = None
        else:
            logger.warning("SerperDevTool not available, falling back to knowledge base")
    
    def _check_api_key(self) -> bool:
        """Check if Serper API key is configured"""
        api_key = os.getenv('SERPER_API_KEY')
        if not api_key or api_key == 'your-serper-api-key-here':
            logger.warning("SERPER_API_KEY not configured. Please set up your API key.")
            return False
        return True
    
    def search(self, query: str) -> List[Dict]:
        """Perform search using Serper API"""
        if not self._check_api_key() or not self.serper_tool:
            return []
        
        try:
            # Use Serper API for search
            raw_results = self.serper_tool.run(search_query=query)
            
            # Parse results
            if isinstance(raw_results, dict):
                parsed_results = raw_results
            elif isinstance(raw_results, str):
                try:
                    parsed_results = json.loads(raw_results)
                except json.JSONDecodeError:
                    parsed_results = {"organic": [{"title": "Search Result", "snippet": raw_results, "link": ""}]}
            else:
                parsed_results = {"organic": [{"title": "Search Result", "snippet": str(raw_results), "link": ""}]}
            
            # Format results
            return self._format_serper_results(parsed_results)
            
        except Exception as e:
            logger.error(f"Serper API search error for query '{query}': {e}")
            return []
    
    def _format_serper_results(self, serper_results: Any) -> List[Dict]:
        """Format Serper API results to standard format"""
        formatted_results = []
        
        try:
            # Handle different Serper response formats
            if isinstance(serper_results, dict):
                organic_results = serper_results.get('organic', [])
                if not organic_results:
                    organic_results = serper_results.get('results', [])
                    if not organic_results:
                        organic_results = [serper_results]
                
                for result in organic_results:
                    if isinstance(result, dict):
                        formatted_result = {
                            'title': result.get('title', result.get('snippet', 'No title')),
                            'body': result.get('snippet', result.get('description', 'No description')),
                            'href': result.get('link', result.get('url', ''))
                        }
                        formatted_results.append(formatted_result)
            
            elif isinstance(serper_results, list):
                for result in serper_results:
                    if isinstance(result, dict):
                        formatted_result = {
                            'title': result.get('title', result.get('snippet', 'No title')),
                            'body': result.get('snippet', result.get('description', result.get('body', 'No description'))),
                            'href': result.get('link', result.get('url', result.get('href', '')))
                        }
                        formatted_results.append(formatted_result)
            
            else:
                formatted_result = {
                    'title': 'Search Result',
                    'body': str(serper_results),
                    'href': ''
                }
                formatted_results.append(formatted_result)
                
        except Exception as e:
            logger.error(f"Error formatting Serper results: {e}")
            formatted_results = [{
                'title': 'Search Error',
                'body': f'Error processing search results: {str(e)}',
                'href': ''
            }]
        
        return formatted_results[:self.max_results_per_search]

# Global Serper tool instance
_serper_tool = SerperSearchTool()

class AgenticSearchTool:
    """
    Advanced search tool that uses iterative refinement and LLM evaluation
    to find the most relevant information through multiple search iterations.
    Now uses Serper API instead of DuckDuckGo.
    """
    
    def __init__(self, max_iterations: int = 3, max_results_per_search: int = 5):
        self.max_iterations = max_iterations
        self.max_results_per_search = max_results_per_search
        self.search_history = []
        self.serper_tool = SerperSearchTool(max_results_per_search)
        
    @lru_cache(maxsize=16)
    def _cached_search(self, query: str) -> str:
        """Cached search to avoid repeated identical searches using Serper API."""
        try:
            results = self.serper_tool.search(query)
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
        Perform agentic search with iterative refinement using Serper API.
        
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
AGENTIC SEARCH RESULTS (via Serper API)
========================================
Search iterations performed: {len(self.search_history)}
Total unique results found: {len(unique_results)}

COMPREHENSIVE FINDINGS:
{''.join(formatted_results)}

SEARCH METHODOLOGY SUMMARY:
- Performed {len(self.search_history)} search iterations
- Used iterative query refinement based on LLM evaluation
- Collected {len(results)} total results, {len(unique_results)} unique
- Applied relevance filtering and deduplication
- Powered by Serper API for reliable Google search results
"""
        
        return summary

# Enhanced traditional search functions for backward compatibility
@lru_cache(maxsize=32)
def search_web(query: str, max_results: int = 5) -> str:
    """
    Enhanced web search using Serper API with caching and improved error handling.
    """
    logger.info(f"Performing web search for: {query}")
    
    try:
        # Use Serper API for search
        serper_tool = SerperSearchTool(max_results)
        results = serper_tool.search(query)
        
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
        
        return f"Search Results for '{query}' (via Serper API):\n" + "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        # Enhanced fallback with knowledge-based content generation
        return _generate_fallback_content(query)

def _generate_fallback_content(query: str) -> str:
    """Generate comprehensive fallback content when search is unavailable."""
    logger.info(f"Generating fallback content for: {query}")
    
    # Check if we have an API key configured
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key or api_key == 'your-serper-api-key-here':
        api_message = """
⚠️ **API Configuration Required**
To enable web search, please configure your Serper API key:
1. Get a free API key at https://serper.dev
2. Set environment variable: SERPER_API_KEY=your-api-key-here
"""
    else:
        api_message = "Search API temporarily unavailable."
    
    # Generate comprehensive content from knowledge base
    fallback_prompt = f"""
Generate comprehensive research content about: {query}

Please provide detailed information covering:
1. Technical overview and key concepts
2. Current state and recent developments
3. Practical applications and use cases
4. Benefits and challenges
5. Future outlook and trends

Format as a structured research summary with clear sections.
"""
    
    try:
        if query_llm:
            knowledge_content = query_llm(fallback_prompt)
            return f"""
{api_message}

**Knowledge Base Research Results for '{query}':**
{knowledge_content}

---
*Note: This content is generated from the knowledge base. For the most current information, please configure web search.*
"""
        else:
            return f"{api_message}\n\nFallback content generation not available."
            
    except Exception as e:
        logger.error(f"Fallback content generation error: {e}")
        return f"{api_message}\n\nError generating fallback content: {str(e)}"

async def async_search_web(query: str, max_results: int = 5) -> str:
    """Async version of web search using Serper API."""
    # For now, wrapping the sync version - can be enhanced with async Serper API later
    return search_web(query, max_results)

def search_web_with_alternatives(primary_query: str, fallback_queries: List[str] = None) -> str:
    """
    Enhanced search with fallback queries using agentic approach with Serper API.
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

# Note: CrewAI tools integration removed - relying solely on superior crawl4ai implementation
# The crawl4ai system provides structured article extraction which is ideal for newsletters

# CrewAI integration status - for compatibility with existing agent code
CREWAI_AVAILABLE = False  # Set to False since we're using custom implementations

# Recommended tool mappings - for compatibility with existing agent code
RECOMMENDED_TOOLS = {
    'search_web': 'search_web',
    'search_knowledge_base': 'search_knowledge_base',
    'agentic_search': 'agentic_search'
}

# Tool registry for agent access
AVAILABLE_TOOLS = {
    # Core search tools
    'search_web': search_web,
    'search_web_with_alternatives': search_web_with_alternatives,
    'agentic_search': AgenticSearchTool().run,  # Make callable
    'hybrid_search_web': search_web_with_alternatives,  # Alias for compatibility
    'async_search_web': async_search_web,
    'search_knowledge_base': search_knowledge_base,
}

# TODO: Convert DDGS call to aiohttp for true async; markers here for future async_execution
