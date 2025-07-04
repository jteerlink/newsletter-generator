"""
CrewAI Web Search Tools

This module provides CrewAI-compatible web search tools to replace DuckDuckGo search.
Uses SerperDevTool for reliable Google search results via CrewAI framework.
"""

from __future__ import annotations

import logging
import time
import os
import json
from typing import List, Dict, Any, Optional
from functools import lru_cache

# CrewAI imports
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew

# Fallback imports for existing functionality
try:
    from src.core.core import query_llm
except ImportError:
    query_llm = None

logger = logging.getLogger(__name__)

class CrewAISearchTool:
    """
    Advanced CrewAI search tool that uses SerperDev for reliable Google search results.
    Provides caching, error handling, and result formatting compatible with existing code.
    """
    
    def __init__(self, max_results_per_search: int = 5, enable_caching: bool = True):
        self.max_results_per_search = max_results_per_search
        self.enable_caching = enable_caching
        self.search_history = []
        
        # Initialize SerperDev tool
        try:
            self.serper_tool = SerperDevTool()
            logger.info("CrewAI SerperDev tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SerperDev tool: {e}")
            self.serper_tool = None
    
    def _check_api_key(self) -> bool:
        """Check if API key is configured"""
        api_key = os.getenv('SERPER_API_KEY')
        if not api_key or api_key == 'your-serper-api-key-here':
            logger.warning("SERPER_API_KEY not configured. Please set up your API key.")
            return False
        return True
    
    @lru_cache(maxsize=32)
    def _cached_search(self, query: str) -> str:
        """Cached search to avoid repeated identical searches."""
        if not self.enable_caching:
            return self._direct_search(query)
        
        try:
            return self._direct_search(query)
        except Exception as e:
            logger.error(f"Cached search error for query '{query}': {e}")
            return json.dumps([])
    
    def _direct_search(self, query: str) -> str:
        """Direct search using SerperDev tool"""
        if not self._check_api_key():
            return json.dumps([])
        
        if not self.serper_tool:
            logger.error("SerperDev tool not initialized")
            return json.dumps([])
        
        try:
            # Use SerperDev tool for search with correct interface
            raw_results = self.serper_tool.run(search_query=query)
            
            # SerperDev returns a dictionary with search results
            if isinstance(raw_results, dict):
                parsed_results = raw_results
            elif isinstance(raw_results, str):
                # Try to parse JSON if it's a string
                try:
                    parsed_results = json.loads(raw_results)
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text and create simple result
                    parsed_results = {"organic": [{"title": "Search Result", "snippet": raw_results, "link": ""}]}
            else:
                # Fallback for unexpected formats
                parsed_results = {"organic": [{"title": "Search Result", "snippet": str(raw_results), "link": ""}]}
            
            # Convert to expected format
            formatted_results = self._format_serper_results(parsed_results)
            return json.dumps(formatted_results[:self.max_results_per_search])
            
        except Exception as e:
            logger.error(f"SerperDev search error for query '{query}': {e}")
            return json.dumps([])
    
    def _format_serper_results(self, serper_results: Any) -> List[Dict]:
        """Format SerperDev results to match DuckDuckGo format"""
        formatted_results = []
        
        try:
            # Handle different SerperDev response formats
            if isinstance(serper_results, dict):
                # Check for organic results
                organic_results = serper_results.get('organic', [])
                if not organic_results:
                    # Try other possible keys
                    organic_results = serper_results.get('results', [])
                    if not organic_results:
                        organic_results = [serper_results]  # Single result
                
                for result in organic_results:
                    if isinstance(result, dict):
                        formatted_result = {
                            'title': result.get('title', result.get('snippet', 'No title')),
                            'body': result.get('snippet', result.get('description', 'No description')),
                            'href': result.get('link', result.get('url', ''))
                        }
                        formatted_results.append(formatted_result)
            
            elif isinstance(serper_results, list):
                # Direct list of results
                for result in serper_results:
                    if isinstance(result, dict):
                        formatted_result = {
                            'title': result.get('title', result.get('snippet', 'No title')),
                            'body': result.get('snippet', result.get('description', result.get('body', 'No description'))),
                            'href': result.get('link', result.get('url', result.get('href', '')))
                        }
                        formatted_results.append(formatted_result)
            
            else:
                # Single result or string
                formatted_result = {
                    'title': 'Search Result',
                    'body': str(serper_results),
                    'href': ''
                }
                formatted_results.append(formatted_result)
                
        except Exception as e:
            logger.error(f"Error formatting SerperDev results: {e}")
            # Fallback result
            formatted_results = [{
                'title': 'Search Error',
                'body': f'Error processing search results: {str(e)}',
                'href': ''
            }]
        
        return formatted_results
    
    def run(self, query: str) -> str:
        """
        Perform web search using CrewAI SerperDev tool.
        
        Args:
            query: The search query
            
        Returns:
            Formatted search results string
        """
        logger.info(f"Starting CrewAI search for: {query}")
        
        try:
            # Perform search
            raw_results = self._cached_search(query) if self.enable_caching else self._direct_search(query)
            results = json.loads(raw_results)
            
            if not results:
                return f"No search results found for query: {query}"
            
            # Store search history
            self.search_history.append({
                "query": query,
                "results": results,
                "result_count": len(results),
                "timestamp": time.time()
            })
            
            # Format results for display
            return self._format_search_results(results, query)
            
        except Exception as e:
            logger.error(f"CrewAI search error: {e}")
            return f"Search temporarily unavailable: {str(e)}"
    
    def _format_search_results(self, results: List[Dict], query: str) -> str:
        """Format search results into a readable string"""
        if not results:
            return f"No search results found for query: {query}"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            # Truncate body for readability
            truncated_body = body[:200] + '...' if len(body) > 200 else body
            
            formatted_results.append(f"""
{i}. **{title}**
   URL: {href}
   Summary: {truncated_body}
""")
        
        return f"Search Results for '{query}':\n" + "\n".join(formatted_results)


class CrewAIAgenticSearchTool(CrewAISearchTool):
    """
    Advanced agentic search tool using CrewAI with iterative refinement.
    Provides compatibility with the existing AgenticSearchTool interface.
    """
    
    def __init__(self, max_iterations: int = 3, max_results_per_search: int = 5):
        super().__init__(max_results_per_search)
        self.max_iterations = max_iterations
    
    def run(self, initial_query: str, target_information: str = "") -> str:
        """
        Perform agentic search with potential iterative refinement.
        
        Args:
            initial_query: The initial search query
            target_information: Description of what information is needed (optional)
            
        Returns:
            Comprehensive search results
        """
        logger.info(f"Starting CrewAI agentic search for: {initial_query}")
        
        # For now, perform a single enhanced search
        # Future enhancement: Add LLM-based result evaluation and query refinement
        
        try:
            # Perform primary search
            primary_results = super().run(initial_query)
            
            # If target information is specified and we have LLM access, could add evaluation
            if target_information and query_llm:
                # Future enhancement: Evaluate results and potentially refine query
                pass
            
            # For now, return primary results with agentic formatting
            return f"""
CREWAI AGENTIC SEARCH RESULTS
=============================
Search query: {initial_query}
Target information: {target_information or "General search"}
Search iterations: 1 (CrewAI enhanced)

{primary_results}

SEARCH METHODOLOGY:
- Used CrewAI SerperDev tool for Google search
- Applied result formatting and error handling
- Cached results for improved performance
"""
        
        except Exception as e:
            logger.error(f"CrewAI agentic search error: {e}")
            return f"Agentic search temporarily unavailable: {str(e)}"


# Global tool instances
_crewai_search_tool = None
_crewai_agentic_tool = None

def get_crewai_search_tool() -> CrewAISearchTool:
    """Get singleton CrewAI search tool instance"""
    global _crewai_search_tool
    if _crewai_search_tool is None:
        _crewai_search_tool = CrewAISearchTool()
    return _crewai_search_tool

def get_crewai_agentic_tool() -> CrewAIAgenticSearchTool:
    """Get singleton CrewAI agentic search tool instance"""
    global _crewai_agentic_tool
    if _crewai_agentic_tool is None:
        _crewai_agentic_tool = CrewAIAgenticSearchTool()
    return _crewai_agentic_tool


# CrewAI-compatible search functions
def crewai_search_web(query: str, max_results: int = 5) -> str:
    """
    CrewAI web search function compatible with existing search_web interface.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results string
    """
    logger.info(f"Performing CrewAI web search for: {query}")
    
    try:
        tool = get_crewai_search_tool()
        tool.max_results_per_search = max_results
        return tool.run(query)
    except Exception as e:
        logger.error(f"CrewAI web search error: {e}")
        return f"CrewAI search temporarily unavailable: {str(e)}"


def crewai_search_web_with_alternatives(primary_query: str, fallback_queries: List[str] = None) -> str:
    """
    Enhanced CrewAI search with fallback queries.
    
    Args:
        primary_query: Primary search query
        fallback_queries: List of fallback queries (not used in current implementation)
        
    Returns:
        Comprehensive search results
    """
    logger.info(f"Performing CrewAI search with alternatives for: {primary_query}")
    
    try:
        tool = get_crewai_agentic_tool()
        target_info = f"Comprehensive information about {primary_query}"
        return tool.run(primary_query, target_info)
    except Exception as e:
        logger.error(f"CrewAI alternative search error: {e}")
        return f"CrewAI alternative search temporarily unavailable: {str(e)}"


async def async_crewai_search_web(query: str, max_results: int = 5) -> str:
    """Async version of CrewAI web search"""
    # For now, wrapping the sync version
    # Future enhancement: Use async CrewAI capabilities
    return crewai_search_web(query, max_results)


# Fallback function that tries CrewAI first, then DuckDuckGo
def hybrid_search_web(query: str, max_results: int = 5) -> str:
    """
    Hybrid search that tries CrewAI first, then falls back to DuckDuckGo if needed.
    """
    try:
        # Try CrewAI first
        result = crewai_search_web(query, max_results)
        if "temporarily unavailable" not in result.lower():
            return result
    except Exception as e:
        logger.warning(f"CrewAI search failed, trying fallback: {e}")
    
    # Fallback to DuckDuckGo (if available)
    try:
        from src.tools.tools import search_web as duckduckgo_search_web
        logger.info("Falling back to DuckDuckGo search")
        return duckduckgo_search_web(query, max_results)
    except Exception as e:
        logger.error(f"Fallback search also failed: {e}")
        return f"All search methods temporarily unavailable for query: {query}"


# Phase 1: Immediate Value CrewAI Tools (No Additional APIs Required)
try:
    from crewai_tools import (
        ScrapeWebsiteTool,
        FileReadTool
    )
    
    # Test which tools actually work without OpenAI API keys
    working_tools = {}
    
    # Test ScrapeWebsiteTool
    try:
        test_scraper = ScrapeWebsiteTool()
        working_tools['ScrapeWebsiteTool'] = ScrapeWebsiteTool
        print("✅ ScrapeWebsiteTool - No API key required")
    except Exception as e:
        print(f"❌ ScrapeWebsiteTool requires API key: {e}")
    
    # Test FileReadTool
    try:
        test_file_reader = FileReadTool()
        working_tools['FileReadTool'] = FileReadTool
        print("✅ FileReadTool - No API key required")
    except Exception as e:
        print(f"❌ FileReadTool requires API key: {e}")
    
    # Try other tools individually with error handling
    optional_tools = [
        'WebsiteSearchTool',
        'DirectorySearchTool',
        'JSONSearchTool',
        'CSVSearchTool',
        'TXTSearchTool',
        'XMLSearchTool',
        'DOCXSearchTool',
        'MDXSearchTool'
    ]
    
    for tool_name in optional_tools:
        try:
            from crewai_tools import *
            tool_class = globals()[tool_name]
            test_instance = tool_class()
            working_tools[tool_name] = tool_class
            print(f"✅ {tool_name} - No API key required")
        except Exception as e:
            print(f"❌ {tool_name} requires API key: {e}")
    
    # Only create processors with working tools
    class CrewAIWebContentExtractor:
        """Enhanced web content extraction using CrewAI tools"""
        
        def __init__(self):
            self.scraper = working_tools.get('ScrapeWebsiteTool')() if 'ScrapeWebsiteTool' in working_tools else None
            self.website_search = working_tools.get('WebsiteSearchTool')() if 'WebsiteSearchTool' in working_tools else None
        
        def scrape_website(self, url: str) -> str:
            """Extract clean content from any website"""
            try:
                if self.scraper:
                    result = self.scraper.run(website_url=url)
                    return result
                else:
                    return "ScrapeWebsiteTool not available"
            except Exception as e:
                return f"Error scraping website: {str(e)}"
        
        def search_website(self, url: str, query: str) -> str:
            """Search within a specific website"""
            try:
                if self.website_search:
                    result = self.website_search.run(website_url=url, search_query=query)
                    return result
                else:
                    return "WebsiteSearchTool not available (requires OpenAI API key)"
            except Exception as e:
                return f"Error searching website: {str(e)}"
    
    # File system operations
    class CrewAIFileManager:
        """File system operations using CrewAI tools"""
        
        def __init__(self):
            self.file_reader = working_tools.get('FileReadTool')() if 'FileReadTool' in working_tools else None
            self.directory_search = working_tools.get('DirectorySearchTool')() if 'DirectorySearchTool' in working_tools else None
        
        def read_file(self, file_path: str, start_line: int = 1, line_count: int = None) -> str:
            """Read file contents with line control"""
            try:
                if self.file_reader:
                    if line_count:
                        result = self.file_reader.run(
                            file_path=file_path,
                            start_line=start_line,
                            line_count=line_count
                        )
                    else:
                        result = self.file_reader.run(file_path=file_path)
                    return result
                else:
                    return "FileReadTool not available"
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        def search_directory(self, directory_path: str, search_query: str) -> str:
            """Search within directory contents"""
            try:
                if self.directory_search:
                    result = self.directory_search.run(
                        directory_path=directory_path,
                        search_query=search_query
                    )
                    return result
                else:
                    return "DirectorySearchTool not available (requires OpenAI API key)"
            except Exception as e:
                return f"Error searching directory: {str(e)}"
    
    # Document processing (simplified)
    class CrewAIDocumentProcessor:
        """CrewAI-powered document processing capabilities"""
        
        def __init__(self):
            self.available_tools = {}
            for tool_name in ['TXTSearchTool', 'XMLSearchTool', 'JSONSearchTool', 'CSVSearchTool', 'DOCXSearchTool', 'MDXSearchTool']:
                if tool_name in working_tools:
                    self.available_tools[tool_name] = working_tools[tool_name]()
        
        def search_document(self, file_path: str, query: str) -> str:
            """Search within documents based on file type"""
            try:
                file_ext = file_path.lower().split('.')[-1]
                tool_mapping = {
                    'txt': 'TXTSearchTool',
                    'xml': 'XMLSearchTool', 
                    'json': 'JSONSearchTool',
                    'csv': 'CSVSearchTool',
                    'docx': 'DOCXSearchTool',
                    'md': 'MDXSearchTool',
                    'mdx': 'MDXSearchTool'
                }
                
                required_tool = tool_mapping.get(file_ext)
                if required_tool and required_tool in self.available_tools:
                    tool = self.available_tools[required_tool]
                    
                    # Use appropriate parameter names for each tool
                    if file_ext == 'txt':
                        return tool.run(txt_path=file_path, search_query=query)
                    elif file_ext == 'xml':
                        return tool.run(xml_path=file_path, search_query=query)
                    elif file_ext == 'json':
                        return tool.run(json_path=file_path, search_query=query)
                    elif file_ext == 'csv':
                        return tool.run(csv_path=file_path, search_query=query)
                    elif file_ext == 'docx':
                        return tool.run(docx_path=file_path, search_query=query)
                    elif file_ext in ['md', 'mdx']:
                        return tool.run(mdx_path=file_path, search_query=query)
                else:
                    return f"Document type '{file_ext}' not supported or requires OpenAI API key"
                    
            except Exception as e:
                return f"Error searching document: {str(e)}"
    
    # Initialize only working tools
    crewai_web_extractor = CrewAIWebContentExtractor()
    crewai_file_manager = CrewAIFileManager()
    crewai_doc_processor = CrewAIDocumentProcessor()
    
    # Enhanced tool functions (only for working tools)
    def crewai_scrape_website(url: str) -> str:
        """Enhanced website scraping using CrewAI ScrapeWebsiteTool"""
        try:
            result = crewai_web_extractor.scrape_website(url)
            return f"Website Content from {url}:\n{result}"
        except Exception as e:
            return f"Error scraping website: {str(e)}"
    
    def crewai_search_website(url: str, query: str) -> str:
        """Search within a specific website using CrewAI WebsiteSearchTool"""
        try:
            result = crewai_web_extractor.search_website(url, query)
            return f"Website Search Results for '{query}' on {url}:\n{result}"
        except Exception as e:
            return f"Error searching website: {str(e)}"
    
    def crewai_read_file(file_path: str, start_line: int = 1, line_count: int = None) -> str:
        """Read file contents using CrewAI FileReadTool"""
        try:
            result = crewai_file_manager.read_file(file_path, start_line, line_count)
            return f"File Content from {file_path}:\n{result}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def crewai_search_directory(directory_path: str, query: str) -> str:
        """Search within directory contents using CrewAI DirectorySearchTool"""
        try:
            result = crewai_file_manager.search_directory(directory_path, query)
            return f"Directory Search Results for '{query}' in {directory_path}:\n{result}"
        except Exception as e:
            return f"Error searching directory: {str(e)}"
    
    def crewai_search_document(file_path: str, query: str) -> str:
        """Search within documents using appropriate CrewAI document tool"""
        try:
            result = crewai_doc_processor.search_document(file_path, query)
            return f"Document Search Results for '{query}' in {file_path}:\n{result}"
        except Exception as e:
            return f"Error searching document: {str(e)}"
    
    # Newsletter-specific functions
    def crewai_extract_article_content(urls: List[str]) -> str:
        """Extract content from multiple article URLs for newsletter compilation"""
        try:
            articles = []
            for url in urls:
                content = crewai_web_extractor.scrape_website(url)
                articles.append(f"=== Article from {url} ===\n{content}\n")
            
            return "\n".join(articles)
        except Exception as e:
            return f"Error extracting articles: {str(e)}"
    
    def crewai_search_news_sites(sites: List[str], query: str) -> str:
        """Search across multiple news sites for newsletter content"""
        try:
            results = []
            for site in sites:
                search_result = crewai_web_extractor.search_website(site, query)
                results.append(f"=== Results from {site} ===\n{search_result}\n")
            
            return "\n".join(results)
        except Exception as e:
            return f"Error searching news sites: {str(e)}"
    
    CREWAI_PHASE1_TOOLS_AVAILABLE = len(working_tools) > 0
    print(f"✅ CrewAI Phase 1 tools loaded: {len(working_tools)} tools available")
    print(f"   Available tools: {list(working_tools.keys())}")
    
except ImportError as e:
    print(f"⚠️  CrewAI Phase 1 tools not available: {e}")
    CREWAI_PHASE1_TOOLS_AVAILABLE = False
    
    # Fallback implementations
    def crewai_scrape_website(url: str) -> str:
        return f"CrewAI scraping not available. URL: {url}"
    
    def crewai_search_website(url: str, query: str) -> str:
        return f"CrewAI website search not available. URL: {url}, Query: {query}"
    
    def crewai_read_file(file_path: str, start_line: int = 1, line_count: int = None) -> str:
        return f"CrewAI file reading not available. File: {file_path}"
    
    def crewai_search_directory(directory_path: str, query: str) -> str:
        return f"CrewAI directory search not available. Directory: {directory_path}, Query: {query}"
    
    def crewai_search_document(file_path: str, query: str) -> str:
        return f"CrewAI document search not available. File: {file_path}, Query: {query}"
    
    def crewai_extract_article_content(urls: List[str]) -> str:
        return f"CrewAI article extraction not available. URLs: {urls}"
    
    def crewai_search_news_sites(sites: List[str], query: str) -> str:
        return f"CrewAI news site search not available. Sites: {sites}, Query: {query}"

# Hybrid Scraping Functions (CrewAI + Crawl4AI Integration)
def hybrid_scraping_validation(url: str) -> dict:
    """
    Quick validation of website accessibility using CrewAI before full crawl4ai extraction
    
    Args:
        url: Website URL to validate
        
    Returns:
        dict: Validation results with accessibility info
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"accessible": True, "method": "skip_validation", "reason": "CrewAI not available"}
        
        # Quick validation with CrewAI ScrapeWebsiteTool
        result = crewai_web_extractor.scrape_website(url)
        
        # Analyze if content was successfully retrieved
        if result and len(result) > 100:  # Minimum content threshold
            # Check for common error indicators
            error_indicators = ["404", "not found", "access denied", "blocked", "error"]
            content_lower = result.lower()
            
            if any(indicator in content_lower for indicator in error_indicators):
                return {
                    "accessible": False,
                    "method": "crewai_validation",
                    "reason": "Error indicators detected",
                    "content_length": len(result)
                }
            
            return {
                "accessible": True,
                "method": "crewai_validation",
                "reason": "Content successfully retrieved",
                "content_length": len(result)
            }
        else:
            return {
                "accessible": False,
                "method": "crewai_validation",
                "reason": "Insufficient content retrieved",
                "content_length": len(result) if result else 0
            }
            
    except Exception as e:
        return {
            "accessible": False,
            "method": "crewai_validation",
            "reason": f"Validation failed: {str(e)}",
            "content_length": 0
        }

def hybrid_fallback_extraction(url: str, crawl4ai_failed: bool = False) -> dict:
    """
    Fallback content extraction using CrewAI when crawl4ai fails
    
    Args:
        url: Website URL to extract from
        crawl4ai_failed: Whether crawl4ai extraction failed
        
    Returns:
        dict: Fallback extraction results
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"success": False, "method": "no_fallback", "content": None}
        
        # Use CrewAI for basic content extraction
        result = crewai_web_extractor.scrape_website(url)
        
        if result and len(result) > 50:
            # Basic article detection in raw content
            potential_articles = []
            
            # Simple heuristics for article detection
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                # Look for lines that might be article titles (reasonable length, not too short)
                if 20 <= len(line) <= 200 and not line.startswith(('http', 'www')):
                    potential_articles.append(line)
            
            return {
                "success": True,
                "method": "crewai_fallback",
                "content": result,
                "content_length": len(result),
                "potential_articles": potential_articles[:10],  # Top 10 potential articles
                "fallback_reason": "crawl4ai_failed" if crawl4ai_failed else "direct_fallback"
            }
        else:
            return {
                "success": False,
                "method": "crewai_fallback",
                "content": result,
                "content_length": len(result) if result else 0,
                "reason": "Insufficient content extracted"
            }
            
    except Exception as e:
        return {
            "success": False,
            "method": "crewai_fallback",
            "content": None,
            "reason": f"Fallback extraction failed: {str(e)}"
        }

def hybrid_single_article_extraction(article_url: str) -> dict:
    """
    Extract content from a single article URL using CrewAI
    This is useful for processing individual article links found by crawl4ai
    
    Args:
        article_url: URL of specific article to extract
        
    Returns:
        dict: Article content extraction results
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"success": False, "method": "no_extraction", "content": None}
        
        # Use CrewAI to extract full article content
        result = crewai_web_extractor.scrape_website(article_url)
        
        if result and len(result) > 100:
            # Extract article-specific information
            lines = result.split('\n')
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            # Estimate article content quality
            total_chars = sum(len(line) for line in clean_lines)
            avg_line_length = total_chars / len(clean_lines) if clean_lines else 0
            
            # Look for article structure indicators
            has_paragraphs = any(len(line) > 100 for line in clean_lines)
            has_title = any(20 <= len(line) <= 200 for line in clean_lines[:5])
            
            quality_score = 0
            if has_paragraphs: quality_score += 1
            if has_title: quality_score += 1
            if avg_line_length > 30: quality_score += 1
            if total_chars > 1000: quality_score += 1
            
            return {
                "success": True,
                "method": "crewai_single_article",
                "content": result,
                "content_length": len(result),
                "quality_score": quality_score,
                "estimated_article_content": has_paragraphs,
                "url": article_url
            }
        else:
            return {
                "success": False,
                "method": "crewai_single_article",
                "content": result,
                "content_length": len(result) if result else 0,
                "reason": "Insufficient article content",
                "url": article_url
            }
            
    except Exception as e:
        return {
            "success": False,
            "method": "crewai_single_article",
            "content": None,
            "reason": f"Single article extraction failed: {str(e)}",
            "url": article_url
        }

def hybrid_content_preview(url: str, max_chars: int = 500) -> dict:
    """
    Quick content preview using CrewAI for rapid assessment
    
    Args:
        url: Website URL to preview
        max_chars: Maximum characters to return in preview
        
    Returns:
        dict: Content preview results
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"success": False, "method": "no_preview", "preview": None}
        
        # Quick content extraction
        result = crewai_web_extractor.scrape_website(url)
        
        if result:
            # Create clean preview
            preview_text = result.strip()[:max_chars]
            
            # Basic content analysis
            word_count = len(preview_text.split())
            
            # Check for content type indicators
            is_news_site = any(indicator in preview_text.lower() for indicator in [
                'news', 'article', 'story', 'report', 'latest', 'breaking'
            ])
            
            is_blog = any(indicator in preview_text.lower() for indicator in [
                'blog', 'post', 'author', 'published', 'category'
            ])
            
            content_type = "unknown"
            if is_news_site:
                content_type = "news"
            elif is_blog:
                content_type = "blog"
            
            return {
                "success": True,
                "method": "crewai_preview",
                "preview": preview_text,
                "word_count": word_count,
                "content_type": content_type,
                "full_length": len(result),
                "url": url
            }
        else:
            return {
                "success": False,
                "method": "crewai_preview",
                "preview": None,
                "reason": "No content retrieved",
                "url": url
            }
            
    except Exception as e:
        return {
            "success": False,
            "method": "crewai_preview",
            "preview": None,
            "reason": f"Preview generation failed: {str(e)}",
            "url": url
        }

# Enhanced newsletter-specific hybrid functions
def enhanced_newsletter_source_validation(sources: List[str]) -> dict:
    """
    Validate multiple newsletter sources quickly using CrewAI
    
    Args:
        sources: List of source URLs to validate
        
    Returns:
        dict: Validation results for all sources
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"validated_sources": sources, "method": "skip_validation"}
        
        results = {
            "accessible": [],
            "inaccessible": [],
            "validation_details": {},
            "total_sources": len(sources),
            "method": "crewai_batch_validation"
        }
        
        for source in sources:
            validation = hybrid_scraping_validation(source)
            results["validation_details"][source] = validation
            
            if validation["accessible"]:
                results["accessible"].append(source)
            else:
                results["inaccessible"].append(source)
        
        results["success_rate"] = len(results["accessible"]) / len(sources) if sources else 0
        
        return results
        
    except Exception as e:
        return {
            "validated_sources": sources,
            "method": "validation_failed",
            "error": str(e)
        }

def enhanced_article_content_extraction(article_urls: List[str]) -> dict:
    """
    Extract content from multiple article URLs using CrewAI
    
    Args:
        article_urls: List of article URLs to extract content from
        
    Returns:
        dict: Content extraction results for all articles
    """
    try:
        if not CREWAI_PHASE1_TOOLS_AVAILABLE:
            return {"success": False, "method": "no_extraction", "articles": []}
        
        results = {
            "success": True,
            "method": "crewai_batch_article_extraction",
            "articles": [],
            "total_urls": len(article_urls),
            "successful_extractions": 0,
            "failed_extractions": 0
        }
        
        for url in article_urls:
            extraction = hybrid_single_article_extraction(url)
            results["articles"].append(extraction)
            
            if extraction["success"]:
                results["successful_extractions"] += 1
            else:
                results["failed_extractions"] += 1
        
        results["success_rate"] = results["successful_extractions"] / len(article_urls) if article_urls else 0
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "method": "batch_extraction_failed",
            "error": str(e),
            "articles": []
        }

# Export hybrid functions for use in tool registry
HYBRID_FUNCTIONS = {
    'hybrid_scraping_validation': hybrid_scraping_validation,
    'hybrid_fallback_extraction': hybrid_fallback_extraction,
    'hybrid_single_article_extraction': hybrid_single_article_extraction,
    'hybrid_content_preview': hybrid_content_preview,
    'enhanced_newsletter_source_validation': enhanced_newsletter_source_validation,
    'enhanced_article_content_extraction': enhanced_article_content_extraction
} 