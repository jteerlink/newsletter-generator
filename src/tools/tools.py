"""
Core tool functions for newsletter generation.

This module provides essential tools for web search, content processing,
and other utilities needed by the newsletter generation system.
"""

import logging
import requests
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse, urljoin
import hashlib
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Check Python version for compatibility
if sys.version_info < (3, 10):
    logger.warning("Python 3.9 detected - some tools may have compatibility issues")

# Available tools mapping
AVAILABLE_TOOLS = {
    'search_web': 'Web search functionality',
    'search_knowledge_base': 'Knowledge base search',
    'extract_content': 'Content extraction from URLs',
    'analyze_content': 'Content analysis and scoring',
    'format_content': 'Content formatting utilities',
    'validate_url': 'URL validation and processing',
    'generate_summary': 'Content summarization',
    'extract_keywords': 'Keyword extraction from text',
    'detect_language': 'Language detection',
    'clean_text': 'Text cleaning and normalization'
}

# Tool availability tracking
_tool_availability = {}

def check_tool_availability() -> Dict[str, bool]:
    """Check availability of all tools."""
    global _tool_availability
    
    if not _tool_availability:
        _tool_availability = {
            'search_web': True,
            'search_knowledge_base': True,
            'extract_content': True,
            'analyze_content': True,
            'format_content': True,
            'validate_url': True,
            'generate_summary': True,
            'extract_keywords': True,
            'detect_language': True,
            'clean_text': True
        }
    
    return _tool_availability.copy()

def search_web(query: str, max_results: int = 5) -> str:
    """
    Perform web search using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results as string
    """
    try:
        # Simple DuckDuckGo search implementation
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
        if 'AbstractText' in data and data['AbstractText']:
            results.append(f"Summary: {data['AbstractText']}")
        
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:max_results]:
                if 'Text' in topic:
                    results.append(f"• {topic['Text']}")
        
        if not results:
            return f"No results found for query: {query}"
        
        return "\n".join(results)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search failed: {str(e)}"

def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the local knowledge base.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Search results as string
    """
    try:
        # Placeholder for knowledge base search
        # This would integrate with the vector store
        return f"Knowledge base search for '{query}' - implementation pending"
        
    except Exception as e:
        logger.error(f"Knowledge base search error: {e}")
        return f"Knowledge base search failed: {str(e)}"

def extract_content(url: str) -> str:
    """
    Extract content from a URL.
    
    Args:
        url: URL to extract content from
        
    Returns:
        Extracted content as string
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Simple text extraction
        content = response.text
        
        # Basic cleaning
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content[:5000]  # Limit content length
        
    except Exception as e:
        logger.error(f"Content extraction error: {e}")
        return f"Content extraction failed: {str(e)}"

def analyze_content(content: str) -> Dict[str, Any]:
    """
    Analyze content and return metrics.
    
    Args:
        content: Content to analyze
        
    Returns:
        Analysis results as dictionary
    """
    try:
        # Basic content analysis
        word_count = len(content.split())
        char_count = len(content)
        
        # Simple readability score (basic implementation)
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        readability_score = max(0, 100 - avg_sentence_length * 2)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability_score,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Content analysis error: {e}")
        return {'error': str(e)}

def format_content(content: str, format_type: str = 'markdown') -> str:
    """
    Format content according to specified format.
    
    Args:
        content: Content to format
        format_type: Desired format ('markdown', 'html', 'plain')
        
    Returns:
        Formatted content
    """
    try:
        if format_type == 'markdown':
            # Basic markdown formatting
            content = re.sub(r'\n\n+', '\n\n', content)
            return content
        elif format_type == 'html':
            # Convert to basic HTML
            content = content.replace('\n', '<br>')
            return f"<div>{content}</div>"
        else:
            return content
            
    except Exception as e:
        logger.error(f"Content formatting error: {e}")
        return content

def validate_url(url: str) -> Dict[str, Any]:
    """
    Validate and process a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Validation results
    """
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            url = 'https://' + url
            parsed = urlparse(url)
        
        return {
            'is_valid': bool(parsed.netloc),
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'normalized_url': url
        }
        
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return {'is_valid': False, 'error': str(e)}

def generate_summary(content: str, max_length: int = 200) -> str:
    """
    Generate a summary of content.
    
    Args:
        content: Content to summarize
        max_length: Maximum summary length
        
    Returns:
        Generated summary
    """
    try:
        # Simple summary: take first few sentences
        sentences = re.split(r'[.!?]+', content)
        summary = '. '.join(sentences[:3]) + '.'
        
        if len(summary) > max_length:
            summary = summary[:max_length-3] + '...'
        
        return summary
        
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return f"Summary generation failed: {str(e)}"

def extract_keywords(content: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from content.
    
    Args:
        content: Content to extract keywords from
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    try:
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most frequent words
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        return []

def detect_language(content: str) -> str:
    """
    Detect the language of content.
    
    Args:
        content: Content to analyze
        
    Returns:
        Detected language code
    """
    try:
        # Simple language detection based on common words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = set(re.findall(r'\b\w+\b', content.lower()))
        english_word_count = len(words.intersection(english_words))
        
        if english_word_count > len(words) * 0.1:
            return 'en'
        else:
            return 'unknown'
            
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return 'unknown'

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Text cleaning error: {e}")
        return text

# Tool registry for easy access
TOOL_REGISTRY = {
    'search_web': search_web,
    'search_knowledge_base': search_knowledge_base,
    'extract_content': extract_content,
    'analyze_content': analyze_content,
    'format_content': format_content,
    'validate_url': validate_url,
    'generate_summary': generate_summary,
    'extract_keywords': extract_keywords,
    'detect_language': detect_language,
    'clean_text': clean_text
}

def get_tool(tool_name: str):
    """Get a tool function by name."""
    return TOOL_REGISTRY.get(tool_name)

def list_available_tools() -> List[str]:
    """List all available tool names."""
    return list(TOOL_REGISTRY.keys())
