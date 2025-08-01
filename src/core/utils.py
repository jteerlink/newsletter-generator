"""
Core utility functions for newsletter generation.

This module provides essential utilities for text processing, formatting,
and other common operations.
"""

import re
import json
import logging
import time
import functools
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

def setup_logging(name: str = None, level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()]', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within the last 100 characters
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    # This is a placeholder - actual embedding would use a model
    # For now, return dummy embeddings
    return [[0.0] * 384 for _ in chunks]  # 384 is a common embedding size

def format_newsletter_content(content: Dict[str, Any]) -> str:
    """
    Format newsletter content into a standardized markdown format.
    
    Args:
        content: Dictionary containing newsletter content
        
    Returns:
        Formatted newsletter as markdown string
    """
    try:
        title = content.get('title', 'Newsletter')
        sections = content.get('sections', [])
        summary = content.get('summary', '')
        timestamp = content.get('timestamp', datetime.now().isoformat())
        
        # Build the newsletter
        newsletter = f"# {title}\n\n"
        
        if summary:
            newsletter += f"## Summary\n{summary}\n\n"
        
        newsletter += f"*Generated on {timestamp}*\n\n"
        newsletter += "---\n\n"
        
        # Add sections
        for i, section in enumerate(sections, 1):
            section_title = section.get('title', f'Section {i}')
            section_content = section.get('content', '')
            
            newsletter += f"## {section_title}\n\n"
            newsletter += f"{section_content}\n\n"
        
        return newsletter
        
    except Exception as e:
        logger.error(f"Error formatting newsletter content: {e}")
        return f"# Newsletter\n\nError formatting content: {str(e)}"

def save_newsletter(content: str, filename: str = None) -> str:
    """
    Save newsletter content to a file.
    
    Args:
        content: Newsletter content to save
        filename: Optional filename, will generate one if not provided
        
    Returns:
        Path to saved file
    """
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"newsletter_{timestamp}.md"
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Newsletter saved to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving newsletter: {e}")
        raise

def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse date string with multiple format support."""
    if not date_str:
        return None
    
    # Common date formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%d %B %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def generate_content_hash(content: str) -> str:
    """Generate a hash for content to detect duplicates."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text)

def count_words(text: str) -> int:
    """Count words in text."""
    if not text:
        return 0
    return len(text.split())

def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score."""
    if not text:
        return 0.0
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    if not words or not sentences:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    
    # Simple Flesch Reading Ease approximation
    # Lower score = more difficult to read
    score = max(0, 100 - (avg_sentence_length * 2))
    
    return min(100, max(0, score))

def validate_content_quality(content: str) -> Dict[str, Any]:
    """Validate content quality and return metrics."""
    if not content:
        return {
            'valid': False,
            'word_count': 0,
            'readability_score': 0,
            'issues': ['Content is empty']
        }
    
    word_count = count_words(content)
    readability_score = calculate_readability_score(content)
    
    issues = []
    if word_count < 50:
        issues.append('Content is too short')
    if readability_score < 30:
        issues.append('Content may be difficult to read')
    
    return {
        'valid': len(issues) == 0,
        'word_count': word_count,
        'readability_score': readability_score,
        'issues': issues
    } 