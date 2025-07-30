"""Consolidated utilities for the newsletter generator."""

import logging
import time
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
import asyncio

from .constants import LOG_FORMAT, LOG_LEVEL
from .exceptions import NewsletterGeneratorError

logger = logging.getLogger(__name__)

def setup_logging(name: str = None, level: str = None) -> logging.Logger:
    """Setup logging with consistent configuration."""
    if name is None:
        name = __name__
    if level is None:
        level = LOG_LEVEL
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        from .constants import LOGS_PATH
        log_file = LOGS_PATH / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry functions on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def async_retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry async functions on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON string with fallback."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default

def safe_json_dumps(data: Any, default: str = "") -> str:
    """Safely serialize data to JSON string with fallback."""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize to JSON: {e}")
        return default

def generate_content_hash(content: str) -> str:
    """Generate a hash for content to detect duplicates."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system operations."""
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized

def parse_date_string(date_string: str) -> Optional[datetime]:
    """Parse date string with multiple format support."""
    if not date_string:
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
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    # Try with dateutil if available
    try:
        from dateutil import parser
        return parser.parse(date_string)
    except ImportError:
        pass
    
    logger.warning(f"Could not parse date string: {date_string}")
    return None

def format_timestamp(timestamp: Union[datetime, str, float]) -> str:
    """Format timestamp consistently."""
    if isinstance(timestamp, str):
        dt = parse_date_string(timestamp)
        if dt is None:
            return timestamp
        timestamp = dt
    elif isinstance(timestamp, float):
        timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    return timestamp.isoformat()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks

def embed_chunks(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """Embed a list of text chunks using a sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(chunks, show_progress_bar=True)
    except ImportError:
        logger.warning("sentence-transformers not available, returning dummy embeddings")
        # Return dummy embeddings for testing
        return [[0.0] * 384 for _ in chunks]  # 384 is typical embedding size
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except ImportError:
        # Fallback to regex if BeautifulSoup is not available
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Calculate estimated reading time in minutes."""
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))

def validate_url(url: str) -> bool:
    """Validate URL format."""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], overwrite: bool = True) -> Dict[str, Any]:
    """Merge two dictionaries with optional overwrite."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, overwrite)
        elif key not in result or overwrite:
            result[key] = value
    
    return result

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension from path."""
    return Path(file_path).suffix.lower()

def is_supported_file_type(file_path: Union[str, Path], supported_types: List[str]) -> bool:
    """Check if file type is supported."""
    ext = get_file_extension(file_path)
    return ext in supported_types

def create_backup(file_path: Union[str, Path]) -> Path:
    """Create a backup of a file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")
    
    import shutil
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    return backup_path

def cleanup_old_backups(directory: Union[str, Path], max_backups: int = 5) -> None:
    """Clean up old backup files, keeping only the most recent ones."""
    directory = Path(directory)
    if not directory.exists():
        return
    
    # Find all backup files
    backup_files = list(directory.glob("*.backup_*"))
    
    if len(backup_files) <= max_backups:
        return
    
    # Sort by modification time (oldest first)
    backup_files.sort(key=lambda x: x.stat().st_mtime)
    
    # Remove oldest files
    files_to_remove = backup_files[:-max_backups]
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            logger.info(f"Removed old backup: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove backup {file_path}: {e}")

def measure_execution_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def async_measure_execution_time(func):
    """Decorator to measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper 