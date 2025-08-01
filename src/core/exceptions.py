"""
Simplified exception classes for the newsletter generator.

This module provides a streamlined exception hierarchy for error handling.
"""

class NewsletterError(Exception):
    """Base exception for newsletter generator errors."""
    pass

class ConfigurationError(NewsletterError):
    """Raised when there's a configuration issue."""
    pass

class StorageError(NewsletterError):
    """Raised when there's a storage operation error."""
    pass

class SearchError(NewsletterError):
    """Raised when there's a search operation error."""
    pass

class ContentError(NewsletterError):
    """Raised when there's a content processing error."""
    pass

class AgentError(NewsletterError):
    """Raised when there's an agent operation error."""
    pass

class ValidationError(NewsletterError):
    """Raised when content validation fails."""
    pass

class ImportError(NewsletterError):
    """Raised when there's an import error."""
    pass

class LLMError(NewsletterError):
    """Raised when there's an LLM operation error."""
    pass 