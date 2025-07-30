"""Custom exceptions for the newsletter generator."""

class NewsletterGeneratorError(Exception):
    """Base exception for newsletter generator."""
    pass

class LLMError(NewsletterGeneratorError):
    """Exception raised for LLM-related errors."""
    pass

class SearchError(NewsletterGeneratorError):
    """Exception raised for search-related errors."""
    pass

class ScrapingError(NewsletterGeneratorError):
    """Exception raised for scraping-related errors."""
    pass

class ValidationError(NewsletterGeneratorError):
    """Exception raised for validation errors."""
    pass

class AgentError(NewsletterGeneratorError):
    """Exception raised for agent-related errors."""
    pass

class ConfigurationError(NewsletterGeneratorError):
    """Exception raised for configuration errors."""
    pass

class QualityGateError(NewsletterGeneratorError):
    """Exception raised for quality gate failures."""
    pass

class ImportError(NewsletterGeneratorError):
    """Exception raised for import-related errors."""
    pass

class StorageError(NewsletterGeneratorError):
    """Exception raised for storage-related errors."""
    pass

class WorkflowError(NewsletterGeneratorError):
    """Exception raised for workflow-related errors."""
    pass

class ContentError(NewsletterGeneratorError):
    """Exception raised for content-related errors."""
    pass

class ToolError(NewsletterGeneratorError):
    """Exception raised for tool-related errors."""
    pass

class FeedbackError(NewsletterGeneratorError):
    """Exception raised for feedback-related errors."""
    pass

class PerformanceError(NewsletterGeneratorError):
    """Exception raised for performance-related errors."""
    pass 