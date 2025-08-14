"""Shared constants for the newsletter generator."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_PATH = PROJECT_ROOT / "logs"
TESTS_PATH = PROJECT_ROOT / "tests"

# Ensure directories exist
for path in [DATA_PATH, LOGS_PATH]:
    path.mkdir(exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# NVIDIA Cloud Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1")

# LLM Settings (applies to all providers)
# CRITICAL FIX: Increased timeout for comprehensive newsletter generation
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))  # Increased from 30s to 3 minutes
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))  # Increased for longer content

# Generation Timeout Configuration
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "600"))  # 10 minutes for full generation
CHECKPOINT_TIMEOUT = int(os.getenv("CHECKPOINT_TIMEOUT", "180"))  # 3 minutes per section
SECTION_GENERATION_TIMEOUT = int(os.getenv("SECTION_GENERATION_TIMEOUT", "120"))  # 2 minutes per section

# Backward compatibility
DEFAULT_LLM_MODEL = OLLAMA_MODEL

# Search Configuration
DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "10"))
SEARCH_MAX_RETRIES = int(os.getenv("SEARCH_MAX_RETRIES", "3"))

# Scraping Configuration
SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "30"))
SCRAPER_MAX_RETRIES = int(os.getenv("SCRAPER_MAX_RETRIES", "3"))
SCRAPER_HEADLESS = os.getenv("SCRAPER_HEADLESS", "true").lower() == "true"

# Quality Configuration
MINIMUM_QUALITY_SCORE = float(os.getenv("MINIMUM_QUALITY_SCORE", "7.0"))
QUALITY_THRESHOLDS = {
    "technical_accuracy": 6.0,
    "readability": 7.0,
    "engagement": 6.5,
    "completeness": 7.0
}

# Newsletter Configuration
DEFAULT_NEWSLETTER_LENGTH = int(os.getenv("DEFAULT_NEWSLETTER_LENGTH", "1500"))
MAX_NEWSLETTER_LENGTH = int(os.getenv("MAX_NEWSLETTER_LENGTH", "3000"))

# Agent Configuration
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "60"))
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Error Messages
ERROR_MESSAGES = {
    "llm_timeout": "LLM request timed out",
    "search_failed": "Search operation failed",
    "scraping_failed": "Web scraping failed",
    "validation_failed": "Content validation failed",
    "agent_failed": "Agent execution failed"
}

# Success Messages
SUCCESS_MESSAGES = {
    "newsletter_generated": "Newsletter generated successfully",
    "content_validated": "Content validation passed",
    "agent_completed": "Agent task completed successfully"
}

# File Extensions
SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".md", ".rst"],
    "code": [".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"],
    "data": [".csv", ".tsv", ".xlsx", ".xls"],
    "documents": [".pdf", ".doc", ".docx"]
}

# Content Types
CONTENT_TYPES = {
    "newsletter": "newsletter",
    "article": "article",
    "summary": "summary",
    "analysis": "analysis",
    "tutorial": "tutorial"
}

# Agent Types
AGENT_TYPES = {
    "research": "research",
    "writer": "writer",
    "editor": "editor",
    "manager": "manager",
    "planner": "planner"
}

# Workflow Types
WORKFLOW_TYPES = {
    "sequential": "sequential",
    "parallel": "parallel",
    "hierarchical": "hierarchical"
}

# Database Configuration
DB_CONFIG = {
    "vector_collection": "newsletter_content",
    "max_results": 100,
    "similarity_threshold": 0.7
}

# API Configuration
API_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "rate_limit": 100  # requests per minute
}

# Tool Usage Enforcement (Phase 5 - PRD: Mandatory Tool Integration)
TOOL_ENFORCEMENT_ENABLED = os.getenv("TOOL_ENFORCEMENT_ENABLED", "false").lower() == "true"

# Minimum thresholds for quality gates (non-blocking in Phase 1)
MIN_VECTOR_QUERIES = int(os.getenv("MIN_VECTOR_QUERIES", "0"))
MIN_WEB_SEARCHES = int(os.getenv("MIN_WEB_SEARCHES", "0"))

# Default mandatory tools per agent type (can be overridden per-agent)
MANDATORY_TOOLS = {
    "research": ["vector_search", "web_search"],
    "writer": ["vector_search"],
    "editor": [],
    "manager": []
}

# Defaults for tool integration behavior
MANDATORY_VECTOR_TOP_K = int(os.getenv("MANDATORY_VECTOR_TOP_K", "5"))
MANDATORY_WEB_MAX_RESULTS = int(os.getenv("MANDATORY_WEB_MAX_RESULTS", "3"))
