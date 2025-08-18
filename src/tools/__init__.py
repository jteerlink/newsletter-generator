"""Collections of tool functions and classes exposed to CrewAI agents."""

from .enhanced_search import EnhancedSearchTool, SearchQuery, SearchResult
from .grammar_linter import GrammarAndStyleLinter, GrammarIssue, LinterResult, StyleIssue
from .tools import *

__all__ = [
    # Core tools
    'search_web',
    'search_knowledge_base',
    'extract_content',
    'analyze_content',
    'format_content',
    'validate_url',
    'generate_summary',
    'extract_keywords',
    'detect_language',
    'clean_text',

    # Enhanced tools (Phase 3)
    'GrammarAndStyleLinter',
    'LinterResult',
    'GrammarIssue',
    'StyleIssue',
    'EnhancedSearchTool',
    'SearchResult',
    'SearchQuery'
]
