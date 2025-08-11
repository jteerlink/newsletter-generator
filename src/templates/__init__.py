"""
Templates module for newsletter generation

This module provides code templates and template management for AI/ML newsletter content.
"""

from .code_templates import (
    CodeTemplate,
    CodeTemplateLibrary,
    TemplateCategory,
    Framework,
    ComplexityLevel,
    template_library,
    get_template,
    search_templates,
    list_frameworks,
    list_categories
)

__all__ = [
    'CodeTemplate',
    'CodeTemplateLibrary', 
    'TemplateCategory',
    'Framework',
    'ComplexityLevel',
    'template_library',
    'get_template',
    'search_templates',
    'list_frameworks',
    'list_categories'
]
