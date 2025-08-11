"""
Quality assurance system for newsletter generation.

This module provides quality validation and improvement tools
for newsletter content generation.
"""

from .base import QualityAssuranceSystem, QualityMetrics, QualityResult

# Export main classes
__all__ = [
    'QualityAssuranceSystem',
    'QualityResult',
    'QualityMetrics'
]
