"""
Unified Quality System for Newsletter Generation

This module provides a consolidated quality validation system that combines
content validation, technical validation, and quality monitoring capabilities.
"""

from .base import QualityValidator, QualityMetrics, QualityStatus, QualityReport
from .content_validator import ContentQualityValidator
from .technical_validator import TechnicalQualityValidator
from .quality_monitor import QualityMonitor

# Legacy compatibility layer
from .compatibility import (
    ContentValidator,
    NewsletterQualityGate,
    QualityGateStatus,
    QualityGateResult,
    QualityAssuranceSystem
)

__all__ = [
    'QualityValidator',
    'QualityMetrics', 
    'QualityStatus',
    'QualityReport',
    'ContentQualityValidator',
    'TechnicalQualityValidator',
    'QualityMonitor',
    # Legacy compatibility
    'ContentValidator',
    'NewsletterQualityGate',
    'QualityGateStatus',
    'QualityGateResult',
    'QualityAssuranceSystem'
]