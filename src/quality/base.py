"""
Simplified Quality System for Newsletter Generation

This module provides a basic quality validation interface that integrates
with the EditorAgent's built-in quality assessment.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QualityStatus(Enum):
    """Simple quality validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class QualityMetrics:
    """Quality metrics for content evaluation."""
    readability_score: float = 0.0
    technical_accuracy: float = 0.0
    engagement_score: float = 0.0
    completeness_score: float = 0.0
    overall_score: float = 0.0
    word_count: int = 0
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class QualityResult:
    """Result of quality evaluation."""
    status: QualityStatus
    metrics: QualityMetrics
    passes_gate: bool = False
    gate_id: str = "basic_quality"
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class QualityAssuranceSystem:
    """Quality assurance system for newsletter content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validators = []
    
    def add_validator(self, validator):
        """Add a validator to the system."""
        self.validators.append(validator)
    
    def evaluate_content(self, content: str, gate_id: str = "basic_quality") -> QualityResult:
        """Evaluate content quality using all registered validators."""
        try:
            from ..agents.editing import EditorAgent
            editor = EditorAgent()
            
            # Use EditorAgent's built-in quality validation
            result = editor.validate_content_quality(content)
            
            # Create metrics
            metrics = QualityMetrics(
                readability_score=result.get('readability_score', 0.0),
                technical_accuracy=result.get('technical_accuracy', 0.0),
                engagement_score=result.get('engagement_score', 0.0),
                completeness_score=result.get('completeness_score', 0.0),
                overall_score=result.get('quality_score', 0.0),
                word_count=result.get('word_count', 0),
                issues=result.get('issues', [])
            )
            
            # Determine status
            passes_gate = result.get('passes_quality_gate', False)
            status = QualityStatus.PASSED if passes_gate else QualityStatus.FAILED
            
            return QualityResult(
                status=status,
                metrics=metrics,
                passes_gate=passes_gate,
                gate_id=gate_id,
                recommendations=result.get('recommendations', [])
            )
        
        except Exception as e:
            self.logger.error(f"Error in quality evaluation: {e}")
            metrics = QualityMetrics(
                overall_score=0.0,
                issues=[f"Quality evaluation error: {e}"]
            )
            return QualityResult(
                status=QualityStatus.FAILED,
                metrics=metrics,
                passes_gate=False,
                gate_id=gate_id,
                recommendations=[f"Error in quality evaluation: {e}"]
            )


class SimpleQualityValidator:
    """Simple quality validator that delegates to EditorAgent methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_content(self, content: str, gate_id: str = "basic_quality") -> Dict[str, Any]:
        """Simple content evaluation using basic quality checks."""
        try:
            from ..agents.editing import EditorAgent
            editor = EditorAgent()
            
            # Use EditorAgent's built-in quality validation
            result = editor.validate_content_quality(content)
            
            # Convert to simple format
            return {
                'status': 'passed' if result.get('passes_quality_gate', False) else 'failed',
                'score': result.get('quality_score', 0.0),
                'passes_gate': result.get('passes_quality_gate', False),
                'issues': result.get('recommendations', []),
                'gate_id': gate_id
            }
        
        except Exception as e:
            self.logger.error(f"Error in quality evaluation: {e}")
            return {
                'status': 'failed',
                'score': 0.0,
                'passes_gate': False,
                'issues': [f"Quality evaluation error: {e}"],
                'gate_id': gate_id
            }


# Create singleton instance for compatibility
_simple_validator = SimpleQualityValidator()


def evaluate_content(content: str, gate_id: str = "basic_quality") -> Dict[str, Any]:
    """Simple function interface for content evaluation."""
    return _simple_validator.evaluate_content(content, gate_id)