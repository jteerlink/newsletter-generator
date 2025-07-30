"""
Compatibility Layer for Legacy Quality Systems

This module provides backward compatibility for the old quality system interfaces
while using the new unified quality system under the hood.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .content_validator import ContentQualityValidator
from .technical_validator import TechnicalQualityValidator
from .base import QualityStatus, QualityMetrics


class ContentValidator:
    """Legacy ContentValidator interface using the new unified system."""
    
    def __init__(self):
        self.content_validator = ContentQualityValidator()
        self.technical_validator = TechnicalQualityValidator()
    
    def assess_content_quality(self, content: str) -> Dict[str, Any]:
        """Assess content quality using the new unified system."""
        report = self.content_validator.validate(content)
        return {
            'overall_score': report.metrics.overall_score,
            'content_quality_score': report.metrics.content_quality_score,
            'readability_score': report.metrics.readability_score,
            'engagement_score': report.metrics.engagement_score,
            'structure_score': report.metrics.structure_score,
            'issues': report.issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations
        }
    
    def detect_repetition(self, content: str) -> Dict[str, Any]:
        """Detect repetition using the new unified system."""
        report = self.content_validator.validate(content)
        return {
            'repetition_score': 1.0 - report.metrics.content_quality_score,
            'repetitive_concepts': [issue for issue in report.issues if 'repetition' in issue.lower()],
            'recommendations': [rec for rec in report.recommendations if 'repetition' in rec.lower()]
        }
    
    def analyze_expert_quotes(self, content: str) -> Dict[str, Any]:
        """Analyze expert quotes using the new unified system."""
        report = self.content_validator.validate(content)
        return {
            'expert_quotes_count': len([issue for issue in report.issues if 'quote' in issue.lower()]),
            'suspicious_quotes': [issue for issue in report.issues if 'suspicious' in issue.lower()],
            'quote_analysis': report.detailed_analysis.get('expert_quotes', {})
        }
    
    def analyze_factual_claims(self, content: str) -> Dict[str, Any]:
        """Analyze factual claims using the new unified system."""
        report = self.content_validator.validate(content)
        return {
            'factual_claims_count': len([issue for issue in report.issues if 'factual' in issue.lower()]),
            'verification_needed': [issue for issue in report.issues if 'verification' in issue.lower()],
            'factual_analysis': report.detailed_analysis.get('factual_claims', {})
        }
    
    def validate(self, content: str) -> Dict[str, Any]:
        """Validate content using the new unified system."""
        report = self.content_validator.validate(content)
        return {
            'is_valid': report.status == QualityStatus.PASSED,
            'score': report.metrics.overall_score,
            'issues': report.issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations
        }
    
    def create_quality_gate(self, minimum_score: float = 7.0) -> Dict[str, Any]:
        """Create a quality gate using the new unified system."""
        return {
            'minimum_overall_score': minimum_score,
            'required_metrics': {
                'content_quality_score': minimum_score - 1.0,
                'readability_score': minimum_score - 1.0,
                'structure_score': minimum_score - 2.0
            },
            'gate_type': 'unified_quality_gate'
        }
    
    def evaluate_quality_gate(self, content: str, quality_gate: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate quality gate using the new unified system."""
        if quality_gate is None:
            quality_gate = self.create_quality_gate()
        
        report = self.content_validator.validate(content)
        assessment = {
            'overall_score': report.metrics.overall_score,
            'content_quality_score': report.metrics.content_quality_score,
            'readability_score': report.metrics.readability_score,
            'structure_score': report.metrics.structure_score
        }
        
        # Check if content passes quality gate
        passes_gate = True
        issues = []
        
        if assessment["overall_score"] < quality_gate["minimum_overall_score"]:
            passes_gate = False
            issues.append(
                f"Overall score ({assessment['overall_score']:.2f}) below minimum ({quality_gate['minimum_overall_score']:.2f})"
            )
        
        for metric, min_score in quality_gate["required_metrics"].items():
            if assessment.get(metric, 0) < min_score:
                passes_gate = False
                issues.append(f"{metric} ({assessment.get(metric, 0):.2f}) below minimum ({min_score:.2f})")
        
        return {
            'passes_gate': passes_gate,
            'assessment': assessment,
            'issues': issues,
            'gate_config': quality_gate
        }


class NewsletterQualityGate:
    """Legacy NewsletterQualityGate interface using the new unified system."""
    
    def __init__(self):
        self.content_validator = ContentQualityValidator()
        self.technical_validator = TechnicalQualityValidator()
    
    def evaluate_content(self, content: str, gate_id: str = "basic_quality") -> Dict[str, Any]:
        """Evaluate content using the new unified system."""
        if gate_id == "basic_quality":
            report = self.content_validator.validate(content)
        elif gate_id == "technical_quality":
            report = self.technical_validator.validate(content)
        else:
            # Default to content validation
            report = self.content_validator.validate(content)
        
        return {
            'status': report.status.value,
            'score': report.metrics.overall_score,
            'passes_gate': report.status == QualityStatus.PASSED,
            'issues': report.issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations,
            'gate_id': gate_id,
            'evaluation_timestamp': datetime.now().isoformat()
        }


class QualityGateStatus:
    """Legacy QualityGateStatus enum compatibility."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


class QualityGateResult:
    """Legacy QualityGateResult compatibility."""
    
    def __init__(self, status: str, score: float, issues: List[str] = None):
        self.status = status
        self.score = score
        self.issues = issues or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'score': self.score,
            'issues': self.issues
        }


class QualityAssuranceSystem:
    """Legacy QualityAssuranceSystem interface using the new unified system."""
    
    def __init__(self):
        self.content_validator = ContentQualityValidator()
        self.technical_validator = TechnicalQualityValidator()
    
    def validate_content(self, content: str) -> Dict[str, Any]:
        """Validate content using the new unified system."""
        content_report = self.content_validator.validate(content)
        technical_report = self.technical_validator.validate(content)
        
        return {
            'content_validation': content_report.to_dict(),
            'technical_validation': technical_report.to_dict(),
            'overall_score': (content_report.metrics.overall_score + technical_report.metrics.overall_score) / 2,
            'passes_quality_gate': content_report.status == QualityStatus.PASSED and technical_report.status == QualityStatus.PASSED
        }
    
    def validate_technical_accuracy(self, content: str) -> Dict[str, Any]:
        """Validate technical accuracy using the new unified system."""
        report = self.technical_validator.validate(content)
        return {
            'accuracy_score': report.metrics.technical_accuracy_score,
            'code_quality_score': report.metrics.code_quality_score,
            'mobile_readability_score': report.metrics.mobile_readability_score,
            'issues': report.issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations
        } 