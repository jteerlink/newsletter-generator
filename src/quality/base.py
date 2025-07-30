"""
Unified Quality Validator Interface

This module provides the abstract base classes and interfaces for the unified
quality validation system, consolidating all quality-related functionality.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class QualityStatus(Enum):
    """Quality validation status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"
    PENDING = "pending"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for content validation."""
    overall_score: float
    technical_accuracy_score: float
    content_quality_score: float
    readability_score: float
    engagement_score: float
    structure_score: float
    code_quality_score: float
    mobile_readability_score: float
    source_credibility_score: float
    content_balance_score: float
    performance_score: float
    validation_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Calculate overall score if not provided."""
        if self.overall_score == 0:
            self.overall_score = self._calculate_weighted_score()
    
    def _calculate_weighted_score(self) -> float:
        """Calculate weighted overall score from individual metrics."""
        weights = {
            'technical_accuracy_score': 0.25,
            'content_quality_score': 0.20,
            'readability_score': 0.15,
            'engagement_score': 0.10,
            'structure_score': 0.10,
            'code_quality_score': 0.10,
            'mobile_readability_score': 0.05,
            'source_credibility_score': 0.03,
            'content_balance_score': 0.02
        }
        
        weighted_sum = sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )
        
        return round(weighted_sum, 2)


@dataclass
class QualityReport:
    """Comprehensive quality validation report."""
    status: QualityStatus
    metrics: QualityMetrics
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    def is_publishable(self, minimum_score: float = 7.0) -> bool:
        """Check if content meets minimum quality standards for publishing."""
        return (
            self.status in [QualityStatus.PASSED, QualityStatus.WARNING] and
            self.metrics.overall_score >= minimum_score and
            len(self.blocking_issues) == 0
        )
    
    def get_grade(self) -> str:
        """Get letter grade based on overall score."""
        score = self.metrics.overall_score
        if score >= 9.0:
            return "A+"
        elif score >= 8.5:
            return "A"
        elif score >= 8.0:
            return "A-"
        elif score >= 7.5:
            return "B+"
        elif score >= 7.0:
            return "B"
        elif score >= 6.5:
            return "B-"
        elif score >= 6.0:
            return "C+"
        elif score >= 5.5:
            return "C"
        elif score >= 5.0:
            return "C-"
        else:
            return "F"


class QualityValidator(ABC):
    """Abstract base class for quality validators."""
    
    def __init__(self, name: str = "QualityValidator"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def validate(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityReport:
        """Validate content and return quality report."""
        pass
    
    @abstractmethod
    def get_metrics(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityMetrics:
        """Extract quality metrics from content."""
        pass
    
    def validate_batch(self, contents: List[Union[str, Dict[str, Any]]], **kwargs) -> List[QualityReport]:
        """Validate multiple content items."""
        reports = []
        for content in contents:
            try:
                report = self.validate(content, **kwargs)
                reports.append(report)
            except Exception as e:
                self.logger.error(f"Error validating content: {e}")
                # Create failed report
                failed_metrics = QualityMetrics(
                    overall_score=0.0,
                    technical_accuracy_score=0.0,
                    content_quality_score=0.0,
                    readability_score=0.0,
                    engagement_score=0.0,
                    structure_score=0.0,
                    code_quality_score=0.0,
                    mobile_readability_score=0.0,
                    source_credibility_score=0.0,
                    content_balance_score=0.0,
                    performance_score=0.0
                )
                failed_report = QualityReport(
                    status=QualityStatus.FAILED,
                    metrics=failed_metrics,
                    issues=[f"Validation error: {str(e)}"]
                )
                reports.append(failed_report)
        
        return reports


class UnifiedQualitySystem:
    """Unified quality system that coordinates multiple validators."""
    
    def __init__(self):
        self.validators: Dict[str, QualityValidator] = {}
        self.logger = logging.getLogger(__name__)
        self.default_thresholds = {
            "minimum_overall_score": 7.0,
            "minimum_technical_accuracy": 6.0,
            "minimum_content_quality": 6.0,
            "minimum_readability": 6.0,
            "maximum_issues": 5,
            "maximum_blocking_issues": 0
        }
    
    def register_validator(self, name: str, validator: QualityValidator):
        """Register a quality validator."""
        self.validators[name] = validator
        self.logger.info(f"Registered quality validator: {name}")
    
    def validate_content(self, content: Union[str, Dict[str, Any]], 
                        validators: List[str] = None, **kwargs) -> QualityReport:
        """Validate content using all registered validators or specified ones."""
        start_time = time.time()
        
        if validators is None:
            validators = list(self.validators.keys())
        
        # Run validators
        validator_reports = {}
        for validator_name in validators:
            if validator_name in self.validators:
                try:
                    report = self.validators[validator_name].validate(content, **kwargs)
                    validator_reports[validator_name] = report
                except Exception as e:
                    self.logger.error(f"Error in validator {validator_name}: {e}")
                    # Create failed report for this validator
                    failed_metrics = QualityMetrics(
                        overall_score=0.0,
                        technical_accuracy_score=0.0,
                        content_quality_score=0.0,
                        readability_score=0.0,
                        engagement_score=0.0,
                        structure_score=0.0,
                        code_quality_score=0.0,
                        mobile_readability_score=0.0,
                        source_credibility_score=0.0,
                        content_balance_score=0.0,
                        performance_score=0.0
                    )
                    validator_reports[validator_name] = QualityReport(
                        status=QualityStatus.FAILED,
                        metrics=failed_metrics,
                        issues=[f"Validator error: {str(e)}"]
                    )
        
        # Combine results
        combined_report = self._combine_reports(validator_reports, content)
        combined_report.metrics.processing_time_seconds = time.time() - start_time
        
        return combined_report
    
    def _combine_reports(self, validator_reports: Dict[str, QualityReport], 
                        content: Union[str, Dict[str, Any]]) -> QualityReport:
        """Combine multiple validator reports into a unified report."""
        if not validator_reports:
            return self._create_empty_report()
        
        # Aggregate metrics
        all_metrics = [report.metrics for report in validator_reports.values()]
        combined_metrics = self._aggregate_metrics(all_metrics)
        
        # Aggregate issues and recommendations
        all_issues = []
        all_warnings = []
        all_recommendations = []
        all_blocking_issues = []
        all_strengths = []
        
        for report in validator_reports.values():
            all_issues.extend(report.issues)
            all_warnings.extend(report.warnings)
            all_recommendations.extend(report.recommendations)
            all_blocking_issues.extend(report.blocking_issues)
            all_strengths.extend(report.strengths)
        
        # Determine overall status
        status = self._determine_overall_status(validator_reports, combined_metrics)
        
        # Create combined report
        combined_report = QualityReport(
            status=status,
            metrics=combined_metrics,
            issues=list(set(all_issues)),  # Remove duplicates
            warnings=list(set(all_warnings)),
            recommendations=list(set(all_recommendations)),
            blocking_issues=list(set(all_blocking_issues)),
            strengths=list(set(all_strengths)),
            detailed_analysis={
                'validator_reports': validator_reports,
                'content_length': len(str(content)),
                'validation_count': len(validator_reports)
            }
        )
        
        return combined_report
    
    def _aggregate_metrics(self, metrics_list: List[QualityMetrics]) -> QualityMetrics:
        """Aggregate multiple quality metrics into a single metrics object."""
        if not metrics_list:
            return self._create_empty_metrics()
        
        # Calculate averages for numeric fields
        avg_metrics = {}
        for field in QualityMetrics.__dataclass_fields__:
            if field in ['overall_score', 'technical_accuracy_score', 'content_quality_score',
                        'readability_score', 'engagement_score', 'structure_score',
                        'code_quality_score', 'mobile_readability_score', 'source_credibility_score',
                        'content_balance_score', 'performance_score']:
                values = [getattr(m, field) for m in metrics_list if getattr(m, field) > 0]
                avg_metrics[field] = sum(values) / len(values) if values else 0.0
        
        # Use the most recent timestamp
        latest_timestamp = max(m.validation_timestamp for m in metrics_list)
        
        return QualityMetrics(
            overall_score=avg_metrics.get('overall_score', 0.0),
            technical_accuracy_score=avg_metrics.get('technical_accuracy_score', 0.0),
            content_quality_score=avg_metrics.get('content_quality_score', 0.0),
            readability_score=avg_metrics.get('readability_score', 0.0),
            engagement_score=avg_metrics.get('engagement_score', 0.0),
            structure_score=avg_metrics.get('structure_score', 0.0),
            code_quality_score=avg_metrics.get('code_quality_score', 0.0),
            mobile_readability_score=avg_metrics.get('mobile_readability_score', 0.0),
            source_credibility_score=avg_metrics.get('source_credibility_score', 0.0),
            content_balance_score=avg_metrics.get('content_balance_score', 0.0),
            performance_score=avg_metrics.get('performance_score', 0.0),
            validation_timestamp=latest_timestamp
        )
    
    def _determine_overall_status(self, validator_reports: Dict[str, QualityReport], 
                                 combined_metrics: QualityMetrics) -> QualityStatus:
        """Determine overall quality status based on validator results and metrics."""
        # Check for any failed validators
        if any(report.status == QualityStatus.FAILED for report in validator_reports.values()):
            return QualityStatus.FAILED
        
        # Check for blocking issues
        all_blocking_issues = []
        for report in validator_reports.values():
            all_blocking_issues.extend(report.blocking_issues)
        
        if all_blocking_issues:
            return QualityStatus.FAILED
        
        # Check score thresholds
        if combined_metrics.overall_score < self.default_thresholds["minimum_overall_score"]:
            return QualityStatus.NEEDS_REVIEW
        
        # Check for warnings
        all_warnings = []
        for report in validator_reports.values():
            all_warnings.extend(report.warnings)
        
        if all_warnings:
            return QualityStatus.WARNING
        
        return QualityStatus.PASSED
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty quality metrics."""
        return QualityMetrics(
            overall_score=0.0,
            technical_accuracy_score=0.0,
            content_quality_score=0.0,
            readability_score=0.0,
            engagement_score=0.0,
            structure_score=0.0,
            code_quality_score=0.0,
            mobile_readability_score=0.0,
            source_credibility_score=0.0,
            content_balance_score=0.0,
            performance_score=0.0
        )
    
    def _create_empty_report(self) -> QualityReport:
        """Create empty quality report."""
        return QualityReport(
            status=QualityStatus.FAILED,
            metrics=self._create_empty_metrics(),
            issues=["No validators available"]
        )