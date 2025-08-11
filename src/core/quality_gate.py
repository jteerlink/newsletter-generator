"""
Quality gate module for newsletter generation.

This module provides quality gate functionality to ensure newsletter content
meets quality standards before publication.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""
    status: QualityGateStatus
    overall_score: float
    grade: str
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class NewsletterQualityGate:
    """Quality gate for newsletter content evaluation."""

    def __init__(self):
        self.minimum_score = 7.0
        self.warning_threshold = 8.0
        self.excellent_threshold = 9.0

        # Quality criteria weights
        self.criteria_weights = {
            'length': 0.2,
            'readability': 0.3,
            'structure': 0.2,
            'engagement': 0.2,
            'accuracy': 0.1
        }

    def evaluate_content(self, content: str) -> QualityGateResult:
        """
        Evaluate newsletter content against quality criteria.

        Args:
            content: The content to evaluate

        Returns:
            QualityGateResult with evaluation details
        """
        if not content or not content.strip():
            return self._create_failed_result("Content is empty")

        # Evaluate different aspects
        length_score = self._evaluate_length(content)
        readability_score = self._evaluate_readability(content)
        structure_score = self._evaluate_structure(content)
        engagement_score = self._evaluate_engagement(content)
        accuracy_score = self._evaluate_accuracy(content)

        # Calculate weighted overall score
        overall_score = (
            length_score * self.criteria_weights['length'] +
            readability_score * self.criteria_weights['readability'] +
            structure_score * self.criteria_weights['structure'] +
            engagement_score * self.criteria_weights['engagement'] +
            accuracy_score * self.criteria_weights['accuracy']
        )

        # Determine status and grade
        status, grade = self._determine_status_and_grade(overall_score)

        # Collect issues and warnings
        blocking_issues = self._collect_blocking_issues(
            content, length_score, readability_score, structure_score,
            engagement_score, accuracy_score
        )

        warnings = self._collect_warnings(
            content, length_score, readability_score, structure_score,
            engagement_score, accuracy_score
        )

        recommendations = self._generate_recommendations(
            blocking_issues, warnings, overall_score
        )

        return QualityGateResult(
            status=status,
            overall_score=overall_score,
            grade=grade,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations
        )

    def _evaluate_length(self, content: str) -> float:
        """Evaluate content length."""
        length = len(content)

        if length < 200:
            return 2.0  # Very poor
        elif length < 500:
            return 5.0  # Poor
        elif length < 1000:
            return 7.0  # Good
        elif length < 2000:
            return 9.0  # Very good
        else:
            return 10.0  # Excellent

    def _evaluate_readability(self, content: str) -> float:
        """Evaluate content readability."""
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 2.0

        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split())
                                  for s in sentences) / len(sentences)

        # Evaluate based on sentence length
        if avg_sentence_length <= 15:
            return 10.0
        elif avg_sentence_length <= 20:
            return 8.0
        elif avg_sentence_length <= 25:
            return 6.0
        elif avg_sentence_length <= 30:
            return 4.0
        else:
            return 2.0

    def _evaluate_structure(self, content: str) -> float:
        """Evaluate content structure."""
        # Check for headers
        headers = content.count('#')
        paragraphs = content.count('\n\n')

        # Check for list items
        list_items = content.count('- ') + content.count('* ')

        # Calculate structure score
        structure_score = 5.0  # Base score

        if headers > 0:
            structure_score += 2.0

        if paragraphs > 2:
            structure_score += 1.5

        if list_items > 0:
            structure_score += 1.5

        return min(10.0, structure_score)

    def _evaluate_engagement(self, content: str) -> float:
        """Evaluate content engagement."""
        # Check for engaging elements
        engagement_score = 5.0  # Base score

        # Check for questions
        if '?' in content:
            engagement_score += 1.0

        # Check for exclamations
        if '!' in content:
            engagement_score += 0.5

        # Check for quotes
        if '"' in content:
            engagement_score += 1.0

        # Check for links or references
        if 'http' in content or 'www' in content:
            engagement_score += 1.0

        # Check for numbers/statistics
        import re
        numbers = re.findall(r'\d+', content)
        if len(numbers) > 2:
            engagement_score += 1.0

        return min(10.0, engagement_score)

    def _evaluate_accuracy(self, content: str) -> float:
        """Evaluate content accuracy indicators."""
        # This is a simplified accuracy evaluation
        # In a real system, this would involve fact-checking

        accuracy_score = 7.0  # Base score

        # Check for citation indicators
        citation_indicators = [
            'according to', 'study shows', 'research indicates',
            'statistics', 'percentage', 'million', 'billion'
        ]

        found_indicators = 0
        for indicator in citation_indicators:
            if indicator.lower() in content.lower():
                found_indicators += 1

        # Adjust score based on indicators
        if found_indicators > 0:
            accuracy_score += min(3.0, found_indicators)

        return min(10.0, accuracy_score)

    def _determine_status_and_grade(
            self, overall_score: float) -> tuple[QualityGateStatus, str]:
        """Determine status and grade based on overall score."""
        if overall_score >= self.excellent_threshold:
            return QualityGateStatus.PASSED, "A"
        elif overall_score >= self.warning_threshold:
            return QualityGateStatus.PASSED, "B"
        elif overall_score >= self.minimum_score:
            return QualityGateStatus.WARNING, "C"
        else:
            return QualityGateStatus.FAILED, "F"

    def _collect_blocking_issues(
            self,
            content: str,
            length_score: float,
            readability_score: float,
            structure_score: float,
            engagement_score: float,
            accuracy_score: float) -> List[str]:
        """Collect blocking issues that prevent publication."""
        issues = []

        if length_score < 3.0:
            issues.append("Content is too short")

        if readability_score < 3.0:
            issues.append("Content is not readable")

        if structure_score < 3.0:
            issues.append("Content lacks proper structure")

        if not content.strip():
            issues.append("Content is empty")

        return issues

    def _collect_warnings(
            self,
            content: str,
            length_score: float,
            readability_score: float,
            structure_score: float,
            engagement_score: float,
            accuracy_score: float) -> List[str]:
        """Collect warnings for content improvement."""
        warnings = []

        if length_score < 6.0:
            warnings.append("Content could be longer")

        if readability_score < 6.0:
            warnings.append("Content readability could be improved")

        if structure_score < 6.0:
            warnings.append("Content structure could be improved")

        if engagement_score < 6.0:
            warnings.append("Content could be more engaging")

        if accuracy_score < 6.0:
            warnings.append("Content could benefit from more citations")

        return warnings

    def _generate_recommendations(
            self,
            blocking_issues: List[str],
            warnings: List[str],
            overall_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if "Content is too short" in blocking_issues:
            recommendations.append(
                "Expand content with more details and examples")

        if "Content is not readable" in blocking_issues:
            recommendations.append(
                "Use shorter sentences and simpler language")

        if "Content lacks proper structure" in blocking_issues:
            recommendations.append(
                "Add headers, paragraphs, and bullet points")

        if "Content could be longer" in warnings:
            recommendations.append("Add more content to reach optimal length")

        if "Content readability could be improved" in warnings:
            recommendations.append(
                "Break up long sentences and use active voice")

        if "Content structure could be improved" in warnings:
            recommendations.append(
                "Organize content with clear sections and subsections")

        if "Content could be more engaging" in warnings:
            recommendations.append(
                "Add questions, examples, and interactive elements")

        if "Content could benefit from more citations" in warnings:
            recommendations.append("Include more references and citations")

        if not recommendations and overall_score >= 8.0:
            recommendations.append("Content quality is excellent")

        return recommendations

    def _create_failed_result(self, reason: str) -> QualityGateResult:
        """Create a failed quality gate result."""
        return QualityGateResult(
            status=QualityGateStatus.FAILED,
            overall_score=0.0,
            grade="F",
            blocking_issues=[reason],
            warnings=[],
            recommendations=["Fix the blocking issue to proceed"]
        )
