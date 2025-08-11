"""
Content validation module for newsletter generation.

This module provides content validation functionality including quality assessment,
repetition analysis, and fact checking capabilities.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation."""
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    repetition_analysis: Dict[str, Any]
    expert_quote_analysis: Dict[str, Any]
    fact_check_analysis: Dict[str, Any]


class ContentValidator:
    """Content validator for newsletter quality assessment."""

    def __init__(self):
        self.min_length = 100
        self.max_length = 10000
        self.quality_thresholds = {
            'min_length': 100,
            'max_repetition': 0.3,
            'min_readability': 0.6
        }

    def validate_content(self, content: str) -> Dict[str, Any]:
        """
        Validate newsletter content and return quality metrics.

        Args:
            content: The content to validate

        Returns:
            Dictionary with validation results
        """
        if not content:
            return self._create_empty_result()

        # Basic validation
        length_score = self._validate_length(content)
        repetition_score = self._analyze_repetition(content)
        readability_score = self._analyze_readability(content)

        # Calculate overall quality score
        quality_score = (length_score + repetition_score +
                         readability_score) / 3

        # Identify issues
        issues = self._identify_issues(
            content,
            length_score,
            repetition_score,
            readability_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, quality_score)

        return {
            'quality_score': quality_score,
            'issues': issues,
            'recommendations': recommendations,
            'repetition_analysis': self._get_repetition_analysis(content),
            'expert_quote_analysis': self._get_expert_quote_analysis(content),
            'fact_check_analysis': self._get_fact_check_analysis(content)
        }

    def _validate_length(self, content: str) -> float:
        """Validate content length."""
        length = len(content)
        if length < self.quality_thresholds['min_length']:
            return 0.1
        elif length > self.max_length:
            return 0.5
        else:
            # Normalize to 0-1 scale
            return min(1.0, length / 1000)

    def _analyze_repetition(self, content: str) -> float:
        """Analyze content for repetition."""
        words = content.lower().split()
        if not words:
            return 0.0

        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return 1.0

        # Calculate repetition ratio
        total_words = len(words)
        unique_words = len(word_counts)
        repetition_ratio = 1 - (unique_words / total_words)

        # Convert to score (lower repetition = higher score)
        return max(0.0, 1.0 - repetition_ratio)

    def _analyze_readability(self, content: str) -> float:
        """Analyze content readability."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Calculate average sentence length
        avg_sentence_length = sum(len(s.split())
                                  for s in sentences) / len(sentences)

        # Simple readability score (shorter sentences = better readability)
        if avg_sentence_length <= 15:
            return 1.0
        elif avg_sentence_length <= 25:
            return 0.8
        elif avg_sentence_length <= 35:
            return 0.6
        else:
            return 0.3

    def _identify_issues(
            self,
            content: str,
            length_score: float,
            repetition_score: float,
            readability_score: float) -> List[str]:
        """Identify content issues."""
        issues = []

        if length_score < 0.5:
            issues.append("Content is too short")

        if repetition_score < 0.7:
            issues.append("Content has too much repetition")

        if readability_score < 0.6:
            issues.append("Content readability could be improved")

        if not content.strip():
            issues.append("Content is empty")

        return issues

    def _generate_recommendations(
            self,
            issues: List[str],
            quality_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if quality_score < 0.5:
            recommendations.append(
                "Consider expanding the content with more details")

        if "Content is too short" in issues:
            recommendations.append(
                "Add more content to reach minimum length requirements")

        if "Content has too much repetition" in issues:
            recommendations.append("Use synonyms and vary sentence structure")

        if "Content readability could be improved" in issues:
            recommendations.append(
                "Use shorter sentences and simpler language")

        if not recommendations:
            recommendations.append("Content quality is good")

        return recommendations

    def _get_repetition_analysis(self, content: str) -> Dict[str, Any]:
        """Get detailed repetition analysis."""
        words = content.lower().split()
        word_counts = {}

        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get most repeated words
        repeated_words = [(word, count)
                          for word, count in word_counts.items() if count > 2]
        repeated_words.sort(key=lambda x: x[1], reverse=True)

        return {
            'total_words': len(words),
            'unique_words': len(word_counts),
            'repetition_ratio': 1 - (len(word_counts) / len(words)) if words else 0,
            'most_repeated': repeated_words[:5]
        }

    def _get_expert_quote_analysis(self, content: str) -> Dict[str, Any]:
        """Analyze content for expert quotes."""
        # Simple quote detection
        quotes = re.findall(r'"([^"]*)"', content)

        return {
            'quote_count': len(quotes),
            'quotes': quotes,
            'has_expert_quotes': len(quotes) > 0
        }

    def _get_fact_check_analysis(self, content: str) -> Dict[str, Any]:
        """Analyze content for fact-checking needs."""
        # Simple fact-checking indicators
        fact_indicators = [
            'according to', 'study shows', 'research indicates',
            'statistics', 'percentage', 'million', 'billion'
        ]

        found_indicators = []
        for indicator in fact_indicators:
            if indicator.lower() in content.lower():
                found_indicators.append(indicator)

        return {
            'fact_indicators': found_indicators,
            'needs_fact_checking': len(found_indicators) > 0,
            'fact_check_score': len(found_indicators) / len(fact_indicators)
        }

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty content."""
        return {
            'quality_score': 0.0,
            'issues': ['Content is empty'],
            'recommendations': ['Add content to the newsletter'],
            'repetition_analysis': {
                'total_words': 0,
                'unique_words': 0,
                'repetition_ratio': 0,
                'most_repeated': []},
            'expert_quote_analysis': {
                'quote_count': 0,
                'quotes': [],
                'has_expert_quotes': False},
            'fact_check_analysis': {
                'fact_indicators': [],
                'needs_fact_checking': False,
                'fact_check_score': 0.0}}
