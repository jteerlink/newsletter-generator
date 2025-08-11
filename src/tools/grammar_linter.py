"""
Grammar and Style Linter Tool

This module provides grammar and style checking capabilities for content validation
in the newsletter generation system. It uses language-tool-python for grammar checking
and custom rules for style validation.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GrammarIssue:
    """Represents a grammar or style issue found in content."""
    message: str
    offset: int
    error_length: int
    rule_id: str
    category: str
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class StyleIssue:
    """Represents a style issue found in content."""
    message: str
    offset: int
    error_length: int
    rule_id: str
    category: str
    severity: str
    suggestion: Optional[str] = None
    context: Optional[str] = None


@dataclass
class LinterResult:
    """Result of grammar and style checking."""
    grammar_issues: List[GrammarIssue]
    style_issues: List[StyleIssue]
    overall_score: float
    grammar_score: float
    style_score: float
    total_issues: int
    suggestions: List[str]
    summary: Dict[str, Any]


class GrammarAndStyleLinter:
    """
    Grammar and style linter for content validation.

    This class provides comprehensive grammar and style checking capabilities
    for newsletter content, including custom style rules and suggestions.
    """

    def __init__(self):
        """Initialize the linter with default rules and configurations."""
        self.grammar_issues = []
        self.style_issues = []
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize grammar and style checking rules."""
        # Grammar rules (simplified - in production would use
        # language-tool-python)
        self.grammar_rules = {
            'spelling': r'\b\w+\b',  # Basic word boundary check
            'punctuation': r'[.!?]+',  # Basic punctuation check
            'capitalization': r'\b[A-Z][a-z]+\b',  # Basic capitalization check
        }

        # Style rules for newsletter content
        self.style_rules = {
            'passive_voice': r'\b(am|is|are|was|were|be|been|being)\s+\w+ed\b',
            'repetitive_words': r'\b(\w+)\s+\1\b',
            'long_sentences': r'[^.!?]{50,}[.!?]',
            'clichés': r'\b(think outside the box|at the end of the day|in this day and age)\b',
            'weak_words': r'\b(very|really|quite|rather|somewhat)\b',
            'jargon': r'\b(synergy|paradigm|leverage|optimize|utilize)\b',
        }

        # Severity mappings
        self.severity_mapping = {
            'spelling': 'HIGH',
            'punctuation': 'MEDIUM',
            'capitalization': 'MEDIUM',
            'passive_voice': 'MEDIUM',
            'repetitive_words': 'LOW',
            'long_sentences': 'MEDIUM',
            'clichés': 'LOW',
            'weak_words': 'LOW',
            'jargon': 'MEDIUM',
        }

    def check_content(self, content: str) -> LinterResult:
        """
        Check grammar and style issues in content.

        Args:
            content: The text content to check

        Returns:
            LinterResult with detailed analysis
        """
        try:
            logger.info("Starting grammar and style check")

            # Reset issues
            self.grammar_issues = []
            self.style_issues = []

            # Check grammar
            self._check_grammar(content)

            # Check style
            self._check_style(content)

            # Calculate scores
            grammar_score = self._calculate_grammar_score(content)
            style_score = self._calculate_style_score(content)
            overall_score = (grammar_score + style_score) / 2

            # Generate suggestions
            suggestions = self._generate_suggestions()

            # Create summary
            summary = self._create_summary(content)

            result = LinterResult(
                grammar_issues=self.grammar_issues,
                style_issues=self.style_issues,
                overall_score=overall_score,
                grammar_score=grammar_score,
                style_score=style_score,
                total_issues=len(self.grammar_issues) + len(self.style_issues),
                suggestions=suggestions,
                summary=summary
            )

            logger.info(
                f"Grammar and style check completed. Score: {
                    overall_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error during grammar and style check: {e}")
            # Return a basic result with error information
            return LinterResult(
                grammar_issues=[],
                style_issues=[],
                overall_score=0.0,
                grammar_score=0.0,
                style_score=0.0,
                total_issues=0,
                suggestions=[f"Error during analysis: {str(e)}"],
                summary={'error': str(e)}
            )

    def _check_grammar(self, content: str):
        """Check for grammar issues in content."""
        # Basic grammar checking (simplified implementation)
        # In production, this would use language-tool-python

        # Check for common grammar issues
        issues = [
            # Double spaces
            (r'\s{2,}', 'DOUBLE_SPACE', 'Remove extra spaces'),
            # Missing periods at end of sentences
            (r'[A-Z][^.!?]*\n', 'MISSING_PERIOD',
             'Add period at end of sentence'),
            # Incorrect apostrophes
            (r'its\s', 'APOSTROPHE_ERROR', 'Check if "its" should be "it\'s"'),
        ]

        for pattern, rule_id, message in issues:
            matches = re.finditer(pattern, content)
            for match in matches:
                issue = GrammarIssue(
                    message=message,
                    offset=match.start(),
                    error_length=match.end() - match.start(),
                    rule_id=rule_id,
                    category='grammar',
                    severity='MEDIUM',
                    suggestion=message,
                    context=content[max(0, match.start() - 20):match.end() + 20]
                )
                self.grammar_issues.append(issue)

    def _check_style(self, content: str):
        """Check for style issues in content."""
        for rule_name, pattern in self.style_rules.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                severity = self.severity_mapping.get(rule_name, 'LOW')
                suggestion = self._get_style_suggestion(
                    rule_name, match.group())

                issue = StyleIssue(
                    message=f"Style issue: {rule_name.replace('_', ' ').title()}",
                    offset=match.start(),
                    error_length=match.end() - match.start(),
                    rule_id=rule_name.upper(),
                    category='style',
                    severity=severity,
                    suggestion=suggestion,
                    context=content[max(0, match.start() - 20):match.end() + 20]
                )
                self.style_issues.append(issue)

    def _get_style_suggestion(self, rule_name: str, matched_text: str) -> str:
        """Get suggestion for style issues."""
        suggestions = {
            'passive_voice': 'Consider using active voice for more engaging content',
            'repetitive_words': 'Avoid repeating the same word in close proximity',
            'long_sentences': 'Break long sentences into shorter, clearer ones',
            'clichés': 'Replace clichés with more original expressions',
            'weak_words': 'Use stronger, more specific words',
            'jargon': 'Replace jargon with clearer, more accessible language',
        }
        return suggestions.get(
            rule_name, 'Review this section for improvement')

    def _calculate_grammar_score(self, content: str) -> float:
        """Calculate grammar score based on issues found."""
        if not content.strip():
            return 0.0

        # Base score
        base_score = 1.0

        # Deduct points for each issue
        deductions = {
            'HIGH': 0.1,
            'MEDIUM': 0.05,
            'LOW': 0.02
        }

        total_deduction = 0
        for issue in self.grammar_issues:
            deduction = deductions.get(issue.severity, 0.02)
            total_deduction += deduction

        score = max(0.0, base_score - total_deduction)
        return round(score, 2)

    def _calculate_style_score(self, content: str) -> float:
        """Calculate style score based on issues found."""
        if not content.strip():
            return 0.0

        # Base score
        base_score = 1.0

        # Deduct points for each issue
        deductions = {
            'HIGH': 0.08,
            'MEDIUM': 0.04,
            'LOW': 0.02
        }

        total_deduction = 0
        for issue in self.style_issues:
            deduction = deductions.get(issue.severity, 0.02)
            total_deduction += deduction

        score = max(0.0, base_score - total_deduction)
        return round(score, 2)

    def _generate_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on issues found."""
        suggestions = []

        # Grammar suggestions
        if self.grammar_issues:
            suggestions.append("Review grammar and punctuation")

        # Style suggestions
        style_categories = set(issue.rule_id for issue in self.style_issues)
        for category in style_categories:
            if category == 'PASSIVE_VOICE':
                suggestions.append(
                    "Use active voice for more engaging content")
            elif category == 'LONG_SENTENCES':
                suggestions.append("Break long sentences into shorter ones")
            elif category == 'JARGON':
                suggestions.append("Replace jargon with clearer language")

        # General suggestions
        if len(self.grammar_issues) + len(self.style_issues) > 10:
            suggestions.append(
                "Consider a comprehensive review of this content")

        return suggestions

    def _create_summary(self, content: str) -> Dict[str, Any]:
        """Create a summary of the linting results."""
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))

        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'average_sentence_length': word_count / max(sentence_count, 1),
            'grammar_issue_count': len(self.grammar_issues),
            'style_issue_count': len(self.style_issues),
            'high_severity_issues': len([i for i in self.grammar_issues + self.style_issues if i.severity == 'HIGH']),
            'medium_severity_issues': len([i for i in self.grammar_issues + self.style_issues if i.severity == 'MEDIUM']),
            'low_severity_issues': len([i for i in self.grammar_issues + self.style_issues if i.severity == 'LOW']),
            'timestamp': datetime.now().isoformat()
        }

    def get_quick_feedback(self, content: str) -> Dict[str, Any]:
        """
        Get quick feedback summary for content.

        Args:
            content: The text content to check

        Returns:
            Dictionary with quick feedback summary
        """
        result = self.check_content(content)

        return {
            'score': result.overall_score,
            'grammar_score': result.grammar_score,
            'style_score': result.style_score,
            'total_issues': result.total_issues,
            'high_priority_issues': len([i for i in result.grammar_issues + result.style_issues if i.severity == 'HIGH']),
            'suggestions': result.suggestions[:3],  # Top 3 suggestions
            'needs_revision': result.overall_score < 0.8
        }
