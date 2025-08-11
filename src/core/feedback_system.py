"""
Structured Feedback System: Enhanced feedback with actionable items.

This module defines the structured feedback system that provides
detailed, actionable feedback for content improvement, including
specific text snippets, issue types, and required actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class IssueType(Enum):
    """Types of issues that can be identified in content."""
    GRAMMAR = "grammar"
    STYLE = "style"
    CLARITY = "clarity"
    ACCURACY = "accuracy"
    TONE = "tone"
    STRUCTURE = "structure"
    ENGAGEMENT = "engagement"
    SEO = "seo"
    BRAND_COMPLIANCE = "brand_compliance"
    ACCESSIBILITY = "accessibility"


class Severity(Enum):
    """Severity levels for feedback items."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RequiredAction(Enum):
    """Required actions for feedback items."""
    REVISION = "REVISION"
    RESEARCH_VERIFICATION = "RESEARCH_VERIFICATION"
    STYLE_ADJUSTMENT = "STYLE_ADJUSTMENT"
    STRUCTURE_REORGANIZATION = "STRUCTURE_REORGANIZATION"
    TONE_ADJUSTMENT = "TONE_ADJUSTMENT"
    NO_ACTION = "NO_ACTION"


@dataclass
class FeedbackItem:
    """Individual feedback item with specific details."""
    text_snippet: str
    issue_type: IssueType
    comment: str
    required_action: RequiredAction
    severity: Severity
    suggestion: Optional[str] = None
    line_number: Optional[int] = None
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert FeedbackItem to dictionary."""
        return {
            'text_snippet': self.text_snippet,
            'issue_type': self.issue_type.value,
            'comment': self.comment,
            'required_action': self.required_action.value,
            'severity': self.severity.value,
            'suggestion': self.suggestion,
            'line_number': self.line_number,
            'confidence_score': self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create FeedbackItem from dictionary."""
        return cls(
            text_snippet=data['text_snippet'],
            issue_type=IssueType(data['issue_type']),
            comment=data['comment'],
            required_action=RequiredAction(data['required_action']),
            severity=Severity(data['severity']),
            suggestion=data.get('suggestion'),
            line_number=data.get('line_number'),
            confidence_score=data.get('confidence_score', 0.0)
        )


@dataclass
class StructuredFeedback:
    """Structured feedback with overall assessment and specific items."""
    overall_score: float
    sub_scores: Dict[str, float]
    feedback_items: List[FeedbackItem]
    required_action: Optional[RequiredAction]
    revision_cycles: int
    summary: str
    improvement_suggestions: List[str]
    quality_metrics: Dict[str, Any]

    def get_high_priority_items(self) -> List[FeedbackItem]:
        """Get feedback items with high severity."""
        return [
            item for item in self.feedback_items if item.severity == Severity.HIGH]

    def get_items_by_type(self, issue_type: IssueType) -> List[FeedbackItem]:
        """Get feedback items by issue type."""
        return [
            item for item in self.feedback_items if item.issue_type == issue_type]

    def get_items_by_action(
            self,
            action: RequiredAction) -> List[FeedbackItem]:
        """Get feedback items by required action."""
        return [
            item for item in self.feedback_items if item.required_action == action]

    def get_average_confidence(self) -> float:
        """Calculate average confidence score across all items."""
        if not self.feedback_items:
            return 0.0
        return sum(item.confidence_score for item in self.feedback_items) / \
            len(self.feedback_items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert StructuredFeedback to dictionary."""
        return {
            'overall_score': self.overall_score,
            'sub_scores': self.sub_scores,
            'feedback_items': [
                item.to_dict() for item in self.feedback_items],
            'required_action': self.required_action.value if self.required_action else None,
            'revision_cycles': self.revision_cycles,
            'summary': self.summary,
            'improvement_suggestions': self.improvement_suggestions,
            'quality_metrics': self.quality_metrics}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredFeedback':
        """Create StructuredFeedback from dictionary."""
        feedback_items = [
            FeedbackItem.from_dict(item_data)
            for item_data in data.get('feedback_items', [])
        ]

        required_action = None
        if data.get('required_action'):
            required_action = RequiredAction(data['required_action'])

        return cls(
            overall_score=data['overall_score'],
            sub_scores=data.get('sub_scores', {}),
            feedback_items=feedback_items,
            required_action=required_action,
            revision_cycles=data.get('revision_cycles', 0),
            summary=data.get('summary', ''),
            improvement_suggestions=data.get('improvement_suggestions', []),
            quality_metrics=data.get('quality_metrics', {})
        )


class FeedbackGenerator:
    """Generator for structured feedback based on content analysis."""

    def __init__(self):
        self.quality_thresholds = {
            'grammar': 0.8,
            'style': 0.7,
            'clarity': 0.75,
            'accuracy': 0.9,
            'tone': 0.7,
            'structure': 0.7,
            'engagement': 0.6,
            'seo': 0.6,
            'brand_compliance': 0.8,
            'accessibility': 0.7
        }

    def generate_feedback(self,
                          content: str,
                          analysis_results: Dict[str,
                                                 Any]) -> StructuredFeedback:
        """Generate structured feedback from content analysis."""
        feedback_items = []
        sub_scores = {}
        quality_metrics = {}

        # Process grammar and style issues
        if 'grammar_issues' in analysis_results:
            grammar_score, grammar_items = self._process_grammar_issues(
                analysis_results['grammar_issues']
            )
            sub_scores['grammar'] = grammar_score
            feedback_items.extend(grammar_items)

        # Process style issues
        if 'style_issues' in analysis_results:
            style_score, style_items = self._process_style_issues(
                analysis_results['style_issues']
            )
            sub_scores['style'] = style_score
            feedback_items.extend(style_items)

        # Process clarity issues
        if 'clarity_issues' in analysis_results:
            clarity_score, clarity_items = self._process_clarity_issues(
                analysis_results['clarity_issues']
            )
            sub_scores['clarity'] = clarity_score
            feedback_items.extend(clarity_items)

        # Process accuracy issues
        if 'accuracy_issues' in analysis_results:
            accuracy_score, accuracy_items = self._process_accuracy_issues(
                analysis_results['accuracy_issues']
            )
            sub_scores['accuracy'] = accuracy_score
            feedback_items.extend(accuracy_items)

        # Calculate overall score
        overall_score = self._calculate_overall_score(sub_scores)

        # Determine required action
        required_action = self._determine_required_action(
            feedback_items, overall_score)

        # Generate summary and suggestions
        summary = self._generate_summary(feedback_items, overall_score)
        improvement_suggestions = self._generate_improvement_suggestions(
            feedback_items)

        return StructuredFeedback(
            overall_score=overall_score,
            sub_scores=sub_scores,
            feedback_items=feedback_items,
            required_action=required_action,
            revision_cycles=0,  # Will be updated by workflow
            summary=summary,
            improvement_suggestions=improvement_suggestions,
            quality_metrics=quality_metrics
        )

    def generate_feedback_from_audit(
            self, audit_results: Dict[str, Any]) -> StructuredFeedback:
        """Generate structured feedback from audit results."""
        feedback_items = []
        sub_scores = {}
        quality_metrics = {}

        # Process grammar issues
        if 'grammar_issues' in audit_results:
            grammar_score, grammar_items = self._process_audit_issues(
                audit_results['grammar_issues'], IssueType.GRAMMAR
            )
            sub_scores['grammar'] = grammar_score
            feedback_items.extend(grammar_items)

        # Process style issues
        if 'style_issues' in audit_results:
            style_score, style_items = self._process_audit_issues(
                audit_results['style_issues'], IssueType.STYLE
            )
            sub_scores['style'] = style_score
            feedback_items.extend(style_items)

        # Process clarity issues
        if 'clarity_issues' in audit_results:
            clarity_score, clarity_items = self._process_audit_issues(
                audit_results['clarity_issues'], IssueType.CLARITY
            )
            sub_scores['clarity'] = clarity_score
            feedback_items.extend(clarity_items)

        # Process structure issues
        if 'structure_issues' in audit_results:
            structure_score, structure_items = self._process_audit_issues(
                audit_results['structure_issues'], IssueType.STRUCTURE
            )
            sub_scores['structure'] = structure_score
            feedback_items.extend(structure_items)

        # Process engagement issues
        if 'engagement_issues' in audit_results:
            engagement_score, engagement_items = self._process_audit_issues(
                audit_results['engagement_issues'], IssueType.ENGAGEMENT
            )
            sub_scores['engagement'] = engagement_score
            feedback_items.extend(engagement_items)

        # Process SEO issues
        if 'seo_issues' in audit_results:
            seo_score, seo_items = self._process_audit_issues(
                audit_results['seo_issues'], IssueType.SEO
            )
            sub_scores['seo'] = seo_score
            feedback_items.extend(seo_items)

        # Process brand compliance issues
        if 'brand_compliance_issues' in audit_results:
            brand_score, brand_items = self._process_audit_issues(
                audit_results['brand_compliance_issues'], IssueType.BRAND_COMPLIANCE)
            sub_scores['brand_compliance'] = brand_score
            feedback_items.extend(brand_items)

        # Process accessibility issues
        if 'accessibility_issues' in audit_results:
            accessibility_score, accessibility_items = self._process_audit_issues(
                audit_results['accessibility_issues'], IssueType.ACCESSIBILITY)
            sub_scores['accessibility'] = accessibility_score
            feedback_items.extend(accessibility_items)

        # Calculate overall score
        overall_score = audit_results.get('overall_score', 7.0)

        # Determine required action
        required_action = self._determine_required_action(
            feedback_items, overall_score)

        # Generate summary and suggestions
        summary = self._generate_summary(feedback_items, overall_score)
        improvement_suggestions = audit_results.get('recommendations', [])

        return StructuredFeedback(
            overall_score=overall_score,
            sub_scores=sub_scores,
            feedback_items=feedback_items,
            required_action=required_action,
            revision_cycles=0,
            summary=summary,
            improvement_suggestions=improvement_suggestions,
            quality_metrics=quality_metrics
        )

    def _process_grammar_issues(
            self, grammar_issues: List[Dict]) -> tuple[float, List[FeedbackItem]]:
        """Process grammar issues and return score and feedback items."""
        items = []
        total_issues = len(grammar_issues)

        for issue in grammar_issues:
            severity = self._determine_grammar_severity(issue)
            required_action = RequiredAction.REVISION if severity == Severity.HIGH else RequiredAction.STYLE_ADJUSTMENT

            items.append(FeedbackItem(
                text_snippet=issue.get('text', ''),
                issue_type=IssueType.GRAMMAR,
                comment=issue.get('message', 'Grammar issue detected'),
                required_action=required_action,
                severity=severity,
                suggestion=issue.get('suggestion'),
                confidence_score=issue.get('confidence', 0.8)
            ))

        # Calculate score based on number and severity of issues
        score = max(0.0, 1.0 - (total_issues * 0.1))
        return score, items

    def _process_style_issues(
            self, style_issues: List[Dict]) -> tuple[float, List[FeedbackItem]]:
        """Process style issues and return score and feedback items."""
        items = []
        total_issues = len(style_issues)

        for issue in style_issues:
            severity = self._determine_style_severity(issue)
            required_action = RequiredAction.STYLE_ADJUSTMENT

            items.append(FeedbackItem(
                text_snippet=issue.get('text', ''),
                issue_type=IssueType.STYLE,
                comment=issue.get('message', 'Style issue detected'),
                required_action=required_action,
                severity=severity,
                suggestion=issue.get('suggestion'),
                confidence_score=issue.get('confidence', 0.7)
            ))

        score = max(0.0, 1.0 - (total_issues * 0.05))
        return score, items

    def _process_clarity_issues(
            self, clarity_issues: List[Dict]) -> tuple[float, List[FeedbackItem]]:
        """Process clarity issues and return score and feedback items."""
        items = []
        total_issues = len(clarity_issues)

        for issue in clarity_issues:
            severity = self._determine_clarity_severity(issue)
            required_action = RequiredAction.REVISION if severity == Severity.HIGH else RequiredAction.STYLE_ADJUSTMENT

            items.append(FeedbackItem(
                text_snippet=issue.get('text', ''),
                issue_type=IssueType.CLARITY,
                comment=issue.get('message', 'Clarity issue detected'),
                required_action=required_action,
                severity=severity,
                suggestion=issue.get('suggestion'),
                confidence_score=issue.get('confidence', 0.6)
            ))

        score = max(0.0, 1.0 - (total_issues * 0.08))
        return score, items

    def _process_accuracy_issues(
            self, accuracy_issues: List[Dict]) -> tuple[float, List[FeedbackItem]]:
        """Process accuracy issues and return score and feedback items."""
        items = []
        total_issues = len(accuracy_issues)

        for issue in accuracy_issues:
            severity = Severity.HIGH  # Accuracy issues are always high priority
            required_action = RequiredAction.RESEARCH_VERIFICATION

            items.append(FeedbackItem(
                text_snippet=issue.get('text', ''),
                issue_type=IssueType.ACCURACY,
                comment=issue.get('message', 'Accuracy issue detected'),
                required_action=required_action,
                severity=severity,
                suggestion=issue.get('suggestion'),
                confidence_score=issue.get('confidence', 0.9)
            ))

        # Accuracy issues heavily penalize score
        score = max(0.0, 1.0 - (total_issues * 0.2))
        return score, items

    def _process_audit_issues(self,
                              issues: List[Dict],
                              issue_type: IssueType) -> tuple[float,
                                                              List[FeedbackItem]]:
        """Process audit issues and return score and feedback items."""
        items = []
        total_issues = len(issues)

        for issue in issues:
            severity = self._determine_audit_severity(issue)
            required_action = self._determine_audit_action(issue, severity)

            items.append(FeedbackItem(
                text_snippet=issue.get('text', ''),
                issue_type=issue_type,
                comment=issue.get('issue_type', 'Issue detected'),
                required_action=required_action,
                severity=severity,
                suggestion=issue.get('suggestion'),
                confidence_score=0.8
            ))

        # Calculate score based on number and severity of issues
        score = max(0.0, 1.0 - (total_issues * 0.1))
        return score, items

    def _determine_grammar_severity(self, issue: Dict) -> Severity:
        """Determine severity of grammar issue."""
        issue_type = issue.get('type', '').lower()
        if issue_type in ['spelling', 'punctuation']:
            return Severity.MEDIUM
        elif issue_type in ['syntax', 'grammar']:
            return Severity.HIGH
        else:
            return Severity.LOW

    def _determine_style_severity(self, issue: Dict) -> Severity:
        """Determine severity of style issue."""
        issue_type = issue.get('type', '').lower()
        if issue_type in ['tone', 'voice']:
            return Severity.MEDIUM
        elif issue_type in ['formatting', 'structure']:
            return Severity.LOW
        else:
            return Severity.MEDIUM

    def _determine_clarity_severity(self, issue: Dict) -> Severity:
        """Determine severity of clarity issue."""
        issue_type = issue.get('type', '').lower()
        if issue_type in ['ambiguity', 'confusion']:
            return Severity.HIGH
        elif issue_type in ['wordiness', 'redundancy']:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _determine_audit_severity(self, issue: Dict) -> Severity:
        """Determine severity for audit issues."""
        issue_type = issue.get('issue_type', '').lower()

        # High severity issues
        high_severity = ['grammar_error', 'critical_error', 'major_issue']
        if any(high in issue_type for high in high_severity):
            return Severity.HIGH

        # Medium severity issues
        medium_severity = ['style_issue', 'clarity_issue', 'structure_issue']
        if any(medium in issue_type for medium in medium_severity):
            return Severity.MEDIUM

        return Severity.LOW

    def _determine_audit_action(
            self,
            issue: Dict,
            severity: Severity) -> RequiredAction:
        """Determine required action for audit issues."""
        if severity == Severity.HIGH:
            return RequiredAction.REVISION
        elif severity == Severity.MEDIUM:
            return RequiredAction.STYLE_ADJUSTMENT
        else:
            return RequiredAction.NO_ACTION

    def _calculate_overall_score(self, sub_scores: Dict[str, float]) -> float:
        """Calculate overall score from sub-scores."""
        if not sub_scores:
            return 0.0

        # Weight different aspects
        weights = {
            'grammar': 0.15,
            'style': 0.10,
            'clarity': 0.20,
            'accuracy': 0.25,
            'tone': 0.10,
            'structure': 0.10,
            'engagement': 0.05,
            'seo': 0.03,
            'brand_compliance': 0.02
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for aspect, score in sub_scores.items():
            weight = weights.get(aspect, 0.05)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_required_action(self,
                                   feedback_items: List[FeedbackItem],
                                   overall_score: float) -> RequiredAction:
        """Determine the required action based on feedback items and overall score."""
        high_priority_items = [
            item for item in feedback_items if item.severity == Severity.HIGH]

        if overall_score < 0.6:
            return RequiredAction.REVISION
        elif any(item.required_action == RequiredAction.RESEARCH_VERIFICATION for item in high_priority_items):
            return RequiredAction.RESEARCH_VERIFICATION
        elif high_priority_items:
            return RequiredAction.REVISION
        elif overall_score < 0.8:
            return RequiredAction.STYLE_ADJUSTMENT
        else:
            return RequiredAction.NO_ACTION

    def _generate_summary(
            self,
            feedback_items: List[FeedbackItem],
            overall_score: float) -> str:
        """Generate a summary of the feedback."""
        if overall_score >= 0.9:
            return "Excellent content with minimal issues."
        elif overall_score >= 0.8:
            return "Good content with some minor improvements needed."
        elif overall_score >= 0.7:
            return "Acceptable content requiring moderate revisions."
        elif overall_score >= 0.6:
            return "Content needs significant improvements."
        else:
            return "Content requires major revisions to meet quality standards."

    def _generate_improvement_suggestions(
            self, feedback_items: List[FeedbackItem]) -> List[str]:
        """Generate improvement suggestions based on feedback items."""
        suggestions = []

        # Group by issue type and generate suggestions
        issue_counts = {}
        for item in feedback_items:
            issue_type = item.issue_type.value
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        if issue_counts.get('grammar', 0) > 0:
            suggestions.append(
                "Review and correct grammar issues throughout the content.")

        if issue_counts.get('style', 0) > 0:
            suggestions.append("Improve writing style and tone consistency.")

        if issue_counts.get('clarity', 0) > 0:
            suggestions.append(
                "Enhance clarity and readability of complex sentences.")

        if issue_counts.get('accuracy', 0) > 0:
            suggestions.append("Verify factual accuracy and source citations.")

        if len(feedback_items) > 10:
            suggestions.append(
                "Consider breaking content into smaller, more focused sections.")

        return suggestions
