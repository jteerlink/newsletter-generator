"""
Feedback Orchestrator: Integrates Phase 3 tools with feedback system.

This module orchestrates the feedback system by integrating the grammar linter
and enhanced search tools from Phase 3 with the structured feedback system
to provide comprehensive content validation and improvement recommendations.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.enhanced_search import EnhancedSearchTool, SearchResult
from tools.grammar_linter import GrammarAndStyleLinter, LinterResult

from .campaign_context import CampaignContext
from .execution_state import ExecutionState
from .feedback_system import FeedbackItem, IssueType, RequiredAction, Severity, StructuredFeedback
from .tool_usage_tracker import get_tool_tracker

logger = logging.getLogger(__name__)


@dataclass
class FeedbackOrchestrationResult:
    """Result of feedback orchestration process."""
    structured_feedback: StructuredFeedback
    linter_result: LinterResult
    search_verifications: List[Dict[str, Any]]
    improvement_recommendations: List[str]
    quality_assessment: Dict[str, Any]
    next_actions: List[str]


class FeedbackOrchestrator:
    """
    Orchestrates feedback generation using Phase 3 tools.

    This class integrates the grammar linter and enhanced search tools
    with the structured feedback system to provide comprehensive
    content validation and improvement recommendations.
    """

    def __init__(self):
        """Initialize the feedback orchestrator."""
        self.linter = GrammarAndStyleLinter()
        self.search_tool = EnhancedSearchTool()
        self.feedback_generator = None  # Will be initialized when needed

        # Phase 2.1: Add tool usage tracking
        self.tool_tracker = get_tool_tracker()

    def orchestrate_feedback(
            self,
            content: str,
            context: CampaignContext,
            execution_state: ExecutionState) -> FeedbackOrchestrationResult:
        """
        Orchestrate comprehensive feedback generation.

        Args:
            content: Content to analyze
            context: Campaign context for quality thresholds
            execution_state: Current execution state

        Returns:
            FeedbackOrchestrationResult with comprehensive analysis
        """
        try:
            logger.info("Starting feedback orchestration")

            # Step 1: Grammar and style analysis with tracking
            with self.tool_tracker.track_tool_usage(
                tool_name="grammar_linter",
                agent_name="FeedbackOrchestrator",
                workflow_id=execution_state.workflow_id,
                session_id=execution_state.session_id,
                input_data={"content_length": len(content)},
                context={"phase": "feedback_orchestration", "step": "grammar_analysis"}
            ):
                linter_result = self.linter.check_content(content)
            logger.info(
                f"Linter analysis completed. Score: {
                    linter_result.overall_score:.2f}")

            # Step 2: Generate structured feedback
            structured_feedback = self._generate_structured_feedback(
                content, linter_result, context
            )

            # Step 3: Search verification for accuracy issues with tracking
            # context
            search_verifications = self._verify_claims(
                content, structured_feedback,
                execution_state.workflow_id, execution_state.session_id
            )

            # Step 4: Generate improvement recommendations
            improvement_recommendations = self._generate_improvement_recommendations(
                structured_feedback, linter_result, context)

            # Step 5: Quality assessment
            quality_assessment = self._assess_quality(
                structured_feedback, linter_result, context
            )

            # Step 6: Determine next actions
            next_actions = self._determine_next_actions(
                structured_feedback, quality_assessment, context
            )

            # Update execution state
            self._update_execution_state(
                execution_state,
                structured_feedback,
                quality_assessment)

            result = FeedbackOrchestrationResult(
                structured_feedback=structured_feedback,
                linter_result=linter_result,
                search_verifications=search_verifications,
                improvement_recommendations=improvement_recommendations,
                quality_assessment=quality_assessment,
                next_actions=next_actions
            )

            logger.info("Feedback orchestration completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in feedback orchestration: {e}")
            # Return a basic result with error information
            return self._create_error_result(str(e))

    def _generate_structured_feedback(
            self,
            content: str,
            linter_result: LinterResult,
            context: CampaignContext) -> StructuredFeedback:
        """Generate structured feedback from linter results."""
        feedback_items = []

        # Process grammar issues
        for issue in linter_result.grammar_issues:
            feedback_item = FeedbackItem(
                text_snippet=issue.context or issue.message,
                issue_type=IssueType.GRAMMAR,
                comment=issue.message,
                required_action=RequiredAction.REVISION,
                severity=Severity(issue.severity),
                suggestion=issue.suggestion,
                confidence_score=0.9  # High confidence for grammar issues
            )
            feedback_items.append(feedback_item)

        # Process style issues
        for issue in linter_result.style_issues:
            feedback_item = FeedbackItem(
                text_snippet=issue.context or issue.message,
                issue_type=IssueType.STYLE,
                comment=issue.message,
                required_action=RequiredAction.STYLE_ADJUSTMENT,
                severity=Severity(issue.severity),
                suggestion=issue.suggestion,
                confidence_score=0.8  # Good confidence for style issues
            )
            feedback_items.append(feedback_item)

        # Calculate scores
        grammar_score = linter_result.grammar_score
        style_score = linter_result.style_score
        overall_score = linter_result.overall_score

        sub_scores = {
            'grammar': grammar_score,
            'style': style_score,
            'clarity': self._assess_clarity(content),
            'engagement': self._assess_engagement(content, context),
            'brand_compliance': self._assess_brand_compliance(content, context)
        }

        # Determine required action
        required_action = self._determine_required_action(
            feedback_items, overall_score, context)

        # Generate summary and suggestions
        summary = self._generate_summary(feedback_items, overall_score)
        improvement_suggestions = self._generate_improvement_suggestions(
            feedback_items)

        # Quality metrics
        quality_metrics = {
            'word_count': linter_result.summary.get('word_count', 0),
            'sentence_count': linter_result.summary.get('sentence_count', 0),
            'average_sentence_length': linter_result.summary.get('average_sentence_length', 0),
            'issue_distribution': {
                'high': len([i for i in feedback_items if i.severity == Severity.HIGH]),
                'medium': len([i for i in feedback_items if i.severity == Severity.MEDIUM]),
                'low': len([i for i in feedback_items if i.severity == Severity.LOW])
            }
        }

        return StructuredFeedback(
            overall_score=overall_score,
            sub_scores=sub_scores,
            feedback_items=feedback_items,
            required_action=required_action,
            revision_cycles=0,  # Will be updated by execution state
            summary=summary,
            improvement_suggestions=improvement_suggestions,
            quality_metrics=quality_metrics
        )

    def _verify_claims(self,
                       content: str,
                       feedback: StructuredFeedback,
                       workflow_id: str = None,
                       session_id: str = None) -> List[Dict[str,
                                                            Any]]:
        """Verify claims in content using enhanced search."""
        verifications = []

        # Extract potential claims for verification
        claims = self._extract_claims(content)

        for claim in claims[:3]:  # Limit to top 3 claims
            try:
                # Search for verification with tracking
                with self.tool_tracker.track_tool_usage(
                    tool_name="enhanced_search",
                    agent_name="FeedbackOrchestrator",
                    workflow_id=workflow_id,
                    session_id=session_id,
                    input_data={"query": claim, "context": "fact verification", "max_results": 3},
                    context={"phase": "feedback_orchestration", "step": "claim_verification"}
                ):
                    search_results = self.search_tool.search_with_confidence(
                        query=claim,
                        context="fact verification",
                        max_results=3
                    )

                verification = {
                    'claim': claim,
                    'search_results': [
                        {
                            'title': result.title,
                            'url': result.url,
                            'confidence': result.confidence_score,
                            'relevance': result.relevance_score
                        }
                        for result in search_results
                    ],
                    'verification_score': self._calculate_verification_score(search_results),
                    'needs_research': len(search_results) == 0 or
                    max(r.confidence_score for r in search_results) < 0.6
                }
                verifications.append(verification)

            except Exception as e:
                logger.warning(f"Error verifying claim '{claim}': {e}")
                verifications.append({
                    'claim': claim,
                    'error': str(e),
                    'needs_research': True
                })

        return verifications

    def _extract_claims(self, content: str) -> List[str]:
        """Extract potential claims from content for verification."""
        # Simple claim extraction - in production, use NLP
        sentences = content.split('.')
        claims = []

        # Look for sentences with specific patterns that might need
        # verification
        claim_indicators = [
            'research shows', 'studies indicate', 'experts say',
            'according to', 'reports suggest', 'data shows',
            'statistics show', 'analysis reveals', 'survey finds'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower()
                   for indicator in claim_indicators):
                claims.append(sentence)

        return claims[:5]  # Return top 5 claims

    def _calculate_verification_score(
            self, search_results: List[SearchResult]) -> float:
        """Calculate verification score based on search results."""
        if not search_results:
            return 0.0

        # Weight by confidence and relevance
        scores = []
        for result in search_results:
            score = (result.confidence_score + result.relevance_score) / 2
            scores.append(score)

        return max(scores) if scores else 0.0

    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity."""
        # Simple clarity assessment
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split())
                                  for s in sentences) / max(len(sentences), 1)

        # Shorter sentences are generally clearer
        if avg_sentence_length <= 15:
            return 0.9
        elif avg_sentence_length <= 25:
            return 0.7
        else:
            return 0.5

    def _assess_engagement(
            self,
            content: str,
            context: CampaignContext) -> float:
        """Assess content engagement based on campaign context."""
        # Simple engagement assessment
        engagement_indicators = [
            'you', 'your', 'imagine', 'consider', 'think about',
            'what if', 'suppose', 'let\'s', 'we can'
        ]

        content_lower = content.lower()
        engagement_count = sum(1 for indicator in engagement_indicators
                               if indicator in content_lower)

        # Normalize by content length
        word_count = len(content.split())
        engagement_score = min(
            engagement_count / max(word_count / 100, 1), 1.0)

        return engagement_score

    def _assess_brand_compliance(
            self,
            content: str,
            context: CampaignContext) -> float:
        """Assess brand compliance based on campaign context."""
        score = 1.0

        # Check for forbidden terminology
        content_lower = content.lower()
        forbidden_count = sum(1 for term in context.forbidden_terminology
                              if term.lower() in content_lower)

        if forbidden_count > 0:
            score -= 0.2 * forbidden_count

        # Check for preferred terminology usage
        preferred_count = sum(1 for term in context.preferred_terminology
                              if term.lower() in content_lower)

        if preferred_count > 0:
            score += 0.1 * preferred_count

        return max(0.0, min(1.0, score))

    def _determine_required_action(self,
                                   feedback_items: List[FeedbackItem],
                                   overall_score: float,
                                   context: CampaignContext) -> RequiredAction:
        """Determine the required action based on feedback and context."""
        high_priority_items = [
            item for item in feedback_items if item.severity == Severity.HIGH]

        # If quality threshold not met, revision is required
        if overall_score < context.get_quality_threshold('minimum'):
            return RequiredAction.REVISION

        # If high priority issues exist, revision is required
        if high_priority_items:
            return RequiredAction.REVISION

        # If accuracy issues exist, research verification might be needed
        accuracy_items = [
            item for item in feedback_items if item.issue_type == IssueType.ACCURACY]
        if accuracy_items:
            return RequiredAction.RESEARCH_VERIFICATION

        # If only style issues, style adjustment is sufficient
        style_items = [
            item for item in feedback_items if item.issue_type == IssueType.STYLE]
        if style_items and not high_priority_items:
            return RequiredAction.STYLE_ADJUSTMENT

        return RequiredAction.NO_ACTION

    def _generate_summary(
            self,
            feedback_items: List[FeedbackItem],
            overall_score: float) -> str:
        """Generate a summary of the feedback."""
        if not feedback_items:
            return "Content meets quality standards."

        high_count = len(
            [i for i in feedback_items if i.severity == Severity.HIGH])
        medium_count = len(
            [i for i in feedback_items if i.severity == Severity.MEDIUM])
        low_count = len(
            [i for i in feedback_items if i.severity == Severity.LOW])

        summary = f"Content quality score: {overall_score:.2f}. "

        if high_count > 0:
            summary += f"Found {
                high_count} high-priority issues requiring immediate attention. "
        if medium_count > 0:
            summary += f"Found {medium_count} medium-priority issues for improvement. "
        if low_count > 0:
            summary += f"Found {low_count} minor issues for consideration. "

        return summary

    def _generate_improvement_suggestions(
            self, feedback_items: List[FeedbackItem]) -> List[str]:
        """Generate improvement suggestions based on feedback items."""
        suggestions = []

        # Group suggestions by issue type
        issue_types = {}
        for item in feedback_items:
            if item.issue_type not in issue_types:
                issue_types[item.issue_type] = []
            issue_types[item.issue_type].append(item)

        # Generate suggestions for each issue type
        for issue_type, items in issue_types.items():
            if issue_type == IssueType.GRAMMAR:
                suggestions.append("Review and correct grammar issues")
            elif issue_type == IssueType.STYLE:
                suggestions.append("Improve writing style and clarity")
            elif issue_type == IssueType.CLARITY:
                suggestions.append("Enhance content clarity and readability")
            elif issue_type == IssueType.ACCURACY:
                suggestions.append("Verify factual claims and data")
            elif issue_type == IssueType.TONE:
                suggestions.append("Adjust tone to match target audience")

        # Add general suggestions
        if len(feedback_items) > 10:
            suggestions.append("Consider comprehensive content review")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _generate_improvement_recommendations(
            self,
            feedback: StructuredFeedback,
            linter_result: LinterResult,
            context: CampaignContext) -> List[str]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []

        # Quality-based recommendations
        if feedback.overall_score < 0.8:
            recommendations.append(
                "Focus on improving overall content quality")

        # Grammar-based recommendations
        if linter_result.grammar_score < 0.8:
            recommendations.append("Review grammar and punctuation")

        # Style-based recommendations
        if linter_result.style_score < 0.8:
            recommendations.append("Enhance writing style and engagement")

        # Context-based recommendations
        if context.content_style.get('tone') == 'professional':
            recommendations.append("Maintain professional tone throughout")

        return recommendations

    def _assess_quality(self,
                        feedback: StructuredFeedback,
                        linter_result: LinterResult,
                        context: CampaignContext) -> Dict[str, Any]:
        """Assess overall quality based on multiple factors."""
        quality_assessment = {
            'overall_score': feedback.overall_score,
            'meets_threshold': feedback.overall_score >= context.get_quality_threshold('minimum'),
            'sub_scores': feedback.sub_scores,
            'issue_count': len(feedback.feedback_items),
            'high_priority_issues': len([i for i in feedback.feedback_items if i.severity == Severity.HIGH]),
            'quality_level': self._determine_quality_level(feedback.overall_score),
            'improvement_potential': self._calculate_improvement_potential(feedback),
            'ready_for_publication': feedback.overall_score >= context.get_quality_threshold('excellent')
        }

        return quality_assessment

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.75:
            return "acceptable"
        elif score >= 0.65:
            return "needs_improvement"
        else:
            return "poor"

    def _calculate_improvement_potential(
            self, feedback: StructuredFeedback) -> float:
        """Calculate improvement potential based on issues."""
        if not feedback.feedback_items:
            return 0.0

        # Calculate potential improvement based on issue severity
        potential = 0.0
        for item in feedback.feedback_items:
            if item.severity == Severity.HIGH:
                potential += 0.1
            elif item.severity == Severity.MEDIUM:
                potential += 0.05
            elif item.severity == Severity.LOW:
                potential += 0.02

        return min(1.0, potential)

    def _determine_next_actions(self,
                                feedback: StructuredFeedback,
                                quality_assessment: Dict[str, Any],
                                context: CampaignContext) -> List[str]:
        """Determine next actions based on feedback and quality assessment."""
        actions = []

        if not quality_assessment['meets_threshold']:
            actions.append("REVISION_REQUIRED")

        if quality_assessment['high_priority_issues'] > 0:
            actions.append("ADDRESS_HIGH_PRIORITY_ISSUES")

        if feedback.required_action == RequiredAction.RESEARCH_VERIFICATION:
            actions.append("VERIFY_CLAIMS")

        if quality_assessment['ready_for_publication']:
            actions.append("READY_FOR_PUBLICATION")
        else:
            actions.append("CONTINUE_REFINEMENT")

        return actions

    def _update_execution_state(self,
                                execution_state: ExecutionState,
                                feedback: StructuredFeedback,
                                quality_assessment: Dict[str, Any]) -> None:
        """Update execution state with feedback results."""
        # Add feedback to execution state
        execution_state.add_feedback({
            'feedback': feedback.to_dict(),
            'quality_assessment': quality_assessment,
            'timestamp': time.time()
        })

        # Update quality scores
        execution_state.update_quality_score(
            'content_quality', feedback.overall_score)

    def _create_error_result(
            self,
            error_message: str) -> FeedbackOrchestrationResult:
        """Create a result with error information."""
        error_feedback = StructuredFeedback(
            overall_score=0.0,
            sub_scores={},
            feedback_items=[],
            required_action=RequiredAction.REVISION,
            revision_cycles=0,
            summary=f"Error during analysis: {error_message}",
            improvement_suggestions=["Resolve system error and retry analysis"],
            quality_metrics={})

        return FeedbackOrchestrationResult(
            structured_feedback=error_feedback,
            linter_result=None,
            search_verifications=[],
            improvement_recommendations=["Resolve system error"],
            quality_assessment={'error': error_message},
            next_actions=["RETRY_ANALYSIS"]
        )
