"""
Iterative Refinement Loop: Automated content improvement workflow.

This module implements the iterative refinement loop that automatically
improves content quality through multiple cycles of feedback and revision,
using the feedback orchestrator and learning system.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .campaign_context import CampaignContext
from .execution_state import ExecutionState
from .feedback_orchestrator import FeedbackOrchestrationResult, FeedbackOrchestrator
from .feedback_system import RequiredAction
from .learning_system import LearningSystem

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Result of the refinement loop execution."""
    final_content: str
    final_score: float
    revision_cycles: int
    improvement_history: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    learning_data: Dict[str, Any]
    status: str  # 'completed', 'max_cycles_reached', 'error'


class RefinementLoop:
    """
    Iterative refinement loop for content improvement.

    This class implements an automated workflow that continuously
    improves content quality through feedback-driven revisions.
    """

    def __init__(self, max_revision_cycles: int = 3):
        """Initialize the refinement loop."""
        self.max_revision_cycles = max_revision_cycles
        self.feedback_orchestrator = FeedbackOrchestrator()
        self.learning_system = LearningSystem()

    def execute_refinement(
            self,
            content: str,
            context: CampaignContext,
            execution_state: ExecutionState) -> RefinementResult:
        """
        Execute the iterative refinement loop.

        Args:
            content: Initial content to refine
            context: Campaign context for quality thresholds
            execution_state: Current execution state

        Returns:
            RefinementResult with final content and metrics
        """
        try:
            logger.info("Starting iterative refinement loop")

            current_content = content
            revision_count = 0
            improvement_history = []
            start_time = time.time()

            while revision_count < self.max_revision_cycles:
                logger.info(f"Starting revision cycle {revision_count + 1}")

                # Step 1: Generate feedback
                feedback_result = self.feedback_orchestrator.orchestrate_feedback(
                    current_content, context, execution_state)

                # Step 2: Check if quality threshold is met
                if self._meets_quality_threshold(feedback_result, context):
                    logger.info("Quality threshold met, refinement complete")
                    return self._create_success_result(
                        current_content, feedback_result, revision_count,
                        improvement_history, start_time, "completed"
                    )

                # Step 3: Handle feedback based on required action
                if feedback_result.structured_feedback.required_action == RequiredAction.REVISION:
                    current_content = self._handle_revision(
                        current_content, feedback_result)
                elif feedback_result.structured_feedback.required_action == RequiredAction.RESEARCH_VERIFICATION:
                    current_content = self._handle_research_verification(
                        current_content, feedback_result)
                elif feedback_result.structured_feedback.required_action == RequiredAction.STYLE_ADJUSTMENT:
                    current_content = self._handle_style_adjustment(
                        current_content, feedback_result)
                else:
                    logger.info("No action required, refinement complete")
                    return self._create_success_result(
                        current_content, feedback_result, revision_count,
                        improvement_history, start_time, "completed"
                    )

                # Step 4: Record improvement
                improvement_record = {
                    'cycle': revision_count + 1,
                    'score': feedback_result.structured_feedback.overall_score,
                    'issues_fixed': len(
                        feedback_result.structured_feedback.feedback_items),
                    'action_taken': feedback_result.structured_feedback.required_action.value,
                    'timestamp': time.time()}
                improvement_history.append(improvement_record)

                # Step 5: Update execution state
                execution_state.increment_revision_cycle('content_refinement')

                revision_count += 1
                logger.info(f"Completed revision cycle {revision_count}")

            # If max cycles reached, flag for human review
            logger.warning(
                f"Max revision cycles ({
                    self.max_revision_cycles}) reached")
            return self._create_success_result(
                current_content, feedback_result, revision_count,
                improvement_history, start_time, "max_cycles_reached"
            )

        except Exception as e:
            logger.error(f"Error in refinement loop: {e}")
            return self._create_error_result(
                str(e), revision_count, improvement_history)

    def _meets_quality_threshold(self,
                                 feedback_result: FeedbackOrchestrationResult,
                                 context: CampaignContext) -> bool:
        """Check if content meets quality threshold."""
        overall_score = feedback_result.structured_feedback.overall_score
        minimum_threshold = context.get_quality_threshold('minimum')

        # Check if score meets minimum threshold
        if overall_score < minimum_threshold:
            return False

        # Check if there are high priority issues
        high_priority_issues = feedback_result.quality_assessment.get(
            'high_priority_issues', 0)
        if high_priority_issues > 0:
            return False

        # Check if required action is NO_ACTION
        if feedback_result.structured_feedback.required_action != RequiredAction.NO_ACTION:
            return False

        return True

    def _handle_revision(self,
                         content: str,
                         feedback_result: FeedbackOrchestrationResult) -> str:
        """Handle revision based on feedback."""
        logger.info("Handling content revision")

        # Get high priority issues first
        high_priority_items = feedback_result.structured_feedback.get_high_priority_items()

        # Apply fixes based on issue types
        revised_content = content

        for item in high_priority_items:
            if item.issue_type.value == 'grammar':
                revised_content = self._fix_grammar_issue(
                    revised_content, item)
            elif item.issue_type.value == 'style':
                revised_content = self._fix_style_issue(revised_content, item)
            elif item.issue_type.value == 'clarity':
                revised_content = self._fix_clarity_issue(
                    revised_content, item)

        # Apply medium priority fixes if time permits
        medium_priority_items = [
            item for item in feedback_result.structured_feedback.feedback_items
            if item.severity.value == 'MEDIUM'
        ][:5]  # Limit to top 5 medium priority items

        for item in medium_priority_items:
            if item.issue_type.value == 'style':
                revised_content = self._fix_style_issue(revised_content, item)

        return revised_content

    def _handle_research_verification(
            self,
            content: str,
            feedback_result: FeedbackOrchestrationResult) -> str:
        """Handle research verification based on feedback."""
        logger.info("Handling research verification")

        # For now, we'll add verification notes to the content
        # In a full implementation, this would trigger additional research
        verification_notes = []

        for verification in feedback_result.search_verifications:
            if verification.get('needs_research', False):
                claim = verification.get('claim', '')
                verification_notes.append(
                    f"Note: Claim '{claim}' requires verification")

        if verification_notes:
            content += "\n\n--- Verification Notes ---\n"
            content += "\n".join(verification_notes)

        return content

    def _handle_style_adjustment(
            self,
            content: str,
            feedback_result: FeedbackOrchestrationResult) -> str:
        """Handle style adjustment based on feedback."""
        logger.info("Handling style adjustment")

        # Apply style improvements
        style_items = [
            item for item in feedback_result.structured_feedback.feedback_items
            if item.issue_type.value == 'style'
        ]

        revised_content = content

        for item in style_items:
            revised_content = self._fix_style_issue(revised_content, item)

        return revised_content

    def _fix_grammar_issue(self, content: str, item) -> str:
        """Fix a grammar issue in content."""
        # Simple grammar fixes - in production, use more sophisticated NLP
        if 'double space' in item.comment.lower():
            # Fix double spaces
            while '  ' in content:
                content = content.replace('  ', ' ')

        if 'missing period' in item.comment.lower():
            # Add periods to sentences that don't end with punctuation
            sentences = content.split('\n')
            fixed_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                fixed_sentences.append(sentence)
            content = '\n'.join(fixed_sentences)

        return content

    def _fix_style_issue(self, content: str, item) -> str:
        """Fix a style issue in content."""
        # Simple style fixes - in production, use more sophisticated NLP
        if 'passive voice' in item.comment.lower():
            # Simple passive voice detection and correction
            # This is a simplified implementation
            content = content.replace(' was optimized ', ' optimized ')
            content = content.replace(' was developed ', ' developed ')
            content = content.replace(' was created ', ' created ')

        if 'jargon' in item.comment.lower():
            # Replace jargon with simpler terms
            jargon_replacements = {
                'leverage': 'use',
                'synergy': 'collaboration',
                'optimize': 'improve',
                'utilize': 'use',
                'paradigm': 'approach'
            }

            for jargon, replacement in jargon_replacements.items():
                content = content.replace(f' {jargon} ', f' {replacement} ')

        if 'weak words' in item.comment.lower():
            # Replace weak words with stronger alternatives
            weak_word_replacements = {
                ' very ': ' ',
                ' really ': ' ',
                ' quite ': ' ',
                ' rather ': ' ',
                ' somewhat ': ' '
            }

            for weak_word, replacement in weak_word_replacements.items():
                content = content.replace(weak_word, replacement)

        return content

    def _fix_clarity_issue(self, content: str, item) -> str:
        """Fix a clarity issue in content."""
        # Simple clarity improvements
        if 'long sentence' in item.comment.lower():
            # Break long sentences (simplified)
            sentences = content.split('.')
            improved_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 30:  # Long sentence
                    # Simple sentence breaking
                    words = sentence.split()
                    mid_point = len(words) // 2
                    part1 = ' '.join(words[:mid_point])
                    part2 = ' '.join(words[mid_point:])
                    improved_sentences.append(f"{part1}. {part2}")
                else:
                    improved_sentences.append(sentence)

            content = '. '.join(improved_sentences)

        return content

    def _create_success_result(self,
                               content: str,
                               feedback_result: FeedbackOrchestrationResult,
                               revision_count: int,
                               improvement_history: List[Dict[str, Any]],
                               start_time: float,
                               status: str) -> RefinementResult:
        """Create a successful refinement result."""
        execution_time = time.time() - start_time

        # Generate learning data
        learning_data = {
            'revision_cycles': revision_count,
            'final_score': feedback_result.structured_feedback.overall_score,
            'improvement_history': improvement_history,
            'execution_time': execution_time,
            'quality_metrics': feedback_result.quality_assessment
        }

        # Update learning system
        self.learning_system._update_learning_data(learning_data)

        return RefinementResult(
            final_content=content,
            final_score=feedback_result.structured_feedback.overall_score,
            revision_cycles=revision_count,
            improvement_history=improvement_history,
            quality_metrics=feedback_result.quality_assessment,
            learning_data=learning_data,
            status=status
        )

    def _create_error_result(self,
                             error_message: str,
                             revision_count: int,
                             improvement_history: List[Dict[str,
                                                            Any]]) -> RefinementResult:
        """Create an error result."""
        return RefinementResult(
            final_content="",
            final_score=0.0,
            revision_cycles=revision_count,
            improvement_history=improvement_history,
            quality_metrics={'error': error_message},
            learning_data={'error': error_message},
            status='error'
        )

    def get_refinement_analytics(
            self, result: RefinementResult) -> Dict[str, Any]:
        """Get analytics for the refinement process."""
        if result.status == 'error':
            return {
                'error': result.quality_metrics.get(
                    'error', 'Unknown error')}

        # Calculate improvement metrics
        if len(result.improvement_history) > 1:
            initial_score = result.improvement_history[0]['score']
            final_score = result.final_score
            improvement = final_score - initial_score
            improvement_percentage = (
                improvement /
                initial_score *
                100) if initial_score > 0 else 0
        else:
            improvement = 0.0
            improvement_percentage = 0.0

        analytics = {
            'status': result.status,
            'revision_cycles': result.revision_cycles,
            'final_score': result.final_score,
            'improvement': improvement,
            'improvement_percentage': improvement_percentage,
            'execution_time': result.learning_data.get(
                'execution_time',
                0),
            'quality_level': self._determine_quality_level(
                result.final_score),
            'efficiency': self._calculate_efficiency(result),
            'improvement_trend': self._analyze_improvement_trend(
                result.improvement_history)}

        return analytics

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

    def _calculate_efficiency(self, result: RefinementResult) -> float:
        """Calculate refinement efficiency."""
        if result.revision_cycles == 0:
            return 1.0

        # Efficiency based on improvement per cycle
        if len(result.improvement_history) > 1:
            total_improvement = result.final_score - \
                result.improvement_history[0]['score']
            efficiency = total_improvement / result.revision_cycles
            return min(1.0, efficiency)

        return 0.5  # Default efficiency

    def _analyze_improvement_trend(
            self, improvement_history: List[Dict[str, Any]]) -> str:
        """Analyze improvement trend across cycles."""
        if len(improvement_history) < 2:
            return "insufficient_data"

        scores = [record['score'] for record in improvement_history]

        # Calculate trend
        improvements = []
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i - 1]
            improvements.append(improvement)

        avg_improvement = sum(improvements) / len(improvements)

        if avg_improvement > 0.05:
            return "strong_improvement"
        elif avg_improvement > 0.02:
            return "moderate_improvement"
        elif avg_improvement > 0:
            return "slight_improvement"
        else:
            return "no_improvement"
