"""
Tests for Phase 4 Feedback System

This module tests the feedback orchestrator and refinement loop
implemented in Phase 4 of the enhanced agent architecture.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.campaign_context import CampaignContext
from core.execution_state import ExecutionState
from core.feedback_orchestrator import FeedbackOrchestrationResult, FeedbackOrchestrator
from core.feedback_system import IssueType, RequiredAction, Severity
from core.refinement_loop import RefinementLoop, RefinementResult


class TestFeedbackOrchestrator:
    """Test cases for the FeedbackOrchestrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = FeedbackOrchestrator()
        self.context = CampaignContext.create_default_context()
        self.execution_state = ExecutionState(workflow_id="test-workflow")
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert hasattr(self.orchestrator, 'linter')
        assert hasattr(self.orchestrator, 'search_tool')
    
    def test_orchestrate_feedback_basic(self):
        """Test basic feedback orchestration."""
        content = "This is a test sentence. It has some basic content."
        result = self.orchestrator.orchestrate_feedback(content, self.context, self.execution_state)
        
        assert isinstance(result, FeedbackOrchestrationResult)
        assert hasattr(result, 'structured_feedback')
        assert hasattr(result, 'linter_result')
        assert hasattr(result, 'search_verifications')
        assert hasattr(result, 'improvement_recommendations')
        assert hasattr(result, 'quality_assessment')
        assert hasattr(result, 'next_actions')
    
    def test_orchestrate_feedback_with_issues(self):
        """Test feedback orchestration with content issues."""
        content = "The project was optimized to leverage synergies.  It has  double  spaces."
        result = self.orchestrator.orchestrate_feedback(content, self.context, self.execution_state)
        
        assert result.structured_feedback.overall_score < 1.0
        assert len(result.structured_feedback.feedback_items) > 0
        assert len(result.improvement_recommendations) > 0
    
    def test_generate_structured_feedback(self):
        """Test structured feedback generation."""
        content = "Test content with issues."
        
        # Mock linter result
        from tools.grammar_linter import GrammarIssue, LinterResult, StyleIssue
        
        mock_linter_result = LinterResult(
            grammar_issues=[
                GrammarIssue(
                    message="Remove extra spaces",
                    offset=10,
                    error_length=2,
                    rule_id="DOUBLE_SPACE",
                    category="grammar",
                    severity="MEDIUM",
                    suggestion="Remove extra spaces",
                    context="Test content with  issues."
                )
            ],
            style_issues=[
                StyleIssue(
                    message="Style issue: Jargon",
                    offset=20,
                    error_length=8,
                    rule_id="JARGON",
                    category="style",
                    severity="LOW",
                    suggestion="Replace jargon with clearer language",
                    context="Test content with jargon issues."
                )
            ],
            overall_score=0.75,
            grammar_score=0.8,
            style_score=0.7,
            total_issues=2,
            suggestions=["Review grammar and punctuation"],
            summary={'word_count': 5, 'sentence_count': 1, 'average_sentence_length': 5.0}
        )
        
        feedback = self.orchestrator._generate_structured_feedback(
            content, mock_linter_result, self.context
        )
        
        assert feedback.overall_score == 0.75
        assert len(feedback.feedback_items) == 2
        assert feedback.sub_scores['grammar'] == 0.8
        assert feedback.sub_scores['style'] == 0.7
    
    def test_extract_claims(self):
        """Test claim extraction from content."""
        content = "Research shows that AI is improving. Studies indicate better performance."
        claims = self.orchestrator._extract_claims(content)
        
        assert len(claims) > 0
        assert any("research shows" in claim.lower() for claim in claims)
        assert any("studies indicate" in claim.lower() for claim in claims)
    
    def test_calculate_verification_score(self):
        """Test verification score calculation."""
        from tools.enhanced_search import SearchResult

        # Mock search results
        mock_results = [
            SearchResult(
                title="Test Result 1",
                url="https://example.com",
                snippet="Test snippet",
                source="test",
                confidence_score=0.8,
                relevance_score=0.7,
                freshness_score=0.6,
                authority_score=0.5,
                overall_score=0.65,
                metadata={},
                timestamp=datetime.now()
            ),
            SearchResult(
                title="Test Result 2",
                url="https://example2.com",
                snippet="Test snippet 2",
                source="test",
                confidence_score=0.9,
                relevance_score=0.8,
                freshness_score=0.7,
                authority_score=0.6,
                overall_score=0.75,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        score = self.orchestrator._calculate_verification_score(mock_results)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good score for these results
    
    def test_assess_clarity(self):
        """Test clarity assessment."""
        short_sentence = "This is a short sentence."
        long_sentence = "This is a very long sentence that contains many words and goes on for quite a while to test the clarity assessment functionality."
        
        short_score = self.orchestrator._assess_clarity(short_sentence)
        long_score = self.orchestrator._assess_clarity(long_sentence)
        
        assert short_score > long_score  # Shorter sentence should score higher
        assert 0.0 <= short_score <= 1.0
        assert 0.0 <= long_score <= 1.0
    
    def test_assess_engagement(self):
        """Test engagement assessment."""
        engaging_content = "You should consider this approach. Imagine the possibilities."
        neutral_content = "The approach is considered. The possibilities are examined."
        
        engaging_score = self.orchestrator._assess_engagement(engaging_content, self.context)
        neutral_score = self.orchestrator._assess_engagement(neutral_content, self.context)
        
        assert engaging_score > neutral_score  # Engaging content should score higher
        assert 0.0 <= engaging_score <= 1.0
        assert 0.0 <= neutral_score <= 1.0
    
    def test_assess_brand_compliance(self):
        """Test brand compliance assessment."""
        # Test with forbidden terminology
        self.context.forbidden_terminology = ["forbidden_term"]
        content_with_forbidden = "This contains forbidden_term which should be avoided."
        content_without_forbidden = "This contains no forbidden terms."
        
        forbidden_score = self.orchestrator._assess_brand_compliance(content_with_forbidden, self.context)
        clean_score = self.orchestrator._assess_brand_compliance(content_without_forbidden, self.context)
        
        assert clean_score > forbidden_score  # Clean content should score higher
        assert 0.0 <= forbidden_score <= 1.0
        assert 0.0 <= clean_score <= 1.0
    
    def test_determine_required_action(self):
        """Test required action determination."""
        from core.feedback_system import FeedbackItem

        # Test with high priority issues
        high_priority_items = [
            FeedbackItem(
                text_snippet="test",
                issue_type=IssueType.GRAMMAR,
                comment="High priority issue",
                required_action=RequiredAction.REVISION,
                severity=Severity.HIGH
            )
        ]
        
        action = self.orchestrator._determine_required_action(high_priority_items, 0.6, self.context)
        assert action == RequiredAction.REVISION
        
        # Test with low score
        action = self.orchestrator._determine_required_action([], 0.5, self.context)
        assert action == RequiredAction.REVISION
        
        # Test with no issues
        action = self.orchestrator._determine_required_action([], 0.9, self.context)
        assert action == RequiredAction.NO_ACTION
    
    def test_generate_summary(self):
        """Test summary generation."""
        from core.feedback_system import FeedbackItem
        
        feedback_items = [
            FeedbackItem(
                text_snippet="test1",
                issue_type=IssueType.GRAMMAR,
                comment="High priority issue",
                required_action=RequiredAction.REVISION,
                severity=Severity.HIGH
            ),
            FeedbackItem(
                text_snippet="test2",
                issue_type=IssueType.STYLE,
                comment="Medium priority issue",
                required_action=RequiredAction.STYLE_ADJUSTMENT,
                severity=Severity.MEDIUM
            )
        ]
        
        summary = self.orchestrator._generate_summary(feedback_items, 0.75)
        
        assert "0.75" in summary
        assert "1" in summary  # Should mention high priority issues
        assert "1" in summary  # Should mention medium priority issues
    
    def test_error_handling(self):
        """Test error handling in feedback orchestration."""
        with patch.object(self.orchestrator.linter, 'check_content', side_effect=Exception("Test error")):
            result = self.orchestrator.orchestrate_feedback("test content", self.context, self.execution_state)
            
            assert result.structured_feedback.overall_score == 0.0
            assert "Error during analysis" in result.structured_feedback.summary

class TestRefinementLoop:
    """Test cases for the RefinementLoop class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.refinement_loop = RefinementLoop(max_revision_cycles=3)
        self.context = CampaignContext.create_default_context()
        self.execution_state = ExecutionState(workflow_id="test-workflow")
    
    def test_initialization(self):
        """Test refinement loop initialization."""
        assert self.refinement_loop is not None
        assert hasattr(self.refinement_loop, 'feedback_orchestrator')
        assert hasattr(self.refinement_loop, 'learning_system')
        assert self.refinement_loop.max_revision_cycles == 3
    
    def test_execute_refinement_basic(self):
        """Test basic refinement execution."""
        content = "This is a test sentence. It has some basic content."
        result = self.refinement_loop.execute_refinement(content, self.context, self.execution_state)
        
        assert isinstance(result, RefinementResult)
        assert hasattr(result, 'final_content')
        assert hasattr(result, 'final_score')
        assert hasattr(result, 'revision_cycles')
        assert hasattr(result, 'improvement_history')
        assert hasattr(result, 'quality_metrics')
        assert hasattr(result, 'learning_data')
        assert hasattr(result, 'status')
    
    def test_meets_quality_threshold(self):
        """Test quality threshold checking."""
        from core.feedback_orchestrator import FeedbackOrchestrationResult
        from core.feedback_system import StructuredFeedback

        # Mock feedback result with good score
        mock_feedback = StructuredFeedback(
            overall_score=0.85,
            sub_scores={},
            feedback_items=[],
            required_action=RequiredAction.NO_ACTION,
            revision_cycles=0,
            summary="Good content",
            improvement_suggestions=[],
            quality_metrics={}
        )
        
        mock_result = FeedbackOrchestrationResult(
            structured_feedback=mock_feedback,
            linter_result=None,
            search_verifications=[],
            improvement_recommendations=[],
            quality_assessment={'high_priority_issues': 0},
            next_actions=[]
        )
        
        meets_threshold = self.refinement_loop._meets_quality_threshold(mock_result, self.context)
        assert meets_threshold == True
    
    def test_handle_revision(self):
        """Test revision handling."""
        content = "The project was optimized to leverage synergies."
        
        # Mock feedback result with style issues
        from core.feedback_orchestrator import FeedbackOrchestrationResult
        from core.feedback_system import FeedbackItem, StructuredFeedback
        
        feedback_items = [
            FeedbackItem(
                text_snippet="was optimized",
                issue_type=IssueType.STYLE,
                comment="Style issue: Passive Voice",
                required_action=RequiredAction.STYLE_ADJUSTMENT,
                severity=Severity.MEDIUM,
                suggestion="Use active voice"
            )
        ]
        
        mock_feedback = StructuredFeedback(
            overall_score=0.7,
            sub_scores={},
            feedback_items=feedback_items,
            required_action=RequiredAction.REVISION,
            revision_cycles=0,
            summary="Needs revision",
            improvement_suggestions=[],
            quality_metrics={}
        )
        
        mock_result = FeedbackOrchestrationResult(
            structured_feedback=mock_feedback,
            linter_result=None,
            search_verifications=[],
            improvement_recommendations=[],
            quality_assessment={'high_priority_issues': 0},
            next_actions=[]
        )
        
        revised_content = self.refinement_loop._handle_revision(content, mock_result)
        
        # Should have applied some fixes
        assert revised_content != content
        assert "optimized" in revised_content  # Should have removed "was"
    
    def test_fix_grammar_issue(self):
        """Test grammar issue fixing."""
        content = "This sentence has  double  spaces."
        
        from core.feedback_system import FeedbackItem
        
        grammar_item = FeedbackItem(
            text_snippet="  double  ",
            issue_type=IssueType.GRAMMAR,
            comment="Remove extra spaces",
            required_action=RequiredAction.REVISION,
            severity=Severity.MEDIUM
        )
        
        fixed_content = self.refinement_loop._fix_grammar_issue(content, grammar_item)
        
        assert "  " not in fixed_content  # Should remove double spaces
        assert "double" in fixed_content  # Should preserve the word
    
    def test_fix_style_issue(self):
        """Test style issue fixing."""
        content = "The project was optimized to leverage synergies."
        
        from core.feedback_system import FeedbackItem
        
        style_item = FeedbackItem(
            text_snippet="was optimized",
            issue_type=IssueType.STYLE,
            comment="Style issue: Passive Voice",
            required_action=RequiredAction.STYLE_ADJUSTMENT,
            severity=Severity.MEDIUM
        )
        
        fixed_content = self.refinement_loop._fix_style_issue(content, style_item)
        
        assert "was optimized" not in fixed_content  # Should remove passive voice
        assert "optimized" in fixed_content  # Should preserve the action
    
    def test_get_refinement_analytics(self):
        """Test refinement analytics generation."""
        # Create a mock refinement result
        result = RefinementResult(
            final_content="Final content",
            final_score=0.85,
            revision_cycles=2,
            improvement_history=[
                {'cycle': 1, 'score': 0.7, 'issues_fixed': 3, 'action_taken': 'REVISION'},
                {'cycle': 2, 'score': 0.85, 'issues_fixed': 1, 'action_taken': 'STYLE_ADJUSTMENT'}
            ],
            quality_metrics={'overall_score': 0.85},
            learning_data={'execution_time': 10.5},
            status='completed'
        )
        
        analytics = self.refinement_loop.get_refinement_analytics(result)
        
        assert analytics['status'] == 'completed'
        assert analytics['revision_cycles'] == 2
        assert analytics['final_score'] == 0.85
        assert analytics['improvement'] > 0
        assert analytics['execution_time'] == 10.5
        assert 'quality_level' in analytics
        assert 'efficiency' in analytics
        assert 'improvement_trend' in analytics
    
    def test_error_handling(self):
        """Test error handling in refinement loop."""
        with patch.object(self.refinement_loop.feedback_orchestrator, 'orchestrate_feedback', 
                         side_effect=Exception("Test error")):
            result = self.refinement_loop.execute_refinement("test content", self.context, self.execution_state)
            
            assert result.status == 'error'
            assert result.final_score == 0.0
            assert "Test error" in result.quality_metrics.get('error', '')

class TestIntegration:
    """Integration tests for Phase 4 feedback system."""
    
    def test_feedback_orchestrator_with_refinement_loop(self):
        """Test integration between feedback orchestrator and refinement loop."""
        orchestrator = FeedbackOrchestrator()
        refinement_loop = RefinementLoop(max_revision_cycles=2)
        context = CampaignContext.create_default_context()
        execution_state = ExecutionState(workflow_id="test-workflow")
        
        # Test content with issues
        content = "The project was optimized to leverage synergies.  It has  double  spaces."
        
        # Execute refinement
        result = refinement_loop.execute_refinement(content, context, execution_state)
        
        assert result.status in ['completed', 'max_cycles_reached']
        assert result.final_content != content  # Should have been improved
        assert result.final_score > 0.0
        assert result.revision_cycles >= 0
    
    def test_learning_integration(self):
        """Test learning system integration."""
        refinement_loop = RefinementLoop()
        context = CampaignContext.create_default_context()
        execution_state = ExecutionState(workflow_id="test-workflow")
        
        content = "Test content for learning integration."
        result = refinement_loop.execute_refinement(content, context, execution_state)
        
        # Check that learning data was generated
        assert 'learning_data' in result.__dict__
        assert result.learning_data is not None
        assert 'revision_cycles' in result.learning_data
        assert 'final_score' in result.learning_data
        assert 'execution_time' in result.learning_data

if __name__ == "__main__":
    pytest.main([__file__]) 