"""
Tests for the enhanced agent architecture core components.

This module tests the new core data structures including CampaignContext,
ExecutionState, and StructuredFeedback.
"""

import time

import pytest

from src.core.campaign_context import CampaignContext
from src.core.config_manager import ConfigManager
from src.core.execution_state import ExecutionState, TaskResult
from src.core.feedback_system import (
    FeedbackItem,
    IssueType,
    RequiredAction,
    Severity,
    StructuredFeedback,
)


class TestCampaignContext:
    """Test CampaignContext functionality."""
    
    def test_create_default_context(self):
        """Test creating a default campaign context."""
        context = CampaignContext.create_default_context()
        
        assert context.content_style['tone'] == 'professional'
        assert 'Increase reader engagement' in context.strategic_goals
        assert context.quality_thresholds['minimum'] == 0.7
        assert isinstance(context.learning_data, dict)
    
    def test_update_learning_data(self):
        """Test updating learning data."""
        context = CampaignContext.create_default_context()
        original_updated_at = context.updated_at
        
        time.sleep(0.1)  # Ensure timestamp difference
        context.update_learning_data({'new_pattern': 'test'})
        
        assert 'new_pattern' in context.learning_data
        assert context.updated_at > original_updated_at
    
    def test_quality_threshold_getter(self):
        """Test getting quality thresholds."""
        context = CampaignContext.create_default_context()
        
        assert context.get_quality_threshold('minimum') == 0.7
        assert context.get_quality_threshold('nonexistent') == 0.7  # Default
    
    def test_forbidden_terminology_check(self):
        """Test forbidden terminology checking."""
        context = CampaignContext.create_default_context()
        context.forbidden_terminology = ['bad_word', 'inappropriate']
        
        assert context.is_forbidden_term('bad_word') == True
        assert context.is_forbidden_term('good_word') == False
        assert context.is_forbidden_term('BAD_WORD') == True  # Case insensitive
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        context = CampaignContext.create_default_context()
        context_dict = context.to_dict()
        
        # Test that we can recreate from dict
        recreated_context = CampaignContext.from_dict(context_dict)
        
        assert recreated_context.content_style == context.content_style
        assert recreated_context.strategic_goals == context.strategic_goals
        assert recreated_context.quality_thresholds == context.quality_thresholds


class TestExecutionState:
    """Test ExecutionState functionality."""
    
    def test_create_execution_state(self):
        """Test creating an execution state."""
        state = ExecutionState(workflow_id="test_workflow")
        
        assert state.workflow_id == "test_workflow"
        assert state.current_phase == "initialized"
        assert len(state.task_results) == 0
    
    def test_add_task_result(self):
        """Test adding task results."""
        state = ExecutionState(workflow_id="test_workflow")
        task_result = TaskResult(
            task_id="task1",
            task_type="research",
            status="completed"
        )
        
        state.add_task_result(task_result)
        
        assert "task1" in state.task_results
        assert state.task_results["task1"] == task_result
    
    def test_update_phase(self):
        """Test updating workflow phase."""
        state = ExecutionState(workflow_id="test_workflow")
        original_updated = state.last_updated
        
        time.sleep(0.1)  # Ensure timestamp difference
        state.update_phase("research")
        
        assert state.current_phase == "research"
        assert state.last_updated > original_updated
    
    def test_increment_revision_cycle(self):
        """Test incrementing revision cycles."""
        state = ExecutionState(workflow_id="test_workflow")
        
        assert state.increment_revision_cycle("task1") == 1
        assert state.increment_revision_cycle("task1") == 2
        assert state.revision_cycles["task1"] == 2
    
    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        state = ExecutionState(workflow_id="test_workflow")
        
        # Add completed task
        completed_task = TaskResult(
            task_id="task1",
            task_type="research",
            status="completed"
        )
        completed_task.mark_completed({"result": "success"})
        state.add_task_result(completed_task)
        
        # Add failed task
        failed_task = TaskResult(
            task_id="task2",
            task_type="writing",
            status="failed"
        )
        failed_task.mark_failed("error occurred")
        state.add_task_result(failed_task)
        
        completed_tasks = state.get_completed_tasks()
        failed_tasks = state.get_failed_tasks()
        
        assert len(completed_tasks) == 1
        assert len(failed_tasks) == 1
        assert completed_tasks[0].task_id == "task1"
        assert failed_tasks[0].task_id == "task2"
    
    def test_average_quality_score(self):
        """Test calculating average quality score."""
        state = ExecutionState(workflow_id="test_workflow")
        
        # No scores yet
        assert state.get_average_quality_score() == 0.0
        
        # Add some scores
        state.update_quality_score("task1", 0.8)
        state.update_quality_score("task2", 0.6)
        
        assert state.get_average_quality_score() == 0.7
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        state = ExecutionState(workflow_id="test_workflow")
        state.update_phase("research")
        state.update_quality_score("task1", 0.8)
        
        state_dict = state.to_dict()
        recreated_state = ExecutionState.from_dict(state_dict)
        
        assert recreated_state.workflow_id == state.workflow_id
        assert recreated_state.current_phase == state.current_phase
        assert recreated_state.quality_scores == state.quality_scores


class TestStructuredFeedback:
    """Test StructuredFeedback functionality."""
    
    def test_create_feedback_item(self):
        """Test creating a feedback item."""
        item = FeedbackItem(
            text_snippet="This is a test sentence.",
            issue_type=IssueType.GRAMMAR,
            comment="Grammar issue detected",
            required_action=RequiredAction.REVISION,
            severity=Severity.HIGH,
            suggestion="Consider rephrasing"
        )
        
        assert item.text_snippet == "This is a test sentence."
        assert item.issue_type == IssueType.GRAMMAR
        assert item.severity == Severity.HIGH
    
    def test_create_structured_feedback(self):
        """Test creating structured feedback."""
        feedback_item = FeedbackItem(
            text_snippet="Test content",
            issue_type=IssueType.STYLE,
            comment="Style issue",
            required_action=RequiredAction.STYLE_ADJUSTMENT,
            severity=Severity.MEDIUM
        )
        
        feedback = StructuredFeedback(
            overall_score=0.75,
            sub_scores={'grammar': 0.8, 'style': 0.7},
            feedback_items=[feedback_item],
            required_action=RequiredAction.STYLE_ADJUSTMENT,
            revision_cycles=1,
            summary="Good content with minor issues",
            improvement_suggestions=["Improve style consistency"],
            quality_metrics={'readability': 0.8}
        )
        
        assert feedback.overall_score == 0.75
        assert len(feedback.feedback_items) == 1
        assert feedback.required_action == RequiredAction.STYLE_ADJUSTMENT
    
    def test_get_high_priority_items(self):
        """Test getting high priority feedback items."""
        high_item = FeedbackItem(
            text_snippet="High priority issue",
            issue_type=IssueType.ACCURACY,
            comment="Accuracy issue",
            required_action=RequiredAction.RESEARCH_VERIFICATION,
            severity=Severity.HIGH
        )
        
        low_item = FeedbackItem(
            text_snippet="Low priority issue",
            issue_type=IssueType.STYLE,
            comment="Style issue",
            required_action=RequiredAction.STYLE_ADJUSTMENT,
            severity=Severity.LOW
        )
        
        feedback = StructuredFeedback(
            overall_score=0.7,
            sub_scores={},
            feedback_items=[high_item, low_item],
            required_action=RequiredAction.REVISION,
            revision_cycles=1,
            summary="Test",
            improvement_suggestions=[],
            quality_metrics={}
        )
        
        high_priority_items = feedback.get_high_priority_items()
        assert len(high_priority_items) == 1
        assert high_priority_items[0].severity == Severity.HIGH
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        feedback_item = FeedbackItem(
            text_snippet="Test content",
            issue_type=IssueType.GRAMMAR,
            comment="Grammar issue",
            required_action=RequiredAction.REVISION,
            severity=Severity.HIGH
        )
        
        feedback = StructuredFeedback(
            overall_score=0.8,
            sub_scores={'grammar': 0.8},
            feedback_items=[feedback_item],
            required_action=RequiredAction.REVISION,
            revision_cycles=1,
            summary="Test feedback",
            improvement_suggestions=["Fix grammar"],
            quality_metrics={}
        )
        
        feedback_dict = feedback.to_dict()
        recreated_feedback = StructuredFeedback.from_dict(feedback_dict)
        
        assert recreated_feedback.overall_score == feedback.overall_score
        assert len(recreated_feedback.feedback_items) == len(feedback.feedback_items)
        assert recreated_feedback.required_action == feedback.required_action


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_create_config_manager(self):
        """Test creating a config manager."""
        config_manager = ConfigManager()
        
        assert config_manager.default_context_id == "default"
        assert isinstance(config_manager.campaign_contexts, dict)
    
    def test_load_default_context(self):
        """Test loading the default context."""
        config_manager = ConfigManager()
        context = config_manager.load_campaign_context("default")
        
        assert isinstance(context, CampaignContext)
        assert context.content_style['tone'] == 'professional'
    
    def test_load_custom_context(self):
        """Test loading a custom context."""
        config_manager = ConfigManager()
        context = config_manager.load_campaign_context("technical")
        
        assert isinstance(context, CampaignContext)
        assert context.content_style['tone'] == 'technical'
        assert context.audience_persona['demographics'] == 'technical professionals aged 25-50'
    
    def test_list_contexts(self):
        """Test listing available contexts."""
        config_manager = ConfigManager()
        contexts = config_manager.list_campaign_contexts()
        
        assert isinstance(contexts, list)
        assert "default" in contexts
    
    def test_context_summary(self):
        """Test getting context summary."""
        config_manager = ConfigManager()
        summary = config_manager.get_context_summary("default")
        
        assert "context_id" in summary
        assert "content_style" in summary
        assert "strategic_goals" in summary
        assert summary["context_id"] == "default"


if __name__ == "__main__":
    pytest.main([__file__]) 