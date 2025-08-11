#!/usr/bin/env python3
"""
Enhanced Architecture Demo

This script demonstrates the new enhanced agent architecture components
including CampaignContext, ExecutionState, StructuredFeedback, and
ConfigManager.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.campaign_context import CampaignContext
from src.core.execution_state import ExecutionState, TaskResult
from src.core.feedback_system import StructuredFeedback, FeedbackItem, IssueType, Severity, RequiredAction, FeedbackGenerator
from src.core.config_manager import ConfigManager
from src.core.learning_system import LearningSystem
import time


def demo_campaign_context():
    """Demonstrate CampaignContext functionality."""
    print("=== CampaignContext Demo ===")
    
    # Create a default context
    context = CampaignContext.create_default_context()
    print(f"Default context tone: {context.content_style['tone']}")
    print(f"Strategic goals: {context.strategic_goals}")
    print(f"Quality threshold: {context.quality_thresholds['minimum']}")
    
    # Update learning data
    context.update_learning_data({
        'successful_patterns': [{'theme': 'technology', 'score': 0.9}],
        'performance_trends': {'quality_score': [0.8, 0.85, 0.9]}
    })
    print(f"Updated learning data: {list(context.learning_data.keys())}")
    
    # Check forbidden terminology
    context.forbidden_terminology = ['outdated', 'legacy']
    print(f"Is 'outdated' forbidden? {context.is_forbidden_term('outdated')}")
    print(f"Is 'modern' forbidden? {context.is_forbidden_term('modern')}")
    
    print()


def demo_execution_state():
    """Demonstrate ExecutionState functionality."""
    print("=== ExecutionState Demo ===")
    
    # Create execution state
    state = ExecutionState(workflow_id="demo_workflow")
    print(f"Workflow ID: {state.workflow_id}")
    print(f"Current phase: {state.current_phase}")
    
    # Add task results
    research_task = TaskResult(
        task_id="research_1",
        task_type="research",
        status="completed"
    )
    research_task.mark_completed({"sources": ["source1", "source2"]}, execution_time=120.5)
    research_task.update_quality_score(0.85)
    
    state.add_task_result(research_task)
    state.update_phase("writing")
    state.update_quality_score("research_1", 0.85)
    
    print(f"Updated phase: {state.current_phase}")
    print(f"Completed tasks: {len(state.get_completed_tasks())}")
    print(f"Average quality score: {state.get_average_quality_score():.2f}")
    print(f"Total execution time: {state.get_total_execution_time():.2f}s")
    
    print()


def demo_structured_feedback():
    """Demonstrate StructuredFeedback functionality."""
    print("=== StructuredFeedback Demo ===")
    
    # Create feedback items
    grammar_item = FeedbackItem(
        text_snippet="The company is doing good.",
        issue_type=IssueType.GRAMMAR,
        comment="Use 'well' instead of 'good'",
        required_action=RequiredAction.REVISION,
        severity=Severity.MEDIUM,
        suggestion="Change 'good' to 'well'"
    )
    
    style_item = FeedbackItem(
        text_snippet="This is a very, very long sentence.",
        issue_type=IssueType.STYLE,
        comment="Sentence is too long",
        required_action=RequiredAction.STYLE_ADJUSTMENT,
        severity=Severity.LOW,
        suggestion="Break into shorter sentences"
    )
    
    # Create structured feedback
    feedback = StructuredFeedback(
        overall_score=0.78,
        sub_scores={'grammar': 0.8, 'style': 0.75, 'clarity': 0.8},
        feedback_items=[grammar_item, style_item],
        required_action=RequiredAction.STYLE_ADJUSTMENT,
        revision_cycles=1,
        summary="Good content with minor style improvements needed",
        improvement_suggestions=["Fix grammar issues", "Improve sentence structure"],
        quality_metrics={'readability': 0.75}
    )
    
    print(f"Overall score: {feedback.overall_score}")
    print(f"High priority items: {len(feedback.get_high_priority_items())}")
    print(f"Required action: {feedback.required_action.value}")
    print(f"Summary: {feedback.summary}")
    
    print()


def demo_config_manager():
    """Demonstrate ConfigManager functionality."""
    print("=== ConfigManager Demo ===")
    
    # Create config manager
    config_manager = ConfigManager()
    
    # List available contexts
    contexts = config_manager.list_campaign_contexts()
    print(f"Available contexts: {contexts}")
    
    # Load different context types
    default_context = config_manager.load_campaign_context("default")
    technical_context = config_manager.load_campaign_context("technical")
    business_context = config_manager.load_campaign_context("business")
    
    print(f"Default context tone: {default_context.content_style['tone']}")
    print(f"Technical context tone: {technical_context.content_style['tone']}")
    print(f"Business context tone: {business_context.content_style['tone']}")
    
    # Get context summary
    summary = config_manager.get_context_summary("default")
    print(f"Default context summary: {summary['content_style']['tone']}")
    
    print()


def demo_learning_system():
    """Demonstrate LearningSystem functionality."""
    print("=== LearningSystem Demo ===")
    
    # Create learning system
    learning_system = LearningSystem()
    
    # Create a context
    context = CampaignContext.create_default_context()
    
    # Simulate performance data
    performance_data = {
        'execution_time': 180.5,
        'quality_score': 0.85,
        'revision_cycles': 2,
        'user_satisfaction': 0.8,
        'theme': 'technology',
        'style': 'informative'
    }
    
    # Update context with learning data
    learning_system.update_campaign_context(context, performance_data)
    
    # Generate improvement recommendations
    recommendations = learning_system.generate_improvement_recommendations(context)
    print(f"Improvement recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Get performance summary
    summary = learning_system.get_performance_summary()
    print(f"Performance status: {summary['status']}")
    
    print()


def demo_feedback_generator():
    """Demonstrate FeedbackGenerator functionality."""
    print("=== FeedbackGenerator Demo ===")
    
    # Create feedback generator
    generator = FeedbackGenerator()
    
    # Simulate content analysis results
    analysis_results = {
        'grammar_issues': [
            {
                'text': 'The company is doing good.',
                'message': 'Use "well" instead of "good"',
                'suggestion': 'Change "good" to "well"',
                'type': 'grammar',
                'confidence': 0.9
            }
        ],
        'style_issues': [
            {
                'text': 'This is a very, very long sentence.',
                'message': 'Sentence is too long',
                'suggestion': 'Break into shorter sentences',
                'type': 'style',
                'confidence': 0.7
            }
        ],
        'clarity_issues': [
            {
                'text': 'The thing that we need to do is...',
                'message': 'Wordy phrase',
                'suggestion': 'Simplify to "We need to..."',
                'type': 'wordiness',
                'confidence': 0.6
            }
        ]
    }
    
    # Generate structured feedback
    content = "The company is doing good. This is a very, very long sentence. The thing that we need to do is improve our processes."
    feedback = generator.generate_feedback(content, analysis_results)
    
    print(f"Overall score: {feedback.overall_score:.2f}")
    print(f"Required action: {feedback.required_action.value}")
    print(f"Feedback items: {len(feedback.feedback_items)}")
    print(f"Summary: {feedback.summary}")
    
    print()


def main():
    """Run all demonstrations."""
    print("Enhanced Agent Architecture Demo")
    print("=" * 40)
    print()
    
    try:
        demo_campaign_context()
        demo_execution_state()
        demo_structured_feedback()
        demo_config_manager()
        demo_learning_system()
        demo_feedback_generator()
        
        print("All demonstrations completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("- CampaignContext: Long-term campaign configuration and learning")
        print("- ExecutionState: Short-term workflow execution tracking")
        print("- StructuredFeedback: Detailed, actionable feedback system")
        print("- ConfigManager: Campaign context management and persistence")
        print("- LearningSystem: Performance tracking and improvement analysis")
        print("- FeedbackGenerator: Automated feedback generation from analysis")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 