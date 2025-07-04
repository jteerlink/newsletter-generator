"""
Tests for Enhanced Coordination and Quality/Feedback System

Tests the complete enhanced system including:
- ManagerAgent coordination
- Quality scoring and assessment
- Feedback collection and learning
- Multi-agent coordination
- Performance monitoring
"""

import pytest
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.agents import (
    ManagerAgent, PlannerAgent, WriterAgent, EditorAgent, 
    ResearchAgent, EnhancedCrew, Task
)
from src.core.feedback_system import (
    FeedbackLogger, FeedbackAnalyzer, FeedbackLearningSystem, FeedbackEntry
)

class TestEnhancedAgentSystem:
    """Test the enhanced agent system with all new agents."""
    
    def test_manager_agent_creation(self):
        """Test ManagerAgent creation and capabilities."""
        manager = ManagerAgent()
        
        assert manager.name == "ManagerAgent"
        assert "Workflow Coordinator" in manager.role
        assert "orchestrate" in manager.goal.lower()
        assert "coordinating" in manager.backstory.lower()
        assert hasattr(manager, 'delegate_task')
        assert hasattr(manager, 'monitor_workflow_progress')
        assert hasattr(manager, 'optimize_workflow_sequence')
        assert hasattr(manager, 'delegation_history')
        assert hasattr(manager, 'agent_performance')
    
    def test_enhanced_editor_quality_scoring(self):
        """Test EditorAgent quality scoring capabilities."""
        editor = EditorAgent()
        
        # Test quality metric extraction
        sample_content = """
        # Test Content
        
        This is a test newsletter about artificial intelligence.
        It contains multiple paragraphs with detailed information.
        
        ## Key Points
        - Point one with details
        - Point two with more information  
        - Point three with comprehensive analysis
        
        The content is structured and informative.
        """
        
        metrics = editor.extract_quality_metrics(sample_content)
        
        # Test the updated metrics structure
        assert 'word_count' in metrics
        assert 'estimated_reading_time' in metrics
        assert 'content_depth' in metrics
        assert 'structure_score' in metrics
        assert 'engagement_indicators' in metrics
        assert 'technical_depth' in metrics
        assert 'example_richness' in metrics
        assert 'readability_score' in metrics
        
        # Verify metric values are reasonable
        assert metrics['word_count'] > 0
        assert metrics['estimated_reading_time'] > 0
        assert 0.0 <= metrics['content_depth'] <= 10.0
        assert 0.0 <= metrics['structure_score'] <= 10.0
        assert 0.0 <= metrics['engagement_indicators'] <= 10.0
        assert 0.0 <= metrics['readability_score'] <= 10.0
    
    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        editor = EditorAgent()
        
        sample_scores = {
            'clarity': 8.0,
            'accuracy': 7.5,
            'engagement': 6.5,
            'completeness': 7.0
        }
        
        quality_analysis = editor.calculate_quality_score({'scores': sample_scores})
        
        assert 'overall_score' in quality_analysis
        assert 'grade' in quality_analysis
        assert 'dimension_scores' in quality_analysis
        
        # Check overall score is reasonable weighted average
        assert 5.0 <= quality_analysis['overall_score'] <= 10.0
        assert quality_analysis['grade'] in ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement', 'Poor']
        
        # Check dimension details
        for dimension in sample_scores.keys():
            assert dimension in quality_analysis['dimension_scores']
            dim_score = quality_analysis['dimension_scores'][dimension]
            assert 'raw_score' in dim_score
            assert 'weight' in dim_score
            assert dim_score['raw_score'] == sample_scores[dimension]
    
    def test_enhanced_crew_coordination(self):
        """Test EnhancedCrew coordination capabilities."""
        # Create agents
        planner = PlannerAgent()
        researcher = ResearchAgent()
        writer = WriterAgent()
        
        # Create tasks
        tasks = [
            Task("Plan content structure", planner, "Focus on organization"),
            Task("Research the topic", researcher, "Find current information"),
            Task("Write the content", writer, "Create engaging content")
        ]
        
        # Create enhanced crew
        crew = EnhancedCrew(
            agents=[planner, researcher, writer],
            tasks=tasks,
            workflow_type="sequential"
        )
        
        assert len(crew.agents) == 3
        assert len(crew.tasks) == 3
        assert crew.workflow_type == "sequential"
        assert hasattr(crew, 'task_results')
        assert hasattr(crew, 'agent_performance')
        
        # Check that agent_performance structure exists (may be empty initially)
        assert isinstance(crew.agent_performance, dict)

class TestFeedbackSystem:
    """Test the feedback collection and learning system."""
    
    def setup_method(self):
        """Setup test feedback file."""
        self.test_feedback_file = "test_feedback.json"
        # Clean up any existing test file
        if os.path.exists(self.test_feedback_file):
            os.remove(self.test_feedback_file)
    
    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.test_feedback_file):
            os.remove(self.test_feedback_file)
    
    def test_feedback_logger_initialization(self):
        """Test FeedbackLogger initialization."""
        logger = FeedbackLogger(self.test_feedback_file)
        
        assert logger.feedback_file.name == self.test_feedback_file
        assert os.path.exists(self.test_feedback_file)
        
        # Check file structure
        with open(self.test_feedback_file, 'r') as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'created' in data
        assert 'total_entries' in data
        assert 'feedback_entries' in data
        assert data['total_entries'] == 0
        assert len(data['feedback_entries']) == 0
    
    def test_feedback_logging(self):
        """Test logging feedback entries."""
        logger = FeedbackLogger(self.test_feedback_file)
        
        # Log test feedback
        session_id = logger.log_feedback(
            topic="Test Topic",
            content="This is test content for feedback logging",
            user_rating="approved",
            quality_scores={"clarity": 8.0, "accuracy": 7.5},
            specific_feedback="Great content!",
            suggestions=["Add more examples"]
        )
        
        assert session_id.startswith("session_")
        
        # Verify data was saved
        history = logger.get_feedback_history()
        assert len(history) == 1
        
        entry = history[0]
        assert entry.newsletter_topic == "Test Topic"
        assert entry.user_rating == "approved"
        assert entry.quality_scores["clarity"] == 8.0
        assert entry.specific_feedback == "Great content!"
        assert "Add more examples" in entry.suggestions
    
    def test_feedback_analysis(self):
        """Test feedback analysis capabilities."""
        logger = FeedbackLogger(self.test_feedback_file)
        analyzer = FeedbackAnalyzer(logger)
        
        # Add sample feedback data
        test_data = [
            {
                "topic": "AI Ethics",
                "rating": "approved",
                "scores": {"clarity": 8.0, "accuracy": 8.5, "engagement": 7.5, "completeness": 8.0}
            },
            {
                "topic": "ML Algorithms", 
                "rating": "rejected",
                "scores": {"clarity": 5.0, "accuracy": 6.0, "engagement": 4.5, "completeness": 5.5}
            },
            {
                "topic": "Data Science",
                "rating": "needs_revision",
                "scores": {"clarity": 6.5, "accuracy": 7.0, "engagement": 6.0, "completeness": 6.5}
            }
        ]
        
        for data in test_data:
            logger.log_feedback(
                topic=data["topic"],
                content="Sample content",
                user_rating=data["rating"],
                quality_scores=data["scores"]
            )
        
        # Test rejection analysis
        analysis = analyzer.analyze_rejection_patterns()
        
        assert 'total_rejected' in analysis
        assert 'rejection_rate' in analysis
        assert analysis['total_rejected'] == 1
        assert analysis['rejection_rate'] == 1/3
        
        # Test improvement recommendations
        recommendations = analyzer.generate_improvement_recommendations(analysis)
        assert len(recommendations) > 0
        assert any("clarity" in rec.lower() for rec in recommendations)
    
    def test_learning_system_integration(self):
        """Test complete learning system integration."""
        learning_system = FeedbackLearningSystem(self.test_feedback_file)
        
        # Test non-interactive feedback collection
        session_id = learning_system.collect_user_feedback(
            topic="Integration Test",
            content="Test content for learning system",
            interactive=False
        )
        
        assert session_id.startswith("session_")
        
        # Test insights generation
        insights = learning_system.generate_learning_insights()
        
        assert 'rejection_analysis' in insights
        assert 'improvement_recommendations' in insights
        assert 'performance_trends' in insights
        assert 'summary' in insights
        
        summary = insights['summary']
        assert 'total_feedback_entries' in summary
        assert 'learning_status' in summary
        assert summary['total_feedback_entries'] >= 1

class TestSystemIntegration:
    """Test integration between all enhanced system components."""
    
    def test_manager_agent_delegation(self):
        """Test ManagerAgent task delegation capabilities."""
        manager = ManagerAgent()
        
        # Create actual agent objects for delegation
        available_agents = [
            ResearchAgent(),
            WriterAgent(), 
            PlannerAgent(),
            EditorAgent()
        ]
        
        # Test delegation decision making
        task_type = "research"  # Use the task type, not full description
        delegation = manager.delegate_task(task_type, "detailed research task", available_agents)
        
        # Check the actual return format
        assert 'agent' in delegation
        assert 'delegation_id' in delegation
        assert 'success' in delegation
        
        # Should successfully delegate to an appropriate agent
        assert delegation['success'] == True
        assert delegation['agent'] in available_agents
        assert isinstance(delegation['delegation_id'], int)
    
    def test_workflow_optimization(self):
        """Test workflow optimization logic."""
        manager = ManagerAgent()
        
        # Test with sample task objects (as the method expects)
        sample_tasks = [
            {"description": "Plan the newsletter structure", "type": "planning"},
            {"description": "Research current trends", "type": "research"}, 
            {"description": "Write engaging content", "type": "writing"},
            {"description": "Edit and review quality", "type": "editing"}
        ]
        
        optimization = manager.optimize_workflow_sequence(sample_tasks)
        
        # The method returns a list of workflow groups, not a dict
        assert isinstance(optimization, list)
        assert len(optimization) > 0
        
        # Each group should have group_type and tasks
        for group in optimization:
            assert 'group_type' in group
            assert 'tasks' in group
            assert group['group_type'] in ['parallel', 'sequential']
            assert isinstance(group['tasks'], list)
    
    def test_end_to_end_enhanced_workflow(self):
        """Test complete enhanced workflow (without LLM calls)."""
        # Create all agents
        manager = ManagerAgent()
        planner = PlannerAgent()
        researcher = ResearchAgent()
        writer = WriterAgent()
        editor = EditorAgent()
        
        agents = [manager, planner, researcher, writer, editor]
        
        # Create test tasks
        tasks = [
            Task("Plan newsletter", planner, "Create strategic plan"),
            Task("Research content", researcher, "Find current information"),
            Task("Write newsletter", writer, "Create engaging content"),
            Task("Edit and review", editor, "Ensure quality standards")
        ]
        
        # Create enhanced crew
        crew = EnhancedCrew(
            agents=agents,
            tasks=tasks,
            workflow_type="hierarchical"
        )
        
        # Verify crew setup
        assert len(crew.agents) == 5
        assert len(crew.tasks) == 4
        assert crew.workflow_type == "hierarchical"
        
        # Check that all agent types are present
        agent_types = [type(agent).__name__ for agent in crew.agents]
        assert 'ManagerAgent' in agent_types
        assert 'PlannerAgent' in agent_types
        assert 'ResearchAgent' in agent_types
        assert 'WriterAgent' in agent_types
        assert 'EditorAgent' in agent_types
    
    def test_quality_feedback_integration(self):
        """Test integration between quality scoring and feedback systems."""
        editor = EditorAgent()
        
        # Clean up test file first
        test_file = "test_integration_feedback.json"
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Initialize the learning system properly
        learning_system = FeedbackLearningSystem(test_file)
        
        try:
            # Generate quality analysis
            sample_content = "Test newsletter content with adequate quality for testing purposes."
            metrics = editor.extract_quality_metrics(sample_content)
            
            sample_scores = {'clarity': 7.0, 'accuracy': 6.5, 'engagement': 6.0, 'completeness': 7.5}
            quality_analysis = editor.calculate_quality_score({'scores': sample_scores})
            
            # Simulate feedback based on quality analysis
            rating = "approved" if quality_analysis['overall_score'] >= 7.0 else "needs_revision"
            
            session_id = learning_system.collect_user_feedback(
                topic="Quality Integration Test",
                content=sample_content,
                interactive=False
            )
            
            # Override with quality-based feedback
            learning_system.logger.log_feedback(
                topic="Quality Integration Test",
                content=sample_content,
                user_rating=rating,
                quality_scores=sample_scores,
                agent_performance={
                    'EditorAgent': {
                        'quality_score': quality_analysis['overall_score'],
                        'issues': []
                    }
                }
            )
            
            # Generate insights
            insights = learning_system.generate_learning_insights()
            
            assert insights['summary']['total_feedback_entries'] >= 1
            
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)

def test_demo_mode_functions():
    """Test the demo mode functions work without errors."""
    from src.main import run_quality_analysis_demo, run_feedback_learning_demo
    
    # These should run without exceptions (though they print output)
    try:
        # Capture and suppress output for testing
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                run_quality_analysis_demo()
                run_feedback_learning_demo()
        
        # If we get here, no exceptions were raised
        assert True
        
    except Exception as e:
        # If there are exceptions, we still want to know about them
        pytest.fail(f"Demo functions raised exception: {e}")
    
    finally:
        # Clean up any files created by demos
        for file in ["logs/feedback_history.json", "logs/learning_report.json"]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 