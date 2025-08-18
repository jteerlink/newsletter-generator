"""
Tests for Modular Agent Classes

This module tests the new modular agent architecture including base classes,
specialized agents, and workflow management.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.agents.base import AgentContext, AgentType, SimpleAgent, TaskResult, TaskStatus
from src.agents.editing import EditorAgent
from src.agents.management import ManagerAgent, WorkflowPlan, WorkflowStep
from src.agents.research import ResearchAgent
from src.agents.writing import WriterAgent
from src.core.template_manager import NewsletterType


class TestBaseAgent:
    """Test the base SimpleAgent class."""
    
    def test_simple_agent_initialization(self):
        """Test SimpleAgent initialization with basic parameters."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
            agent_type=AgentType.RESEARCH,
            tools=["test_tool"]
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Role"
        assert agent.goal == "Test Goal"
        assert agent.backstory == "Test Backstory"
        assert agent.agent_type == AgentType.RESEARCH
        assert agent.tools == ["test_tool"]
        assert len(agent.execution_history) == 0
    
    def test_simple_agent_execute_task(self):
        """Test SimpleAgent task execution."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
            agent_type=AgentType.RESEARCH
        )
        
        # Test basic functionality without mocking
        result = agent.execute_task("Test task")
            
        assert result is not None
        assert len(agent.execution_history) == 1
        assert agent.execution_history[0].task_id is not None
        assert agent.execution_history[0].status == TaskStatus.COMPLETED
    
    def test_simple_agent_execution_history(self):
        """Test execution history tracking."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
            agent_type=AgentType.RESEARCH
        )
        
        # Test execution history tracking
        agent.execute_task("Task 1")
        agent.execute_task("Task 2")
        
        assert len(agent.execution_history) == 2
        assert agent.execution_history[0].result is not None
        assert agent.execution_history[1].result is not None
        assert agent.execution_history[0].task_id != agent.execution_history[1].task_id
    
    def test_simple_agent_error_handling(self):
        """Test error handling in task execution."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
            agent_type=AgentType.RESEARCH
        )
        
        # Test that agent can handle tasks without errors
        result = agent.execute_task("Test task")
        
        assert result is not None
        assert len(agent.execution_history) == 1
        assert agent.execution_history[0].status == TaskStatus.COMPLETED


class TestResearchAgent:
    """Test the ResearchAgent class."""
    
    def test_research_agent_initialization(self):
        """Test ResearchAgent initialization."""
        agent = ResearchAgent()
        
        assert agent.name == "ResearchAgent"
        assert agent.role == "Research Specialist"
        assert agent.agent_type == AgentType.RESEARCH
        assert "search_web" in agent.tools
        assert "search_knowledge_base" in agent.tools
    
    def test_research_agent_execute_task(self):
        """Test ResearchAgent task execution."""
        agent = ResearchAgent()
        
        with patch('src.core.core.query_llm') as mock_query, \
             patch('src.tools.tools.search_web') as mock_search, \
             patch('src.tools.tools.search_knowledge_base') as mock_kb:
            
            mock_query.return_value = "Synthesized research results"
            mock_search.return_value = "Web search results"
            mock_kb.return_value = "Knowledge base results"
            
            result = agent.execute_task("Research AI trends")
            
            assert "Synthesized research results" in result
            mock_search.assert_called()
            mock_kb.assert_called()
    
    def test_research_agent_generate_search_queries(self):
        """Test search query generation."""
        agent = ResearchAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "query1\nquery2\nquery3"
            queries = agent._generate_search_queries("AI trends", "context")
            
            assert len(queries) == 3
            assert "query1" in queries
            assert "query2" in queries
            assert "query3" in queries
    
    def test_research_agent_synthesize_results(self):
        """Test research results synthesis."""
        agent = ResearchAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "Synthesized content"
            result = agent._synthesize_research_results(
                "Knowledge results", "Web results", "Test topic"
            )
            
            assert "Synthesized content" in result


class TestWriterAgent:
    """Test the WriterAgent class."""
    
    def test_writer_agent_initialization(self):
        """Test WriterAgent initialization."""
        agent = WriterAgent()
        
        assert agent.name == "WriterAgent"
        assert agent.role == "Content Writer"
        assert agent.agent_type == AgentType.WRITER
        assert "search_web" in agent.tools
    
    def test_writer_agent_execute_task(self):
        """Test WriterAgent task execution."""
        agent = WriterAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "Generated newsletter content"
            result = agent.execute_task("Write about AI", template_type=NewsletterType.TECHNICAL_DEEP_DIVE)
            
            assert "Generated newsletter content" in result
    
    def test_writer_agent_template_instructions(self):
        """Test template-specific instructions."""
        agent = WriterAgent()
        
        technical_instructions = agent._get_template_instructions(NewsletterType.TECHNICAL_DEEP_DIVE)
        business_instructions = agent._get_template_instructions(NewsletterType.TREND_ANALYSIS)
        
        assert "technical" in technical_instructions.lower()
        assert "business" in business_instructions.lower()
    
    def test_writer_agent_writing_guidelines(self):
        """Test writing guidelines generation."""
        agent = WriterAgent()
        
        guidelines = agent._get_writing_guidelines("professional", "technical")
        
        assert "professional" in guidelines.lower()
        assert "technical" in guidelines.lower()
    
    def test_writer_agent_generate_headlines(self):
        """Test headline generation."""
        agent = WriterAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "Headline 1\nHeadline 2\nHeadline 3"
            headlines = agent.generate_headlines("Test content", count=3)
            
            assert len(headlines) == 3
            assert "Headline 1" in headlines


class TestEditorAgent:
    """Test the EditorAgent class."""
    
    def test_editor_agent_initialization(self):
        """Test EditorAgent initialization."""
        agent = EditorAgent()
        
        assert agent.name == "EditorAgent"
        assert agent.role == "Content Editor"
        assert agent.agent_type == AgentType.EDITOR
        assert len(agent.tools) == 0  # Editors don't need external tools
    
    def test_editor_agent_execute_task(self):
        """Test EditorAgent task execution."""
        agent = EditorAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "Improved content"
            result = agent.execute_task("Edit this content")
            
            assert "Improved content" in result
    
    def test_editor_agent_extract_quality_metrics(self):
        """Test quality metrics extraction."""
        agent = EditorAgent()
        
        content = "This is a test content. It has multiple sentences. And paragraphs."
        metrics = agent.extract_quality_metrics(content)
        
        assert 'word_count' in metrics
        assert 'sentence_count' in metrics
        assert 'paragraph_count' in metrics
        assert 'scores' in metrics
        assert metrics['word_count'] > 0
    
    def test_editor_agent_calculate_quality_score(self):
        """Test quality score calculation."""
        agent = EditorAgent()
        
        content_analysis = {
            'scores': {
                'clarity': 8.0,
                'accuracy': 9.0,
                'engagement': 7.0,
                'completeness': 8.0,
                'structure': 7.0,
                'grammar': 8.0
            }
        }
        
        quality_analysis = agent.calculate_quality_score(content_analysis)
        
        assert 'overall_score' in quality_analysis
        assert 'grade' in quality_analysis
        assert 'improvement_areas' in quality_analysis
        assert quality_analysis['overall_score'] > 0
    
    def test_editor_agent_validate_content_quality(self):
        """Test content quality validation."""
        agent = EditorAgent()
        
        content = "This is a test content for quality validation."
        validation = agent.validate_content_quality(content)
        
        assert 'passes_quality_gate' in validation
        assert 'quality_score' in validation
        assert 'recommendations' in validation
        assert isinstance(validation['recommendations'], list)


class TestManagerAgent:
    """Test the ManagerAgent class."""
    
    def test_manager_agent_initialization(self):
        """Test ManagerAgent initialization."""
        agent = ManagerAgent()
        
        assert agent.name == "ManagerAgent"
        assert agent.role == "Workflow Manager"
        assert agent.agent_type == AgentType.MANAGER
        assert len(agent.tools) == 0  # Managers coordinate rather than execute
        assert len(agent.active_workflows) == 0
    
    def test_manager_agent_create_workflow_plan(self):
        """Test workflow plan creation."""
        agent = ManagerAgent()
        
        with patch('src.core.core.query_llm') as mock_query:
            mock_query.return_value = "Workflow plan created"
            result = agent._create_workflow_plan("Create workflow about AI", "context", "standard")
            
            assert "Workflow Plan:" in result
            assert len(agent.active_workflows) > 0
    
    def test_manager_agent_create_simple_workflow(self):
        """Test simple workflow creation."""
        agent = ManagerAgent()
        
        workflow = agent._create_simple_workflow("Test topic")
        
        assert workflow['topic'] == "Test topic"
        assert workflow['complexity'] == "simple"
        assert 'streams' in workflow
        assert 'estimated_time' in workflow
    
    def test_manager_agent_create_standard_workflow(self):
        """Test standard workflow creation."""
        agent = ManagerAgent()
        
        workflow = agent._create_standard_workflow("Test topic")
        
        assert workflow['topic'] == "Test topic"
        assert workflow['complexity'] == "standard"
        assert len(workflow['streams']) == 2  # Research and Content Creation
        assert workflow['estimated_time'] > 0
    
    def test_manager_agent_create_complex_workflow(self):
        """Test complex workflow creation."""
        agent = ManagerAgent()
        
        workflow = agent._create_complex_workflow("Test topic")
        
        assert workflow['topic'] == "Test topic"
        assert workflow['complexity'] == "complex"
        assert len(workflow['streams']) == 3  # Research, Content, Quality
        assert workflow['estimated_time'] > 0
    
    def test_manager_agent_determine_quality_gates(self):
        """Test quality gate determination."""
        agent = ManagerAgent()
        
        simple_gates = agent._determine_quality_gates("simple")
        standard_gates = agent._determine_quality_gates("standard")
        complex_gates = agent._determine_quality_gates("complex")
        
        assert len(simple_gates) == 1
        assert len(standard_gates) == 2
        assert len(complex_gates) == 3
    
    def test_manager_agent_extract_topic_from_task(self):
        """Test topic extraction from task."""
        agent = ManagerAgent()
        
        topic1 = agent._extract_topic_from_task("Create workflow about AI trends")
        topic2 = agent._extract_topic_from_task("Plan newsletter on machine learning")
        topic3 = agent._extract_topic_from_task("General task")
        
        assert "AI trends" in topic1
        assert "machine learning" in topic2
        assert topic3 == "general newsletter"
    
    def test_manager_agent_select_template_for_topic(self):
        """Test template selection for topics."""
        agent = ManagerAgent()
        
        technical_template = agent._select_template_for_topic("technical programming")
        business_template = agent._select_template_for_topic("business market analysis")
        quick_template = agent._select_template_for_topic("quick summary")
        general_template = agent._select_template_for_topic("general topic")
        
        assert technical_template == NewsletterType.TECHNICAL_DEEP_DIVE
        assert business_template == NewsletterType.TREND_ANALYSIS
        assert quick_template == NewsletterType.TUTORIAL_GUIDE
        assert general_template == NewsletterType.RESEARCH_SUMMARY
    
    def test_manager_agent_monitor_workflows(self):
        """Test workflow monitoring."""
        agent = ManagerAgent()
        
        # Create a test workflow
        agent.active_workflows['test_workflow'] = WorkflowPlan(
            workflow_id='test_workflow',
            topic='Test Topic',
            complexity='standard',
            template_type=NewsletterType.RESEARCH_SUMMARY,
            steps=[],
            estimated_total_time=60,
            quality_gates=['basic_quality']
        )
        
        result = agent._monitor_workflows("Monitor workflows", "")
        
        assert "test_workflow" in result
        assert "Test Topic" in result
        assert "Not Started" in result


class TestWorkflowComponents:
    """Test workflow-related components."""
    
    def test_workflow_step_creation(self):
        """Test WorkflowStep creation."""
        step = WorkflowStep(
            step_id="step_1",
            name="Test Step",
            description="Test description",
            agent_type="research",
            dependencies=["step_0"],
            estimated_time=30
        )
        
        assert step.step_id == "step_1"
        assert step.name == "Test Step"
        assert step.agent_type == "research"
        assert step.dependencies == ["step_0"]
        assert step.estimated_time == 30
        assert step.status == TaskStatus.PENDING
    
    def test_workflow_plan_creation(self):
        """Test WorkflowPlan creation."""
        steps = [
            WorkflowStep(
                step_id="step_1",
                name="Test Step",
                description="Test description",
                agent_type="research"
            )
        ]
        
        plan = WorkflowPlan(
            workflow_id="test_workflow",
            topic="Test Topic",
            complexity="standard",
            template_type=NewsletterType.RESEARCH_SUMMARY,
            steps=steps,
            estimated_total_time=60,
            quality_gates=["basic_quality"]
        )
        
        assert plan.workflow_id == "test_workflow"
        assert plan.topic == "Test Topic"
        assert plan.complexity == "standard"
        assert len(plan.steps) == 1
        assert plan.estimated_total_time == 60
        assert "basic_quality" in plan.quality_gates


class TestAgentIntegration:
    """Test integration between different agent types."""
    
    def test_agent_creation_function(self):
        """Test the convenience function for creating agents."""
        from src.agents.agents import create_agent, get_available_agent_types

        # Test agent creation
        research_agent = create_agent('research')
        writer_agent = create_agent('writer')
        editor_agent = create_agent('editor')
        manager_agent = create_agent('manager')
        
        assert isinstance(research_agent, ResearchAgent)
        assert isinstance(writer_agent, WriterAgent)
        assert isinstance(editor_agent, EditorAgent)
        assert isinstance(manager_agent, ManagerAgent)
        
        # Test available agent types
        agent_types = get_available_agent_types()
        assert 'research' in agent_types
        assert 'writer' in agent_types
        assert 'editor' in agent_types
        assert 'manager' in agent_types
    
    def test_agent_error_handling(self):
        """Test error handling for unknown agent types."""
        from src.agents.agents import create_agent
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent('unknown_agent_type')
    
    def test_agent_analytics(self):
        """Test agent analytics functionality."""
        research_agent = ResearchAgent()
        writer_agent = WriterAgent()
        editor_agent = EditorAgent()
        manager_agent = ManagerAgent()
        
        # Test research analytics
        research_analytics = research_agent.get_research_analytics()
        assert 'research_sessions' in research_analytics
        assert 'tool_usage_breakdown' in research_analytics
        
        # Test writing analytics
        writing_analytics = writer_agent.get_writing_analytics()
        assert 'writing_sessions' in writing_analytics
        assert 'content_quality_metrics' in writing_analytics
        
        # Test editing analytics
        editing_analytics = editor_agent.get_editing_analytics()
        assert 'editing_sessions' in editing_analytics
        assert 'quality_improvement_metrics' in editing_analytics
        
        # Test management analytics
        management_analytics = manager_agent.get_management_analytics()
        assert 'workflows_managed' in management_analytics
        assert 'workflow_complexity_distribution' in management_analytics


if __name__ == "__main__":
    pytest.main([__file__]) 