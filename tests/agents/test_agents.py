"""Tests for agent system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.agents import (
    SimpleAgent, ResearchAgent, WriterAgent, EditorAgent, 
    ManagerAgent, Task, EnhancedCrew
)
from src.core.exceptions import AgentError

class TestSimpleAgent:
    """Test the base SimpleAgent class."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Test Role"
        assert agent.goal == "Test Goal"
        assert agent.backstory == "Test backstory"
        assert isinstance(agent.tools, list)
    
    @patch('src.agents.agents.query_llm')
    def test_execute_task_success(self, mock_query_llm):
        """Test successful task execution."""
        mock_query_llm.return_value = "Test response"
        
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        result = agent.execute_task("Test task")
        assert result == "Test response"
        mock_query_llm.assert_called_once()
    
    @patch('src.agents.agents.query_llm')
    def test_execute_task_with_tools(self, mock_query_llm):
        """Test task execution with tools."""
        mock_query_llm.side_effect = [
            "I need to search for information",
            "Based on search results: Test response"
        ]
        
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory",
            tools=["search_web"]
        )
        
        with patch.object(agent, '_execute_tools') as mock_tools:
            mock_tools.return_value = "Search results"
            result = agent.execute_task("Test task")
            
            assert result == "Based on search results: Test response"
            assert mock_query_llm.call_count == 2
    
    def test_execute_task_error_handling(self):
        """Test error handling in task execution."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        with patch('src.agents.agents.query_llm', side_effect=Exception("Test error")):
            result = agent.execute_task("Test task")
            assert "Error in agent" in result
    
    def test_build_prompt(self):
        """Test prompt building functionality."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        prompt = agent._build_prompt("Test task", "Test context")
        
        assert "You are a Test Role" in prompt
        assert "Test Goal" in prompt
        assert "Test backstory" in prompt
        assert "Test task" in prompt
        assert "Test context" in prompt
    
    def test_should_use_tools(self):
        """Test tool need detection."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test backstory"
        )
        
        assert agent._should_use_tools("I need to search for information")
        assert agent._should_use_tools("I NEED_TOOLS to complete this")
        assert not agent._should_use_tools("I can complete this without tools")

class TestResearchAgent:
    """Test the ResearchAgent class."""
    
    def test_research_agent_initialization(self):
        """Test research agent initialization."""
        agent = ResearchAgent()
        
        assert agent.name == "ResearchAgent"
        assert "research" in agent.role.lower()
        assert "search" in agent.goal.lower()
        assert "search_web_with_alternatives" in agent.tools
        assert "agentic_search" in agent.tools
    
    @patch('src.agents.agents.query_llm')
    def test_research_task_execution(self, mock_query_llm):
        """Test research task execution."""
        mock_query_llm.return_value = "Research findings: AI is advancing rapidly"
        
        agent = ResearchAgent()
        result = agent.execute_task("Research latest AI developments")
        
        assert "Research findings" in result
        mock_query_llm.assert_called_once()
    
    def test_research_agent_tools(self):
        """Test that research agent has appropriate tools."""
        agent = ResearchAgent()
        
        expected_tools = ['search_web_with_alternatives', 'agentic_search', 
                         'hybrid_search_web']
        
        for tool in expected_tools:
            assert tool in agent.tools or tool in agent.available_tools

class TestWriterAgent:
    """Test the WriterAgent class."""
    
    def test_writer_agent_initialization(self):
        """Test writer agent initialization."""
        agent = WriterAgent()
        
        assert agent.name == "WriterAgent"
        assert "content" in agent.role.lower()
        assert "content" in agent.goal.lower()
    
    @patch('src.agents.agents.query_llm')
    def test_writing_task_execution(self, mock_query_llm):
        """Test writing task execution."""
        mock_query_llm.return_value = "Written content about AI developments"
        
        agent = WriterAgent()
        result = agent.execute_task("Write about AI developments")
        
        assert "Written content" in result
        # WriterAgent may call LLM twice if content enhancement is triggered
        assert mock_query_llm.call_count >= 1
    
    def test_writer_agent_role(self):
        """Test writer agent role and goal."""
        agent = WriterAgent()
        
        assert "Content Creator" in agent.role
        assert "engaging" in agent.goal.lower()
        assert "compelling" in agent.goal.lower()

class TestEditorAgent:
    """Test the EditorAgent class."""
    
    def test_editor_agent_initialization(self):
        """Test editor agent initialization."""
        agent = EditorAgent()
        
        assert agent.name == "EditorAgent"
        assert "editor" in agent.role.lower()
        assert "quality" in agent.goal.lower()
    
    def test_quality_metrics_extraction(self):
        """Test quality metrics extraction."""
        agent = EditorAgent()
        
        content = """
        # Test Content
        
        This is a test newsletter about artificial intelligence.
        It contains multiple paragraphs with detailed information.
        
        ## Key Points
        - Point one with details
        - Point two with more information
        """
        
        metrics = agent.extract_quality_metrics(content)
        
        assert 'word_count' in metrics
        assert 'estimated_reading_time' in metrics
        assert 'content_depth' in metrics
        assert metrics['word_count'] > 0
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        agent = EditorAgent()
        
        sample_scores = {
            'clarity': 8.0,
            'accuracy': 7.5,
            'engagement': 6.5,
            'completeness': 7.0
        }
        
        quality_analysis = agent.calculate_quality_score({'scores': sample_scores})
        
        assert 'overall_score' in quality_analysis
        assert 'grade' in quality_analysis
        assert quality_analysis['overall_score'] > 0
    
    @patch('src.agents.agents.query_llm')
    def test_editing_task_execution(self, mock_query_llm):
        """Test editing task execution."""
        mock_query_llm.return_value = "Edited and improved content"
        
        agent = EditorAgent()
        result = agent.execute_task("Edit this content")
        
        assert "Edited and improved" in result
        mock_query_llm.assert_called_once()

class TestManagerAgent:
    """Test the ManagerAgent class."""
    
    def test_manager_agent_initialization(self):
        """Test manager agent initialization."""
        agent = ManagerAgent()
        
        assert agent.name == "ManagerAgent"
        assert "manager" in agent.role.lower()
        assert "orchestrate" in agent.goal.lower()
    
    def test_workflow_optimization(self):
        """Test workflow optimization functionality."""
        agent = ManagerAgent()
        
        tasks = [
            {'type': 'research', 'description': 'Research AI trends'},
            {'type': 'writing', 'description': 'Write newsletter'},
            {'type': 'editing', 'description': 'Edit content'}
        ]
        
        optimized_tasks = agent.optimize_workflow_sequence(tasks)
        
        assert len(optimized_tasks) == len(tasks)
        assert 'group_type' in optimized_tasks[0]
    
    def test_task_delegation(self):
        """Test task delegation functionality."""
        agent = ManagerAgent()
        
        # Create real agents instead of mocks
        from src.agents.agents import ResearchAgent, WriterAgent
        
        research_agent = ResearchAgent()
        writer_agent = WriterAgent()
        
        agents = [research_agent, writer_agent]
        
        # Test delegating a single research task
        delegation = agent.delegate_task("research", "Research AI trends", agents)
        
        assert delegation["success"] == True
        assert "agent" in delegation
        assert "delegation_id" in delegation

class TestTask:
    """Test the Task class."""
    
    def test_task_creation(self):
        """Test task creation."""
        agent = Mock()
        task = Task("Test task description", agent, context="Test context")
        
        assert task.description == "Test task description"
        assert task.agent == agent
        assert task.context == "Test context"
    
    def test_task_execution(self):
        """Test task execution."""
        agent = Mock()
        agent.execute_task.return_value = "Task result"
        
        task = Task("Test task", agent)
        result = task.execute()
        
        assert result == "Task result"
        agent.execute_task.assert_called_once_with("Test task", "")
    
    def test_task_with_context(self):
        """Test task execution with context."""
        agent = Mock()
        agent.execute_task.return_value = "Task result with context"
        
        task = Task("Test task", agent, context="Test context")
        result = task.execute()
        
        assert result == "Task result with context"
        agent.execute_task.assert_called_once_with("Test task", "Test context")

class TestEnhancedCrew:
    """Test the EnhancedCrew class."""
    
    def test_crew_initialization(self):
        """Test crew initialization."""
        agents = [Mock(), Mock()]
        tasks = [Mock(), Mock()]
        
        crew = EnhancedCrew(agents, tasks)
        
        assert crew.agents == agents
        assert crew.tasks == tasks
        assert crew.workflow_type == "sequential"
    
    @patch('src.agents.agents.query_llm')
    def test_crew_kickoff(self, mock_query_llm):
        """Test crew kickoff."""
        mock_query_llm.return_value = "Agent response"
        
        agent1 = Mock()
        agent1.name = "Agent1"
        agent1.execute_task.return_value = "Agent 1 result"
        agent2 = Mock()
        agent2.name = "WriterAgent"  # This will be the final result
        agent2.execute_task.return_value = "Agent 2 result"
        
        task1 = Task("Task 1", agent1)
        task2 = Task("Task 2", agent2)
        
        crew = EnhancedCrew([agent1, agent2], [task1, task2])
        result = crew.kickoff()
        
        # EnhancedCrew returns only the final newsletter content (from WriterAgent)
        assert "Agent 2 result" in result
    
    def test_crew_with_context_passing(self):
        """Test crew with context passing between tasks."""
        agent1 = Mock()
        agent1.execute_task.return_value = "Research: AI trends in 2024"
        agent2 = Mock()
        agent2.execute_task.return_value = "Content based on: Research: AI trends in 2024"
        
        task1 = Task("Research AI trends", agent1)
        task2 = Task("Write newsletter", agent2)
        
        crew = EnhancedCrew([agent1, agent2], [task1, task2])
        result = crew.kickoff()
        
        assert "Research: AI trends in 2024" in result
        assert "Content based on:" in result
    
    def test_crew_error_handling(self):
        """Test crew error handling."""
        agent1 = Mock()
        agent1.name = "Agent1"
        agent1.execute_task.side_effect = Exception("Task failed")
        agent2 = Mock()
        agent2.name = "WriterAgent"
        agent2.execute_task.return_value = "Final newsletter content"
        
        task1 = Task("Task 1", agent1)
        task2 = Task("Task 2", agent2)
        
        crew = EnhancedCrew([agent1, agent2], [task1, task2])
        result = crew.kickoff()
        
        # EnhancedCrew should handle errors gracefully and continue with next task
        assert "Final newsletter content" in result
        # Second agent should still be called even if first fails
        agent2.execute_task.assert_called_once()

class TestAgentCoordination:
    """Test agent coordination and communication."""
    
    def test_agent_tool_sharing(self):
        """Test that agents can share tools and results."""
        research_agent = ResearchAgent()
        writer_agent = WriterAgent()
        
        # Research agent should have search tools, Writer agent may not
        assert 'search_web_with_alternatives' in research_agent.tools
        # Writer agent doesn't have search tools by default (focuses on content creation)
        assert len(writer_agent.tools) == 0
    
    def test_agent_performance_tracking(self):
        """Test agent performance tracking."""
        agent = SimpleAgent("Test", "Tester", "Test", "Test")
        
        # Execute a task and check if performance is tracked
        with patch('src.agents.agents.query_llm', return_value="Test result"):
            result = agent.execute_task("Test task")
            
            assert result == "Test result"
            # Performance tracking would be implemented here
    
    def test_agent_error_recovery(self):
        """Test agent error recovery mechanisms."""
        agent = SimpleAgent("Test", "Tester", "Test", "Test")
        
        # Test that agent can handle errors gracefully
        with patch('src.agents.agents.query_llm', side_effect=Exception("Test error")):
            result = agent.execute_task("Test task")
            
            assert "Error in agent" in result
            assert "Test error" in result

# class TestEndToEndWorkflow:
#     """Test complete end-to-end newsletter generation."""
    
#     @patch('src.core.core.query_llm')
#     def test_complete_newsletter_generation(self, mock_query_llm):
#         """Test complete newsletter generation from start to finish."""
#         # Mock LLM responses for different stages
#         mock_query_llm.side_effect = [
#             "Research findings: AI is advancing rapidly",
#             "Written content about AI advancements",
#             "Edited and improved newsletter content"
#         ]
        
#         # Create real agents
#         research_agent = ResearchAgent()
#         writer_agent = WriterAgent()
#         editor_agent = EditorAgent()
        
#         # Create tasks
#         research_task = Task("Research latest AI developments", research_agent)
#         writing_task = Task("Write newsletter about AI", writer_agent)
#         editing_task = Task("Edit newsletter", editor_agent)
        
#         # Create crew
#         crew = EnhancedCrew(
#             [research_agent, writer_agent, editor_agent],
#             [research_task, writing_task, editing_task]
#         )
        
#         # Execute workflow
#         result = crew.kickoff()
        
#         # Verify the complete workflow - EnhancedCrew returns final newsletter content
#         assert "Edited Newsletter" in result
#         assert "AI" in result
#         assert "Written content" in result
#         assert "Edited and improved" in result
        
#         # Verify LLM was called for each stage
#         assert mock_query_llm.call_count >= 3 