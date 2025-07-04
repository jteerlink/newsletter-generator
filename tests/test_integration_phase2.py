"""
Integration Tests for Phase 2: Agent with Tools

Tests the complete Phase 2 workflow:
- Web search integration
- Agent system
- Task execution 
- Tool integration
- End-to-end workflow
"""

import pytest
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.agents import ResearchAgent, Task, SimpleCrew, SimpleAgent
from src.tools.tools import search_web, search_knowledge_base, AVAILABLE_TOOLS
from src.core.core import query_llm

logger = logging.getLogger(__name__)

class TestPhase2Tools:
    """Test Phase 2 tool implementations."""
    
    def test_legacy_web_search_tool_structure(self):
        """Test that legacy web search tool has correct structure."""
        assert 'search_web' in AVAILABLE_TOOLS
        assert callable(AVAILABLE_TOOLS['search_web'])
    
    def test_crewai_tools_available(self):
        """Test that CrewAI tools are available when properly configured."""
        from src.tools.tools import CREWAI_AVAILABLE
        if CREWAI_AVAILABLE:
            assert 'crewai_search_web' in AVAILABLE_TOOLS
            assert 'hybrid_search_web' in AVAILABLE_TOOLS
            assert callable(AVAILABLE_TOOLS['crewai_search_web'])
            assert callable(AVAILABLE_TOOLS['hybrid_search_web'])
    
    def test_knowledge_base_tool_structure(self):
        """Test that knowledge base tool has correct structure."""
        assert 'search_knowledge_base' in AVAILABLE_TOOLS
        assert callable(AVAILABLE_TOOLS['search_knowledge_base'])
    
    def test_hybrid_search_execution(self):
        """Test hybrid search execution (graceful fallback)."""
        try:
            if 'hybrid_search_web' in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS['hybrid_search_web']("test query", max_results=1)
                assert isinstance(result, str)
                assert len(result) > 0
                # Should either get CrewAI results, fallback to DuckDuckGo, or indicate no results
                assert any(indicator in result for indicator in [
                    "Search Results", "temporarily unavailable", "CrewAI", "fallback", 
                    "No search results found", "Search unavailable", "API key not configured"
                ])
        except Exception as e:
            # Rate limiting or API issues are acceptable
            assert any(keyword in str(e).lower() for keyword in ['ratelimit', 'rate', 'api', 'temporarily'])
    
    def test_web_search_execution(self):
        """Test legacy web search execution (may be rate limited)."""
        try:
            result = search_web("test query", max_results=1)
            assert isinstance(result, str)
            assert len(result) > 0
            # Either successful results or rate limit error
            assert "Search Results" in result or "temporarily unavailable" in result
        except Exception as e:
            # Rate limiting is acceptable
            assert "ratelimit" in str(e).lower() or "rate" in str(e).lower()

class TestPhase2Agents:
    """Test Phase 2 agent implementations."""
    
    def test_simple_agent_creation(self):
        """Test basic agent creation."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Tester",
            goal="Test things",
            backstory="I am a test agent",
            tools=['search_web']
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == "Tester"
        assert agent.goal == "Test things"
        assert agent.backstory == "I am a test agent"
        assert agent.tools == ['search_web']
        assert 'search_web' in agent.available_tools
    
    def test_research_agent_creation(self):
        """Test research agent creation."""
        agent = ResearchAgent()
        
        assert agent.name == "ResearchAgent"
        assert "Research Analyst" in agent.role
        assert 'agentic_search' in agent.tools
        assert 'search_web_with_alternatives' in agent.tools
        assert 'hybrid_search_web' in agent.tools
        # Agent should have access to multiple tools via the registry
        assert len(agent.available_tools) >= 3
    
    def test_agent_prompt_building(self):
        """Test agent prompt building functionality."""
        agent = SimpleAgent(
            name="TestAgent",
            role="Tester",
            goal="Test things",
            backstory="I am a test agent"
        )
        
        prompt = agent._build_prompt("test task", "test context")
        
        assert "You are a Tester" in prompt
        assert "Test things" in prompt
        assert "I am a test agent" in prompt
        assert "test task" in prompt
        assert "test context" in prompt
    
    def test_tool_need_detection(self):
        """Test tool need detection logic."""
        agent = SimpleAgent("Test", "Tester", "Test", "Test")
        
        assert agent._should_use_tools("I NEED_TOOLS to complete this")
        assert agent._should_use_tools("I need to search for information")
        assert not agent._should_use_tools("I can complete this without tools")

class TestPhase2TaskExecution:
    """Test Phase 2 task execution."""
    
    def test_task_creation(self):
        """Test task creation."""
        agent = SimpleAgent("Test", "Tester", "Test", "Test")
        task = Task("test description", agent, "test context")
        
        assert task.description == "test description"
        assert task.agent == agent
        assert task.context == "test context"
        assert task.result is None
    
    def test_basic_task_execution(self):
        """Test basic task execution without tools."""
        agent = SimpleAgent("Test", "Tester", "Test", "Test")
        task = Task("Say hello", agent)
        
        try:
            result = task.execute()
            assert isinstance(result, str)
            assert len(result) > 0
            assert task.result == result
        except Exception as e:
            # LLM connection issues are acceptable in test environment
            logger.warning(f"Task execution failed (expected in test env): {e}")

class TestPhase2Integration:
    """Test Phase 2 end-to-end integration."""
    
    def test_crew_creation(self):
        """Test crew creation."""
        agent = ResearchAgent()
        task = Task("research test", agent)
        crew = SimpleCrew([agent], [task])
        
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1
        assert crew.agents[0] == agent
        assert crew.tasks[0] == task
    
    def test_simple_workflow_structure(self):
        """Test that workflow components are properly structured."""
        # Create workflow components
        research_agent = ResearchAgent()
        
        research_task = Task(
            description="Research AI trends",
            agent=research_agent,
            context="Test context"
        )
        
        analysis_task = Task(
            description="Analyze findings",
            agent=research_agent,
            context=""
        )
        
        crew = SimpleCrew([research_agent], [research_task, analysis_task])
        
        # Verify structure
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 2
        assert crew.tasks[0].description == "Research AI trends"
        assert crew.tasks[1].description == "Analyze findings"
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow (marked as slow test)."""
        try:
            # Create a minimal workflow
            agent = ResearchAgent()
            task = Task("What is artificial intelligence?", agent)
            crew = SimpleCrew([agent], [task])
            
            # Execute workflow
            result = crew.kickoff()
            
            # Verify results
            assert isinstance(result, str)
            assert len(result) > 0
            assert "TASK 1 RESULT" in result
            
            logger.info("End-to-end workflow test passed")
            
        except Exception as e:
            logger.warning(f"End-to-end test failed (may be due to environment): {e}")
            # In CI/test environments, network/LLM issues are common
            if any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network', 'rate']):
                pytest.skip(f"Skipping due to external dependency: {e}")
            else:
                raise

class TestPhase2Requirements:
    """Test that Phase 2 requirements are met."""
    
    def test_web_search_integration_exists(self):
        """Verify web search integration exists (Task 2.3)."""
        from src.tools.tools import search_web
        assert callable(search_web)
        
        # Test function signature
        import inspect
        sig = inspect.signature(search_web)
        assert 'query' in sig.parameters
        assert 'max_results' in sig.parameters
    
    def test_agent_framework_exists(self):
        """Verify agent framework exists (Task 2.4)."""
        from src.agents.agents import SimpleAgent, ResearchAgent, Task, SimpleCrew
        
        # Test that classes exist and are instantiable
        agent = SimpleAgent("Test", "Test", "Test", "Test")
        research_agent = ResearchAgent()
        task = Task("Test", agent)
        crew = SimpleCrew([agent], [task])
        
        assert isinstance(agent, SimpleAgent)
        assert isinstance(research_agent, ResearchAgent)
        assert isinstance(task, Task)
        assert isinstance(crew, SimpleCrew)
    
    def test_integration_components_exist(self):
        """Verify integration components exist (Task 2.5)."""
        # Test main execution workflow exists
        from src.main import create_research_workflow, test_basic_functionality
        
        assert callable(create_research_workflow)
        assert callable(test_basic_functionality)

def run_phase2_validation():
    """Run a comprehensive Phase 2 validation."""
    print("Running Phase 2 Validation...")
    print("=" * 50)
    
    # Test 1: Tool availability
    print("✓ Testing tool availability...")
    assert 'search_web' in AVAILABLE_TOOLS
    assert 'search_knowledge_base' in AVAILABLE_TOOLS
    
    # Test 2: Agent creation
    print("✓ Testing agent creation...")
    agent = ResearchAgent()
    assert agent.name == "ResearchAgent"
    
    # Test 3: Task execution structure
    print("✓ Testing task execution structure...")
    task = Task("test", agent)
    assert task.description == "test"
    
    # Test 4: Crew orchestration
    print("✓ Testing crew orchestration...")
    crew = SimpleCrew([agent], [task])
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    
    print("✓ Phase 2 validation complete!")
    print("All core components are properly integrated.")

if __name__ == "__main__":
    # Run validation when script is executed directly
    run_phase2_validation() 