"""
Integration Tests for Phase 2: Agent with Tools

Tests the complete Phase 2 workflow:
- Web search integration
- Agent system
- Task execution 
- Tool integration
- End-to-end workflow
"""

import logging
import os
import sys

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.agents import ResearchAgent, SimpleAgent
from src.core.core import query_llm
from src.tools.tools import AVAILABLE_TOOLS, search_knowledge_base, search_web

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
            assert callable(AVAILABLE_TOOLS['crewai_search_web'])
    
    def test_knowledge_base_tool_structure(self):
        """Test that knowledge base tool has correct structure."""
        assert 'search_knowledge_base' in AVAILABLE_TOOLS
        assert callable(AVAILABLE_TOOLS['search_knowledge_base'])
    

    
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
        # Agent should have access to multiple tools via the registry
        assert len(agent.available_tools) >= 2
    
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
        from src.agents.agents import ResearchAgent, SimpleAgent

        # Test that classes exist and are instantiable
        agent = SimpleAgent("Test", "Test", "Test", "Test")
        research_agent = ResearchAgent()
        
        assert isinstance(agent, SimpleAgent)
        assert isinstance(research_agent, ResearchAgent)
    
    def test_integration_components_exist(self):
        """Verify integration components exist (Task 2.5)."""
        # Test main execution workflow exists
        from src.main import test_basic_functionality
        
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
    
    # Test 3: Agent execution
    print("✓ Testing agent execution...")
    result = agent.execute_task("test task")
    assert isinstance(result, str)
    
    print("✓ Phase 2 validation complete!")
    print("All core components are properly integrated.")

if __name__ == "__main__":
    # Run validation when script is executed directly
    run_phase2_validation() 