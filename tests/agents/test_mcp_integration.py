"""
Tests for MCP tool integration with agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.base.agent_base import AgentBase
from src.agents.base.agent_registry import AgentRegistry
from src.agents.base.mcp_integration import MCPClient, MCPToolManager
from src.agents.writing.research_summary_agent import ResearchSummaryAgent


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class TestAgent(AgentBase):
    """Concrete test agent for testing AgentBase functionality."""
    
    async def run(self):
        """Test implementation of run method."""
        pass
    
    def receive_message(self, message: dict):
        """Test implementation of receive_message method."""
        pass
    
    def send_message(self, recipient_id: str, message: dict):
        """Test implementation of send_message method."""
        pass


class TestAgentBase:
    """Test MCP tool integration in AgentBase."""
    
    def test_agent_base_initialization(self):
        """Test agent base initialization with MCP support."""
        agent = TestAgent("test_agent")
        assert agent.agent_id == "test_agent"
        assert agent.mcp_client is None
        assert agent.available_tools == []
    
    def test_set_mcp_client(self):
        """Test setting MCP client for agent."""
        agent = TestAgent("test_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        
        agent.set_mcp_client(mock_client)
        assert agent.mcp_client == mock_client
    
    def test_get_available_tools(self):
        """Test getting available tools from MCP client."""
        agent = TestAgent("test_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        
        agent.set_mcp_client(mock_client)
        tools = agent.get_available_tools()
        assert tools == ["web_search", "vector_search"]
    
    def test_get_available_tools_no_client(self):
        """Test getting available tools when no MCP client is set."""
        agent = TestAgent("test_agent")
        tools = agent.get_available_tools()
        assert tools == []
    
    def test_can_use_tool(self):
        """Test checking if agent can use a specific tool."""
        agent = TestAgent("test_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        
        agent.set_mcp_client(mock_client)
        assert agent.can_use_tool("web_search") is True
        assert agent.can_use_tool("nonexistent_tool") is False
    
    def test_get_tool_capabilities(self):
        """Test getting agent's tool capabilities."""
        agent = TestAgent("test_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        
        agent.set_mcp_client(mock_client)
        capabilities = agent.get_tool_capabilities()
        
        assert capabilities["agent_id"] == "test_agent"
        assert capabilities["available_tools"] == ["web_search", "vector_search"]
        assert capabilities["mcp_client_configured"] is True
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_success(self):
        """Test successful MCP tool call."""
        agent = TestAgent("test_agent")
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {"result": "success"}
        
        agent.set_mcp_client(mock_client)
        result = await agent.call_mcp_tool("test_tool", {"param": "value"})
        
        assert result == {"result": "success"}
        mock_client.call_tool.assert_called_once_with("test_tool", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_no_client(self):
        """Test MCP tool call when no client is configured."""
        agent = TestAgent("test_agent")
        result = await agent.call_mcp_tool("test_tool", {"param": "value"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_call_mcp_tool_failure(self):
        """Test MCP tool call failure."""
        agent = TestAgent("test_agent")
        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = Exception("Tool call failed")
        
        agent.set_mcp_client(mock_client)
        result = await agent.call_mcp_tool("test_tool", {"param": "value"})
        
        assert result is None


class TestAgentRegistry:
    """Test enhanced AgentRegistry with MCP integration."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()
        assert registry.agents == {}
        assert registry.mcp_client is None
        assert registry.agent_capabilities == {}
    
    def test_register_agent_with_mcp_client(self):
        """Test registering agent with MCP client configured."""
        registry = AgentRegistry()
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search"]
        registry.set_mcp_client(mock_client)
        
        agent = TestAgent("test_agent")
        success = registry.register_agent(agent)
        
        assert success is True
        assert "test_agent" in registry.agents
        assert agent.mcp_client == mock_client
    
    def test_register_agent_without_mcp_client(self):
        """Test registering agent without MCP client."""
        registry = AgentRegistry()
        agent = TestAgent("test_agent")
        success = registry.register_agent(agent)
        
        assert success is True
        assert "test_agent" in registry.agents
        assert agent.mcp_client is None
    
    def test_set_mcp_client_for_existing_agents(self):
        """Test setting MCP client for existing agents."""
        registry = AgentRegistry()
        agent = TestAgent("test_agent")
        registry.register_agent(agent)
        
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search"]
        registry.set_mcp_client(mock_client)
        
        assert agent.mcp_client == mock_client
        assert registry.agent_capabilities["test_agent"]["mcp_client_configured"] is True
    
    def test_get_agents_with_tool(self):
        """Test getting agents that can use a specific tool."""
        registry = AgentRegistry()
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        registry.set_mcp_client(mock_client)
        
        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        agents_with_web_search = registry.get_agents_with_tool("web_search")
        assert len(agents_with_web_search) == 2
        assert "agent1" in agents_with_web_search
        assert "agent2" in agents_with_web_search
    
    def test_get_tool_usage_summary(self):
        """Test getting tool usage summary."""
        registry = AgentRegistry()
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search", "vector_search"]
        registry.set_mcp_client(mock_client)
        
        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")
        registry.register_agent(agent1)
        registry.register_agent(agent2)
        
        summary = registry.get_tool_usage_summary()
        assert "web_search" in summary
        assert "vector_search" in summary
        assert len(summary["web_search"]) == 2
        assert len(summary["vector_search"]) == 2
    
    def test_validate_agent_tools(self):
        """Test validating agent tool access."""
        registry = AgentRegistry()
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search"]
        registry.set_mcp_client(mock_client)
        
        agent = TestAgent("test_agent")
        registry.register_agent(agent)
        
        validation = registry.validate_agent_tools("test_agent", ["web_search", "vector_search"])
        assert validation["web_search"] is True
        assert validation["vector_search"] is False
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old method names."""
        registry = AgentRegistry()
        agent = TestAgent("test_agent")
        
        # Test old method names still work
        registry.register(agent)
        assert registry.get("test_agent") == agent
        assert "test_agent" in registry.list_agents()
        assert registry.count() == 1
        assert registry.remove("test_agent") is True


class TestMCPClient:
    """Test MCP client functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """Test MCP client initialization."""
        client = MCPClient("http://localhost:8001")
        assert client.server_url == "http://localhost:8001"
        assert client.session is None
        assert client.available_tools == []
        assert client._initialized is False
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialize_success(self):
        """Test successful MCP client initialization."""
        client = MCPClient("http://localhost:8001")
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            from unittest.mock import MagicMock
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Properly mock the async context manager and async json method
            class MockResponse:
                status = 200
                async def json(self):
                    return {"tools": [{"name": "web_search"}, {"name": "vector_search"}]}
            
            class MockContextManager:
                async def __aenter__(self):
                    return MockResponse()
                async def __aexit__(self, exc_type, exc, tb):
                    return False
            
            mock_session.get.return_value = MockContextManager()
            
            await client.initialize()
            
            assert client._initialized is True
            assert client.available_tools == ["web_search", "vector_search"]
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialize_failure(self):
        """Test MCP client initialization failure."""
        client = MCPClient("http://localhost:8001")
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = Exception("Connection failed")
            
            await client.initialize()
            
            assert client._initialized is False
    
    def test_get_available_tools(self):
        """Test getting available tools."""
        client = MCPClient("http://localhost:8001")
        client.available_tools = ["web_search", "vector_search"]
        
        tools = client.get_available_tools()
        assert tools == ["web_search", "vector_search"]
        # Should return a copy
        assert tools is not client.available_tools


class TestMCPToolManager:
    """Test MCP tool manager functionality."""
    
    def test_tool_manager_initialization(self):
        """Test tool manager initialization."""
        mock_client = Mock()
        manager = MCPToolManager(mock_client)
        
        assert manager.mcp_client == mock_client
        assert manager.tool_usage_stats == {}
        assert manager.rate_limits == {}
    
    @pytest.mark.asyncio
    async def test_call_tool_with_retry_success(self):
        """Test successful tool call with retry."""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = {"result": "success"}
        
        manager = MCPToolManager(mock_client)
        result = await manager.call_tool_with_retry("test_tool", {"param": "value"})
        
        assert result == {"result": "success"}
        assert manager.tool_usage_stats["test_tool"]["successes"] == 1
        assert manager.tool_usage_stats["test_tool"]["failures"] == 0
    
    @pytest.mark.asyncio
    async def test_call_tool_with_retry_failure(self):
        """Test tool call failure with retry."""
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = None
        
        manager = MCPToolManager(mock_client)
        result = await manager.call_tool_with_retry("test_tool", {"param": "value"})
        
        assert result is None
        assert manager.tool_usage_stats["test_tool"]["successes"] == 0
        assert manager.tool_usage_stats["test_tool"]["failures"] == 3  # 3 retry attempts = 3 failures
    
    def test_get_tool_stats(self):
        """Test getting tool usage statistics."""
        mock_client = Mock()
        manager = MCPToolManager(mock_client)
        manager.tool_usage_stats = {
            "web_search": {"calls": 10, "successes": 8, "failures": 2}
        }
        
        stats = manager.get_tool_stats()
        assert stats["web_search"]["calls"] == 10
        assert stats["web_search"]["successes"] == 8
        assert stats["web_search"]["failures"] == 2
    
    def test_get_tool_success_rate(self):
        """Test getting tool success rate."""
        mock_client = Mock()
        manager = MCPToolManager(mock_client)
        manager.tool_usage_stats = {
            "web_search": {"calls": 10, "successes": 8, "failures": 2}
        }
        
        success_rate = manager.get_tool_success_rate("web_search")
        assert success_rate == 0.8
    
    def test_get_tool_success_rate_no_calls(self):
        """Test getting tool success rate with no calls."""
        mock_client = Mock()
        manager = MCPToolManager(mock_client)
        
        success_rate = manager.get_tool_success_rate("web_search")
        assert success_rate == 0.0


class TestResearchSummaryAgent:
    """Test ResearchSummaryAgent with MCP tool integration."""
    
    def test_research_agent_initialization(self):
        """Test research agent initialization."""
        agent = ResearchSummaryAgent("research_agent")
        assert agent.agent_id == "research_agent"
        assert "academic" in agent.summary_templates
        assert "business" in agent.summary_templates
        assert "technical" in agent.summary_templates
    
    @pytest.mark.asyncio
    async def test_search_recent_research_with_tools(self):
        """Test searching recent research with MCP tools."""
        agent = ResearchSummaryAgent("research_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["web_search"]
        agent.set_mcp_client(mock_client)
        
        # Mock web search results
        agent.web_search = AsyncMock()
        agent.web_search.return_value = {
            "results": [
                {"title": "Test Paper", "snippet": "Test content", "url": "http://test.com"}
            ]
        }
        
        results = await agent._search_recent_research("AI research")
        
        assert "results" in results
        assert len(results["results"]) > 0
        assert "sources" in results
    
    @pytest.mark.asyncio
    async def test_search_recent_research_no_tools(self):
        """Test searching recent research without MCP tools."""
        agent = ResearchSummaryAgent("research_agent")
        
        results = await agent._search_recent_research("AI research")
        
        assert results["results"] == []
        assert results["sources"] == []
    
    @pytest.mark.asyncio
    async def test_search_local_content_with_tools(self):
        """Test searching local content with MCP tools."""
        agent = ResearchSummaryAgent("research_agent")
        mock_client = Mock()
        mock_client.get_available_tools.return_value = ["vector_search"]
        agent.set_mcp_client(mock_client)
        
        # Mock vector search results
        agent.vector_search = AsyncMock()
        agent.vector_search.return_value = {
            "results": [
                {"title": "Local Content", "content": "Test content", "url": "http://local.com"}
            ]
        }
        
        results = await agent._search_local_content("AI research")
        
        assert "results" in results
        assert len(results["results"]) > 0
    
    def test_extract_key_findings(self):
        """Test extracting key findings from search results."""
        agent = ResearchSummaryAgent("research_agent")
        
        research_results = {
            "results": [
                {"title": "Test Paper", "snippet": "Test content", "url": "http://test.com", "relevance": 0.9}
            ]
        }
        
        local_content = {
            "results": [
                {"title": "Local Content", "content": "Test content", "url": "http://local.com", "similarity": 0.8}
            ]
        }
        
        findings = agent._extract_key_findings(research_results, local_content)
        
        assert len(findings) == 2
        assert findings[0]["title"] == "Test Paper"  # Higher relevance score
        assert findings[1]["title"] == "Local Content"
    
    def test_summary_templates(self):
        """Test summary template generation."""
        agent = ResearchSummaryAgent("research_agent")
        
        findings = [
            {"title": "Test Finding", "snippet": "Test content", "source_type": "web_search"}
        ]
        
        # Test academic template
        academic_summary = agent._academic_summary_template("AI Research", findings, 1000)
        assert "Research Summary: AI Research" in academic_summary
        assert "Key Findings" in academic_summary
        
        # Test business template
        business_summary = agent._business_summary_template("AI Research", findings, 1000)
        assert "Business Impact Analysis" in business_summary
        assert "Market Developments" in business_summary
        
        # Test technical template
        technical_summary = agent._technical_summary_template("AI Research", findings, 1000)
        assert "Technical Overview" in technical_summary
        assert "Recent Developments" in technical_summary
    
    @pytest.mark.asyncio
    async def test_get_research_capabilities(self):
        """Test getting research agent capabilities."""
        agent = ResearchSummaryAgent("research_agent")
        
        capabilities = await agent.get_research_capabilities()
        
        assert capabilities["agent_id"] == "research_agent"
        assert "academic_research_summary" in capabilities["capabilities"]
        assert "web_search_integration" in capabilities["capabilities"]
        assert "vector_search_integration" in capabilities["capabilities"]
        assert "academic" in capabilities["templates"]
        assert "business" in capabilities["templates"]
        assert "technical" in capabilities["templates"]


if __name__ == "__main__":
    pytest.main([__file__]) 