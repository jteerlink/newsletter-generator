from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from .agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class AgentBase(ABC):
    """
    Abstract base class for all agents with MCP tool integration capabilities.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {}
        self.memory = AgentMemory(agent_id=agent_id)  # Per-agent memory/state with agent_id
        self.mcp_client = None  # Will be set by agent registry
        self.available_tools = []

    @abstractmethod
    def run(self):
        """Main execution loop for the agent."""
        pass

    @abstractmethod
    def receive_message(self, message: dict):
        """Handle incoming messages."""
        pass

    @abstractmethod
    def send_message(self, recipient_id: str, message: dict):
        """Send a message to another agent."""
        pass

    def set_mcp_client(self, mcp_client):
        """Set the MCP client for tool access."""
        self.mcp_client = mcp_client
        logger.info(f"Agent {self.agent_id} MCP client configured")

    def get_available_tools(self) -> list:
        """Get list of available MCP tools."""
        if self.mcp_client:
            return self.mcp_client.get_available_tools()
        return []

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Call an MCP tool with given parameters.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            
        Returns:
            Tool response or None if failed
        """
        if not self.mcp_client:
            logger.warning(f"Agent {self.agent_id} has no MCP client configured")
            return None
        
        try:
            logger.info(f"Agent {self.agent_id} calling MCP tool: {tool_name}")
            response = await self.mcp_client.call_tool(tool_name, parameters)
            logger.info(f"Agent {self.agent_id} received response from {tool_name}")
            return response
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to call MCP tool {tool_name}: {e}")
            return None

    async def web_search(self, query: str, max_results: int = 10, 
                        date_range: str = "week", content_type: str = "all") -> Optional[Dict[str, Any]]:
        """
        Perform web search using MCP web search tool.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            date_range: Time range (day, week, month, year)
            content_type: Type of content (news, research, blog, all)
            
        Returns:
            Search results or None if failed
        """
        parameters = {
            "query": query,
            "max_results": max_results,
            "date_range": date_range,
            "content_type": content_type
        }
        return await self.call_mcp_tool("web_search", parameters)

    async def vector_search(self, query: str, max_results: int = 20,
                           similarity_threshold: float = 0.7,
                           filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Perform vector database search using MCP vector search tool.
        
        Args:
            query: Semantic search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Additional filters (date_range, source_types, topics)
            
        Returns:
            Search results or None if failed
        """
        parameters = {
            "query": query,
            "max_results": max_results,
            "similarity_threshold": similarity_threshold
        }
        if filters:
            parameters["filters"] = filters
        
        return await self.call_mcp_tool("vector_search", parameters)

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use a specific MCP tool."""
        available_tools = self.get_available_tools()
        return tool_name in available_tools

    def get_tool_capabilities(self) -> Dict[str, Any]:
        """Get agent's tool usage capabilities."""
        return {
            "agent_id": self.agent_id,
            "available_tools": self.get_available_tools(),
            "mcp_client_configured": self.mcp_client is not None
        }

    def get_memory(self) -> AgentMemory:
        """Access the agent's memory/state object."""
        return self.memory 

    def request_delegation(self, agent_registry, agent_type: str, task: dict):
        """Request delegation of a task to another agent of the given type (stub)."""
        return agent_registry.delegate_task(agent_type, task)

    def set_persistence(self, persistence):
        """Set persistence layer for the agent's memory."""
        self.memory.persistence = persistence
        # Reload memory from persistence
        self.memory._load_from_persistence() 