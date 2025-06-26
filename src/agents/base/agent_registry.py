from typing import Dict, List, Optional
from .agent_base import AgentBase

class AgentRegistry:
    """
    Registry for managing agent instances and their lifecycle.
    """
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}

    def register(self, agent: AgentBase) -> None:
        """Register an agent in the registry."""
        self.agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[AgentBase]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self.agents.keys())

    def remove(self, agent_id: str) -> bool:
        """Remove an agent from the registry. Returns True if agent was found and removed."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    def count(self) -> int:
        """Get the number of registered agents."""
        return len(self.agents) 