from abc import ABC, abstractmethod

class AgentBase(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {}

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