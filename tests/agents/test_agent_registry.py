import pytest
from src.agents.base.agent_registry import AgentRegistry
from src.agents.base.agent_base import AgentBase

class DummyAgent(AgentBase):
    def run(self):
        return "running"
    def receive_message(self, message):
        return "received"
    def send_message(self, recipient_id, message):
        return "sent"

def test_agent_registry():
    registry = AgentRegistry()
    agent1 = DummyAgent("agent1")
    agent2 = DummyAgent("agent2")
    
    # Test registration
    registry.register(agent1)
    registry.register(agent2)
    assert registry.count() == 2
    assert "agent1" in registry.list_agents()
    assert "agent2" in registry.list_agents()
    
    # Test retrieval
    retrieved = registry.get("agent1")
    assert retrieved is agent1
    assert registry.get("nonexistent") is None
    
    # Test removal
    assert registry.remove("agent1") is True
    assert registry.count() == 1
    assert "agent1" not in registry.list_agents()
    assert registry.remove("nonexistent") is False 