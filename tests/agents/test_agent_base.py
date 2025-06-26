import pytest
from src.agents.base.agent_base import AgentBase

class DummyAgent(AgentBase):
    def run(self):
        return "running"
    def receive_message(self, message):
        self.state['last_message'] = message
        return "received"
    def send_message(self, recipient_id, message):
        return f"sent to {recipient_id}"

def test_dummy_agent_instantiation():
    agent = DummyAgent(agent_id="test_agent")
    assert agent.agent_id == "test_agent"
    assert agent.run() == "running"
    assert agent.receive_message({"foo": "bar"}) == "received"
    assert agent.state['last_message'] == {"foo": "bar"}
    assert agent.send_message("other_agent", {"msg": 1}) == "sent to other_agent" 