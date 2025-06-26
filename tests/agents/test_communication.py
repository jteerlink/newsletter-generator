import pytest
from src.agents.base.communication import Message, InMemoryMessageBus

class DummyAgent:
    def __init__(self, agent_id, bus):
        self.agent_id = agent_id
        self.bus = bus
        self.received = []
    def send_message(self, recipient_id, message_type, payload):
        msg = Message(sender=self.agent_id, recipient=recipient_id, message_type=message_type, payload=payload)
        self.bus.send(msg)
    def receive_messages(self):
        msgs = self.bus.receive(self.agent_id)
        self.received.extend(msgs)
        return msgs

def test_agent_communication():
    bus = InMemoryMessageBus()
    agent_a = DummyAgent("A", bus)
    agent_b = DummyAgent("B", bus)
    # Agent A sends a message to B
    agent_a.send_message("B", "task", {"foo": "bar"})
    # Agent B receives it
    msgs = agent_b.receive_messages()
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg.sender == "A"
    assert msg.recipient == "B"
    assert msg.message_type == "task"
    assert msg.payload == {"foo": "bar"}
    assert msg.correlation_id
    assert msg.timestamp 