import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class MessageType(Enum):
    """Message types for agent communication, including critique, delegation, escalation, and feedback."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    TASK = "task"
    RESULT = "result"
    CRITIQUE = "critique"
    DELEGATION = "delegation"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"


class Message:
    """
    Represents a message for agent communication.
    """
    def __init__(self, sender: str, recipient: str, type: MessageType, content: Dict[str, Any], correlation_id: str = None):
        self.sender = sender
        self.recipient = recipient
        self.type = type
        self.content = content
        self.timestamp = datetime.utcnow().isoformat()
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class InMemoryMessageBus:
    """
    Simple in-memory message bus for agent-to-agent communication (for testing).
    """
    def __init__(self):
        self.messages: List[Message] = []

    def send(self, message: Message):
        self.messages.append(message)

    def receive(self, recipient_id: str) -> List[Message]:
        received = [m for m in self.messages if m.recipient == recipient_id]
        self.messages = [m for m in self.messages if m.recipient != recipient_id]
        return received 