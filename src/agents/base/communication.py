import uuid
from datetime import datetime
from typing import Dict, Any, List

class Message:
    """
    Represents a message for agent communication.
    """
    def __init__(self, sender: str, recipient: str, message_type: str, payload: Dict[str, Any], correlation_id: str = None):
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type  # e.g., 'task', 'result', 'error', 'status'
        self.payload = payload
        self.timestamp = datetime.utcnow().isoformat()
        self.correlation_id = correlation_id or str(uuid.uuid4())

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
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