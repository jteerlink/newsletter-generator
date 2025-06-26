import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    """
    Represents a task that can be assigned to an agent.
    """
    def __init__(self, task_type: str, payload: Dict[str, Any], task_id: str = None):
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.payload = payload
        self.status = TaskStatus.PENDING
        self.assigned_agent: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.assigned_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def assign(self, agent_id: str) -> None:
        """Assign this task to an agent."""
        self.assigned_agent = agent_id
        self.status = TaskStatus.ASSIGNED
        self.assigned_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark this task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def fail(self) -> None:
        """Mark this task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        } 