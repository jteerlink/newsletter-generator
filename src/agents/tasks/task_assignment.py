from typing import List, Optional
from .task_definition import Task, TaskStatus

class TaskAssigner:
    """
    Handles task assignment and management.
    """
    def __init__(self):
        self.tasks: List[Task] = []

    def create_task(self, task_type: str, payload: dict) -> Task:
        """Create a new task."""
        task = Task(task_type, payload)
        self.tasks.append(task)
        return task

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent. Returns True if successful."""
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.assign(agent_id)
            return True
        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]

    def get_agent_tasks(self, agent_id: str) -> List[Task]:
        """Get all tasks assigned to a specific agent."""
        return [task for task in self.tasks if task.assigned_agent == agent_id]

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed. Returns True if successful."""
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.ASSIGNED:
            task.complete()
            return True
        return False

    def fail_task(self, task_id: str) -> bool:
        """Mark a task as failed. Returns True if successful."""
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.ASSIGNED:
            task.fail()
            return True
        return False 