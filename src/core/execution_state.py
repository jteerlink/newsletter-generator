"""
ExecutionState: Short-term workflow execution state.

This module defines the ExecutionState dataclass that tracks the current
state of workflow execution, including task results, revision cycles,
quality scores, and feedback history.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskResult:
    """Result of a specific task execution."""
    task_id: str
    task_type: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    quality_score: Optional[float] = None
    feedback: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def mark_completed(
            self, result: Dict[str, Any], execution_time: float = 0.0) -> None:
        """Mark task as completed with result."""
        self.status = 'completed'
        self.result = result
        self.execution_time = execution_time
        self.completed_at = time.time()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error message."""
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = time.time()

    def update_quality_score(self, score: float) -> None:
        """Update the quality score for this task."""
        self.quality_score = score

    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add feedback to this task."""
        self.feedback = feedback


@dataclass
class ExecutionState:
    """Short-term workflow execution state."""
    workflow_id: str
    current_phase: str = "initialized"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    revision_cycles: Dict[str, int] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize workflow_id if not provided."""
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())

    def add_task_result(self, task_result: TaskResult) -> None:
        """Add a task result to the execution state."""
        self.task_results[task_result.task_id] = task_result
        self.last_updated = time.time()

    def update_phase(self, new_phase: str) -> None:
        """Update the current workflow phase."""
        self.current_phase = new_phase
        self.last_updated = time.time()

    def increment_revision_cycle(self, task_id: str) -> int:
        """Increment revision cycle count for a task."""
        current_count = self.revision_cycles.get(task_id, 0)
        self.revision_cycles[task_id] = current_count + 1
        self.last_updated = time.time()
        return self.revision_cycles[task_id]

    def update_quality_score(self, task_id: str, score: float) -> None:
        """Update quality score for a task."""
        self.quality_scores[task_id] = score
        self.last_updated = time.time()

    def add_feedback(self, feedback: Dict[str, Any]) -> None:
        """Add feedback to the history."""
        feedback['timestamp'] = time.time()
        self.feedback_history.append(feedback)
        self.last_updated = time.time()

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID."""
        return self.task_results.get(task_id)

    def get_completed_tasks(self) -> List[TaskResult]:
        """Get all completed tasks."""
        return [task for task in self.task_results.values()
                if task.status == 'completed']

    def get_failed_tasks(self) -> List[TaskResult]:
        """Get all failed tasks."""
        return [task for task in self.task_results.values()
                if task.status == 'failed']

    def get_pending_tasks(self) -> List[TaskResult]:
        """Get all pending tasks."""
        return [task for task in self.task_results.values()
                if task.status == 'pending']

    def get_average_quality_score(self) -> float:
        """Calculate average quality score across all tasks."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores.values()) / len(self.quality_scores)

    def get_total_execution_time(self) -> float:
        """Calculate total execution time."""
        return time.time() - self.start_time

    def get_phase_duration(self) -> float:
        """Get duration of current phase."""
        # This could be enhanced to track phase start times
        return time.time() - self.start_time

    def is_workflow_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.current_phase == "completed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExecutionState to dictionary for serialization."""
        return {
            'workflow_id': self.workflow_id,
            'current_phase': self.current_phase,
            'task_results': {
                task_id: {
                    'task_id': result.task_id,
                    'task_type': result.task_type,
                    'status': result.status,
                    'result': result.result,
                    'error_message': result.error_message,
                    'execution_time': result.execution_time,
                    'quality_score': result.quality_score,
                    'feedback': result.feedback,
                    'created_at': result.created_at,
                    'completed_at': result.completed_at
                }
                for task_id, result in self.task_results.items()
            },
            'revision_cycles': self.revision_cycles,
            'quality_scores': self.quality_scores,
            'feedback_history': self.feedback_history,
            'start_time': self.start_time,
            'last_updated': self.last_updated
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionState':
        """Create ExecutionState from dictionary."""
        # Reconstruct TaskResult objects
        task_results = {}
        for task_id, task_data in data.get('task_results', {}).items():
            task_results[task_id] = TaskResult(
                task_id=task_data['task_id'],
                task_type=task_data['task_type'],
                status=task_data['status'],
                result=task_data.get('result'),
                error_message=task_data.get('error_message'),
                execution_time=task_data.get('execution_time', 0.0),
                quality_score=task_data.get('quality_score'),
                feedback=task_data.get('feedback'),
                created_at=task_data.get('created_at', time.time()),
                completed_at=task_data.get('completed_at')
            )

        return cls(
            workflow_id=data['workflow_id'],
            current_phase=data.get('current_phase', 'initialized'),
            task_results=task_results,
            revision_cycles=data.get('revision_cycles', {}),
            quality_scores=data.get('quality_scores', {}),
            feedback_history=data.get('feedback_history', []),
            start_time=data.get('start_time', time.time()),
            last_updated=data.get('last_updated', time.time())
        )
