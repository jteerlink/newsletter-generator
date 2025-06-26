import pytest
from src.agents.tasks.task_assignment import TaskAssigner
from src.agents.tasks.task_definition import TaskStatus

def test_task_assignment():
    assigner = TaskAssigner()
    
    # Create tasks
    task1 = assigner.create_task("research", {"topic": "AI"})
    task2 = assigner.create_task("write", {"content": "summary"})
    
    # Test pending tasks
    pending = assigner.get_pending_tasks()
    assert len(pending) == 2
    assert task1.status == TaskStatus.PENDING
    assert task2.status == TaskStatus.PENDING
    
    # Assign tasks
    assert assigner.assign_task(task1.task_id, "agent1") is True
    assert assigner.assign_task(task2.task_id, "agent2") is True
    
    # Test assignment
    assert task1.assigned_agent == "agent1"
    assert task1.status == TaskStatus.ASSIGNED
    assert task1.assigned_at is not None
    
    # Test agent tasks
    agent1_tasks = assigner.get_agent_tasks("agent1")
    assert len(agent1_tasks) == 1
    assert agent1_tasks[0] is task1
    
    # Test completion
    assert assigner.complete_task(task1.task_id) is True
    assert task1.status == TaskStatus.COMPLETED
    assert task1.completed_at is not None
    
    # Test failure
    assert assigner.fail_task(task2.task_id) is True
    assert task2.status == TaskStatus.FAILED 