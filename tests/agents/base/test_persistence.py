import pytest
import tempfile
import os
import json
from agents.base.persistence import AgenticPersistence

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)

@pytest.fixture
def persistence(temp_db):
    """Create a persistence instance with temporary database."""
    return AgenticPersistence(db_path=temp_db, backup_dir=tempfile.mkdtemp())

def test_persistence_initialization(persistence):
    """Test that persistence layer initializes correctly."""
    assert persistence.db_path is not None
    assert os.path.exists(persistence.db_path)

def test_agent_memory_save_load(persistence):
    """Test saving and loading agent memory."""
    agent_id = "test_agent"
    memory_type = "queries"
    data = {"query": "test query", "refined": "refined query"}
    
    persistence.save_agent_memory(agent_id, memory_type, data)
    loaded_data = persistence.load_agent_memory(agent_id, memory_type)
    
    assert len(loaded_data) == 1
    assert loaded_data[0]["query"] == "test query"

def test_workflow_state_save_load(persistence):
    """Test saving and loading workflow state."""
    workflow_id = "test_workflow"
    state_data = {"input": {"query": "test"}, "iterations": 1, "history": []}
    
    persistence.save_workflow_state(workflow_id, state_data)
    loaded_state = persistence.load_workflow_state(workflow_id)
    
    assert loaded_state["input"]["query"] == "test"
    assert loaded_state["iterations"] == 1

def test_logs_save_load(persistence):
    """Test saving and loading logs."""
    action = "test_action"
    details = {"test": "data"}
    
    persistence.save_log(action, details)
    logs = persistence.load_logs()
    
    assert len(logs) == 1
    assert logs[0]["action"] == action
    assert logs[0]["details"]["test"] == "data"

def test_metrics_save_load(persistence):
    """Test saving and loading metrics."""
    metric_type = "test_metric"
    value = 0.85
    
    persistence.save_metric(metric_type, value)
    metrics = persistence.load_metrics(metric_type)
    
    assert len(metrics) == 1
    assert metrics[0] == value

def test_user_feedback_save_load(persistence):
    """Test saving and loading user feedback."""
    workflow_id = "test_workflow"
    feedback_data = {"rating": 0.9, "comment": "Great!"}
    
    persistence.save_user_feedback(workflow_id, feedback_data)
    feedback = persistence.load_user_feedback(workflow_id)
    
    assert len(feedback) == 1
    assert feedback[0]["feedback_data"]["rating"] == 0.9

def test_backup_creation(persistence):
    """Test backup creation functionality."""
    # Add some test data
    persistence.save_agent_memory("agent1", "queries", {"test": "data"})
    persistence.save_workflow_state("workflow1", {"test": "state"})
    persistence.save_log("test_action", {"test": "details"})
    persistence.save_metric("test_metric", 0.75)
    persistence.save_user_feedback("workflow1", {"rating": 0.8})
    
    # Create backup
    backup_path = persistence.create_backup()
    
    assert os.path.exists(backup_path)
    
    # Verify backup content
    with open(backup_path, 'r') as f:
        backup_data = json.load(f)
    
    assert "agent_memory" in backup_data
    assert "workflow_states" in backup_data
    assert "logs" in backup_data
    assert "metrics" in backup_data
    assert "user_feedback" in backup_data
    assert "backup_timestamp" in backup_data

def test_persistence_with_agent_memory(persistence):
    """Test that AgentMemory works with persistence."""
    from agents.base.agent_memory import AgentMemory
    
    agent_id = "test_agent"
    memory = AgentMemory(agent_id=agent_id, persistence=persistence)
    
    # Add data
    memory.add_query({"original": "test", "refined": "refined"})
    memory.add_feedback({"rating": 0.9})
    
    # Create new memory instance (simulates restart)
    memory2 = AgentMemory(agent_id=agent_id, persistence=persistence)
    
    # Verify data was persisted
    queries = memory2.get_queries()
    feedback = memory2.get_feedback()
    
    assert len(queries) == 1
    assert queries[0]["original"] == "test"
    assert len(feedback) == 1
    assert feedback[0]["rating"] == 0.9 