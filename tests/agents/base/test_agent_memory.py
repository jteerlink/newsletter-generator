import pytest
from agents.base.agent_memory import AgentMemory

def test_agent_memory_add_and_get():
    mem = AgentMemory()
    mem.add_query({'q': 1})
    mem.add_response({'r': 2})
    mem.add_context({'c': 3})
    mem.add_evaluation({'e': 4})
    assert mem.get_queries() == [{'q': 1}]
    assert mem.get_responses() == [{'r': 2}]
    assert mem.get_contexts() == [{'c': 3}]
    assert mem.get_evaluations() == [{'e': 4}]

def test_agent_memory_clear_and_summary():
    mem = AgentMemory()
    mem.add_query({'q': 1})
    mem.add_response({'r': 2})
    mem.clear()
    assert mem.get_queries() == []
    assert mem.get_responses() == []
    summary = mem.summary()
    assert summary == {'queries': 0, 'responses': 0, 'contexts': 0, 'evaluations': 0, 'feedback': 0}

def test_agent_memory_feedback():
    mem = AgentMemory()
    feedback1 = {'workflow_id': 'w1', 'feedback': {'rating': 1.0}}
    feedback2 = {'workflow_id': 'w2', 'feedback': {'rating': 0.0}}
    mem.add_feedback(feedback1)
    mem.add_feedback(feedback2)
    feedbacks = mem.get_feedback()
    assert feedbacks == [feedback1, feedback2] 