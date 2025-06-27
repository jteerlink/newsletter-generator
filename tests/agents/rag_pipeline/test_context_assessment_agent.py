from agents.rag_pipeline.context_assessment_agent import ContextAssessmentAgent

def test_context_assessment_agent_assess_and_memory():
    agent = ContextAssessmentAgent()
    msg = {'content': {'query': 'short'}, 'sender': 'user'}
    response = agent.receive_message(msg)
    assert response['content']['context_needed'] is True
    mem = agent.get_memory().get_contexts()
    assert mem[0]['query'] == 'short'
    assert mem[0]['context_needed'] is True 