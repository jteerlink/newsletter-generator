from agents.rag_pipeline.source_selector_agent import SourceSelectorAgent

def test_source_selector_agent_select_and_memory():
    agent = SourceSelectorAgent()
    msg = {'content': {'query': 'find sources'}, 'sender': 'user'}
    response = agent.receive_message(msg)
    assert response['content']['sources'] == ['vector_db']
    assert 'rationale' in response['content']
    mem = agent.get_memory().get_contexts()
    assert mem[0]['query'] == 'find sources'
    assert mem[0]['sources'] == ['vector_db'] 