from agents.rag_pipeline.query_writer_agent import QueryWriterAgent

def test_query_writer_agent_refine_and_memory():
    agent = QueryWriterAgent()
    msg = {'content': {'query': '  test query  '}, 'sender': 'user'}
    response = agent.receive_message(msg)
    assert response['content']['refined_query'] == 'test query'
    mem = agent.get_memory().get_queries()
    assert mem[0]['original'] == '  test query  '
    assert mem[0]['refined'] == 'test query' 