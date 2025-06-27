from agents.rag_pipeline.response_evaluator_agent import ResponseEvaluatorAgent

def test_response_evaluator_agent_evaluate_and_memory():
    agent = ResponseEvaluatorAgent()
    msg = {'content': {'response': 'some output'}, 'sender': 'llm'}
    response = agent.receive_message(msg)
    eval_result = response['content']['evaluation']
    assert 'relevance' in eval_result and 'completeness' in eval_result and 'factuality' in eval_result
    mem = agent.get_memory().get_evaluations()
    assert mem[0]['response'] == 'some output'
    assert 'evaluation' in mem[0] 