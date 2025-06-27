from agents.rag_pipeline.prompt_builder import PromptBuilder

def test_prompt_builder_build_and_log():
    pb = PromptBuilder()
    prompt = pb.build_prompt('query', 'context')
    assert 'Context:' in prompt and 'Query:' in prompt
    log = pb.get_log()
    assert log[0]['refined_query'] == 'query'
    assert log[0]['context'] == 'context'
    assert 'prompt' in log[0] 