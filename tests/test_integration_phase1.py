from src.core.core import query_llm


def test_query_llm_integration():
    result = query_llm("Hello")
    print("LLM result:", result)
    assert isinstance(result, str)
    assert result.strip() != ""
    # If Ollama is not running, the result should contain an error message
    assert (
        "An error occurred while querying the LLM." not in result
        and "An unexpected error occurred while querying the LLM." not in result
    ), (
        "LLM query failed. Error message returned: {}".format(result)
    )
