from src.core import prompts


def test_get_research_topic_prompt():
    topic = "AI"
    result = prompts.get_research_topic_prompt(topic)
    assert isinstance(result, str)
    assert topic in result


def test_get_research_topic_prompt_empty():
    result = prompts.get_research_topic_prompt("")
    assert isinstance(result, str)
    assert result.strip() != ""


def test_get_summary_prompt():
    text = "Sample text."
    result = prompts.get_summary_prompt(text)
    assert isinstance(result, str)
    assert text in result


def test_get_summary_prompt_empty():
    result = prompts.get_summary_prompt("")
    assert isinstance(result, str)
    assert result.strip() != ""


def test_get_haiku_prompt():
    subject = "the ocean"
    result = prompts.get_haiku_prompt(subject)
    assert isinstance(result, str)
    assert subject in result


def test_get_haiku_prompt_empty():
    result = prompts.get_haiku_prompt("")
    assert isinstance(result, str)
    assert result.strip() != ""


def test_get_benefits_prompt():
    activity = "exercise"
    result = prompts.get_benefits_prompt(activity)
    assert isinstance(result, str)
    assert activity in result


def test_get_benefits_prompt_empty():
    result = prompts.get_benefits_prompt("")
    assert isinstance(result, str)
    assert result.strip() != ""


def test_get_explanation_prompt():
    concept = "quantum computing"
    result = prompts.get_explanation_prompt(concept)
    assert isinstance(result, str)
    assert concept in result


def test_get_explanation_prompt_empty():
    result = prompts.get_explanation_prompt("")
    assert isinstance(result, str)
    assert result.strip() != ""
