"""Prompt templates for Phase-1 prompt engineering experiments."""

from __future__ import annotations


# Example placeholder prompt builder
def get_research_topic_prompt(topic: str) -> str:
    """Return a formatted prompt asking the LLM to research a topic."""
    return f"Research the following topic in detail: {topic}"


def get_summary_prompt(text: str) -> str:
    """
    Return a prompt asking the LLM to summarize the given text in 3 sentences.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The formatted summary prompt.
    """
    return f"Summarize the following text in 3 sentences:\n{text}"


def get_haiku_prompt(subject: str) -> str:
    """
    Return a prompt asking the LLM to write a haiku about the given subject.

    Args:
        subject (str): The subject of the haiku.

    Returns:
        str: The formatted haiku prompt.
    """
    return f"Write a haiku about {subject}."


def get_benefits_prompt(activity: str) -> str:
    """
    Return a prompt asking the LLM to list three benefits of the given activity.

    Args:
        activity (str): The activity to list benefits for.

    Returns:
        str: The formatted benefits prompt.
    """
    return f"List three benefits of {activity}."


def get_explanation_prompt(concept: str) -> str:
    """
    Return a prompt asking the LLM to explain a concept in simple terms.

    Args:
        concept (str): The concept to explain.

    Returns:
        str: The formatted explanation prompt.
    """
    return f"Explain {concept} in simple terms."
