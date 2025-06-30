"""Prompt templates for Phase-1 prompt engineering experiments."""

from __future__ import annotations

# Example placeholder prompt builder
def get_research_topic_prompt(topic: str) -> str:
    """Return a formatted prompt asking the LLM to research a topic."""
    return (
        f"You are a research assistant. Provide a concise overview of the topic: {topic}. "
        "Include key facts, recent developments, and cite primary sources."
    ) 