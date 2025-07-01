"""Entry-point stub that will eventually orchestrate multi-agent workflows."""

from __future__ import annotations

from src.agents.agents import BaseAgent
from src.core.prompts import get_research_topic_prompt
from src.core.core import query_llm


def demo() -> None:  # noqa: D401
    prompt = get_research_topic_prompt("The role of AI in climate science")
    response = query_llm(prompt)
    print(response)


if __name__ == "__main__":
    demo()
