"""Stub definitions for CrewAI agents (Phases 2-3).

The concrete agent classes will be added as soon as CrewAI is integrated.
This stub prevents import errors for now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass
class BaseAgent:
    """Very light placeholder for a CrewAI Agent-like object."""
    name: str
    role: str
    goal: str
    tools: List[str] | None = None

    def act(self, task: str) -> str:  # noqa: D401
        return f"[Agent {self.name} would handle task: {task}]" 