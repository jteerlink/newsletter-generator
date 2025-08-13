"""
Newsletter Generation Agents

This module provides the main agent classes for newsletter generation.
It imports and re-exports the modular agent classes for backward compatibility.
"""

from __future__ import annotations

# Import base classes
from .base import AgentContext, AgentType, SimpleAgent, TaskResult, TaskStatus
from .editing import EditorAgent
from .management import ManagerAgent, WorkflowPlan, WorkflowStep

# Import specialized agents
from .research import ResearchAgent
from .writing import WriterAgent

# Re-export all agent classes for backward compatibility
__all__ = [
    # Base classes
    'SimpleAgent',
    'AgentType',
    'TaskResult',
    'TaskStatus',
    'AgentContext',

    # Specialized agents
    'ResearchAgent',
    'WriterAgent',
    'EditorAgent',
    'ManagerAgent',
    'WorkflowStep',
    'WorkflowPlan'
]

def create_agent(agent_type: str) -> SimpleAgent:
    """Factory for tests expecting this convenience creator."""
    at = agent_type.lower()
    if at == 'research':
        return ResearchAgent()
    if at == 'writer':
        return WriterAgent()
    if at == 'editor':
        return EditorAgent()
    if at == 'manager':
        return ManagerAgent()
    raise ValueError(f"Unknown agent type: {agent_type}")

def get_available_agent_types() -> List[str]:
    return ['research', 'writer', 'editor', 'manager']
