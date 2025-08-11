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

# Legacy aliases for backward compatibility
ResearchAgent = ResearchAgent
WriterAgent = WriterAgent
EditorAgent = EditorAgent
ManagerAgent = ManagerAgent

# Legacy classes for backward compatibility (to be refactored in future phases)


class PlannerAgent(SimpleAgent):
    """Legacy PlannerAgent for backward compatibility."""

    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.get(
                'name',
                'PlannerAgent'),
            role="Content Planner",
            goal="Plan and structure newsletter content",
            backstory="You are an experienced content planner specializing in newsletter strategy.",
            agent_type=AgentType.MANAGER,
            **kwargs)
