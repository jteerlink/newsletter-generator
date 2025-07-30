"""
Newsletter Generation Agents

This module provides the main agent classes for newsletter generation.
It imports and re-exports the modular agent classes for backward compatibility.
"""

from __future__ import annotations

# Import base classes
from .base import (
    SimpleAgent,
    AgentType,
    TaskResult,
    TaskStatus,
    AgentContext
)

# Import specialized agents
from .research import ResearchAgent
from .writing import WriterAgent
from .editing import EditorAgent
from .management import ManagerAgent, WorkflowStep, WorkflowPlan

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
            name=kwargs.get('name', 'PlannerAgent'),
            role="Content Planner",
            goal="Plan and structure newsletter content",
            backstory="You are an experienced content planner specializing in newsletter strategy.",
            agent_type=AgentType.MANAGER,
            **kwargs
        )

class Task:
    """Legacy Task class for backward compatibility."""
    def __init__(self, description: str, agent=None, context="", **kwargs):
        self.description = description
        self.agent = agent
        self.context = context
        self.result = None
        self.kwargs = kwargs

class EnhancedCrew:
    """Legacy EnhancedCrew class for backward compatibility."""
    def __init__(self, agents, tasks, workflow_type="sequential"):
        self.agents = agents
        self.tasks = tasks
        self.workflow_type = workflow_type
        self.task_results = []
        self.agent_performance = {}
    
    def run(self):
        """Simple implementation for backward compatibility."""
        results = []
        for i, task in enumerate(self.tasks):
            if task.agent:
                print(f"🔍 DEBUG: Executing task {i+1}: {task.agent.name}")
                print(f"🔍 DEBUG: Task description: {task.description[:200]}...")
                result = task.agent.execute_task(task.description)
                print(f"🔍 DEBUG: Task {i+1} result length: {len(result)} characters, {len(result.split())} words")
                results.append(result)
                result = task.agent.execute_task(task.description)
                results.append(result)
        return results
    
    def kickoff(self):
        """Execute the workflow and return the final result."""
        results = self.run()
        # Return the last result (typically the final newsletter content)
        return "\n\n".join(results) if results else ""

# Convenience function to create agent instances
def create_agent(agent_type: str, **kwargs) -> SimpleAgent:
    """
    Create an agent instance based on type.
    
    Args:
        agent_type: Type of agent to create ('research', 'writer', 'editor', 'manager')
        **kwargs: Additional arguments to pass to agent constructor
        
    Returns:
        Agent instance
    """
    agent_map = {
        'research': ResearchAgent,
        'writer': WriterAgent,
        'editor': EditorAgent, 
        'manager': ManagerAgent
        # 'rag': AgenticRAGAgent,
        # 'optimizer': ContentFormatOptimizer,
        # 'pipeline': DailyQuickPipeline,
        # 'workflow': HybridWorkflowManager,
        # 'qa': QualityAssuranceSystem
    }
    
    agent_class = agent_map.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(**kwargs)

# Convenience function to get all available agent types
def get_available_agent_types() -> list[str]:
    """Get list of all available agent types."""
    return [
        'research',
        'writer', 
        'editor',
        'manager'
        # 'rag',
        # 'optimizer',
        # 'pipeline',
        # 'workflow',
        # 'qa'
    ]
