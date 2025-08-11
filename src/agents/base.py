"""
Base Agent Classes for Newsletter Generation

This module provides the foundational classes for all agents in the newsletter generation system.
It includes abstract base classes, common utilities, and standardized interfaces.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.core import query_llm
from src.core.exceptions import AgentError
from src.core.tool_usage_tracker import get_tool_tracker
from src.tools.tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of core agent types."""
    RESEARCH = "research"
    WRITER = "writer"
    EDITOR = "editor"
    MANAGER = "manager"


class TaskStatus(Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentContext:
    """Context information for agent execution."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    agent_name: str
    status: TaskStatus
    result: str
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        agent_type: AgentType,
        tools: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout: int = 60
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.agent_type = agent_type
        self.tools = tools or []
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize tools
        self.available_tools = {
            name: func for name, func in AVAILABLE_TOOLS.items()
            if name in self.tools
        }

        # Initialize tracking
        self.tool_tracker = get_tool_tracker()
        self.context = AgentContext()
        self.execution_history: List[TaskResult] = []

        logger.info(
            f"Initialized agent: {
                self.name} ({
                self.agent_type.value})")

    @abstractmethod
    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute a task. Must be implemented by subclasses."""
        pass

    def set_context(
            self,
            workflow_id: str = None,
            session_id: str = None,
            **metadata):
        """Set execution context for the agent."""
        if workflow_id:
            self.context.workflow_id = workflow_id
        if session_id:
            self.context.session_id = session_id
        if metadata:
            self.context.metadata.update(metadata)

    def get_execution_history(self, limit: int = 10) -> List[TaskResult]:
        """Get recent execution history."""
        return self.execution_history[-limit:
                                      ] if self.execution_history else []

    def get_tool_usage_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get tool usage analytics for this agent."""
        analytics = self.tool_tracker.generate_usage_analytics(hours_back)
        agent_specific_entries = self.tool_tracker.get_tool_usage_history(
            agent_name=self.name,
            hours_back=hours_back
        )

        return {
            "agent_name": self.name,
            "agent_type": self.agent_type.value,
            "total_tool_calls": len(agent_specific_entries),
            "session_id": self.context.session_id,
            "workflow_id": self.context.workflow_id,
            "agent_tool_usage": [entry.to_dict() for entry in agent_specific_entries[:10]],
            "system_analytics": analytics
        }

    def _record_execution(
            self,
            task_id: str,
            result: str,
            error: Optional[str] = None,
            execution_time: float = 0.0,
            **metadata) -> TaskResult:
        """Record task execution result."""
        status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        task_result = TaskResult(
            task_id=task_id,
            agent_name=self.name,
            status=status,
            result=result,
            error_message=error,
            execution_time=execution_time,
            metadata=metadata
        )
        self.execution_history.append(task_result)
        return task_result

    def _execute_with_retry(self, task_func, *args, **kwargs) -> str:
        """Execute a function with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                result = task_func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    f"Agent {
                        self.name} completed task successfully (attempt {
                        attempt + 1})")
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Agent {
                        self.name} attempt {
                        attempt +
                        1} failed: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        error_msg = f"Agent {
            self.name} failed after {
            self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise AgentError(error_msg) from last_error


class SimpleAgent(BaseAgent):
    """Simple agent implementation with basic LLM and tool capabilities."""

    def __init__(self, name: str, role: str, goal: str, backstory: str,
                 agent_type: AgentType = AgentType.RESEARCH,
                 tools: Optional[List[str]] = None, **kwargs):
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            agent_type=agent_type,
            tools=tools,
            **kwargs
        )

    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute a task using available tools and LLM reasoning."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Phase 1 Debug Logging - Workflow-level task execution tracking
            logger.info(f"Agent {self.name} executing task: {task}")
            logger.info(
                f"Agent {
                    self.name} context - Session: {
                    self.context.session_id}, Workflow: {
                    self.context.workflow_id}")
            logger.info(
                f"Agent {
                    self.name} available tools: {
                    list(
                        self.available_tools.keys())}")

            # Create the prompt for the agent
            prompt = self._build_prompt(task, context)
            logger.debug(
                f"Agent {
                    self.name} generated prompt length: {
                    len(prompt)} chars")

            # Query the LLM
            logger.info(f"Agent {self.name} querying LLM for initial response")
            response = query_llm(prompt)
            logger.debug(
                f"Agent {
                    self.name} received LLM response length: {
                    len(response)} chars")

            # Check if the agent needs to use tools
            logger.info(f"Agent {self.name} checking if tools should be used")
            if self._should_use_tools(response):
                logger.info(
                    f"Agent {
                        self.name} WILL use tools - executing tool phase")
                tool_output = self._execute_tools(task)
                logger.info(
                    f"Agent {
                        self.name} tool execution completed, re-querying LLM")
                # Re-query with tool results
                enhanced_prompt = self._build_prompt_with_tools(
                    task, context, tool_output)
                response = query_llm(enhanced_prompt)
                logger.info(
                    f"Agent {
                        self.name} final response with tools completed")
            else:
                logger.info(
                    f"Agent {
                        self.name} will NOT use tools - using LLM-only response")

            execution_time = time.time() - start_time
            self._record_execution(
                task_id, response, execution_time=execution_time)

            logger.info(
                f"Agent {
                    self.name} completed task in {
                    execution_time:.2f}s")
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error in agent {self.name}: {str(e)}"
            self._record_execution(
                task_id,
                "",
                error=error_msg,
                execution_time=execution_time)
            logger.error(error_msg)
            return error_msg

    def _build_prompt(self, task: str, context: str = "") -> str:
        """Build the initial prompt for the agent."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Task: {task}"
        ]

        if context:
            prompt_parts.extend([
                "",
                f"Context: {context}"
            ])

        if self.tools:
            prompt_parts.extend([
                "",
                "🛠️ AVAILABLE TOOLS:",
                *[f"- {tool}" for tool in self.tools],
                "",
                "⚡ TOOL USAGE GUIDELINES:",
                f"- As a {self.agent_type} agent, you should actively consider using tools",
                "- For research tasks: ALWAYS search for current information",
                "- For writing tasks: Verify key facts and statistics",
                "- For editing tasks: Fact-check important claims",
                "- When in doubt, USE TOOLS to ensure accuracy and completeness"
            ])

        prompt_parts.extend([
            "",
            "📋 TASK EXECUTION:",
            "1. Analyze if this task would benefit from tool usage",
            "2. If yes, explicitly state: 'I need to search for...' or 'I should verify...'",
            "3. Use appropriate tools to gather information",
            "4. Provide a comprehensive response based on your findings",
            "",
            "Please execute this task thoughtfully and use tools when they would improve the quality of your response."
        ])

        return "\n".join(prompt_parts)

    def _should_use_tools(self, response: str) -> bool:
        """Determine if tools should be used based on LLM response and agent type."""
        tool_indicators = [
            "need to search", "need to look up", "need to find",
            "search for", "look up", "find information",
            "use search", "use tools", "need tools",
            "search the web", "search online", "research",
            "verify", "check", "confirm", "validate",
            "current", "latest", "recent", "updated",
            "statistics", "data", "facts", "studies"
        ]

        response_lower = response.lower()
        indicator_found = any(
            indicator in response_lower for indicator in tool_indicators)

        # Phase 2.2: Enhanced tool detection based on agent type and task
        agent_should_use_tools = False

        # Research agents should use tools more aggressively
        if self.agent_type == AgentType.RESEARCH and len(
                self.available_tools) > 0:
            research_triggers = [
                "research", "information", "data", "facts",
                "current", "latest", "trends", "developments",
                "statistics", "analysis", "insights"
            ]
            agent_should_use_tools = any(
                trigger in response_lower for trigger in research_triggers)

        # Writers should verify facts occasionally
        elif self.agent_type == AgentType.WRITER and len(self.available_tools) > 0:
            writer_triggers = [
                "verify", "check", "confirm", "validate",
                "accurate", "current", "latest", "recent"
            ]
            agent_should_use_tools = any(
                trigger in response_lower for trigger in writer_triggers)

        # Editors should verify claims and facts
        elif self.agent_type == AgentType.EDITOR and len(self.available_tools) > 0:
            editor_triggers = [
                "verify", "fact-check", "accuracy", "validate",
                "confirm", "check", "ensure", "accurate"
            ]
            agent_should_use_tools = any(
                trigger in response_lower for trigger in editor_triggers)

        # Final decision: use tools if either indicator found OR agent type
        # suggests it
        should_use = indicator_found or agent_should_use_tools

        # Phase 1 Debug Logging - Tool Usage Decision Making
        logger.info(f"Agent {self.name} tool usage decision for '{
                    response[:100]}...': {should_use}")
        logger.debug(f"Response indicators found: {
                     [ind for ind in tool_indicators if ind in response_lower]}")
        logger.debug(f"Agent type triggers: {
                     agent_should_use_tools}, General indicators: {indicator_found}")
        logger.debug(f"Available tools count: {len(self.available_tools)}")

        return should_use

    def _execute_tools(self, task: str) -> str:
        """Execute available tools based on the task."""
        # Phase 1 Debug Logging - Tool Call Investigation
        logger.info(f"Agent {self.name} checking {
                    len(self.available_tools)} available tools")
        logger.info(f"Available tools: {list(self.available_tools.keys())}")
        logger.info(
            f"Agent context - Session: {
                self.context.session_id}, Workflow: {
                self.context.workflow_id}")

        # Phase 2.2: Enhanced tool tracking with proper context manager
        from src.core.tool_usage_tracker import get_tool_tracker
        tool_tracker = get_tool_tracker()

        tool_results = []

        for tool_name, tool_func in self.available_tools.items():
            try:
                # Enhanced tool tracking with context manager
                with tool_tracker.track_tool_usage(
                    tool_name=tool_name,
                    agent_name=self.name,
                    workflow_id=self.context.workflow_id,
                    session_id=self.context.session_id,
                    input_data={"task": task},
                    context={"agent_type": str(self.agent_type), "tool_execution": "agent_level"}
                ):
                    # Execute tool based on type
                    if tool_name == "search_web":
                        search_query = self._extract_search_query(task)
                        result = tool_func(search_query)
                    elif tool_name == "search_knowledge_base":
                        result = tool_func(task)
                    else:
                        result = tool_func(task)

                tool_results.append(f"{tool_name}: {result}")
                logger.info(
                    f"Agent {
                        self.name} successfully used tool {tool_name}")

            except Exception as e:
                error_msg = f"Error using tool {tool_name}: {e}"
                logger.error(error_msg)
                tool_results.append(f"{tool_name}: Error - {error_msg}")

        return "\n\n".join(
            tool_results) if tool_results else "No tools were used."

    def _get_effective_tool_name(self, requested_tool: str) -> str:
        """Get the effective tool name from a requested tool string."""
        requested_lower = requested_tool.lower()

        if "search" in requested_lower and "web" in requested_lower:
            return "search_web"
        elif "search" in requested_lower and "knowledge" in requested_lower:
            return "search_knowledge_base"
        elif "search" in requested_lower:
            return "search_web"  # Default to web search
        else:
            return requested_tool

    def _extract_search_query(self, task: str) -> str:
        """Extract a search query from a task description."""
        # Simple extraction - can be enhanced with more sophisticated NLP
        words = task.split()

        # Remove common words that don't add value to search
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by"}
        search_words = [
            word for word in words if word.lower() not in stop_words]

        # Take first 5-8 words as search query
        query = " ".join(search_words[:8])
        return query if query else task

    def _build_prompt_with_tools(
            self,
            task: str,
            context: str,
            tool_output: str) -> str:
        """Build prompt that includes tool results."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Task: {task}"
        ]

        if context:
            prompt_parts.extend([
                "",
                f"Context: {context}"
            ])

        prompt_parts.extend([
            "",
            "Tool Results:",
            tool_output,
            "",
            "Based on the tool results above, please provide a comprehensive response to the task.",
            "Integrate the information from the tools into your response."
        ])

        return "\n".join(prompt_parts)


class AgentFactory:
    """Simplified factory for creating core agents only."""

    @staticmethod
    def create_agent(agent_type: AgentType, **kwargs) -> BaseAgent:
        """Create an agent of the specified type (core agents only)."""
        if agent_type == AgentType.RESEARCH:
            from .research import ResearchAgent
            return ResearchAgent(**kwargs)
        elif agent_type == AgentType.WRITER:
            from .writing import WriterAgent
            return WriterAgent(**kwargs)
        elif agent_type == AgentType.EDITOR:
            from .editing import EditorAgent
            return EditorAgent(**kwargs)
        elif agent_type == AgentType.MANAGER:
            from .management import ManagerAgent
            return ManagerAgent(**kwargs)
        else:
            raise ValueError(f"Unknown core agent type: {
                             agent_type}. Only RESEARCH, WRITER, EDITOR, and MANAGER are supported.")

    @staticmethod
    def create_simple_agent(name: str,
                            role: str,
                            goal: str,
                            backstory: str,
                            tools: Optional[List[str]] = None,
                            **kwargs) -> SimpleAgent:
        """Create a simple agent with custom configuration."""
        return SimpleAgent(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            **kwargs
        )
