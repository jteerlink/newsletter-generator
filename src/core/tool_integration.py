"""
Tool Integration Engine

Provides a centralized engine for mandatory tool consultation prior to agent
task execution. Designed to be lightweight and config-gated for Phase 1.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.constants import (
    MANDATORY_TOOLS,
    MANDATORY_VECTOR_TOP_K,
    MANDATORY_WEB_MAX_RESULTS,
)
from core.tool_usage_tracker import get_tool_tracker
from storage import get_storage_provider
from tools.tools import search_web


logger = logging.getLogger(__name__)


class ToolIntegrationEngine:
    """Engine to perform mandatory tool consultations per PRD FR1.1.

    This engine does not enforce blocking in Phase 1; it returns gathered
    context that can be injected into prompts. Enforcement can be layered by
    callers using quality gates.
    """

    def __init__(self):
        self.tool_tracker = get_tool_tracker()
        self.vector_store = get_storage_provider()

    def mandatory_tool_consultation(self, task: str, agent_type: str,
                                     agent_name: str,
                                     workflow_id: Optional[str] = None,
                                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform mandatory tool usage before task execution.

        Returns a dict of gathered contexts, e.g. vector search snippets and
        web search summaries.
        """
        results: Dict[str, Any] = {}

        required_tools: List[str] = MANDATORY_TOOLS.get(agent_type.lower(), [])

        # Vector database query
        if 'vector_search' in required_tools:
            query_text = task
            try:
                with self.tool_tracker.track_tool_usage(
                    tool_name='vector_search',
                    agent_name=agent_name,
                    workflow_id=workflow_id,
                    session_id=session_id,
                    input_data={"query": query_text, "top_k": MANDATORY_VECTOR_TOP_K},
                    context={"integration": "pre_task"}
                ):
                    vector_results = self.vector_store.search(
                        query=query_text, top_k=MANDATORY_VECTOR_TOP_K
                    )
                # Convert to lightweight context string
                summarized = []
                for r in vector_results:
                    title = getattr(r.metadata, 'title', '') if r.metadata else ''
                    snippet = (r.content or '')[:200]
                    summarized.append(f"- {title}: {snippet}")
                results['vector_context'] = "\n".join(summarized) if summarized else ""
                results['vector_queries'] = 1
                results['vector_results_count'] = len(vector_results) if vector_results else 0
            except Exception as e:
                logger.warning(f"Vector search failed during mandatory consultation: {e}")
                results['vector_context'] = ""
                results['vector_queries'] = results.get('vector_queries', 0)
                results['vector_results_count'] = results.get('vector_results_count', 0)

        # Web search validation
        if 'web_search' in required_tools:
            try:
                with self.tool_tracker.track_tool_usage(
                    tool_name='web_search',
                    agent_name=agent_name,
                    workflow_id=workflow_id,
                    session_id=session_id,
                    input_data={"query": task, "max_results": MANDATORY_WEB_MAX_RESULTS},
                    context={"integration": "pre_task"}
                ):
                    web_summary = search_web(task, max_results=MANDATORY_WEB_MAX_RESULTS)
                results['web_validation'] = web_summary
                results['web_searches'] = 1
            except Exception as e:
                logger.warning(f"Web search failed during mandatory consultation: {e}")
                results['web_validation'] = ""
                results['web_searches'] = results.get('web_searches', 0)

        return results


