"""Tool definitions (Phase 2).

This file will eventually host implementations for Web search, vector DB search,
and other utilities decorated with @tool for CrewAI.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Placeholder function so imports succeed
def search_web(query: str) -> str:  # type: ignore
    """Stubbed web search tool – replace with real DuckDuckGo implementation."""
    logger.debug("search_web called with query: %s", query)
    return "[web search results placeholder]"
