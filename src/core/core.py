"""Phase-1 core module

Contains a thin wrapper around Ollama's chat endpoint. The real logic will be
implemented in Phase 1; for now this module only defines signatures so other
modules can import without errors.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

def query_llm(prompt: str) -> str:  # noqa: D401
    """Temporary stub for querying the local LLM.

    Until the actual Ollama integration is written, this stub just logs the
    prompt and returns a static placeholder string so that the rest of the
    pipeline can import and run.
    """
    logger.debug("query_llm called with prompt: %s", prompt)
    return "[LLM response placeholder]" 