"""Core module for LLM interactions and logging setup."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .constants import ERROR_MESSAGES, LLM_MAX_RETRIES
from .exceptions import LLMError
from .llm_providers import get_llm_provider, reset_llm_provider
from .utils import retry_on_failure, setup_logging

load_dotenv()

# Setup logging
logger = setup_logging(__name__)


@retry_on_failure(max_retries=LLM_MAX_RETRIES)
def query_llm(prompt: str, model: str | None = None, **kwargs) -> str:
    """
    Query the configured LLM model with a user prompt and return the response.

    Args:
        prompt (str): The prompt string to send to the LLM.
        model (str, optional): The model to use. Ignored for provider-based routing.
        **kwargs: Additional parameters passed to the provider.

    Returns:
        str: The LLM's response as a string.

    Raises:
        LLMError: If the LLM query fails.
    """
    try:
        provider = get_llm_provider()
        messages = [{"role": "user", "content": prompt}]

        # Pass through any additional parameters
        result = provider.chat(messages, **kwargs)
        logger.info("LLM query successful")
        return result
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise LLMError(f"{ERROR_MESSAGES['llm_timeout']}: {e}")


def query_llm_with_messages(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Query the LLM with a full conversation history.

    Args:
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        **kwargs: Additional parameters passed to the provider.

    Returns:
        str: The LLM's response as a string.

    Raises:
        LLMError: If the LLM query fails.
    """
    try:
        provider = get_llm_provider()
        result = provider.chat(messages, **kwargs)
        logger.info("LLM conversation query successful")
        return result
    except Exception as e:
        logger.error(f"LLM conversation query failed: {e}")
        raise LLMError(f"LLM conversation query failed: {e}")


def get_llm_provider_info() -> Dict[str, str]:
    """
    Get information about the current LLM provider.

    Returns:
        Dict[str, str]: Provider information including name and model.
    """
    try:
        provider = get_llm_provider()
        provider_type = type(provider).__name__.replace('Provider', '').lower()

        if hasattr(provider, 'model'):
            model = provider.model
        else:
            model = "unknown"

        return {
            "provider": provider_type,
            "model": model,
            "available": provider.is_available()
        }
    except Exception as e:
        logger.error(f"Failed to get provider info: {e}")
        return {
            "provider": "unknown",
            "model": "unknown",
            "available": False,
            "error": str(e)
        }


def reconfigure_llm_provider():
    """
    Reconfigure the LLM provider (useful when environment variables change).
    """
    reset_llm_provider()
    logger.info("LLM provider reconfigured")
