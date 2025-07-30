"""Core module for LLM interactions and logging setup."""

from __future__ import annotations

import ollama
import logging
from dotenv import load_dotenv

from .constants import DEFAULT_LLM_MODEL, LLM_TIMEOUT, LLM_MAX_RETRIES, ERROR_MESSAGES
from .exceptions import LLMError
from .utils import setup_logging, retry_on_failure

load_dotenv()

# Setup logging
logger = setup_logging(__name__)


@retry_on_failure(max_retries=LLM_MAX_RETRIES)
def query_llm(prompt: str, model: str | None = None) -> str:
    """
    Query the configured LLM model with a user prompt and return the response.

    Args:
        prompt (str): The prompt string to send to the LLM.
        model (str, optional): The model to use. Defaults to DEFAULT_LLM_MODEL.

    Returns:
        str: The LLM's response as a string.

    Raises:
        LLMError: If the LLM query fails.
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    
    try:
        response = ollama.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": LLM_TIMEOUT}
        )
        result = response["message"]["content"]
        logger.info(f"LLM query successful with model {model}")
        return result
    except ollama.ResponseError as e:
        logger.error(f"LLM ResponseError: {e}")
        raise LLMError(f"{ERROR_MESSAGES['llm_timeout']}: {e}")
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}")
        raise LLMError(f"Unexpected error in LLM query: {e}")
