"""Phase-1 core module

Contains a thin wrapper around Ollama's chat endpoint. The real logic will be
implemented in Phase 1; for now this module only defines signatures so other
modules can import without errors.
"""

from __future__ import annotations

import ollama
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="logs/interaction.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


def query_llm(prompt: str) -> str:
    """
    Query the configured LLM model with a user prompt and return the response.

    Args:
        prompt (str): The prompt string to send to the LLM.

    Returns:
        str: The LLM's response as a string, or an error message if the query fails.
    """
    model = os.getenv("OLLAMA_MODEL", "llama3")
    try:
        response = ollama.chat(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"]
        logging.info(f"Prompt: {prompt}\nResponse: {result}")
        return result
    except ollama.ResponseError as e:
        logging.error(f"Prompt: {prompt}\nError: {e}")
        return "An error occurred while querying the LLM."
    except Exception as e:
        logging.error(f"Prompt: {prompt}\nUnexpected error: {e}")
        return "An unexpected error occurred while querying the LLM."
