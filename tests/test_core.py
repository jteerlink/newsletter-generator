import os
from unittest.mock import MagicMock, patch

import ollama
import pytest

from src.core.core import query_llm
from src.core.exceptions import LLMError


def test_query_llm_success():
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Test response"}}
        result = query_llm("Test prompt")
        assert result == "Test response"


def test_query_llm_error():
    with patch("ollama.chat", side_effect=ollama.ResponseError("Error")):
        with pytest.raises(LLMError):
            query_llm("Test prompt")


def test_query_llm_empty_prompt():
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Empty prompt response"}}
        result = query_llm("")
        assert result == "Empty prompt response"


def test_query_llm_invalid_model():
    with patch.dict(os.environ, {"OLLAMA_MODEL": "invalid_model"}):
        with patch("ollama.chat", side_effect=ollama.ResponseError("Model not found")):
            with pytest.raises(LLMError):
                query_llm("Test prompt")


def test_query_llm_missing_env_var():
    with patch.dict(os.environ, {}, clear=True):
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "Default model response"}}
            result = query_llm("Test prompt")
            assert result == "Default model response"
