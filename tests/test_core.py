import pytest
from unittest.mock import patch, MagicMock
from src.core.core import query_llm
import ollama
import os


def test_query_llm_success():
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Test response"}}
        result = query_llm("Test prompt")
        assert result == "Test response"


def test_query_llm_error():
    with patch("ollama.chat", side_effect=ollama.ResponseError("Error")):
        result = query_llm("Test prompt")
        assert result == "An error occurred while querying the LLM."


def test_query_llm_empty_prompt():
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Empty prompt response"}}
        result = query_llm("")
        assert result == "Empty prompt response"


def test_query_llm_invalid_model():
    with patch.dict(os.environ, {"OLLAMA_MODEL": "invalid_model"}):
        with patch("ollama.chat", side_effect=ollama.ResponseError("Model not found")):
            result = query_llm("Test prompt")
            assert result == "An error occurred while querying the LLM."


def test_query_llm_missing_env_var():
    with patch.dict(os.environ, {}, clear=True):
        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "Default model response"}}
            result = query_llm("Test prompt")
            assert result == "Default model response"
