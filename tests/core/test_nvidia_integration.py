"""Tests for NVIDIA API integration."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.llm_providers import NvidiaProvider, OllamaProvider, LLMProviderFactory
from src.core.llm_config_validator import LLMConfigValidator
from src.core.core import query_llm, get_llm_provider_info
from src.core.exceptions import LLMError


class TestNvidiaProvider:
    """Test NVIDIA provider implementation."""

    def test_nvidia_provider_init_with_api_key(self):
        """Test NVIDIA provider initialization with API key."""
        provider = NvidiaProvider(api_key="test-key")
        assert provider.model == "openai/gpt-oss-20b"
        assert provider.client is not None

    def test_nvidia_provider_init_without_api_key(self):
        """Test NVIDIA provider initialization without API key."""
        with pytest.raises(ValueError, match="NVIDIA API key is required"):
            NvidiaProvider(api_key=None)

    @patch('src.core.llm_providers.OpenAI')
    def test_nvidia_provider_chat_success(self, mock_openai_class):
        """Test successful NVIDIA provider chat."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.reasoning_content = None
        
        mock_chunk2 = Mock()  
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk2.choices[0].delta.reasoning_content = None
        
        mock_chunk3 = Mock()
        mock_chunk3.choices = [Mock()]
        mock_chunk3.choices[0].delta.content = None
        mock_chunk3.choices[0].delta.reasoning_content = None
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
        
        provider = NvidiaProvider(api_key="test-key")
        messages = [{"role": "user", "content": "test"}]
        
        result = provider.chat(messages)
        assert result == "Hello World"

    @patch('src.core.llm_providers.OpenAI')
    def test_nvidia_provider_chat_with_reasoning(self, mock_openai_class):
        """Test NVIDIA provider chat with reasoning content."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock streaming response with reasoning
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = None
        mock_chunk1.choices[0].delta.reasoning_content = "Thinking..."
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = "Response"
        mock_chunk2.choices[0].delta.reasoning_content = None
        
        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
        
        provider = NvidiaProvider(api_key="test-key")
        messages = [{"role": "user", "content": "test"}]
        
        result = provider.chat(messages)
        assert result == "Response"

    @patch('src.core.llm_providers.OpenAI')
    def test_nvidia_provider_chat_error(self, mock_openai_class):
        """Test NVIDIA provider chat error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        provider = NvidiaProvider(api_key="test-key")
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(LLMError, match="NVIDIA API request failed"):
            provider.chat(messages)

    @patch('src.core.llm_providers.OpenAI')
    def test_nvidia_provider_is_available(self, mock_openai_class):
        """Test NVIDIA provider availability check."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock successful test call
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = NvidiaProvider(api_key="test-key")
        assert provider.is_available() == True

    @patch('src.core.llm_providers.OpenAI')
    def test_nvidia_provider_is_not_available(self, mock_openai_class):
        """Test NVIDIA provider unavailable."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        
        provider = NvidiaProvider(api_key="test-key")
        assert provider.is_available() == False


class TestLLMProviderFactory:
    """Test LLM provider factory."""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = LLMProviderFactory.create_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_create_nvidia_provider_with_key(self):
        """Test creating NVIDIA provider with API key."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            provider = LLMProviderFactory.create_provider("nvidia")
            assert isinstance(provider, NvidiaProvider)

    def test_create_nvidia_provider_without_key(self):
        """Test creating NVIDIA provider without API key."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=True):
            with pytest.raises(LLMError, match="NVIDIA API key not configured"):
                LLMProviderFactory.create_provider("nvidia")

    def test_create_unknown_provider(self):
        """Test creating unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create_provider("unknown")

    @patch.object(LLMProviderFactory, 'create_provider')
    def test_create_provider_with_fallback_success(self, mock_create):
        """Test provider creation with fallback - primary succeeds."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_create.return_value = mock_provider
        
        provider = LLMProviderFactory.create_provider_with_fallback("nvidia")
        assert provider == mock_provider

    @patch.object(LLMProviderFactory, 'create_provider')
    def test_create_provider_with_fallback_fails_primary(self, mock_create):
        """Test provider creation with fallback - primary fails, fallback succeeds."""
        mock_primary = Mock()
        mock_primary.is_available.return_value = False
        
        mock_fallback = Mock()
        mock_fallback.is_available.return_value = True
        
        mock_create.side_effect = [mock_primary, mock_fallback]
        
        provider = LLMProviderFactory.create_provider_with_fallback("nvidia", "ollama")
        assert provider == mock_fallback

    @patch.object(LLMProviderFactory, 'create_provider')
    def test_create_provider_with_fallback_both_fail(self, mock_create):
        """Test provider creation with fallback - both fail."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = False
        mock_create.return_value = mock_provider
        
        with pytest.raises(LLMError, match="Both primary .* and fallback .* providers are unavailable"):
            LLMProviderFactory.create_provider_with_fallback("nvidia", "ollama")


class TestConfigValidator:
    """Test configuration validator."""

    def test_validate_nvidia_config_with_key(self):
        """Test NVIDIA configuration validation with API key."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "test-key-with-length"
        }):
            validator = LLMConfigValidator()
            result = validator.validate_configuration()
            
            assert result.valid == True
            assert result.provider == "nvidia"
            assert len(result.errors) == 0

    def test_validate_nvidia_config_without_key(self):
        """Test NVIDIA configuration validation without API key."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": ""
        }, clear=True):
            validator = LLMConfigValidator()
            result = validator.validate_configuration()
            
            assert result.valid == False
            assert "NVIDIA API key is required" in str(result.errors)

    def test_validate_ollama_config(self):
        """Test Ollama configuration validation."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            validator = LLMConfigValidator()
            result = validator.validate_configuration()
            
            assert result.valid == True
            assert result.provider == "ollama"

    def test_validate_unknown_provider(self):
        """Test validation with unknown provider."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}):
            validator = LLMConfigValidator()
            result = validator.validate_configuration()
            
            assert result.valid == False
            assert "Unknown LLM provider" in str(result.errors)


class TestCoreIntegration:
    """Test core module integration with providers."""

    @patch('src.core.core.get_llm_provider')
    def test_query_llm_success(self, mock_get_provider):
        """Test query_llm with successful provider call."""
        mock_provider = Mock()
        mock_provider.chat.return_value = "Test response"
        mock_get_provider.return_value = mock_provider
        
        result = query_llm("test prompt")
        assert result == "Test response"
        
        # Verify provider was called correctly
        mock_provider.chat.assert_called_once()
        call_args = mock_provider.chat.call_args[0][0]
        assert call_args == [{"role": "user", "content": "test prompt"}]

    @patch('src.core.core.get_llm_provider')
    def test_query_llm_error(self, mock_get_provider):
        """Test query_llm with provider error."""
        mock_provider = Mock()
        mock_provider.chat.side_effect = Exception("Provider failed")
        mock_get_provider.return_value = mock_provider
        
        with pytest.raises(LLMError, match="LLM request timed out"):
            query_llm("test prompt")

    @patch('src.core.core.get_llm_provider')
    def test_get_llm_provider_info(self, mock_get_provider):
        """Test get_llm_provider_info function."""
        mock_provider = Mock()
        mock_provider.model = "test-model"
        mock_provider.is_available.return_value = True
        mock_get_provider.return_value = mock_provider
        
        # Mock the provider class name
        type(mock_provider).__name__ = "TestProvider"
        
        info = get_llm_provider_info()
        
        assert info["provider"] == "test"
        assert info["model"] == "test-model"
        assert info["available"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])