"""LLM provider abstractions supporting multiple backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import ollama
from openai import OpenAI

from .constants import (
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from .exceptions import LLMError
from .utils import setup_logging

logger = setup_logging(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send messages to the LLM and return the response."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""

    def __init__(self, model: str = OLLAMA_MODEL,
                 base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send messages to Ollama and return the response."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "timeout": kwargs.get("timeout", LLM_TIMEOUT),
                    "temperature": kwargs.get("temperature", LLM_TEMPERATURE),
                    "top_p": kwargs.get("top_p", LLM_TOP_P),
                    "num_predict": kwargs.get("max_tokens", LLM_MAX_TOKENS)
                }
            )
            result = response["message"]["content"]
            logger.info(f"Ollama query successful with model {self.model}")
            return result
        except ollama.ResponseError as e:
            logger.error(f"Ollama ResponseError: {e}")
            raise LLMError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            raise LLMError(f"Unexpected error in Ollama query: {e}")

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            ollama.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False


class NvidiaProvider(LLMProvider):
    """NVIDIA Cloud API provider using OpenAI-compatible interface."""

    def __init__(
        self,
        api_key: Optional[str] = NVIDIA_API_KEY,
        model: str = NVIDIA_MODEL,
        base_url: str = NVIDIA_BASE_URL
    ):
        if not api_key:
            raise ValueError("NVIDIA API key is required but not provided")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send messages to NVIDIA API and return the response."""
        try:
            # Handle streaming response like in the provided example
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", LLM_TEMPERATURE),
                top_p=kwargs.get("top_p", LLM_TOP_P),
                max_tokens=kwargs.get("max_tokens", LLM_MAX_TOKENS),
                stream=True
            )

            # Collect the streamed response
            full_content = ""
            for chunk in completion:
                # Handle reasoning content if present
                reasoning = getattr(
                    chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    logger.debug(f"NVIDIA reasoning: {reasoning}")

                # Collect main content
                if chunk.choices[0].delta.content is not None:
                    full_content += chunk.choices[0].delta.content

            logger.info(f"NVIDIA query successful with model {self.model}")
            return full_content.strip()

        except Exception as e:
            logger.error(f"NVIDIA API error: {e}")
            raise LLMError(f"NVIDIA API request failed: {e}")

    def is_available(self) -> bool:
        """Check if NVIDIA API is available."""
        try:
            # Test with a minimal request
            test_messages = [{"role": "user", "content": "test"}]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"NVIDIA API not available: {e}")
            return False


class LLMProviderFactory:
    """Factory for creating and managing LLM providers."""

    _providers = {
        "ollama": OllamaProvider,
        "nvidia": NvidiaProvider
    }

    @classmethod
    def create_provider(
            cls,
            provider_name: Optional[str] = None) -> LLMProvider:
        """Create an LLM provider instance."""
        if provider_name is None:
            provider_name = LLM_PROVIDER.lower()

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]

        try:
            if provider_name == "nvidia":
                if not NVIDIA_API_KEY:
                    raise ValueError(
                        "NVIDIA API key not configured. Please set NVIDIA_API_KEY environment variable."
                    )
                return provider_class()
            else:
                return provider_class()
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")
            raise LLMError(f"Failed to initialize {
                           provider_name} provider: {e}")

    @classmethod
    def create_provider_with_fallback(
        cls,
        primary_provider: Optional[str] = None,
        fallback_provider: str = "ollama"
    ) -> LLMProvider:
        """Create a provider with fallback to another provider if the primary fails."""
        if primary_provider is None:
            primary_provider = LLM_PROVIDER.lower()

        try:
            provider = cls.create_provider(primary_provider)
            if provider.is_available():
                logger.info(f"Using primary provider: {primary_provider}")
                return provider
            else:
                logger.warning(f"Primary provider {
                               primary_provider} not available, falling back to {fallback_provider}")
        except Exception as e:
            logger.warning(f"Failed to create primary provider {
                           primary_provider}: {e}")

        # Fallback to secondary provider
        try:
            fallback = cls.create_provider(fallback_provider)
            if fallback.is_available():
                logger.info(f"Using fallback provider: {fallback_provider}")
                return fallback
            else:
                raise LLMError(f"Both primary ({primary_provider}) and fallback ({
                               fallback_provider}) providers are unavailable")
        except Exception as e:
            raise LLMError(f"All providers failed. Last error: {e}")


# Global provider instance (lazy initialization)
_provider_instance: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider instance (singleton pattern)."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = LLMProviderFactory.create_provider_with_fallback()
    return _provider_instance


def reset_llm_provider():
    """Reset the provider instance (useful for testing or reconfiguration)."""
    global _provider_instance
    _provider_instance = None
