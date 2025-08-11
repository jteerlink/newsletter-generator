"""LLM configuration validation and health checking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import (
    LLM_MAX_RETRIES,
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
from .llm_providers import LLMProvider, LLMProviderFactory
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    provider: str
    model: str
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]


@dataclass
class HealthCheckResult:
    """Result of provider health check."""
    healthy: bool
    provider: str
    model: str
    response_time: Optional[float]
    test_response: Optional[str]
    error: Optional[str]


class LLMConfigValidator:
    """Validates LLM configuration and performs health checks."""

    def __init__(self):
        self.logger = setup_logging(__name__)

    def validate_configuration(self) -> ValidationResult:
        """
        Validate the current LLM configuration.

        Returns:
            ValidationResult: Comprehensive validation results.
        """
        warnings = []
        errors = []
        recommendations = []

        provider_name = LLM_PROVIDER.lower()
        model = "unknown"

        # Validate provider selection
        if provider_name not in ["ollama", "nvidia"]:
            errors.append(f"Unknown LLM provider: {
                          provider_name}. Must be 'ollama' or 'nvidia'")

        # Provider-specific validation
        if provider_name == "nvidia":
            model = NVIDIA_MODEL
            if not NVIDIA_API_KEY:
                errors.append(
                    "NVIDIA API key is required but not configured (NVIDIA_API_KEY)")
                recommendations.append(
                    "Set NVIDIA_API_KEY environment variable or update .env file")
            elif len(NVIDIA_API_KEY) < 10:
                warnings.append(
                    "NVIDIA API key appears to be too short or invalid")

            if not NVIDIA_BASE_URL:
                warnings.append(
                    "NVIDIA base URL not configured, using default")
            elif not NVIDIA_BASE_URL.startswith("https://"):
                warnings.append("NVIDIA base URL should use HTTPS")

        elif provider_name == "ollama":
            model = OLLAMA_MODEL
            if not OLLAMA_BASE_URL:
                warnings.append(
                    "Ollama base URL not configured, using default")
            elif not OLLAMA_BASE_URL.startswith("http"):
                warnings.append(
                    "Ollama base URL should include protocol (http/https)")

        # Validate general LLM settings
        if LLM_TIMEOUT < 10:
            warnings.append(
                f"LLM timeout ({LLM_TIMEOUT}s) might be too short for complex queries")
        elif LLM_TIMEOUT > 120:
            warnings.append(
                f"LLM timeout ({LLM_TIMEOUT}s) is very high, consider reducing")

        if LLM_MAX_RETRIES > 5:
            warnings.append(f"Max retries ({LLM_MAX_RETRIES}) is quite high")
        elif LLM_MAX_RETRIES < 1:
            errors.append("Max retries must be at least 1")

        if not (0.0 <= LLM_TEMPERATURE <= 2.0):
            warnings.append(
                f"Temperature ({LLM_TEMPERATURE}) outside typical range (0.0-2.0)")

        if not (0.0 <= LLM_TOP_P <= 1.0):
            warnings.append(
                f"Top P ({LLM_TOP_P}) outside valid range (0.0-1.0)")

        if LLM_MAX_TOKENS < 100:
            warnings.append(
                f"Max tokens ({LLM_MAX_TOKENS}) might be too low for newsletter generation")
        elif LLM_MAX_TOKENS > 8192:
            warnings.append(
                f"Max tokens ({LLM_MAX_TOKENS}) is very high, may impact performance")

        # Add general recommendations
        if provider_name == "ollama" and not warnings and not errors:
            recommendations.append(
                "Consider trying NVIDIA Cloud API for potentially better performance")

        is_valid = len(errors) == 0

        return ValidationResult(
            valid=is_valid,
            provider=provider_name,
            model=model,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )

    def health_check(
            self,
            provider: Optional[LLMProvider] = None) -> HealthCheckResult:
        """
        Perform a health check on the configured LLM provider.

        Args:
            provider: Optional provider instance to test. If None, uses default.

        Returns:
            HealthCheckResult: Health check results.
        """
        import time

        try:
            if provider is None:
                provider = LLMProviderFactory.create_provider()

            provider_name = type(provider).__name__.replace(
                'Provider', '').lower()
            model = getattr(provider, 'model', 'unknown')

            # Test simple query
            start_time = time.time()
            test_messages = [{"role": "user",
                              "content": "Hello, respond with 'OK'"}]

            try:
                response = provider.chat(test_messages, max_tokens=10)
                response_time = time.time() - start_time

                # Check if response is reasonable
                if response and len(response.strip()) > 0:
                    return HealthCheckResult(
                        healthy=True,
                        provider=provider_name,
                        model=model,
                        response_time=response_time,
                        test_response=response.strip()[:50],
                        error=None
                    )
                else:
                    return HealthCheckResult(
                        healthy=False,
                        provider=provider_name,
                        model=model,
                        response_time=response_time,
                        test_response=None,
                        error="Empty or invalid response"
                    )

            except Exception as e:
                response_time = time.time() - start_time
                return HealthCheckResult(
                    healthy=False,
                    provider=provider_name,
                    model=model,
                    response_time=response_time,
                    test_response=None,
                    error=str(e)
                )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                provider="unknown",
                model="unknown",
                response_time=None,
                test_response=None,
                error=f"Failed to create provider: {e}"
            )

    def validate_with_health_check(
            self) -> Tuple[ValidationResult, HealthCheckResult]:
        """
        Perform both configuration validation and health check.

        Returns:
            Tuple of (ValidationResult, HealthCheckResult).
        """
        validation = self.validate_configuration()

        # Only do health check if configuration is valid
        if validation.valid:
            try:
                health = self.health_check()
            except Exception as e:
                health = HealthCheckResult(
                    healthy=False,
                    provider=validation.provider,
                    model=validation.model,
                    response_time=None,
                    test_response=None,
                    error=f"Health check failed: {e}"
                )
        else:
            health = HealthCheckResult(
                healthy=False,
                provider=validation.provider,
                model=validation.model,
                response_time=None,
                test_response=None,
                error="Configuration invalid, skipping health check"
            )

        return validation, health

    def get_configuration_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive configuration report.

        Returns:
            Dict containing full configuration status.
        """
        validation, health = self.validate_with_health_check()

        report = {
            "configuration": {
                "provider": validation.provider,
                "model": validation.model,
                "valid": validation.valid,
                "warnings_count": len(validation.warnings),
                "errors_count": len(validation.errors)
            },
            "health": {
                "healthy": health.healthy,
                "response_time": health.response_time,
                "test_passed": bool(health.test_response)
            },
            "settings": {
                "timeout": LLM_TIMEOUT,
                "max_retries": LLM_MAX_RETRIES,
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "max_tokens": LLM_MAX_TOKENS
            },
            "details": {
                "warnings": validation.warnings,
                "errors": validation.errors,
                "recommendations": validation.recommendations,
                "health_error": health.error
            }
        }

        return report

    def print_configuration_status(self):
        """Print a formatted configuration status report."""
        print("\n🔧 LLM Configuration Status")
        print("=" * 50)

        validation, health = self.validate_with_health_check()

        # Provider info
        print(f"📡 Provider: {validation.provider}")
        print(f"🤖 Model: {validation.model}")

        # Configuration status
        status_emoji = "✅" if validation.valid else "❌"
        print(f"{status_emoji} Configuration: {
              'Valid' if validation.valid else 'Invalid'}")

        # Health status
        health_emoji = "🟢" if health.healthy else "🔴"
        print(
            f"{health_emoji} Health: {
                'Healthy' if health.healthy else 'Unhealthy'}")

        if health.response_time:
            print(f"⏱️  Response Time: {health.response_time:.2f}s")

        # Warnings
        if validation.warnings:
            print(f"\n⚠️  Warnings ({len(validation.warnings)}):")
            for warning in validation.warnings:
                print(f"   • {warning}")

        # Errors
        if validation.errors:
            print(f"\n❌ Errors ({len(validation.errors)}):")
            for error in validation.errors:
                print(f"   • {error}")

        # Health error
        if health.error:
            print(f"\n🏥 Health Error: {health.error}")

        # Recommendations
        if validation.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in validation.recommendations:
                print(f"   • {rec}")

        print()


# Global validator instance
_validator_instance: Optional[LLMConfigValidator] = None


def get_validator() -> LLMConfigValidator:
    """Get the global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = LLMConfigValidator()
    return _validator_instance


def validate_llm_config() -> ValidationResult:
    """Quick validation of LLM configuration."""
    return get_validator().validate_configuration()


def check_llm_health() -> HealthCheckResult:
    """Quick health check of LLM provider."""
    return get_validator().health_check()
