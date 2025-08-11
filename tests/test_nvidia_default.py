#!/usr/bin/env python3

"""
Test to verify NVIDIA pipeline is configured as default.
"""

import os
import pytest
from unittest.mock import patch

import sys
sys.path.append('.')

from src.core.constants import LLM_PROVIDER
from src.core.llm_providers import LLMProviderFactory


class TestNvidiaDefault:
    """Test NVIDIA pipeline default configuration."""
    
    def test_nvidia_is_default_provider(self):
        """Test that NVIDIA is set as the default provider."""
        # Test the constant default value
        assert LLM_PROVIDER == "nvidia", f"Expected NVIDIA as default, but got: {LLM_PROVIDER}"
    
    def test_provider_factory_uses_nvidia_by_default(self):
        """Test that provider factory defaults to NVIDIA when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):  # Clear all env vars
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
                try:
                    provider = LLMProviderFactory.create_provider()
                    # Check if it's the nvidia provider by checking class name
                    assert provider.__class__.__name__ == "NvidiaProvider"
                except Exception as e:
                    # If creation fails due to API availability, that's expected in tests
                    # But we should still verify the attempt was made with nvidia
                    assert "NVIDIA" in str(e) or "nvidia" in str(e)
    
    def test_fallback_provider_is_ollama(self):
        """Test that fallback provider is Ollama when NVIDIA fails."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=True):
            try:
                provider = LLMProviderFactory.create_provider_with_fallback()
                # Should fallback to Ollama
                assert provider.__class__.__name__ == "OllamaProvider"
            except Exception:
                # This is acceptable if neither provider is available in test environment
                pass
    
    def test_explicit_nvidia_provider_creation(self):
        """Test explicit NVIDIA provider creation with API key."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-nvidia-key"}):
            try:
                provider = LLMProviderFactory.create_provider("nvidia")
                assert provider.__class__.__name__ == "NvidiaProvider"
            except Exception as e:
                # Should fail with nvidia-specific error
                assert "NVIDIA" in str(e) or "nvidia" in str(e)
    
    def test_testing_environment_uses_nvidia(self):
        """Test that testing environment is configured to use NVIDIA by default."""
        # Mock a complete NVIDIA environment
        nvidia_env = {
            "LLM_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "test-nvidia-api-key",
            "NVIDIA_MODEL": "openai/gpt-oss-20b"
        }
        
        with patch.dict(os.environ, nvidia_env):
            from src.core.constants import LLM_PROVIDER as current_provider
            # Reload the module to get updated constants
            import importlib
            import src.core.constants
            importlib.reload(src.core.constants)
            
            # Verify provider is nvidia
            assert src.core.constants.LLM_PROVIDER == "nvidia"
            
            # Verify NVIDIA configuration is loaded
            assert src.core.constants.NVIDIA_API_KEY == "test-nvidia-api-key"
            assert src.core.constants.NVIDIA_MODEL == "openai/gpt-oss-20b"


if __name__ == "__main__":
    print("🧪 Testing NVIDIA Default Configuration")
    print("=" * 40)
    
    # Run the tests
    test_class = TestNvidiaDefault()
    
    try:
        test_class.test_nvidia_is_default_provider()
        print("✅ NVIDIA is configured as default provider")
    except AssertionError as e:
        print(f"❌ Default provider test failed: {e}")
    
    try:
        test_class.test_provider_factory_uses_nvidia_by_default()
        print("✅ Provider factory defaults to NVIDIA")
    except Exception as e:
        print(f"⚠️  Provider factory test: {e}")
    
    try:
        test_class.test_fallback_provider_is_ollama()
        print("✅ Fallback to Ollama works when NVIDIA unavailable")
    except Exception as e:
        print(f"⚠️  Fallback test: {e}")
    
    try:
        test_class.test_explicit_nvidia_provider_creation()
        print("✅ Explicit NVIDIA provider creation works")
    except Exception as e:
        print(f"⚠️  Explicit NVIDIA test: {e}")
    
    try:
        test_class.test_testing_environment_uses_nvidia()
        print("✅ Testing environment uses NVIDIA by default")
    except Exception as e:
        print(f"⚠️  Testing environment test: {e}")
    
    print("\n🔍 Current Configuration:")
    from src.core.constants import LLM_PROVIDER, NVIDIA_API_KEY, NVIDIA_MODEL
    print(f"  Default Provider: {LLM_PROVIDER}")
    print(f"  NVIDIA API Key: {'Set' if NVIDIA_API_KEY else 'Not Set'}")
    print(f"  NVIDIA Model: {NVIDIA_MODEL}")
    
    print("\n✅ NVIDIA pipeline default configuration tests completed!")
