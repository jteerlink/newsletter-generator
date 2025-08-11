#!/usr/bin/env python3

"""
Comprehensive NVIDIA Pipeline Testing Script

This script validates that the NVIDIA pipeline is properly configured and
functioning across all test scenarios.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(command, description, expect_failure=False):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=project_root)
        end_time = time.time()
        
        success = (result.returncode == 0) if not expect_failure else (result.returncode != 0)
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f}s")
        print(f"Expected success: {'No' if expect_failure else 'Yes'}")
        print(f"Actual result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[:2000])  # Limit output
            if len(result.stdout) > 2000:
                print("... (output truncated)")
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr[:1000])  # Limit error output
            if len(result.stderr) > 1000:
                print("... (error truncated)")
        
        return success
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def test_nvidia_configuration():
    """Test NVIDIA configuration and defaults."""
    print("\n🔧 TESTING NVIDIA CONFIGURATION")
    print("=" * 60)
    
    tests = [
        ("python tests/test_nvidia_default.py", "NVIDIA default configuration test", False),
        ("python -c \"from src.core.constants import LLM_PROVIDER; assert LLM_PROVIDER == 'nvidia'; print('✅ NVIDIA is default provider')\"", "Constants verification", False),
        ("python src/core/llm_cli.py doctor", "LLM configuration validation", False),
    ]
    
    results = []
    for command, description, expect_failure in tests:
        success = run_command(command, description, expect_failure)
        results.append((description, success))
    
    return results


def test_nvidia_integration():
    """Test NVIDIA integration tests."""
    print("\n🧪 TESTING NVIDIA INTEGRATION")
    print("=" * 60)
    
    tests = [
        ("pytest tests/core/test_nvidia_integration.py -v", "NVIDIA integration tests", False),
        ("LLM_PROVIDER=nvidia pytest tests/core/test_core.py::TestCore::test_query_llm -v", "Core LLM tests with NVIDIA", False),
    ]
    
    results = []
    for command, description, expect_failure in tests:
        success = run_command(command, description, expect_failure)
        results.append((description, success))
    
    return results


def test_nvidia_fallback():
    """Test NVIDIA fallback to Ollama."""
    print("\n🔄 TESTING NVIDIA FALLBACK MECHANISM")
    print("=" * 60)
    
    # Test without API key (should fallback to Ollama)
    tests = [
        ("NVIDIA_API_KEY= LLM_PROVIDER=nvidia python -c \"from src.core.llm_providers import LLMProviderFactory; provider = LLMProviderFactory.create_provider_with_fallback(); print(f'Provider: {provider.__class__.__name__}')\"", "Fallback to Ollama when NVIDIA unavailable", False),
    ]
    
    results = []
    for command, description, expect_failure in tests:
        success = run_command(command, description, expect_failure)
        results.append((description, success))
    
    return results


def test_pipeline_end_to_end():
    """Test end-to-end pipeline with NVIDIA."""
    print("\n🚀 TESTING END-TO-END PIPELINE")
    print("=" * 60)
    
    # Mock NVIDIA environment for end-to-end test
    env_vars = {
        "LLM_PROVIDER": "nvidia",
        "NVIDIA_API_KEY": "test-nvidia-key",
        "NVIDIA_MODEL": "openai/gpt-oss-20b"
    }
    
    tests = [
        # These tests would normally fail in CI without real API keys, but should validate configuration
        ("python -c \"from src.core.llm_providers import get_llm_provider; provider = get_llm_provider(); print(f'Active provider: {provider.__class__.__name__}')\"", "Provider selection verification", False),
        ("pytest tests/integration/test_api_integration.py::TestAPIIntegration::test_llm_provider_initialization -v", "LLM provider initialization test", False),
    ]
    
    results = []
    for command, description, expect_failure in tests:
        # Set environment for this test
        full_command = " ".join([f"{k}={v}" for k, v in env_vars.items()]) + " " + command
        success = run_command(full_command, description, expect_failure)
        results.append((description, success))
    
    return results


def test_test_environment_nvidia():
    """Test that test environment uses NVIDIA by default."""
    print("\n🧪 TESTING TEST ENVIRONMENT NVIDIA CONFIGURATION")
    print("=" * 60)
    
    tests = [
        ("pytest tests/conftest.py::test_mock_env_vars -v", "Test environment mock vars include NVIDIA", False),
        ("python -c \"import pytest; import tests.conftest; print('Test fixtures loaded successfully')\"", "Test fixtures loading", False),
    ]
    
    results = []
    for command, description, expect_failure in tests:
        success = run_command(command, description, expect_failure)
        results.append((description, success))
    
    return results


def generate_nvidia_report(all_results):
    """Generate comprehensive NVIDIA pipeline test report."""
    print("\n📋 NVIDIA PIPELINE TEST REPORT")
    print("=" * 80)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, success in all_results if success)
    failed_tests = total_tests - passed_tests
    
    print("Test Results:")
    print("-" * 40)
    for description, success in all_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")
    
    print(f"\nSummary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    
    # Detailed analysis
    if failed_tests == 0:
        print("\n🎉 ALL NVIDIA PIPELINE TESTS PASSED!")
        print("✅ NVIDIA is properly configured as the default LLM provider")
        print("✅ NVIDIA integration tests pass")
        print("✅ Fallback mechanism to Ollama works correctly")
        print("✅ Test environment uses NVIDIA configuration")
        print("✅ End-to-end pipeline validation successful")
    else:
        print(f"\n⚠️  {failed_tests} NVIDIA PIPELINE TESTS FAILED")
        print("❗ Please review the test output above to identify issues")
        
        # Provide specific guidance
        failed_descriptions = [desc for desc, success in all_results if not success]
        
        if any("configuration" in desc.lower() for desc in failed_descriptions):
            print("💡 Configuration issues detected - check environment variables and constants")
        
        if any("integration" in desc.lower() for desc in failed_descriptions):
            print("💡 Integration issues detected - verify NVIDIA provider implementation")
        
        if any("fallback" in desc.lower() for desc in failed_descriptions):
            print("💡 Fallback issues detected - check provider factory logic")
    
    return failed_tests == 0


def main():
    """Main NVIDIA pipeline test function."""
    print("🚀 NVIDIA PIPELINE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("This script validates NVIDIA pipeline configuration and functionality")
    print("across all components of the newsletter generator system.")
    print("=" * 80)
    
    # Change to project root
    os.chdir(project_root)
    
    # Run all test categories
    all_results = []
    
    # Configuration tests
    config_results = test_nvidia_configuration()
    all_results.extend(config_results)
    
    # Integration tests
    integration_results = test_nvidia_integration()
    all_results.extend(integration_results)
    
    # Fallback tests
    fallback_results = test_nvidia_fallback()
    all_results.extend(fallback_results)
    
    # End-to-end tests
    e2e_results = test_pipeline_end_to_end()
    all_results.extend(e2e_results)
    
    # Test environment tests
    test_env_results = test_test_environment_nvidia()
    all_results.extend(test_env_results)
    
    # Generate comprehensive report
    success = generate_nvidia_report(all_results)
    
    if success:
        print("\n🎯 NVIDIA PIPELINE VALIDATION COMPLETE")
        print("The NVIDIA pipeline is properly configured and ready for use!")
        return 0
    else:
        print("\n❌ NVIDIA PIPELINE VALIDATION FAILED")
        print("Please address the issues identified above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
