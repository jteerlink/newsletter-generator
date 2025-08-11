#!/usr/bin/env python3
"""CLI utility for managing LLM providers and configuration."""

from src.core.llm_providers import LLMProviderFactory, get_llm_provider
from src.core.llm_config_validator import get_validator
from src.core.core import get_llm_provider_info, reconfigure_llm_provider
import argparse
import os
import sys
from pathlib import Path

# Add src to path so imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def cmd_status():
    """Show current LLM configuration status."""
    validator = get_validator()
    validator.print_configuration_status()


def cmd_test():
    """Test current LLM provider with a simple query."""
    print("🧪 Testing LLM Provider")
    print("=" * 30)

    try:
        # Import here to avoid circular imports
        from src.core.core import query_llm

        print("Sending test query...")
        response = query_llm(
            "Hello! Please respond with 'LLM test successful' to confirm you're working.")

        print(f"✅ Test successful!")
        print(f"📝 Response: {response[:100]}...")

        # Show provider info
        info = get_llm_provider_info()
        print(f"📡 Provider: {info['provider']}")
        print(f"🤖 Model: {info['model']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

    return 0


def cmd_switch(provider_name: str):
    """Switch to a different LLM provider."""
    print(f"🔄 Switching to {provider_name} provider")
    print("=" * 40)

    if provider_name.lower() not in ["ollama", "nvidia"]:
        print(f"❌ Unknown provider: {provider_name}")
        print("Available providers: ollama, nvidia")
        return 1

    # Update environment variable
    os.environ["LLM_PROVIDER"] = provider_name.lower()

    # Reconfigure provider
    reconfigure_llm_provider()

    # Validate new configuration
    validator = get_validator()
    validation, health = validator.validate_with_health_check()

    if validation.valid:
        print(f"✅ Successfully switched to {provider_name}")
        if health.healthy:
            print(
                f"🟢 Provider is healthy (response time: {
                    health.response_time:.2f}s)")
        else:
            print(
                f"⚠️  Provider switched but health check failed: {
                    health.error}")
    else:
        print(f"❌ Failed to switch to {provider_name}")
        for error in validation.errors:
            print(f"   • {error}")

    return 0 if validation.valid else 1


def cmd_env_template():
    """Generate a .env template with LLM configuration."""
    env_content = """# LLM Configuration
# NVIDIA is the default provider, with Ollama as fallback
LLM_PROVIDER=nvidia

# NVIDIA Configuration (primary/default)
# Get your API key from: https://build.nvidia.com/
NVIDIA_API_KEY=your-nvidia-api-key-here
NVIDIA_MODEL=openai/gpt-oss-20b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Ollama Configuration (fallback)
OLLAMA_MODEL=deepseek-r1
OLLAMA_BASE_URL=http://localhost:11434

# Alternative: To use Ollama instead, uncomment and set:
# LLM_PROVIDER=ollama

# LLM Settings (applies to both providers)
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3
LLM_TEMPERATURE=1.0
LLM_TOP_P=1.0
LLM_MAX_TOKENS=4096

# Other configuration...
DEFAULT_SEARCH_RESULTS=5
SEARCH_TIMEOUT=10
SCRAPER_TIMEOUT=30
MINIMUM_QUALITY_SCORE=7.0
DEFAULT_NEWSLETTER_LENGTH=1500
LOG_LEVEL=INFO
"""

    env_path = Path(".env.template")
    env_path.write_text(env_content)
    print(f"📄 Environment template created: {env_path}")
    print("   Edit this file with your settings and rename to '.env'")

    return 0


def cmd_doctor():
    """Run comprehensive diagnostics on LLM configuration."""
    print("🏥 LLM Configuration Doctor")
    print("=" * 40)

    validator = get_validator()
    report = validator.get_configuration_report()

    # Check configuration
    print("1. Configuration Check...")
    if report["configuration"]["valid"]:
        print("   ✅ Configuration is valid")
    else:
        print(
            f"   ❌ Configuration has {
                report['configuration']['errors_count']} errors")

    if report["configuration"]["warnings_count"] > 0:
        print(f"   ⚠️  {report['configuration']
              ['warnings_count']} warnings found")

    # Check health
    print("\n2. Health Check...")
    if report["health"]["healthy"]:
        print(
            f"   ✅ Provider is healthy ({
                report['health']['response_time']:.2f}s)")
    else:
        print("   ❌ Provider health check failed")
        if report["details"]["health_error"]:
            print(f"      Error: {report['details']['health_error']}")

    # Check dependencies
    print("\n3. Dependencies Check...")
    try:
        import ollama
        print("   ✅ ollama package available")
    except ImportError:
        print("   ❌ ollama package not installed")

    try:
        import openai
        print("   ✅ openai package available")
    except ImportError:
        print("   ⚠️  openai package not installed (needed for NVIDIA provider)")

    # Environment check
    print("\n4. Environment Check...")
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file found")
    else:
        print("   ⚠️  .env file not found (using system environment)")

    # Show detailed issues if any
    if report["details"]["errors"]:
        print(f"\n❌ Errors:")
        for error in report["details"]["errors"]:
            print(f"   • {error}")

    if report["details"]["warnings"]:
        print(f"\n⚠️  Warnings:")
        for warning in report["details"]["warnings"]:
            print(f"   • {warning}")

    if report["details"]["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["details"]["recommendations"]:
            print(f"   • {rec}")

    # Overall status
    overall_healthy = (report["configuration"]["valid"] and
                       report["health"]["healthy"] and
                       report["configuration"]["errors_count"] == 0)

    print(f"\n{'✅' if overall_healthy else '❌'} Overall Status: {
          'Healthy' if overall_healthy else 'Issues Found'}")

    return 0 if overall_healthy else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Provider Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/core/llm_cli.py status          # Show current status
  python src/core/llm_cli.py test            # Test current provider
  python src/core/llm_cli.py switch nvidia   # Switch to NVIDIA provider
  python src/core/llm_cli.py doctor          # Run comprehensive diagnostics
  python src/core/llm_cli.py env-template    # Generate .env template
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Status command
    subparsers.add_parser(
        'status',
        help='Show current LLM configuration status')

    # Test command
    subparsers.add_parser('test', help='Test current LLM provider')

    # Switch command
    switch_parser = subparsers.add_parser('switch', help='Switch LLM provider')
    switch_parser.add_argument('provider', choices=['ollama', 'nvidia'],
                               help='Provider to switch to')

    # Doctor command
    subparsers.add_parser('doctor', help='Run comprehensive diagnostics')

    # Environment template command
    subparsers.add_parser('env-template', help='Generate .env template')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'status':
        return cmd_status()
    elif args.command == 'test':
        return cmd_test()
    elif args.command == 'switch':
        return cmd_switch(args.provider)
    elif args.command == 'doctor':
        return cmd_doctor()
    elif args.command == 'env-template':
        return cmd_env_template()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
