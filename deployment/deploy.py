#!/usr/bin/env python3
"""
Deployment Script for Section-Aware Newsletter Generation System

Command-line tool for deploying and managing the section-aware newsletter
generation system across different environments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from section_aware_deployment import (
    SectionAwareDeployment,
    DeploymentEnvironment,
    create_production_environment,
    create_development_environment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(args) -> int:
    """Set up deployment environment."""
    logger.info(f"Setting up {args.environment} environment")
    
    # Create environment configuration
    if args.environment == "production":
        env_config = create_production_environment()
    elif args.environment == "development":
        env_config = create_development_environment()
    else:
        logger.error(f"Unknown environment: {args.environment}")
        return 1
    
    # Initialize deployment manager
    deployment = SectionAwareDeployment(args.project_root)
    
    # Setup environment
    success = deployment.setup_environment(env_config)
    
    if success:
        logger.info("✅ Environment setup completed successfully")
        return 0
    else:
        logger.error("❌ Environment setup failed")
        return 1


def check_health(args) -> int:
    """Perform health check."""
    logger.info("Performing health check")
    
    # Initialize deployment manager
    deployment = SectionAwareDeployment(args.project_root)
    
    # Perform health check
    health_checks = deployment.perform_health_check()
    
    # Display results
    print("\n" + "="*60)
    print("HEALTH CHECK RESULTS")
    print("="*60)
    
    healthy_count = 0
    warning_count = 0
    error_count = 0
    
    for check in health_checks:
        status_icon = {
            "healthy": "✅",
            "warning": "⚠️",
            "error": "❌"
        }.get(check.status, "❓")
        
        print(f"{status_icon} {check.component}: {check.message}")
        
        if check.response_time_ms:
            print(f"   Response time: {check.response_time_ms:.1f}ms")
        
        if check.details and args.verbose:
            print(f"   Details: {check.details}")
        
        if check.status == "healthy":
            healthy_count += 1
        elif check.status == "warning":
            warning_count += 1
        else:
            error_count += 1
        
        print()
    
    # Summary
    print("="*60)
    print(f"SUMMARY: {healthy_count} healthy, {warning_count} warnings, {error_count} errors")
    print("="*60)
    
    if error_count > 0:
        logger.error("Health check failed - system has errors")
        return 1
    elif warning_count > 0:
        logger.warning("Health check completed with warnings")
        return 0
    else:
        logger.info("Health check passed - system is healthy")
        return 0


def get_status(args) -> int:
    """Get deployment status."""
    logger.info(f"Getting deployment status for {args.environment} environment")
    
    # Initialize deployment manager
    deployment = SectionAwareDeployment(args.project_root)
    
    # Get status
    status = deployment.get_deployment_status(args.environment)
    
    # Display status
    print("\n" + "="*60)
    print("DEPLOYMENT STATUS")
    print("="*60)
    
    status_icon = {
        "healthy": "✅",
        "degraded": "⚠️",
        "error": "❌"
    }.get(status.status, "❓")
    
    print(f"Environment: {status.environment}")
    print(f"Status: {status_icon} {status.status.upper()}")
    print(f"Version: {status.version}")
    print(f"Deployment Time: {status.deployment_time}")
    print()
    
    # Metrics
    print("METRICS:")
    for key, value in status.metrics.items():
        print(f"  {key}: {value}")
    print()
    
    # Health checks summary
    if args.verbose:
        print("HEALTH CHECKS:")
        for check in status.health_checks:
            status_icon = {
                "healthy": "✅",
                "warning": "⚠️", 
                "error": "❌"
            }.get(check.status, "❓")
            print(f"  {status_icon} {check.component}: {check.message}")
        print()
    
    # Save report if requested
    if args.output:
        output_file = deployment.save_deployment_report(status, args.output)
        print(f"📊 Status report saved to: {output_file}")
    
    return 0 if status.status in ["healthy", "degraded"] else 1


def check_environment(args) -> int:
    """Check environment requirements."""
    logger.info(f"Checking {args.environment} environment requirements")
    
    # Create environment configuration
    if args.environment == "production":
        env_config = create_production_environment()
    elif args.environment == "development":
        env_config = create_development_environment()
    else:
        logger.error(f"Unknown environment: {args.environment}")
        return 1
    
    # Initialize deployment manager
    deployment = SectionAwareDeployment(args.project_root)
    
    # Check environment
    health_checks = deployment.check_environment(env_config)
    
    # Display results
    print("\n" + "="*60)
    print(f"ENVIRONMENT CHECK: {args.environment.upper()}")
    print("="*60)
    
    required_failed = 0
    optional_failed = 0
    
    for check in health_checks:
        status_icon = {
            "healthy": "✅",
            "warning": "⚠️",
            "error": "❌"
        }.get(check.status, "❓")
        
        print(f"{status_icon} {check.component}: {check.message}")
        
        if args.verbose and check.details:
            print(f"   Details: {check.details}")
        
        # Track failures
        if check.status == "error":
            if "package_" in check.component and check.details.get("required", True):
                required_failed += 1
            elif "package_" not in check.component:
                required_failed += 1
        elif check.status == "warning":
            optional_failed += 1
        
        print()
    
    # Summary
    print("="*60)
    if required_failed > 0:
        print(f"❌ Environment check FAILED: {required_failed} critical issues")
        return 1
    elif optional_failed > 0:
        print(f"⚠️ Environment check PASSED with {optional_failed} warnings")
        return 0
    else:
        print("✅ Environment check PASSED")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Section-Aware Newsletter Generation Deployment Tool"
    )
    
    # Global options
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Project root directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup environment command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Set up deployment environment"
    )
    setup_parser.add_argument(
        "environment",
        choices=["production", "development"],
        help="Environment to set up"
    )
    
    # Health check command
    health_parser = subparsers.add_parser(
        "health",
        help="Perform health check"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Get deployment status"
    )
    status_parser.add_argument(
        "--environment", "-e",
        type=str,
        default="production",
        help="Environment name for status report"
    )
    status_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for status report"
    )
    
    # Environment check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check environment requirements"
    )
    check_parser.add_argument(
        "environment",
        choices=["production", "development"],
        help="Environment to check"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if args.command == "setup":
        return setup_environment(args)
    elif args.command == "health":
        return check_health(args)
    elif args.command == "status":
        return get_status(args)
    elif args.command == "check":
        return check_environment(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())