"""
Deployment Utilities for Section-Aware Newsletter Generation System

This module provides deployment utilities for the section-aware newsletter
generation system, including environment setup, health checks, and monitoring.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.section_aware_config import (
    ConfigurationManager, SectionAwareConfig, get_current_config
)

logger = logging.getLogger(__name__)


@dataclass
class DeploymentEnvironment:
    """Deployment environment configuration."""
    name: str
    python_version: str = "3.9+"
    required_packages: List[str] = field(default_factory=lambda: [
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ])
    optional_packages: List[str] = field(default_factory=lambda: [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0"
    ])
    environment_variables: Dict[str, str] = field(default_factory=dict)
    health_check_endpoints: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "memory_mb": 1024,
        "cpu_cores": 2,
        "disk_mb": 5120,
        "timeout_seconds": 300
    })


@dataclass 
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # "healthy", "warning", "error"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    response_time_ms: Optional[float] = None


@dataclass
class DeploymentStatus:
    """Overall deployment status."""
    environment: str
    status: str  # "healthy", "degraded", "error"
    version: str
    deployment_time: str
    health_checks: List[HealthCheckResult] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class SectionAwareDeployment:
    """Deployment manager for section-aware newsletter generation system."""
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize deployment manager."""
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.config_dir = self.project_root / "config"
        self.deployment_dir = self.project_root / "deployment"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.deployment_dir.mkdir(exist_ok=True)
        
        self.config_manager = ConfigurationManager(str(self.config_dir))
        
        logger.info(f"Deployment manager initialized for project: {self.project_root}")
    
    def check_environment(self, environment: DeploymentEnvironment) -> List[HealthCheckResult]:
        """Check deployment environment requirements."""
        logger.info(f"Checking environment: {environment.name}")
        
        health_checks = []
        
        # Check Python version
        health_checks.append(self._check_python_version(environment.python_version))
        
        # Check required packages
        for package in environment.required_packages:
            health_checks.append(self._check_package(package, required=True))
        
        # Check optional packages
        for package in environment.optional_packages:
            health_checks.append(self._check_package(package, required=False))
        
        # Check source files
        health_checks.append(self._check_source_files())
        
        # Check configuration
        health_checks.append(self._check_configuration())
        
        # Check resource availability
        health_checks.append(self._check_resources(environment.resource_limits))
        
        return health_checks
    
    def _check_python_version(self, required_version: str) -> HealthCheckResult:
        """Check Python version compatibility."""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        try:
            # Parse required version (simple check for X.Y+ format)
            if required_version.endswith('+'):
                min_version = required_version[:-1]
                major, minor = map(int, min_version.split('.'))
                
                if sys.version_info.major > major or (
                    sys.version_info.major == major and sys.version_info.minor >= minor
                ):
                    return HealthCheckResult(
                        component="python_version",
                        status="healthy",
                        message=f"Python {current_version} meets requirement {required_version}",
                        details={"current": current_version, "required": required_version}
                    )
                else:
                    return HealthCheckResult(
                        component="python_version",
                        status="error",
                        message=f"Python {current_version} does not meet requirement {required_version}",
                        details={"current": current_version, "required": required_version}
                    )
            else:
                return HealthCheckResult(
                    component="python_version",
                    status="warning",
                    message=f"Cannot parse version requirement: {required_version}",
                    details={"current": current_version, "required": required_version}
                )
        
        except Exception as e:
            return HealthCheckResult(
                component="python_version",
                status="error",
                message=f"Error checking Python version: {e}",
                details={"current": current_version, "error": str(e)}
            )
    
    def _check_package(self, package_spec: str, required: bool = True) -> HealthCheckResult:
        """Check if a package is installed and meets version requirements."""
        try:
            # Parse package specification
            if ">=" in package_spec:
                package_name, min_version = package_spec.split(">=")
            else:
                package_name = package_spec
                min_version = None
            
            # Try to import package
            try:
                __import__(package_name)
                
                # Get version if available
                try:
                    import importlib.metadata
                    installed_version = importlib.metadata.version(package_name)
                except Exception:
                    installed_version = "unknown"
                
                status = "healthy"
                message = f"Package {package_name} is installed"
                
                if min_version and installed_version != "unknown":
                    # Simple version comparison (would need proper semver for production)
                    message += f" (version {installed_version})"
                
                return HealthCheckResult(
                    component=f"package_{package_name}",
                    status=status,
                    message=message,
                    details={
                        "package": package_name,
                        "installed_version": installed_version,
                        "required_version": min_version,
                        "required": required
                    }
                )
            
            except ImportError:
                status = "error" if required else "warning"
                message = f"Package {package_name} is not installed"
                
                return HealthCheckResult(
                    component=f"package_{package_name}",
                    status=status,
                    message=message,
                    details={
                        "package": package_name,
                        "required": required,
                        "error": "not_installed"
                    }
                )
        
        except Exception as e:
            return HealthCheckResult(
                component=f"package_{package_spec}",
                status="error",
                message=f"Error checking package {package_spec}: {e}",
                details={"package": package_spec, "error": str(e)}
            )
    
    def _check_source_files(self) -> HealthCheckResult:
        """Check that required source files exist."""
        required_files = [
            "src/core/section_aware_prompts.py",
            "src/core/section_aware_refinement.py",
            "src/core/section_quality_metrics.py",
            "src/core/continuity_validator.py",
            "src/config/section_aware_config.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            return HealthCheckResult(
                component="source_files",
                status="error",
                message=f"Missing {len(missing_files)} required source files",
                details={"missing_files": missing_files}
            )
        else:
            return HealthCheckResult(
                component="source_files",
                status="healthy",
                message=f"All {len(required_files)} required source files found",
                details={"checked_files": required_files}
            )
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check configuration validity."""
        start_time = time.time()
        
        try:
            config = self.config_manager.load_config()
            issues = config.validate()
            
            response_time = (time.time() - start_time) * 1000
            
            if issues:
                return HealthCheckResult(
                    component="configuration",
                    status="warning",
                    message=f"Configuration loaded with {len(issues)} validation issues",
                    details={"validation_issues": issues},
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="configuration",
                    status="healthy",
                    message="Configuration loaded and validated successfully",
                    details={"config_sections": list(config.to_dict().keys())},
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="configuration",
                status="error",
                message=f"Error loading configuration: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def _check_resources(self, limits: Dict[str, Any]) -> HealthCheckResult:
        """Check system resource availability."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available // (1024 * 1024)
            
            # Check disk space
            disk = psutil.disk_usage(str(self.project_root))
            available_disk_mb = disk.free // (1024 * 1024)
            
            # Check CPU
            cpu_count = psutil.cpu_count()
            
            issues = []
            if available_memory_mb < limits.get("memory_mb", 512):
                issues.append(f"Low memory: {available_memory_mb}MB available, {limits['memory_mb']}MB required")
            
            if available_disk_mb < limits.get("disk_mb", 1024):
                issues.append(f"Low disk space: {available_disk_mb}MB available, {limits['disk_mb']}MB required")
            
            if cpu_count < limits.get("cpu_cores", 1):
                issues.append(f"Insufficient CPU cores: {cpu_count} available, {limits['cpu_cores']} required")
            
            status = "error" if issues else "healthy"
            message = "Resource check passed" if not issues else f"{len(issues)} resource issues found"
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                details={
                    "available_memory_mb": available_memory_mb,
                    "available_disk_mb": available_disk_mb,
                    "cpu_cores": cpu_count,
                    "limits": limits,
                    "issues": issues
                }
            )
        
        except ImportError:
            return HealthCheckResult(
                component="system_resources",
                status="warning",
                message="psutil not available - cannot check system resources",
                details={"error": "psutil_not_installed"}
            )
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status="error",
                message=f"Error checking system resources: {e}",
                details={"error": str(e)}
            )
    
    def perform_health_check(self) -> List[HealthCheckResult]:
        """Perform comprehensive health check of deployed system."""
        logger.info("Performing comprehensive health check")
        
        health_checks = []
        
        # Check core components
        health_checks.append(self._health_check_prompt_engine())
        health_checks.append(self._health_check_refinement_system())
        health_checks.append(self._health_check_quality_metrics())
        health_checks.append(self._health_check_continuity_validator())
        
        # Check configuration
        health_checks.append(self._check_configuration())
        
        # Check integration
        health_checks.append(self._health_check_integration())
        
        return health_checks
    
    def _health_check_prompt_engine(self) -> HealthCheckResult:
        """Health check for section-aware prompt engine."""
        start_time = time.time()
        
        try:
            from core.section_aware_prompts import SectionAwarePromptManager, SectionType
            
            manager = SectionAwarePromptManager()
            
            # Test basic functionality
            context = {
                'topic': 'Health Check Test',
                'audience': 'Test Audience',
                'content_focus': 'Test Content',
                'word_count': 1000
            }
            
            prompt = manager.get_section_prompt(SectionType.INTRODUCTION, context)
            
            response_time = (time.time() - start_time) * 1000
            
            if len(prompt) > 100 and 'Health Check Test' in prompt:
                return HealthCheckResult(
                    component="prompt_engine",
                    status="healthy",
                    message="Section-aware prompt engine is functional",
                    details={"prompt_length": len(prompt), "test_passed": True},
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="prompt_engine",
                    status="error",
                    message="Prompt engine test failed - invalid output",
                    details={"prompt_length": len(prompt), "test_passed": False},
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="prompt_engine",
                status="error",
                message=f"Error testing prompt engine: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def _health_check_refinement_system(self) -> HealthCheckResult:
        """Health check for section-aware refinement system."""
        start_time = time.time()
        
        try:
            from core.section_aware_refinement import SectionAwareRefinementLoop, SectionBoundaryDetector
            
            detector = SectionBoundaryDetector()
            refinement_loop = SectionAwareRefinementLoop(max_iterations=1)
            
            # Test boundary detection
            test_content = """# Introduction
            Test content for health check.
            
            ## Analysis
            More test content here."""
            
            boundaries = detector.detect_boundaries(test_content)
            
            response_time = (time.time() - start_time) * 1000
            
            if len(boundaries) >= 1:
                return HealthCheckResult(
                    component="refinement_system",
                    status="healthy",
                    message="Section refinement system is functional",
                    details={"boundaries_detected": len(boundaries), "test_passed": True},
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="refinement_system",
                    status="error",
                    message="Refinement system test failed - no boundaries detected",
                    details={"boundaries_detected": len(boundaries), "test_passed": False},
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="refinement_system",
                status="error",
                message=f"Error testing refinement system: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def _health_check_quality_metrics(self) -> HealthCheckResult:
        """Health check for section quality metrics system."""
        start_time = time.time()
        
        try:
            from core.section_quality_metrics import SectionQualityAnalyzer, SectionAwareQualitySystem
            from src.core.section_aware_prompts import SectionType
            
            analyzer = SectionQualityAnalyzer()
            
            # Test quality analysis
            test_content = "This is a test section for quality analysis. It contains multiple sentences."
            metrics = analyzer.analyze_section(test_content, SectionType.ANALYSIS)
            
            response_time = (time.time() - start_time) * 1000
            
            if 0.0 <= metrics.overall_score <= 1.0 and metrics.word_count > 0:
                return HealthCheckResult(
                    component="quality_metrics",
                    status="healthy",
                    message="Quality metrics system is functional",
                    details={
                        "quality_score": metrics.overall_score,
                        "word_count": metrics.word_count,
                        "test_passed": True
                    },
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="quality_metrics",
                    status="error",
                    message="Quality metrics test failed - invalid results",
                    details={
                        "quality_score": metrics.overall_score,
                        "word_count": metrics.word_count,
                        "test_passed": False
                    },
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="quality_metrics",
                status="error",
                message=f"Error testing quality metrics: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def _health_check_continuity_validator(self) -> HealthCheckResult:
        """Health check for continuity validation system."""
        start_time = time.time()
        
        try:
            from core.continuity_validator import ContinuityValidator
            from src.core.section_aware_prompts import SectionType
            
            validator = ContinuityValidator()
            
            # Test continuity validation
            test_sections = {
                SectionType.INTRODUCTION: "This is an introduction section.",
                SectionType.ANALYSIS: "This is an analysis section with more content."
            }
            
            report = validator.validate_newsletter_continuity(test_sections)
            
            response_time = (time.time() - start_time) * 1000
            
            if (0.0 <= report.overall_continuity_score <= 1.0 and 
                report.sections_analyzed == len(test_sections)):
                return HealthCheckResult(
                    component="continuity_validator",
                    status="healthy",
                    message="Continuity validator is functional",
                    details={
                        "continuity_score": report.overall_continuity_score,
                        "sections_analyzed": report.sections_analyzed,
                        "test_passed": True
                    },
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="continuity_validator",
                    status="error",
                    message="Continuity validator test failed - invalid results",
                    details={
                        "continuity_score": report.overall_continuity_score,
                        "sections_analyzed": report.sections_analyzed,
                        "test_passed": False
                    },
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="continuity_validator",
                status="error",
                message=f"Error testing continuity validator: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def _health_check_integration(self) -> HealthCheckResult:
        """Health check for component integration."""
        start_time = time.time()
        
        try:
            # Test basic integration workflow
            from core.section_aware_prompts import SectionAwarePromptManager, SectionType
            from src.core.section_aware_refinement import SectionBoundaryDetector
            from src.core.section_quality_metrics import SectionQualityAnalyzer
            from core.continuity_validator import ContinuityValidator
            
            # Initialize components
            prompt_manager = SectionAwarePromptManager()
            detector = SectionBoundaryDetector()
            analyzer = SectionQualityAnalyzer()
            validator = ContinuityValidator()
            
            # Test integration workflow
            test_content = """# Test Newsletter
            This is a test introduction.
            
            ## Analysis Section
            This is test analysis content."""
            
            # Detect boundaries
            boundaries = detector.detect_boundaries(test_content)
            
            # Generate prompt
            context = {'topic': 'Test', 'audience': 'Test', 'content_focus': 'Test', 'word_count': 500}
            prompt = prompt_manager.get_section_prompt(SectionType.INTRODUCTION, context)
            
            # Analyze quality
            metrics = analyzer.analyze_section("Test content", SectionType.INTRODUCTION)
            
            # Validate continuity
            sections = {SectionType.INTRODUCTION: "Test intro", SectionType.ANALYSIS: "Test analysis"}
            report = validator.validate_newsletter_continuity(sections)
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if all components worked
            if (len(boundaries) > 0 and len(prompt) > 0 and 
                metrics.overall_score >= 0.0 and report.sections_analyzed > 0):
                return HealthCheckResult(
                    component="integration",
                    status="healthy",
                    message="Component integration is functional",
                    details={
                        "boundaries_detected": len(boundaries),
                        "prompt_generated": len(prompt) > 0,
                        "quality_analyzed": metrics.overall_score >= 0.0,
                        "continuity_validated": report.sections_analyzed > 0,
                        "test_passed": True
                    },
                    response_time_ms=response_time
                )
            else:
                return HealthCheckResult(
                    component="integration",
                    status="error",
                    message="Integration test failed - some components not working",
                    details={
                        "boundaries_detected": len(boundaries),
                        "prompt_generated": len(prompt) > 0,
                        "quality_analyzed": metrics.overall_score >= 0.0,
                        "continuity_validated": report.sections_analyzed > 0,
                        "test_passed": False
                    },
                    response_time_ms=response_time
                )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="integration",
                status="error",
                message=f"Error testing integration: {e}",
                details={"error": str(e)},
                response_time_ms=response_time
            )
    
    def get_deployment_status(self, environment_name: str = "production") -> DeploymentStatus:
        """Get comprehensive deployment status."""
        logger.info(f"Getting deployment status for environment: {environment_name}")
        
        # Perform health checks
        health_checks = self.perform_health_check()
        
        # Determine overall status
        error_count = sum(1 for check in health_checks if check.status == "error")
        warning_count = sum(1 for check in health_checks if check.status == "warning")
        
        if error_count > 0:
            overall_status = "error"
        elif warning_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Get configuration
        try:
            config = get_current_config()
            config_dict = config.to_dict()
        except Exception:
            config_dict = {"error": "Could not load configuration"}
        
        # Calculate metrics
        response_times = [
            check.response_time_ms for check in health_checks 
            if check.response_time_ms is not None
        ]
        
        metrics = {
            "total_health_checks": len(health_checks),
            "healthy_checks": sum(1 for check in health_checks if check.status == "healthy"),
            "warning_checks": warning_count,
            "error_checks": error_count,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0
        }
        
        return DeploymentStatus(
            environment=environment_name,
            status=overall_status,
            version="1.0.0",  # Would be populated from version management
            deployment_time=datetime.utcnow().isoformat(),
            health_checks=health_checks,
            configuration=config_dict,
            metrics=metrics
        )
    
    def save_deployment_report(self, status: DeploymentStatus, 
                              output_file: Optional[str] = None) -> str:
        """Save deployment status report to file."""
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.deployment_dir / f"deployment_status_{timestamp}.json")
        
        try:
            # Convert to serializable format
            report_data = {
                "environment": status.environment,
                "status": status.status,
                "version": status.version,
                "deployment_time": status.deployment_time,
                "health_checks": [
                    {
                        "component": check.component,
                        "status": check.status,
                        "message": check.message,
                        "details": check.details,
                        "timestamp": check.timestamp,
                        "response_time_ms": check.response_time_ms
                    }
                    for check in status.health_checks
                ],
                "configuration": status.configuration,
                "metrics": status.metrics
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Deployment report saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving deployment report: {e}")
            raise
    
    def setup_environment(self, environment: DeploymentEnvironment) -> bool:
        """Set up deployment environment."""
        logger.info(f"Setting up environment: {environment.name}")
        
        try:
            # Set environment variables
            for key, value in environment.environment_variables.items():
                os.environ[key] = value
                logger.info(f"Set environment variable: {key}")
            
            # Create default configuration if it doesn't exist
            config_file = self.config_dir / "section_aware_config.yaml"
            if not config_file.exists():
                logger.info("Creating default configuration file")
                self.config_manager.create_default_config_file(str(config_file))
            
            # Verify setup
            health_checks = self.check_environment(environment)
            errors = [check for check in health_checks if check.status == "error"]
            
            if errors:
                logger.error(f"Environment setup failed with {len(errors)} errors")
                for error in errors:
                    logger.error(f"  {error.component}: {error.message}")
                return False
            else:
                logger.info("Environment setup completed successfully")
                return True
        
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False


def create_production_environment() -> DeploymentEnvironment:
    """Create production deployment environment configuration."""
    return DeploymentEnvironment(
        name="production",
        python_version="3.9+",
        required_packages=[
            "pydantic>=2.0.0",
            "python-dotenv>=1.0.0", 
            "pyyaml>=6.0.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "psutil>=5.8.0"
        ],
        environment_variables={
            "LOG_LEVEL": "INFO",
            "ENABLE_METRICS": "true",
            "ENABLE_PERFORMANCE_OPTIMIZATION": "true"
        },
        resource_limits={
            "memory_mb": 2048,
            "cpu_cores": 2,
            "disk_mb": 10240,
            "timeout_seconds": 300
        }
    )


def create_development_environment() -> DeploymentEnvironment:
    """Create development deployment environment configuration."""
    return DeploymentEnvironment(
        name="development",
        python_version="3.9+",
        required_packages=[
            "pydantic>=2.0.0",
            "python-dotenv>=1.0.0",
            "pyyaml>=6.0.0",
            "numpy>=1.21.0"
        ],
        optional_packages=[
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0"
        ],
        environment_variables={
            "LOG_LEVEL": "DEBUG",
            "ENABLE_METRICS": "true",
            "ENABLE_ADVANCED_ANALYTICS": "true"
        },
        resource_limits={
            "memory_mb": 1024,
            "cpu_cores": 1,
            "disk_mb": 5120,
            "timeout_seconds": 600
        }
    )


# Export main classes and functions
__all__ = [
    'SectionAwareDeployment',
    'DeploymentEnvironment',
    'DeploymentStatus',
    'HealthCheckResult',
    'create_production_environment',
    'create_development_environment'
]