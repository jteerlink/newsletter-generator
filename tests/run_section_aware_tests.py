#!/usr/bin/env python3
"""
Test Runner for Section-Aware Newsletter Generation System

Comprehensive test runner for the Phase 1 section-aware components including:
- Unit tests for all components
- Integration tests
- Performance validation
- Coverage reporting
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class SectionAwareTestRunner:
    """Test runner for section-aware newsletter generation system."""
    
    def __init__(self, project_root: str):
        """Initialize test runner."""
        self.project_root = Path(project_root)
        self.test_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        
        # Section-aware test modules
        self.section_aware_tests = [
            "test_section_aware_prompts.py",
            "test_section_aware_refinement.py", 
            "test_section_quality_metrics.py",
            "test_continuity_validator.py",
            "test_integration.py"
        ]
        
        logger.info(f"Initialized test runner for project: {self.project_root}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import pytest
            import coverage
            logger.info("Required test dependencies found")
            return True
        except ImportError as e:
            logger.error(f"Missing test dependency: {e}")
            logger.info("Install test dependencies with: pip install pytest pytest-cov coverage")
            return False
    
    def check_source_files(self) -> bool:
        """Check if section-aware source files exist."""
        required_files = [
            "src/core/section_aware_prompts.py",
            "src/core/section_aware_refinement.py",
            "src/core/section_quality_metrics.py", 
            "src/core/continuity_validator.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required source files: {missing_files}")
            return False
        
        logger.info("All required source files found")
        return True
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> Dict[str, Any]:
        """Run unit tests for section-aware components."""
        logger.info("Running section-aware unit tests...")
        
        # Prepare pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test files
        for test_file in self.section_aware_tests:
            test_path = self.test_dir / test_file
            if test_path.exists():
                cmd.append(str(test_path))
            else:
                logger.warning(f"Test file not found: {test_file}")
        
        # Add options
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=src/core",
                "--cov-report=html:tests/coverage_html",
                "--cov-report=json:tests/coverage.json",
                "--cov-report=term"
            ])
        
        # Add markers to exclude slow tests by default
        cmd.extend(["-m", "not slow"])
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            test_results = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                logger.info("Unit tests passed successfully")
            else:
                logger.error(f"Unit tests failed with return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Unit tests timed out after 5 minutes")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Error running unit tests: {e}")
            return {"success": False, "error": str(e)}
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "test_integration.py"),
            "-m", "integration"
        ]
        
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for integration tests
            )
            
            test_results = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                logger.info("Integration tests passed successfully")
            else:
                logger.error("Integration tests failed")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Integration tests timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Error running integration tests: {e}")
            return {"success": False, "error": str(e)}
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "test_integration.py"),
            "-m", "performance",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout for performance tests
            )
            
            test_results = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                logger.info("Performance tests completed successfully")
            else:
                logger.warning("Some performance tests may have failed or been skipped")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Performance tests timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Error running performance tests: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate and parse coverage report."""
        coverage_file = self.project_root / "tests" / "coverage.json"
        
        if not coverage_file.exists():
            logger.warning("Coverage report not found")
            return {"success": False, "error": "no_coverage_file"}
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            # Extract key metrics
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            file_coverage = {}
            
            for file_path, file_data in coverage_data.get('files', {}).items():
                if 'src/core' in file_path:
                    file_coverage[file_path] = file_data.get('summary', {}).get('percent_covered', 0)
            
            coverage_report = {
                "success": True,
                "total_coverage": total_coverage,
                "file_coverage": file_coverage,
                "coverage_file": str(coverage_file)
            }
            
            logger.info(f"Total test coverage: {total_coverage:.1f}%")
            
            return coverage_report
            
        except Exception as e:
            logger.error(f"Error parsing coverage report: {e}")
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self, verbose: bool = False, include_slow: bool = False) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report."""
        logger.info("Starting comprehensive test suite for section-aware newsletter generation")
        
        # Check prerequisites
        if not self.check_dependencies():
            return {"success": False, "error": "missing_dependencies"}
        
        if not self.check_source_files():
            return {"success": False, "error": "missing_source_files"}
        
        # Initialize results
        results = {
            "success": True,
            "timestamp": str(Path(__file__).stat().st_mtime),
            "test_suites": {}
        }
        
        # Run unit tests
        logger.info("=" * 60)
        logger.info("RUNNING UNIT TESTS")
        logger.info("=" * 60)
        
        unit_results = self.run_unit_tests(verbose=verbose, coverage=True)
        results["test_suites"]["unit_tests"] = unit_results
        
        if not unit_results["success"]:
            results["success"] = False
            logger.error("Unit tests failed - stopping test execution")
            return results
        
        # Run integration tests
        logger.info("=" * 60)
        logger.info("RUNNING INTEGRATION TESTS")
        logger.info("=" * 60)
        
        integration_results = self.run_integration_tests(verbose=verbose)
        results["test_suites"]["integration_tests"] = integration_results
        
        # Run performance tests (non-blocking)
        if include_slow:
            logger.info("=" * 60)
            logger.info("RUNNING PERFORMANCE TESTS")
            logger.info("=" * 60)
            
            performance_results = self.run_performance_tests(verbose=verbose)
            results["test_suites"]["performance_tests"] = performance_results
        
        # Generate coverage report
        logger.info("=" * 60)
        logger.info("GENERATING COVERAGE REPORT")
        logger.info("=" * 60)
        
        coverage_report = self.generate_coverage_report()
        results["coverage"] = coverage_report
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        total_success = all(
            suite_results.get("success", False) 
            for suite_name, suite_results in results["test_suites"].items()
            if suite_name != "performance_tests"  # Performance tests are informational
        )
        
        results["success"] = total_success
        
        if total_success:
            logger.info("✅ All critical tests passed successfully!")
        else:
            logger.error("❌ Some critical tests failed")
        
        if coverage_report["success"]:
            logger.info(f"📊 Test coverage: {coverage_report['total_coverage']:.1f}%")
        
        return results
    
    def save_test_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Save test results to JSON file."""
        if output_file is None:
            output_file = str(self.project_root / "tests" / "section_aware_test_results.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test report saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
            return None


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Section-Aware Newsletter Generation Test Runner")
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose test output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        default=True,
        help="Generate coverage report (default: True)"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow/performance tests"
    )
    
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests"
    )
    
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for test results"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=os.path.dirname(os.path.dirname(__file__)),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = SectionAwareTestRunner(args.project_root)
    
    # Run tests based on arguments
    if args.unit_only:
        logger.info("Running unit tests only")
        results = {
            "success": True,
            "test_suites": {
                "unit_tests": runner.run_unit_tests(verbose=args.verbose, coverage=args.coverage)
            }
        }
        if args.coverage:
            results["coverage"] = runner.generate_coverage_report()
    
    elif args.integration_only:
        logger.info("Running integration tests only")
        results = {
            "success": True,
            "test_suites": {
                "integration_tests": runner.run_integration_tests(verbose=args.verbose)
            }
        }
    
    else:
        # Run all tests
        results = runner.run_all_tests(verbose=args.verbose, include_slow=args.include_slow)
    
    # Save results
    output_file = runner.save_test_report(results, args.output)
    
    # Exit with appropriate code
    exit_code = 0 if results["success"] else 1
    
    if exit_code == 0:
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.error("💥 Some tests failed - check output for details")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()