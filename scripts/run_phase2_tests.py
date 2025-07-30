#!/usr/bin/env python3
"""Test runner script for Phase 2: Testing Infrastructure."""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f}s")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def run_unit_tests():
    """Run unit tests."""
    print("\n🧪 PHASE 2: UNIT TESTS")
    print("=" * 60)
    
    tests = [
        ("pytest tests/agents/ -v", "Agent unit tests"),
        ("pytest tests/core/ -v", "Core functionality tests"),
        ("pytest tests/tools/ -v", "Tools unit tests"),
        ("pytest tests/scrapers/ -v", "Scrapers unit tests"),
    ]
    
    results = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))
    
    return results

def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 PHASE 2: INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("pytest tests/integration/test_workflows.py -v", "Workflow integration tests"),
        ("pytest tests/integration/test_api_integration.py -v", "API integration tests"),
        ("pytest tests/integration/test_error_handling.py -v", "Error handling tests"),
    ]
    
    results = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))
    
    return results

def run_performance_tests():
    """Run performance tests."""
    print("\n⚡ PHASE 2: PERFORMANCE TESTS")
    print("=" * 60)
    
    tests = [
        ("pytest tests/performance/test_performance.py -v", "Performance tests"),
    ]
    
    results = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))
    
    return results

def run_coverage_analysis():
    """Run coverage analysis."""
    print("\n📊 PHASE 2: COVERAGE ANALYSIS")
    print("=" * 60)
    
    command = "pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --tb=no -q"
    success = run_command(command, "Coverage analysis")
    
    return [("Coverage analysis", success)]

def run_linting_checks():
    """Run linting checks."""
    print("\n🔍 PHASE 2: LINTING CHECKS")
    print("=" * 60)
    
    tests = [
        ("flake8 src/ --max-line-length=88 --extend-ignore=E203,W503", "Flake8 linting"),
        ("black --check src/", "Black formatting check"),
        ("mypy src/ --ignore-missing-imports", "Type checking"),
    ]
    
    results = []
    for command, description in tests:
        success = run_command(command, description)
        results.append((description, success))
    
    return results

def generate_test_report(results, phase_name):
    """Generate a test report."""
    print(f"\n📋 {phase_name} TEST REPORT")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    return passed, failed

def main():
    """Main test runner function."""
    print("🚀 PHASE 2: TESTING INFRASTRUCTURE VALIDATION")
    print("=" * 80)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run all test phases
    all_results = []
    
    # Unit tests
    unit_results = run_unit_tests()
    all_results.extend(unit_results)
    
    # Integration tests
    integration_results = run_integration_tests()
    all_results.extend(integration_results)
    
    # Performance tests
    performance_results = run_performance_tests()
    all_results.extend(performance_results)
    
    # Coverage analysis
    coverage_results = run_coverage_analysis()
    all_results.extend(coverage_results)
    
    # Linting checks
    linting_results = run_linting_checks()
    all_results.extend(linting_results)
    
    # Generate final report
    passed, failed = generate_test_report(all_results, "PHASE 2 COMPLETE")
    
    # Overall summary
    print(f"\n🎯 PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed / len(all_results) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 PHASE 2: ALL TESTS PASSED!")
        print("Testing infrastructure is ready for production use.")
        return 0
    else:
        print(f"\n⚠️  PHASE 2: {failed} TESTS FAILED")
        print("Please review and fix failing tests before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 