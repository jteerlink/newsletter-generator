#!/usr/bin/env python3

"""
Phase 4.3: Production Readiness Validation Suite

This module provides comprehensive validation tests to ensure the tool tracking
system is ready for production deployment.
"""

import pytest
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.append('.')

from src.core.tool_usage_tracker import get_tool_tracker
from src.core.workflow_orchestrator import WorkflowOrchestrator


class ProductionValidationSuite:
    """Comprehensive validation for production readiness."""
    
    def __init__(self):
        self.tracker = get_tool_tracker()
        self.validation_results = {}
        
    def validate_all(self) -> Dict[str, Any]:
        """Run all production validation tests."""
        print("🔍 Running Production Readiness Validation Suite...")
        
        validations = {
            'workflow_consistency': self.validate_workflow_consistency(),
            'data_integrity': self.validate_data_integrity(),
            'error_recovery': self.validate_error_recovery(),
            'performance_benchmarks': self.validate_performance_benchmarks(),
            'deployment_readiness': self.validate_deployment_readiness()
        }
        
        # Calculate overall readiness score
        validations['overall_readiness'] = self._calculate_readiness_score(validations)
        
        return validations
    
    def validate_workflow_consistency(self) -> Dict[str, Any]:
        """Validate consistent workflow tracking across multiple runs."""
        print("  ✓ Validating workflow consistency...")
        
        results = []
        test_topic = "Production Validation Test"
        
        for run in range(2):  # Run 2 workflows to test consistency
            print(f"    Run {run + 1}/2...")
            
            try:
                initial_count = len(self.tracker.get_tool_usage_history(hours_back=1))
                
                orchestrator = WorkflowOrchestrator()
                result = orchestrator.execute_newsletter_generation(
                    topic=f"{test_topic} Run {run + 1}",
                    context_id="default",
                    output_format="markdown"
                )
                
                final_count = len(self.tracker.get_tool_usage_history(hours_back=1))
                entries_added = final_count - initial_count
                
                # Get workflow-specific entries
                workflow_entries = [
                    e for e in self.tracker.get_tool_usage_history(hours_back=1)
                    if hasattr(e, 'workflow_id') and e.workflow_id == result.workflow_id
                ]
                
                run_result = {
                    'run_number': run + 1,
                    'workflow_id': result.workflow_id,
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'entries_added': entries_added,
                    'workflow_entries': len(workflow_entries),
                    'tools_tracked': list(set(e.tool_name for e in workflow_entries))
                }
                results.append(run_result)
                
            except Exception as e:
                results.append({
                    'run_number': run + 1,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Analyze consistency
        successful_runs = [r for r in results if 'error' not in r]
        consistency_score = len(successful_runs) / len(results) * 100
        
        if len(successful_runs) >= 2:
            # Check if tracking patterns are reasonably consistent (allow some variance due to LLM non-determinism)
            all_tools = set()
            for run in successful_runs:
                all_tools.update(run['tools_tracked'])
            
            # Tools consistency: at least 70% overlap in tools used
            min_tools = min(len(r['tools_tracked']) for r in successful_runs)
            max_tools = max(len(r['tools_tracked']) for r in successful_runs)
            tools_consistency = (min_tools / max_tools) >= 0.7 if max_tools > 0 else True
            
            entry_variance = abs(successful_runs[0]['workflow_entries'] - successful_runs[1]['workflow_entries'])
        else:
            tools_consistency = False
            entry_variance = float('inf')
        
        return {
            'results': results,
            'consistency_score': consistency_score,
            'tools_consistency': tools_consistency,
            'entry_variance': entry_variance,
            'validation_passed': consistency_score >= 90 and tools_consistency and entry_variance <= 5
        }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity in tracking logs."""
        print("  ✓ Validating data integrity...")
        
        # Check log file integrity
        log_file_valid = self.tracker.log_file.exists()
        
        try:
            # Attempt to read and parse log file as single JSON object
            if log_file_valid:
                with open(self.tracker.log_file, 'r') as f:
                    try:
                        data = json.load(f)
                        # Validate expected structure
                        if isinstance(data, dict) and 'entries' in data and isinstance(data['entries'], list):
                            total_entries = len(data['entries'])
                            valid_entries = 0
                            
                            # Validate each entry structure
                            required_fields = ['timestamp', 'tool_name', 'agent_name', 'execution_time', 'status']
                            for entry in data['entries']:
                                if isinstance(entry, dict) and all(field in entry for field in required_fields):
                                    valid_entries += 1
                            
                            data_integrity_score = (valid_entries / total_entries * 100) if total_entries > 0 else 100
                        else:
                            data_integrity_score = 0
                            valid_entries = 0
                            total_entries = 1  # File exists but wrong format
                    except json.JSONDecodeError:
                        data_integrity_score = 0
                        valid_entries = 0
                        total_entries = 1
            else:
                data_integrity_score = 0
                valid_entries = 0
                total_entries = 0
        
        except Exception as e:
            data_integrity_score = 0
            valid_entries = 0
            total_entries = 0
            log_file_valid = False
        
        # Validate recent entries structure
        recent_entries = self.tracker.get_tool_usage_history(hours_back=24)
        valid_structure_count = 0
        
        required_fields = ['tool_name', 'agent_name', 'execution_time', 'status']
        
        for entry in recent_entries:
            if all(hasattr(entry, field) for field in required_fields):
                valid_structure_count += 1
        
        structure_integrity = (valid_structure_count / len(recent_entries) * 100) if recent_entries else 100
        
        return {
            'log_file_exists': log_file_valid,
            'data_integrity_score': data_integrity_score,
            'valid_entries': valid_entries,
            'total_entries': total_entries,
            'structure_integrity': structure_integrity,
            'recent_entries_count': len(recent_entries),
            'validation_passed': data_integrity_score >= 95 and structure_integrity >= 95
        }
    
    def validate_error_recovery(self) -> Dict[str, Any]:
        """Validate system recovery from various error conditions."""
        print("  ✓ Validating error recovery...")
        
        recovery_tests = []
        
        # Test 1: Tool execution error recovery
        try:
            with self.tracker.track_tool_usage(
                tool_name="error_recovery_test",
                agent_name="ValidationAgent",
                workflow_id="error-recovery-test",
                session_id="error-session"
            ):
                raise ValueError("Intentional test error")
        except ValueError:
            # This is expected - check if tracking still works
            recovery_tests.append({
                'test': 'tool_execution_error',
                'passed': True  # Expected to raise and be caught
            })
        except Exception:
            recovery_tests.append({
                'test': 'tool_execution_error', 
                'passed': False
            })
        
        # Test 2: Rapid error succession
        error_count = 0
        success_count = 0
        
        for i in range(10):
            try:
                with self.tracker.track_tool_usage(
                    tool_name=f"rapid_error_test_{i}",
                    agent_name="ValidationAgent",
                    workflow_id="rapid-error-test",
                    session_id="rapid-error-session"
                ):
                    if i % 3 == 0:  # Every 3rd iteration fails
                        raise RuntimeError(f"Test error {i}")
                success_count += 1
            except RuntimeError:
                error_count += 1
            except Exception:
                # Unexpected error
                pass
        
        recovery_tests.append({
            'test': 'rapid_error_succession',
            'error_count': error_count,
            'success_count': success_count,
            'passed': success_count >= 6  # Should handle most successes
        })
        
        # Test 3: System continues after errors
        try:
            # Normal operation after errors
            with self.tracker.track_tool_usage(
                tool_name="post_error_test",
                agent_name="ValidationAgent", 
                workflow_id="post-error-test",
                session_id="post-error-session"
            ):
                time.sleep(0.1)
            
            recovery_tests.append({
                'test': 'post_error_operation',
                'passed': True
            })
        except Exception:
            recovery_tests.append({
                'test': 'post_error_operation',
                'passed': False
            })
        
        passed_tests = sum(1 for test in recovery_tests if test['passed'])
        recovery_score = (passed_tests / len(recovery_tests) * 100) if recovery_tests else 0
        
        return {
            'recovery_tests': recovery_tests,
            'recovery_score': recovery_score,
            'passed_tests': passed_tests,
            'total_tests': len(recovery_tests),
            'validation_passed': recovery_score >= 80
        }
    
    def validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Validate performance meets production benchmarks."""
        print("  ✓ Validating performance benchmarks...")
        
        # Define production benchmarks
        benchmarks = {
            'max_tracking_overhead_ms': 50,    # 50ms max overhead per operation
            'max_memory_per_entry_kb': 10,     # 10KB max memory per entry
            'max_log_entry_size_bytes': 2000,  # 2KB max per log entry
            'min_requests_per_second': 100     # 100 requests/sec minimum
        }
        
        # Test tracking overhead
        overhead_times = []
        for _ in range(10):
            start_time = time.time()
            with self.tracker.track_tool_usage(
                tool_name="benchmark_test",
                agent_name="BenchmarkAgent",
                workflow_id="benchmark-test",
                session_id="benchmark-session"
            ):
                pass
            overhead_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        avg_overhead_ms = sum(overhead_times) / len(overhead_times)
        
        # Test requests per second
        start_time = time.time()
        request_count = 50
        
        for i in range(request_count):
            with self.tracker.track_tool_usage(
                tool_name=f"rps_test_{i}",
                agent_name="RPSAgent",
                workflow_id="rps-test",
                session_id="rps-session"
            ):
                pass
        
        elapsed_time = time.time() - start_time
        requests_per_second = request_count / elapsed_time if elapsed_time > 0 else 0
        
        # Check benchmark compliance
        benchmark_results = {
            'tracking_overhead_ms': {
                'measured': avg_overhead_ms,
                'benchmark': benchmarks['max_tracking_overhead_ms'],
                'passed': avg_overhead_ms <= benchmarks['max_tracking_overhead_ms']
            },
            'requests_per_second': {
                'measured': requests_per_second,
                'benchmark': benchmarks['min_requests_per_second'],
                'passed': requests_per_second >= benchmarks['min_requests_per_second']
            }
        }
        
        passed_benchmarks = sum(1 for result in benchmark_results.values() if result['passed'])
        benchmark_score = (passed_benchmarks / len(benchmark_results) * 100)
        
        return {
            'benchmark_results': benchmark_results,
            'benchmark_score': benchmark_score,
            'passed_benchmarks': passed_benchmarks,
            'total_benchmarks': len(benchmark_results),
            'validation_passed': benchmark_score >= 75
        }
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate system is ready for production deployment."""
        print("  ✓ Validating deployment readiness...")
        
        readiness_checks = []
        
        # Check 1: Required files exist
        required_files = [
            'src/core/tool_usage_tracker.py',
            'src/core/workflow_orchestrator.py',
            'src/core/feedback_orchestrator.py',
            'logs/tool_usage.json'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        readiness_checks.append({
            'check': 'required_files',
            'passed': len(missing_files) == 0,
            'missing_files': missing_files
        })
        
        # Check 2: Configuration is valid
        try:
            tracker = get_tool_tracker()
            config_valid = (
                hasattr(tracker, 'log_file') and
                hasattr(tracker, 'track_tool_usage') and
                tracker.log_file.parent.exists()
            )
        except Exception:
            config_valid = False
        
        readiness_checks.append({
            'check': 'configuration_valid',
            'passed': config_valid
        })
        
        # Check 3: Basic functionality works
        try:
            with tracker.track_tool_usage(
                tool_name="readiness_test",
                agent_name="DeploymentAgent",
                workflow_id="deployment-test",
                session_id="deployment-session"
            ):
                pass
            functionality_works = True
        except Exception:
            functionality_works = False
        
        readiness_checks.append({
            'check': 'basic_functionality',
            'passed': functionality_works
        })
        
        # Check 4: Log directory is writable
        try:
            test_file = tracker.log_file.parent / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            log_writable = True
        except Exception:
            log_writable = False
        
        readiness_checks.append({
            'check': 'log_directory_writable',
            'passed': log_writable
        })
        
        passed_checks = sum(1 for check in readiness_checks if check['passed'])
        readiness_score = (passed_checks / len(readiness_checks) * 100)
        
        return {
            'readiness_checks': readiness_checks,
            'readiness_score': readiness_score,
            'passed_checks': passed_checks,
            'total_checks': len(readiness_checks),
            'validation_passed': readiness_score >= 90,
            'deployment_ready': readiness_score == 100
        }
    
    def _calculate_readiness_score(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall production readiness score."""
        
        # Weight different validation categories
        weights = {
            'workflow_consistency': 0.25,
            'data_integrity': 0.25,
            'error_recovery': 0.20,
            'performance_benchmarks': 0.15,
            'deployment_readiness': 0.15
        }
        
        total_score = 0
        validation_summary = {}
        
        for category, weight in weights.items():
            if category in validations:
                validation_data = validations[category]
                if 'validation_passed' in validation_data:
                    category_score = 100 if validation_data['validation_passed'] else 0
                elif 'benchmark_score' in validation_data:
                    category_score = validation_data['benchmark_score']
                elif 'readiness_score' in validation_data:
                    category_score = validation_data['readiness_score']
                else:
                    category_score = 0
                
                weighted_score = category_score * weight
                total_score += weighted_score
                
                validation_summary[category] = {
                    'score': category_score,
                    'weight': weight,
                    'weighted_score': weighted_score,
                    'passed': category_score >= 75
                }
        
        # Determine overall readiness level
        if total_score >= 90:
            readiness_level = "PRODUCTION_READY"
        elif total_score >= 75:
            readiness_level = "STAGING_READY"
        elif total_score >= 60:
            readiness_level = "DEVELOPMENT_READY"
        else:
            readiness_level = "NOT_READY"
        
        return {
            'overall_score': total_score,
            'readiness_level': readiness_level,
            'validation_summary': validation_summary,
            'recommendations': self._generate_recommendations(validation_summary, total_score)
        }
    
    def _generate_recommendations(self, validation_summary: Dict[str, Any], overall_score: float) -> List[str]:
        """Generate recommendations for improving production readiness."""
        recommendations = []
        
        if overall_score < 90:
            recommendations.append("Overall score below production threshold (90%)")
        
        for category, data in validation_summary.items():
            if not data['passed']:
                if category == 'workflow_consistency':
                    recommendations.append("Improve workflow tracking consistency across multiple runs")
                elif category == 'data_integrity':
                    recommendations.append("Address data integrity issues in tracking logs")
                elif category == 'error_recovery':
                    recommendations.append("Enhance error recovery mechanisms")
                elif category == 'performance_benchmarks':
                    recommendations.append("Optimize performance to meet production benchmarks")
                elif category == 'deployment_readiness':
                    recommendations.append("Complete deployment readiness requirements")
        
        if not recommendations:
            recommendations.append("System meets all production readiness criteria")
        
        return recommendations


def main():
    """Run production validation suite."""
    print("🚀 Phase 4.3: Production Readiness Validation")
    print("=" * 50)
    
    validator = ProductionValidationSuite()
    
    try:
        results = validator.validate_all()
        
        # Save results
        results_file = Path("tests/phase4/production_validation_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\\n📋 PRODUCTION READINESS SUMMARY")
        print("=" * 40)
        
        overall = results['overall_readiness']
        print(f"Overall Score: {overall['overall_score']:.1f}/100")
        print(f"Readiness Level: {overall['readiness_level']}")
        
        print("\\n🔍 VALIDATION BREAKDOWN:")
        for category, data in overall['validation_summary'].items():
            status = "✅" if data['passed'] else "❌"
            print(f"  {status} {category.replace('_', ' ').title()}: {data['score']:.1f}/100")
        
        print("\\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(overall['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\\n📄 Full results saved to: {results_file}")
        
        if overall['readiness_level'] == "PRODUCTION_READY":
            print("\\n🎉 System is PRODUCTION READY!")
        else:
            print(f"\\n⚠️  System readiness level: {overall['readiness_level']}")
        
        return results
        
    except Exception as e:
        print(f"\\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
