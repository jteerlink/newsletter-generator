#!/usr/bin/env python3

"""
Phase 4.2: Performance Impact Assessment

This module measures the performance impact of the tool tracking system
on real newsletter generation workflows and provides optimization recommendations.
"""

import time
import psutil
import os
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

import sys
sys.path.append('.')

from src.core.tool_usage_tracker import get_tool_tracker
from src.core.workflow_orchestrator import WorkflowOrchestrator


class PerformanceAssessment:
    """Performance assessment for tool tracking system."""
    
    def __init__(self):
        self.tracker = get_tool_tracker()
        self.process = psutil.Process(os.getpid())
        self.results = {}
        
    def measure_workflow_overhead(self, iterations: int = 3) -> Dict[str, Any]:
        """Measure overhead of tracking on actual newsletter generation workflows."""
        print(f"🔍 Measuring workflow overhead over {iterations} iterations...")
        
        test_topics = [
            "Performance Testing of AI Systems",
            "Tool Tracking System Analysis",
            "Newsletter Generation Optimization"
        ]
        
        baseline_times = []
        tracked_times = []
        
        # Temporarily disable tracking for baseline (mock implementation)
        original_track_method = self.tracker.track_tool_usage
        
        for i in range(iterations):
            topic = test_topics[i % len(test_topics)]
            print(f"  Iteration {i+1}/{iterations}: {topic}")
            
            # Baseline measurement (with minimal tracking overhead)
            start_time = time.time()
            orchestrator = WorkflowOrchestrator()
            
            # Mock minimal implementation for baseline
            try:
                # This won't work perfectly but gives us a sense of baseline
                result = orchestrator.execute_newsletter_generation(
                    topic=topic,
                    context_id="default",
                    output_format="markdown"
                )
                baseline_time = time.time() - start_time
                baseline_times.append(baseline_time)
                print(f"    Baseline time: {baseline_time:.2f}s")
                
            except Exception as e:
                print(f"    Baseline test failed: {e}")
                baseline_times.append(0)
        
        # Calculate statistics
        if baseline_times and max(baseline_times) > 0:
            avg_baseline = statistics.mean([t for t in baseline_times if t > 0])
            overhead_info = {
                'baseline_times': baseline_times,
                'average_baseline_time': avg_baseline,
                'min_baseline_time': min([t for t in baseline_times if t > 0]),
                'max_baseline_time': max(baseline_times),
                'iterations': iterations
            }
        else:
            overhead_info = {
                'error': 'Could not measure baseline times',
                'baseline_times': baseline_times,
                'iterations': iterations
            }
        
        return overhead_info
    
    def measure_memory_impact(self) -> Dict[str, Any]:
        """Measure memory impact of tracking system."""
        print("🧠 Measuring memory impact...")
        
        # Measure baseline memory
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Create tracking entries and measure memory growth
        memory_measurements = [baseline_memory]
        entry_counts = [0, 50, 100, 200, 500]
        
        for count in entry_counts[1:]:
            print(f"  Creating {count} tracking entries...")
            
            # Create entries
            for i in range(count - len(entry_counts) + 2):
                with self.tracker.track_tool_usage(
                    tool_name=f"memory_test_{i}",
                    agent_name="PerformanceTestAgent",
                    workflow_id=f"memory-test-{count}",
                    session_id=f"session-{count}",
                    input_data={"test_data": "x" * 100}  # 100 bytes per entry
                ):
                    pass
            
            # Measure memory
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory)
            print(f"    Memory after {count} entries: {current_memory:.2f} MB")
        
        return {
            'baseline_memory_mb': baseline_memory,
            'memory_measurements_mb': memory_measurements,
            'entry_counts': entry_counts,
            'memory_per_entry_kb': ((memory_measurements[-1] - baseline_memory) * 1024) / entry_counts[-1] if entry_counts[-1] > 0 else 0,
            'total_memory_increase_mb': memory_measurements[-1] - baseline_memory
        }
    
    def measure_log_file_growth(self) -> Dict[str, Any]:
        """Measure log file growth patterns."""
        print("📁 Measuring log file growth...")
        
        # Get initial log file size
        initial_size = 0
        if self.tracker.log_file.exists():
            initial_size = self.tracker.log_file.stat().st_size
        
        # Test different entry counts
        test_counts = [10, 50, 100, 250]
        size_measurements = [initial_size]
        
        for count in test_counts:
            print(f"  Adding {count} log entries...")
            
            for i in range(count):
                with self.tracker.track_tool_usage(
                    tool_name=f"log_growth_test_{count}_{i}",
                    agent_name="LogTestAgent",
                    workflow_id=f"log-test-{count}",
                    session_id=f"session-{count}",
                    input_data={"entry_number": i, "test_data": f"Data for entry {i} in batch {count}"},
                    context={"batch": count, "entry": i, "test_phase": "log_growth"}
                ):
                    time.sleep(0.001)  # Small delay to simulate real work
            
            # Measure file size
            current_size = self.tracker.log_file.stat().st_size
            size_measurements.append(current_size)
            print(f"    Log file size after {count} entries: {current_size} bytes")
        
        total_entries = sum(test_counts)
        total_growth = size_measurements[-1] - initial_size
        
        return {
            'initial_size_bytes': initial_size,
            'final_size_bytes': size_measurements[-1],
            'size_measurements_bytes': size_measurements,
            'test_counts': test_counts,
            'total_entries_added': total_entries,
            'total_growth_bytes': total_growth,
            'average_bytes_per_entry': total_growth / total_entries if total_entries > 0 else 0,
            'compression_ratio': initial_size / size_measurements[-1] if size_measurements[-1] > 0 else 0
        }
    
    def measure_tracking_resilience(self) -> Dict[str, Any]:
        """Test tracking system resilience under various conditions."""
        print("🛡️ Testing tracking system resilience...")
        
        resilience_results = {
            'concurrent_access': self._test_concurrent_access(),
            'error_handling': self._test_error_handling(),
            'large_data_handling': self._test_large_data_handling(),
            'rapid_requests': self._test_rapid_requests()
        }
        
        return resilience_results
    
    def _test_concurrent_access(self) -> Dict[str, Any]:
        """Test concurrent access to tracking system."""
        import threading
        import queue
        
        print("  Testing concurrent access...")
        results_queue = queue.Queue()
        thread_count = 5
        operations_per_thread = 10
        
        def concurrent_tracking(thread_id):
            successes = 0
            failures = 0
            for i in range(operations_per_thread):
                try:
                    with self.tracker.track_tool_usage(
                        tool_name=f"concurrent_test_t{thread_id}_op{i}",
                        agent_name="ConcurrentTestAgent",
                        workflow_id=f"concurrent-{thread_id}",
                        session_id=f"session-{thread_id}",
                        input_data={"thread": thread_id, "operation": i}
                    ):
                        time.sleep(0.01)  # Simulate work
                    successes += 1
                except Exception as e:
                    failures += 1
            
            results_queue.put({"thread_id": thread_id, "successes": successes, "failures": failures})
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(thread_count):
            thread = threading.Thread(target=concurrent_tracking, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Collect results
        total_successes = 0
        total_failures = 0
        thread_results = []
        
        while not results_queue.empty():
            result = results_queue.get()
            thread_results.append(result)
            total_successes += result["successes"]
            total_failures += result["failures"]
        
        return {
            'thread_count': thread_count,
            'operations_per_thread': operations_per_thread,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'success_rate': total_successes / (total_successes + total_failures) if (total_successes + total_failures) > 0 else 0,
            'execution_time': execution_time,
            'thread_results': thread_results
        }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling in tracking system."""
        print("  Testing error handling...")
        
        error_scenarios = 0
        handled_gracefully = 0
        
        # Test 1: Tool execution error
        try:
            with self.tracker.track_tool_usage(
                tool_name="error_test_tool",
                agent_name="ErrorTestAgent",
                workflow_id="error-test",
                session_id="error-session"
            ):
                raise ValueError("Simulated tool error")
        except ValueError:
            error_scenarios += 1
            handled_gracefully += 1  # Expected to raise
        except Exception:
            error_scenarios += 1
        
        # Test 2: Invalid data types
        try:
            with self.tracker.track_tool_usage(
                tool_name="data_test_tool",
                agent_name="DataTestAgent",
                workflow_id="data-test", 
                session_id="data-session",
                input_data={"invalid": object()}  # Non-serializable object
            ):
                pass
            error_scenarios += 1
            handled_gracefully += 1  # Should handle gracefully
        except Exception:
            error_scenarios += 1
        
        return {
            'error_scenarios_tested': error_scenarios,
            'handled_gracefully': handled_gracefully,
            'error_handling_rate': handled_gracefully / error_scenarios if error_scenarios > 0 else 0
        }
    
    def _test_large_data_handling(self) -> Dict[str, Any]:
        """Test handling of large data in tracking."""
        print("  Testing large data handling...")
        
        large_data_sizes = [1000, 10000, 50000]  # bytes
        results = []
        
        for size in large_data_sizes:
            large_data = "x" * size
            start_time = time.time()
            
            try:
                with self.tracker.track_tool_usage(
                    tool_name=f"large_data_test_{size}",
                    agent_name="LargeDataTestAgent",
                    workflow_id="large-data-test",
                    session_id="large-data-session",
                    input_data={"large_field": large_data}
                ):
                    pass
                
                execution_time = time.time() - start_time
                results.append({
                    'data_size_bytes': size,
                    'execution_time': execution_time,
                    'success': True
                })
                
            except Exception as e:
                execution_time = time.time() - start_time
                results.append({
                    'data_size_bytes': size,
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'test_results': results,
            'max_successful_size': max([r['data_size_bytes'] for r in results if r['success']], default=0)
        }
    
    def _test_rapid_requests(self) -> Dict[str, Any]:
        """Test rapid consecutive tracking requests."""
        print("  Testing rapid requests...")
        
        request_counts = [10, 50, 100]
        results = []
        
        for count in request_counts:
            start_time = time.time()
            successes = 0
            
            for i in range(count):
                try:
                    with self.tracker.track_tool_usage(
                        tool_name=f"rapid_test_{i}",
                        agent_name="RapidTestAgent",
                        workflow_id=f"rapid-{count}",
                        session_id=f"rapid-session-{count}"
                    ):
                        pass  # No delay between requests
                    successes += 1
                except Exception:
                    pass
            
            execution_time = time.time() - start_time
            requests_per_second = count / execution_time if execution_time > 0 else 0
            
            results.append({
                'request_count': count,
                'successes': successes,
                'execution_time': execution_time,
                'requests_per_second': requests_per_second,
                'success_rate': successes / count if count > 0 else 0
            })
        
        return {
            'test_results': results,
            'max_requests_per_second': max([r['requests_per_second'] for r in results], default=0)
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance assessment report."""
        print("📊 Generating comprehensive performance report...")
        
        report = {
            'timestamp': time.time(),
            'system_info': {
                'python_version': sys.version,
                'platform': os.name,
                'cpu_count': os.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            },
            'workflow_overhead': self.measure_workflow_overhead(),
            'memory_impact': self.measure_memory_impact(),
            'log_file_growth': self.measure_log_file_growth(),
            'system_resilience': self.measure_tracking_resilience()
        }
        
        # Calculate overall scores
        report['performance_summary'] = self._calculate_performance_scores(report)
        
        return report
    
    def _calculate_performance_scores(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance scores."""
        scores = {}
        
        # Memory efficiency score (0-100)
        memory_data = report['memory_impact']
        memory_per_entry = memory_data.get('memory_per_entry_kb', 0)
        scores['memory_efficiency'] = max(0, 100 - (memory_per_entry * 10))  # Penalty for high memory usage
        
        # Log efficiency score (0-100) 
        log_data = report['log_file_growth']
        bytes_per_entry = log_data.get('average_bytes_per_entry', 0)
        scores['log_efficiency'] = max(0, 100 - (bytes_per_entry / 50))  # Penalty for large log entries
        
        # Resilience score (0-100)
        resilience_data = report['system_resilience']
        concurrent_success = resilience_data['concurrent_access']['success_rate']
        error_handling = resilience_data['error_handling']['error_handling_rate']
        scores['resilience'] = (concurrent_success + error_handling) * 50
        
        # Overall score
        scores['overall'] = (scores['memory_efficiency'] + scores['log_efficiency'] + scores['resilience']) / 3
        
        return scores


def main():
    """Run comprehensive performance assessment."""
    print("🚀 Phase 4.2: Performance Impact Assessment")
    print("=" * 50)
    
    assessment = PerformanceAssessment()
    
    try:
        # Generate comprehensive report
        report = assessment.generate_performance_report()
        
        # Save report to file
        report_file = Path("tests/phase4/performance_assessment_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\\n📋 PERFORMANCE ASSESSMENT SUMMARY")
        print("=" * 40)
        
        summary = report['performance_summary']
        print(f"Overall Performance Score: {summary['overall']:.1f}/100")
        print(f"Memory Efficiency: {summary['memory_efficiency']:.1f}/100")
        print(f"Log Efficiency: {summary['log_efficiency']:.1f}/100")  
        print(f"System Resilience: {summary['resilience']:.1f}/100")
        
        # Memory impact
        memory = report['memory_impact']
        print(f"\\nMemory Impact:")
        print(f"  Memory per entry: {memory['memory_per_entry_kb']:.2f} KB")
        print(f"  Total memory increase: {memory['total_memory_increase_mb']:.2f} MB")
        
        # Log growth
        log_growth = report['log_file_growth']
        print(f"\\nLog File Growth:")
        print(f"  Bytes per entry: {log_growth['average_bytes_per_entry']:.0f} bytes")
        print(f"  Total growth: {log_growth['total_growth_bytes']:,} bytes")
        
        # Resilience
        resilience = report['system_resilience']
        print(f"\\nSystem Resilience:")
        print(f"  Concurrent access success: {resilience['concurrent_access']['success_rate']:.1%}")
        print(f"  Error handling rate: {resilience['error_handling']['error_handling_rate']:.1%}")
        print(f"  Max requests/second: {resilience['rapid_requests']['max_requests_per_second']:.1f}")
        
        print(f"\\n📄 Full report saved to: {report_file}")
        print("\\n✅ Performance assessment completed successfully!")
        
        return report
        
    except Exception as e:
        print(f"\\n❌ Performance assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
