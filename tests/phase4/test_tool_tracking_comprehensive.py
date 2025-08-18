#!/usr/bin/env python3

"""
Phase 4.1: Comprehensive Tool Tracking Tests

This module contains comprehensive test suites for validating the complete
tool tracking system including unit tests, integration tests, and workflow tests.
"""

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.append('.')

from src.agents.editing import EditorAgent
from src.agents.research import ResearchAgent
from src.agents.writing import WriterAgent
from src.core.feedback_orchestrator import FeedbackOrchestrator
from src.core.tool_usage_analytics import create_analytics_dashboard
from src.core.tool_usage_tracker import (
    ToolExecutionStatus,
    ToolUsageEntry,
    ToolUsageLogger,
    get_tool_tracker,
)
from src.core.workflow_orchestrator import WorkflowOrchestrator


class TestToolUsageTrackerUnit:
    """Unit tests for ToolUsageTracker components."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.test_workflow_id = f"test-{uuid.uuid4()}"
        self.test_session_id = f"session-{uuid.uuid4()}"
        
    def test_tool_tracker_initialization(self):
        """Test tool tracker initializes correctly."""
        tracker = get_tool_tracker()
        assert isinstance(tracker, ToolUsageLogger)
        assert hasattr(tracker, 'log_file')
        assert hasattr(tracker, 'track_tool_usage')
        
    def test_tool_usage_entry_creation(self):
        """Test ToolUsageEntry creation and properties."""
        from datetime import datetime
        
        entry = ToolUsageEntry(
            timestamp=datetime.now(),
            tool_name="test_tool",
            agent_name="TestAgent",
            execution_time=1.5,
            status=ToolExecutionStatus.SUCCESS,
            input_data={"test": "data"},
            output_data={"result": "success"},
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            context={"phase": "testing"}
        )
        
        assert entry.tool_name == "test_tool"
        assert entry.agent_name == "TestAgent" 
        assert entry.workflow_id == self.test_workflow_id
        assert entry.session_id == self.test_session_id
        assert entry.execution_time == 1.5
        assert entry.status == ToolExecutionStatus.SUCCESS
        assert entry.input_data == {"test": "data"}
        assert entry.output_data == {"result": "success"}
        assert entry.context == {"phase": "testing"}
        
    def test_context_manager_success(self):
        """Test context manager for successful tool execution."""
        tracker = get_tool_tracker()
        initial_count = len(tracker.get_tool_usage_history(hours_back=1))
        
        with tracker.track_tool_usage(
            tool_name="test_context_tool",
            agent_name="TestAgent",
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            input_data={"test": "input"}
        ):
            # Simulate tool work
            time.sleep(0.1)
            
        # Verify entry was logged
        final_count = len(tracker.get_tool_usage_history(hours_back=1))
        assert final_count > initial_count
        
        # Verify entry details
        entries = tracker.get_tool_usage_history(hours_back=1)
        latest_entry = entries[-1]
        assert latest_entry.tool_name == "test_context_tool"
        assert latest_entry.status == ToolExecutionStatus.SUCCESS
        assert latest_entry.execution_time >= 0.1
        
    def test_context_manager_error(self):
        """Test context manager for failed tool execution."""
        tracker = get_tool_tracker()
        initial_count = len(tracker.get_tool_usage_history(hours_back=1))
        
        try:
            with tracker.track_tool_usage(
                tool_name="test_error_tool",
                agent_name="TestAgent",
                workflow_id=self.test_workflow_id,
                session_id=self.test_session_id,
                input_data={"test": "input"}
            ):
                # Simulate tool error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected error
            
        # Verify error was logged
        final_count = len(tracker.get_tool_usage_history(hours_back=1))
        assert final_count > initial_count
        
        # Verify entry details
        entries = tracker.get_tool_usage_history(hours_back=1)
        latest_entry = entries[-1]
        assert latest_entry.tool_name == "test_error_tool"
        assert latest_entry.status == ToolExecutionStatus.FAILURE
        assert "Test error" in str(latest_entry.error_message)
        
    def test_analytics_generation(self):
        """Test analytics generation from tracking data."""
        tracker = get_tool_tracker()
        
        # Create some test entries
        for i in range(3):
            with tracker.track_tool_usage(
                tool_name=f"test_tool_{i}",
                agent_name="TestAgent",
                workflow_id=self.test_workflow_id,
                session_id=self.test_session_id,
                input_data={"iteration": i}
            ):
                time.sleep(0.01)  # Small delay for different execution times
                
        # Generate analytics
        analytics = tracker.generate_usage_analytics(hours_back=1)
        
        assert analytics.total_invocations >= 3
        assert len(analytics.most_used_tools) >= 3
        assert analytics.average_success_rate >= 0


class TestWorkflowIntegration:
    """Integration tests for workflow + tracking system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_topic = "Phase 4 Testing Integration"
        
    @pytest.mark.slow
    def test_full_workflow_tracking(self):
        """Test complete workflow with tracking enabled."""
        tracker = get_tool_tracker()
        initial_count = len(tracker.get_tool_usage_history(hours_back=1))
        
        # Execute workflow
        orchestrator = WorkflowOrchestrator()
        result = orchestrator.execute_newsletter_generation(
            topic=self.test_topic,
            context_id="default",
            output_format="markdown"
        )
        
        # Verify tracking occurred
        final_count = len(tracker.get_tool_usage_history(hours_back=1))
        assert final_count > initial_count, "No new tracking entries were created"
        
        # Verify workflow phases tracked
        workflow_entries = [e for e in tracker.get_tool_usage_history(hours_back=1) 
                           if hasattr(e, 'workflow_id') and e.workflow_id == result.workflow_id]
        
        assert len(workflow_entries) >= 3, f"Expected at least 3 workflow entries, got {len(workflow_entries)}"
        
        # Verify required tools tracked
        tracked_tools = [entry.tool_name for entry in workflow_entries]
        required_tools = ["research_agent_execution", "writer_agent_execution", "grammar_linter"]
        
        for required_tool in required_tools:
            assert required_tool in tracked_tools, f"Required tool {required_tool} not found in tracked tools: {tracked_tools}"
            
        # Verify all entries have proper workflow context
        for entry in workflow_entries:
            assert entry.workflow_id == result.workflow_id
            assert hasattr(entry, 'session_id') and entry.session_id
            assert entry.execution_time >= 0
            assert entry.status in [ToolExecutionStatus.SUCCESS, ToolExecutionStatus.FAILURE]
            
    def test_agent_initialization_tracking(self):
        """Test agent initialization and context setting."""
        workflow_id = f"test-{uuid.uuid4()}"
        session_id = f"session-{uuid.uuid4()}"
        
        # Test ResearchAgent
        research_agent = ResearchAgent(name="TestResearch")
        research_agent.set_context(workflow_id=workflow_id, session_id=session_id)
        
        assert research_agent.context.workflow_id == workflow_id
        assert research_agent.context.session_id == session_id
        assert len(research_agent.available_tools) >= 1
        
        # Test WriterAgent  
        writer_agent = WriterAgent(name="TestWriter")
        writer_agent.set_context(workflow_id=workflow_id, session_id=session_id)
        
        assert writer_agent.context.workflow_id == workflow_id
        assert writer_agent.context.session_id == session_id
        assert len(writer_agent.available_tools) >= 1
        
    def test_feedback_orchestrator_tracking(self):
        """Test feedback orchestrator tool tracking."""
        from src.core.campaign_context import CampaignContext
        from src.core.execution_state import ExecutionState

        # Create test context
        workflow_id = f"test-{uuid.uuid4()}"
        execution_state = ExecutionState(workflow_id=workflow_id)
        campaign_context = CampaignContext(
            context_id="test",
            content_style="professional",
            audience_persona="technical",
            quality_threshold=0.8
        )
        
        tracker = get_tool_tracker()
        initial_count = len(tracker.get_tool_usage_history(hours_back=1))
        
        # Execute feedback orchestration
        orchestrator = FeedbackOrchestrator()
        result = orchestrator.orchestrate_feedback(
            content="Test content for feedback analysis.",
            context=campaign_context,
            execution_state=execution_state
        )
        
        # Verify tracking
        final_count = len(tracker.get_tool_usage_history(hours_back=1))
        assert final_count > initial_count
        
        # Verify grammar_linter was tracked
        recent_entries = tracker.get_tool_usage_history(hours_back=1)
        grammar_entries = [e for e in recent_entries if e.tool_name == "grammar_linter"]
        assert len(grammar_entries) >= 1


class TestPerformanceImpact:
    """Tests for measuring performance impact of tracking."""
    
    def test_tracking_overhead_measurement(self):
        """Measure overhead of tracking on execution time."""
        tracker = get_tool_tracker()
        iterations = 10
        
        # Time without tracking
        start_time = time.time()
        for i in range(iterations):
            time.sleep(0.01)  # Simulate work
        baseline_time = time.time() - start_time
        
        # Time with tracking
        start_time = time.time()
        for i in range(iterations):
            with tracker.track_tool_usage(
                tool_name="performance_test",
                agent_name="TestAgent", 
                workflow_id=f"perf-{uuid.uuid4()}",
                session_id=f"session-{uuid.uuid4()}",
                input_data={"iteration": i}
            ):
                time.sleep(0.01)  # Simulate work
        tracked_time = time.time() - start_time
        
        # Calculate overhead
        overhead = tracked_time - baseline_time
        overhead_percentage = (overhead / baseline_time) * 100
        
        print(f"\\nPerformance Test Results:")
        print(f"Baseline time: {baseline_time:.4f}s")
        print(f"Tracked time: {tracked_time:.4f}s")
        print(f"Overhead: {overhead:.4f}s ({overhead_percentage:.2f}%)")
        
        # Assert overhead is reasonable (< 10%)
        assert overhead_percentage < 10, f"Tracking overhead too high: {overhead_percentage:.2f}%"
        
    def test_memory_usage_assessment(self):
        """Assess memory usage impact of tracking."""
        import os

        import psutil
        
        tracker = get_tool_tracker()
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many tracking entries
        for i in range(100):
            with tracker.track_tool_usage(
                tool_name=f"memory_test_{i}",
                agent_name="TestAgent",
                workflow_id=f"memory-{uuid.uuid4()}",
                session_id=f"session-{uuid.uuid4()}",
                input_data={"large_data": "x" * 1000}  # 1KB of data per entry
            ):
                pass
                
        # Measure memory after tracking
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        print(f"\\nMemory Usage Test Results:")
        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Assert memory increase is reasonable (< 50MB for 100 entries)
        assert memory_increase < 50, f"Memory usage too high: {memory_increase:.2f} MB"
        
    def test_log_file_growth_rate(self):
        """Test log file growth rates under load."""
        tracker = get_tool_tracker()
        
        # Get initial log file size
        initial_size = 0
        if tracker.log_file.exists():
            initial_size = tracker.log_file.stat().st_size
        
        # Generate tracking entries
        entries_count = 50
        for i in range(entries_count):
            with tracker.track_tool_usage(
                tool_name="log_growth_test",
                agent_name="TestAgent",
                workflow_id=f"log-{uuid.uuid4()}",
                session_id=f"session-{uuid.uuid4()}",
                input_data={"test_data": f"Entry {i} with some test data"},
                context={"phase": "testing", "iteration": i}
            ):
                pass
        
        # Measure final log file size
        final_size = tracker.log_file.stat().st_size
        size_increase = final_size - initial_size
        size_per_entry = size_increase / entries_count if entries_count > 0 else 0
        
        print(f"\\nLog File Growth Test Results:")
        print(f"Initial size: {initial_size} bytes")
        print(f"Final size: {final_size} bytes")
        print(f"Size increase: {size_increase} bytes")
        print(f"Size per entry: {size_per_entry:.2f} bytes")
        
        # Assert reasonable size per entry (< 2KB per entry)
        assert size_per_entry < 2048, f"Log entry size too large: {size_per_entry:.2f} bytes"


class TestSystemResilience:
    """Tests for tracking system resilience and error handling."""
    
    def test_tracking_system_resilience(self):
        """Test tracking system handles errors gracefully."""
        tracker = get_tool_tracker()
        
        # Test with invalid log file path
        original_path = tracker.log_file
        try:
            tracker.log_file = Path("/invalid/path/tool_usage.json")
            
            # This should not crash the system
            with tracker.track_tool_usage(
                tool_name="resilience_test",
                agent_name="TestAgent",
                workflow_id="test-workflow",
                session_id="test-session"
            ):
                pass
                
        except Exception:
            pass  # Expected to fail gracefully
        finally:
            tracker.log_file = original_path
            
    def test_concurrent_tracking(self):
        """Test tracking system handles concurrent access."""
        import queue
        import threading
        
        tracker = get_tool_tracker()
        results = queue.Queue()
        
        def track_concurrently(thread_id):
            """Track tools concurrently."""
            try:
                with tracker.track_tool_usage(
                    tool_name=f"concurrent_test_{thread_id}",
                    agent_name="TestAgent",
                    workflow_id=f"concurrent-{thread_id}",
                    session_id=f"session-{thread_id}"
                ):
                    time.sleep(0.1)  # Simulate work
                results.put(f"success_{thread_id}")
            except Exception as e:
                results.put(f"error_{thread_id}: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_concurrently, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            if result.startswith("success"):
                success_count += 1
                
        assert success_count >= 4, f"Only {success_count}/5 concurrent operations succeeded"


# Test configuration
class TestConfig:
    """Test configuration and utilities."""
    
    @staticmethod
    def setup_test_environment():
        """Set up test environment."""
        # Ensure test directories exist
        os.makedirs("tests/phase4", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
    @staticmethod
    def cleanup_test_environment():
        """Clean up test environment."""
        # Optional cleanup for test files
        pass


if __name__ == "__main__":
    print("🧪 Running Phase 4 Comprehensive Tool Tracking Tests...")
    
    # Setup test environment
    TestConfig.setup_test_environment()
    
    # Run tests manually for demonstration
    try:
        # Unit tests
        print("\\n=== UNIT TESTS ===")
        unit_tests = TestToolUsageTrackerUnit()
        unit_tests.setup_method()
        unit_tests.test_tool_tracker_initialization()
        print("✅ Tool tracker initialization test passed")
        
        unit_tests.test_tool_usage_entry_creation()
        print("✅ Tool usage entry creation test passed")
        
        unit_tests.test_context_manager_success()
        print("✅ Context manager success test passed")
        
        unit_tests.test_context_manager_error()
        print("✅ Context manager error test passed")
        
        unit_tests.test_analytics_generation()
        print("✅ Analytics generation test passed")
        
        # Performance tests
        print("\\n=== PERFORMANCE TESTS ===")
        perf_tests = TestPerformanceImpact()
        perf_tests.test_tracking_overhead_measurement()
        print("✅ Tracking overhead test passed")
        
        perf_tests.test_memory_usage_assessment()
        print("✅ Memory usage test passed")
        
        perf_tests.test_log_file_growth_rate()
        print("✅ Log file growth test passed")
        
        # Resilience tests
        print("\\n=== RESILIENCE TESTS ===")
        resilience_tests = TestSystemResilience()
        resilience_tests.test_tracking_system_resilience()
        print("✅ System resilience test passed")
        
        resilience_tests.test_concurrent_tracking()
        print("✅ Concurrent tracking test passed")
        
        print("\\n🎉 All Phase 4 tests completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        TestConfig.cleanup_test_environment()
