"""
Comprehensive Test Suite for Tool Usage Tracking System

Tests the enhanced tool usage tracking functionality including:
- ToolUsageTracker class functionality
- Agent integration
- MCP orchestrator tracking
- Feedback system integration
- Analytics dashboard
- Performance and reliability
"""

import unittest
import asyncio
import tempfile
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from core.tool_usage_tracker import (
    ToolUsageTracker, ToolUsageEntry, ToolExecutionStatus, 
    track_tool_call, get_tool_tracker
)
from core.feedback_system import FeedbackLearningSystem, FeedbackLogger
from core.tool_usage_analytics import ToolUsageAnalyticsDashboard
from agents.agents import SimpleAgent
from tools.tools import AVAILABLE_TOOLS

class TestToolUsageTracker(unittest.TestCase):
    """Test the core ToolUsageTracker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_tool_usage.jsonl")
        self.tracker = ToolUsageTracker(log_file=self.log_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertIsInstance(self.tracker, ToolUsageTracker)
        self.assertTrue(os.path.exists(self.log_file))
        self.assertEqual(len(self.tracker.get_tool_usage_history()), 0)
    
    def test_log_tool_usage(self):
        """Test logging tool usage."""
        entry = self.tracker.log_tool_usage(
            tool_name="test_tool",
            agent_name="test_agent",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=1.5,
            input_data={"query": "test"},
            output_data={"result": "success"},
            session_id="test_session"
        )
        
        self.assertIsInstance(entry, ToolUsageEntry)
        self.assertEqual(entry.tool_name, "test_tool")
        self.assertEqual(entry.agent_name, "test_agent")
        self.assertEqual(entry.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(entry.execution_time, 1.5)
        
        # Verify it's stored
        history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].tool_name, "test_tool")
    
    def test_get_tool_usage_history_filtering(self):
        """Test filtering tool usage history."""
        # Add multiple entries
        base_time = datetime.now()
        
        for i in range(5):
            self.tracker.log_tool_usage(
                tool_name=f"tool_{i}",
                agent_name=f"agent_{i}",
                status=ToolExecutionStatus.SUCCESS,
                execution_time=float(i),
                session_id=f"session_{i}"
            )
        
        # Test filtering by hours
        all_history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(all_history), 5)
        
        recent_history = self.tracker.get_tool_usage_history(hours_back=0.001)  # Very recent
        self.assertEqual(len(recent_history), 5)  # All should be recent
        
        # Test filtering by tool name
        tool_0_history = self.tracker.get_tool_usage_history(tool_name="tool_0")
        self.assertEqual(len(tool_0_history), 1)
        self.assertEqual(tool_0_history[0].tool_name, "tool_0")
        
        # Test filtering by agent name
        agent_1_history = self.tracker.get_tool_usage_history(agent_name="agent_1")
        self.assertEqual(len(agent_1_history), 1)
        self.assertEqual(agent_1_history[0].agent_name, "agent_1")
    
    def test_generate_usage_analytics(self):
        """Test analytics generation."""
        # Add test data
        for i in range(10):
            status = ToolExecutionStatus.SUCCESS if i % 3 != 0 else ToolExecutionStatus.FAILURE
            self.tracker.log_tool_usage(
                tool_name=f"tool_{i % 3}",  # 3 different tools
                agent_name=f"agent_{i % 2}",  # 2 different agents
                status=status,
                execution_time=float(i + 1),
                session_id=f"session_{i}"
            )
        
        analytics = self.tracker.generate_usage_analytics()
        
        # Verify analytics structure
        self.assertIn("total_tool_calls", analytics)
        self.assertIn("overall_success_rate", analytics)
        self.assertIn("average_execution_time", analytics)
        self.assertIn("tool_breakdown", analytics)
        self.assertIn("agent_breakdown", analytics)
        self.assertIn("most_used_tools", analytics)
        self.assertIn("fastest_tools", analytics)
        self.assertIn("most_reliable_tools", analytics)
        
        # Verify calculated values
        self.assertEqual(analytics["total_tool_calls"], 10)
        self.assertGreater(analytics["overall_success_rate"], 0.5)  # Should be around 0.7
        self.assertGreater(analytics["average_execution_time"], 0)
    
    def test_track_tool_call_decorator(self):
        """Test the tool call tracking decorator."""
        
        @track_tool_call(tool_name="decorated_tool", agent_name="test_agent")
        def sample_tool_function(x, y):
            time.sleep(0.1)  # Simulate work
            return x + y
        
        result = sample_tool_function(2, 3)
        self.assertEqual(result, 5)
        
        # Check that it was logged
        history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].tool_name, "decorated_tool")
        self.assertEqual(history[0].agent_name, "test_agent")
        self.assertEqual(history[0].status, ToolExecutionStatus.SUCCESS)
        self.assertGreater(history[0].execution_time, 0.1)
        self.assertIn("x", history[0].input_data)
        self.assertIn("y", history[0].input_data)
        self.assertEqual(history[0].output_data, 5)
    
    def test_track_tool_call_decorator_with_error(self):
        """Test decorator handling errors."""
        
        @track_tool_call(tool_name="error_tool", agent_name="test_agent")
        def error_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            error_function()
        
        # Check that error was logged
        history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].tool_name, "error_tool")
        self.assertEqual(history[0].status, ToolExecutionStatus.FAILURE)
        self.assertIn("error_type", history[0].error_details)
        self.assertIn("error_message", history[0].error_details)
        self.assertEqual(history[0].error_details["error_message"], "Test error")

class TestAgentIntegration(unittest.TestCase):
    """Test integration with agent classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_agent_usage.jsonl")
        
        # Mock query_llm to avoid external dependencies
        self.mock_query_llm_patcher = patch('agents.agents.query_llm')
        self.mock_query_llm = self.mock_query_llm_patcher.start()
        self.mock_query_llm.return_value = "Mocked LLM response"
        
        # Create agent with tracking
        self.agent = SimpleAgent(
            name="test_agent",
            role="researcher",
            goal="test research",
            backstory="test backstory",
            tools=["search_web"]
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_query_llm_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_agent_tool_execution_tracking(self):
        """Test that agent tool executions are tracked."""
        # Mock the web search tool
        with patch.dict(AVAILABLE_TOOLS, {"search_web": Mock(return_value="Search results")}):
            # Execute a task that should use tools
            result = self.agent.execute_task("Search for AI news")
            
            # Verify tracking occurred
            tracker = get_tool_tracker()
            history = tracker.get_tool_usage_history(hours_back=1)
            
            # Should have tracked the tool usage
            tool_entries = [entry for entry in history if entry.agent_name == "test_agent"]
            self.assertGreater(len(tool_entries), 0)
            
            # Verify tool name is tracked correctly
            search_entries = [entry for entry in tool_entries if "search_web" in entry.tool_name]
            self.assertGreater(len(search_entries), 0)

class TestFeedbackIntegration(unittest.TestCase):
    """Test integration with feedback system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_file = os.path.join(self.temp_dir, "test_feedback.json")
        self.tool_log_file = os.path.join(self.temp_dir, "test_tool_usage.jsonl")
        
        self.feedback_system = FeedbackLearningSystem(feedback_file=self.feedback_file)
        self.tool_tracker = ToolUsageTracker(log_file=self.tool_log_file)
        
        # Connect them
        self.feedback_system.set_tool_tracker(self.tool_tracker)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_tool_usage_correlation_analysis(self):
        """Test correlation analysis between tool usage and feedback quality."""
        # Create test data - tool usage entries
        session_ids = ["session_1", "session_2", "session_3"]
        
        for i, session_id in enumerate(session_ids):
            # Simulate different tool usage patterns
            for j in range(i + 1):  # Different usage amounts
                self.tool_tracker.log_tool_usage(
                    tool_name=f"tool_{j}",
                    agent_name="test_agent",
                    status=ToolExecutionStatus.SUCCESS,
                    execution_time=1.0,
                    session_id=session_id
                )
        
        # Create corresponding feedback entries
        for i, session_id in enumerate(session_ids):
            quality_score = 7.0 + i  # Higher scores for more tool usage
            self.feedback_system.logger.log_feedback(
                topic=f"Topic {i}",
                content=f"Content {i}",
                user_rating="approved",
                quality_scores={"clarity": quality_score, "accuracy": quality_score},
                agent_performance={"test_agent": {"status": "good"}},
                suggestions=[]
            )
            
            # Update metadata to include session_id
            feedback_history = self.feedback_system.logger.get_feedback_history()
            if feedback_history:
                feedback_history[-1].metadata["session_id"] = session_id
                # Save back the updated feedback
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                data["feedback_entries"][-1]["metadata"]["session_id"] = session_id
                with open(self.feedback_file, 'w') as f:
                    json.dump(data, f, indent=2)
        
        # Test correlation analysis
        correlations = self.feedback_system.analyze_tool_usage_correlations()
        
        self.assertIn("tool_correlations", correlations)
        self.assertIn("analysis_summary", correlations)
        self.assertIn("analysis_metadata", correlations)
        
        # Should detect positive correlation for tools used more
        if correlations["tool_correlations"]:
            tool_data = list(correlations["tool_correlations"].values())[0]
            self.assertIn("average_quality_with_tool", tool_data)
            self.assertIn("quality_difference", tool_data)
            self.assertIn("recommendation", tool_data)

class TestAnalyticsDashboard(unittest.TestCase):
    """Test the analytics dashboard functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool_log_file = os.path.join(self.temp_dir, "test_tool_usage.jsonl")
        self.feedback_file = os.path.join(self.temp_dir, "test_feedback.json")
        
        self.tool_tracker = ToolUsageTracker(log_file=self.tool_log_file)
        self.feedback_system = FeedbackLearningSystem(feedback_file=self.feedback_file)
        self.feedback_system.set_tool_tracker(self.tool_tracker)
        
        self.dashboard = ToolUsageAnalyticsDashboard(
            tool_tracker=self.tool_tracker,
            feedback_system=self.feedback_system
        )
        
        # Add sample data
        self._create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample data for testing."""
        tools = ["search_web", "search_knowledge_base", "analyze_content"]
        agents = ["researcher", "writer", "editor"]
        
        for i in range(20):
            tool = tools[i % len(tools)]
            agent = agents[i % len(agents)]
            status = ToolExecutionStatus.SUCCESS if i % 4 != 0 else ToolExecutionStatus.FAILURE
            
            self.tool_tracker.log_tool_usage(
                tool_name=tool,
                agent_name=agent,
                status=status,
                execution_time=float(1 + i % 5),
                session_id=f"session_{i // 5}"
            )
    
    def test_dashboard_data_refresh(self):
        """Test dashboard data refresh."""
        dashboard_data = self.dashboard.refresh_dashboard_data(hours_back=24)
        
        self.assertIn("last_refresh", dashboard_data)
        self.assertIn("system_analytics", dashboard_data)
        self.assertIn("agent_analytics", dashboard_data)
        self.assertIn("performance_insights", dashboard_data)
        self.assertIn("summary", dashboard_data)
        
        # Verify system analytics
        system_analytics = dashboard_data["system_analytics"]
        self.assertIn("total_tool_calls", system_analytics)
        self.assertIn("overall_success_rate", system_analytics)
        self.assertIn("tool_breakdown", system_analytics)
        
        # Verify summary
        summary = dashboard_data["summary"]
        self.assertIn("health_status", summary)
        self.assertIn("key_metrics", summary)
    
    def test_real_time_metrics(self):
        """Test real-time metrics generation."""
        metrics = self.dashboard.get_real_time_metrics()
        
        self.assertIn("timestamp", metrics)
        self.assertIn("last_hour", metrics)
        self.assertIn("active_sessions", metrics)
        self.assertIn("system_status", metrics)
        
        # Verify last hour metrics
        last_hour = metrics["last_hour"]
        self.assertIn("total_tool_calls", last_hour)
        self.assertIn("success_rate", last_hour)
        self.assertIn("average_execution_time", last_hour)
    
    def test_tool_performance_report(self):
        """Test individual tool performance report."""
        report = self.dashboard.get_tool_performance_report("search_web")
        
        self.assertIn("tool_name", report)
        self.assertIn("usage_statistics", report)
        self.assertIn("usage_patterns", report)
        self.assertIn("error_analysis", report)
        self.assertIn("recommendations", report)
        
        # Verify usage statistics
        usage_stats = report["usage_statistics"]
        self.assertIn("total_calls", usage_stats)
        self.assertIn("success_rate", usage_stats)
        self.assertIn("average_execution_time", usage_stats)
    
    def test_agent_performance_comparison(self):
        """Test agent performance comparison."""
        comparison = self.dashboard.get_agent_performance_comparison()
        
        self.assertIn("agent_comparison", comparison)
        self.assertIn("rankings", comparison)
        self.assertIn("insights", comparison)
        
        # Verify rankings
        rankings = comparison["rankings"]
        self.assertIn("by_efficiency", rankings)
        self.assertIn("by_success_rate", rankings)
        
        # Should have data for our test agents
        agent_comparison = comparison["agent_comparison"]
        self.assertIn("researcher", agent_comparison)
        self.assertIn("writer", agent_comparison)
        self.assertIn("editor", agent_comparison)
    
    def test_executive_summary(self):
        """Test executive summary generation."""
        summary = self.dashboard.generate_executive_summary()
        
        self.assertIn("report_period", summary)
        self.assertIn("key_metrics", summary)
        self.assertIn("performance_trend", summary)
        self.assertIn("top_insights", summary)
        self.assertIn("recommendations", summary)
        
        # Verify key metrics
        key_metrics = summary["key_metrics"]
        self.assertIn("total_tool_calls", key_metrics)
        self.assertIn("system_success_rate", key_metrics)
        self.assertIn("average_response_time", key_metrics)
    
    def test_export_analytics_data(self):
        """Test analytics data export."""
        # Test JSON export
        json_file = self.dashboard.export_analytics_data(format_type="json", output_dir=self.temp_dir)
        self.assertTrue(os.path.exists(json_file))
        
        with open(json_file, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("system_analytics", exported_data)
        self.assertIn("summary", exported_data)

class TestPerformanceAndReliability(unittest.TestCase):
    """Test performance and reliability of the tracking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "perf_test_tool_usage.jsonl")
        self.tracker = ToolUsageTracker(log_file=self.log_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_high_volume_logging(self):
        """Test performance with high volume of log entries."""
        start_time = time.time()
        
        # Log 1000 entries
        for i in range(1000):
            self.tracker.log_tool_usage(
                tool_name=f"tool_{i % 10}",
                agent_name=f"agent_{i % 5}",
                status=ToolExecutionStatus.SUCCESS,
                execution_time=float(i % 10),
                session_id=f"session_{i % 50}"
            )
        
        logging_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(logging_time, 10.0)
        
        # Verify all entries were logged
        history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(history), 1000)
        
        # Test analytics generation performance
        start_time = time.time()
        analytics = self.tracker.generate_usage_analytics()
        analytics_time = time.time() - start_time
        
        # Analytics should be fast even with large dataset
        self.assertLess(analytics_time, 5.0)
        self.assertEqual(analytics["total_tool_calls"], 1000)
    
    def test_concurrent_logging(self):
        """Test thread safety of logging."""
        import threading
        import random
        
        def log_entries(thread_id, count):
            for i in range(count):
                self.tracker.log_tool_usage(
                    tool_name=f"tool_{thread_id}_{i}",
                    agent_name=f"agent_{thread_id}",
                    status=ToolExecutionStatus.SUCCESS,
                    execution_time=random.uniform(0.1, 2.0),
                    session_id=f"session_{thread_id}_{i}"
                )
        
        # Create 10 threads each logging 100 entries
        threads = []
        for thread_id in range(10):
            thread = threading.Thread(target=log_entries, args=(thread_id, 100))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(concurrent_time, 15.0)
        
        # Verify all entries were logged
        history = self.tracker.get_tool_usage_history()
        self.assertEqual(len(history), 1000)  # 10 threads * 100 entries each
    
    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add large number of entries
        for i in range(5000):
            self.tracker.log_tool_usage(
                tool_name=f"tool_{i % 20}",
                agent_name=f"agent_{i % 10}",
                status=ToolExecutionStatus.SUCCESS,
                execution_time=float(i % 10),
                input_data={"large_data": "x" * 1000},  # Add some bulk
                session_id=f"session_{i % 100}"
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 5000 entries)
        self.assertLess(memory_increase, 100)
        
        # Test analytics generation doesn't consume excessive memory
        analytics_start_memory = process.memory_info().rss / 1024 / 1024
        analytics = self.tracker.generate_usage_analytics()
        analytics_end_memory = process.memory_info().rss / 1024 / 1024
        
        analytics_memory_increase = analytics_end_memory - analytics_start_memory
        self.assertLess(analytics_memory_increase, 50)  # Should be less than 50MB
        
        # Verify analytics are correct
        self.assertEqual(analytics["total_tool_calls"], 5000)

class TestIntegrationScenarios(unittest.TestCase):
    """Test end-to-end integration scenarios."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tool_log_file = os.path.join(self.temp_dir, "integration_tool_usage.jsonl")
        self.feedback_file = os.path.join(self.temp_dir, "integration_feedback.json")
        
        # Set up complete system
        self.tool_tracker = ToolUsageTracker(log_file=self.tool_log_file)
        self.feedback_system = FeedbackLearningSystem(feedback_file=self.feedback_file)
        self.feedback_system.set_tool_tracker(self.tool_tracker)
        self.dashboard = ToolUsageAnalyticsDashboard(
            tool_tracker=self.tool_tracker,
            feedback_system=self.feedback_system
        )
        
        # Mock external dependencies
        self.mock_query_llm_patcher = patch('agents.agents.query_llm')
        self.mock_query_llm = self.mock_query_llm_patcher.start()
        self.mock_query_llm.return_value = "Mocked LLM response"
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.mock_query_llm_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_newsletter_workflow_tracking(self):
        """Test tracking through a complete newsletter generation workflow."""
        # Simulate a complete workflow
        session_id = "workflow_test_session"
        
        # 1. Research phase
        self.tool_tracker.log_tool_usage(
            tool_name="search_web",
            agent_name="researcher",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=2.5,
            input_data={"query": "AI news"},
            output_data={"articles": ["article1", "article2"]},
            session_id=session_id,
            workflow_id="newsletter_generation"
        )
        
        # 2. Knowledge base search
        self.tool_tracker.log_tool_usage(
            tool_name="search_knowledge_base",
            agent_name="researcher",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=1.8,
            input_data={"query": "AI trends"},
            output_data={"docs": ["doc1", "doc2"]},
            session_id=session_id,
            workflow_id="newsletter_generation"
        )
        
        # 3. Content generation
        self.tool_tracker.log_tool_usage(
            tool_name="generate_content",
            agent_name="writer",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=5.2,
            input_data={"articles": ["article1", "article2"]},
            output_data={"newsletter": "Generated newsletter content"},
            session_id=session_id,
            workflow_id="newsletter_generation"
        )
        
        # 4. Quality check
        self.tool_tracker.log_tool_usage(
            tool_name="quality_check",
            agent_name="editor",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=1.2,
            input_data={"content": "Generated newsletter content"},
            output_data={"quality_score": 8.5},
            session_id=session_id,
            workflow_id="newsletter_generation"
        )
        
        # 5. Add feedback
        self.feedback_system.logger.log_feedback(
            topic="AI News Newsletter",
            content="Generated newsletter content with high quality",
            user_rating="approved",
            quality_scores={"clarity": 8.5, "accuracy": 9.0, "engagement": 8.0},
            agent_performance={
                "researcher": {"status": "excellent", "tools_used": 2},
                "writer": {"status": "good", "efficiency": "high"},
                "editor": {"status": "excellent", "quality_improvement": 0.5}
            },
            suggestions=["Great work overall"]
        )
        
        # Update feedback metadata with session ID
        feedback_history = self.feedback_system.logger.get_feedback_history()
        if feedback_history:
            feedback_history[-1].metadata["session_id"] = session_id
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            data["feedback_entries"][-1]["metadata"]["session_id"] = session_id
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Test analytics across the workflow
        dashboard_data = self.dashboard.refresh_dashboard_data(hours_back=1)
        
        # Verify workflow tracking
        workflow_analytics = dashboard_data["workflow_analytics"]
        self.assertIn("newsletter_generation", workflow_analytics["workflow_breakdown"])
        
        workflow_data = workflow_analytics["workflow_breakdown"]["newsletter_generation"]
        self.assertEqual(workflow_data["total_tool_calls"], 4)
        self.assertEqual(workflow_data["unique_tools"], 4)
        
        # Verify agent analytics
        agent_analytics = dashboard_data["agent_analytics"]["agent_breakdown"]
        self.assertIn("researcher", agent_analytics)
        self.assertIn("writer", agent_analytics)
        self.assertIn("editor", agent_analytics)
        
        # Researcher should have 2 calls
        self.assertEqual(agent_analytics["researcher"]["total_calls"], 2)
        
        # Test correlation analysis
        correlations = self.feedback_system.analyze_tool_usage_correlations()
        self.assertIn("tool_correlations", correlations)
        
        # Generate executive summary
        summary = self.dashboard.generate_executive_summary()
        self.assertIn("key_metrics", summary)
        self.assertGreater(summary["key_metrics"]["total_tool_calls"], 0)
    
    def test_error_handling_and_recovery(self):
        """Test system behavior during errors and recovery."""
        session_id = "error_test_session"
        
        # Simulate normal operation
        self.tool_tracker.log_tool_usage(
            tool_name="search_web",
            agent_name="researcher",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=1.5,
            session_id=session_id
        )
        
        # Simulate tool failure
        self.tool_tracker.log_tool_usage(
            tool_name="search_web",
            agent_name="researcher",
            status=ToolExecutionStatus.FAILURE,
            execution_time=0.5,
            error_details={
                "error_type": "TimeoutError",
                "error_message": "Connection timeout",
                "retry_count": 2
            },
            session_id=session_id
        )
        
        # Simulate recovery with alternative tool
        self.tool_tracker.log_tool_usage(
            tool_name="search_knowledge_base",
            agent_name="researcher",
            status=ToolExecutionStatus.SUCCESS,
            execution_time=2.0,
            session_id=session_id
        )
        
        # Verify analytics handle errors correctly
        analytics = self.tool_tracker.generate_usage_analytics()
        
        # Overall success rate should reflect the failure
        self.assertLess(analytics["overall_success_rate"], 1.0)
        self.assertEqual(analytics["total_tool_calls"], 3)
        
        # Tool-specific analytics should show the failure
        tool_breakdown = analytics["tool_breakdown"]
        search_web_stats = tool_breakdown["search_web"]
        self.assertEqual(search_web_stats["total_calls"], 2)
        self.assertEqual(search_web_stats["success_rate"], 0.5)  # 1 success, 1 failure
        
        # Knowledge base should show 100% success
        kb_stats = tool_breakdown["search_knowledge_base"]
        self.assertEqual(kb_stats["success_rate"], 1.0)
        
        # Dashboard should reflect system status appropriately
        dashboard_data = self.dashboard.refresh_dashboard_data(hours_back=1)
        summary = dashboard_data["summary"]
        
        # Should detect the reliability issue
        if summary["alerts"]:
            self.assertTrue(any("success" in alert.lower() for alert in summary["alerts"]))

def run_all_tests():
    """Run all test suites."""
    print("Running Enhanced Tool Usage Tracking Test Suite...")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        TestToolUsageTracker,
        TestAgentIntegration,
        TestFeedbackIntegration,
        TestAnalyticsDashboard,
        TestPerformanceAndReliability,
        TestIntegrationScenarios
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_suite in test_suites:
        print(f"\nRunning {test_suite.__name__}...")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"FAILURES in {test_suite.__name__}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print(f"ERRORS in {test_suite.__name__}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! Tool usage tracking system is ready for production.")
    else:
        print(f"\n⚠️  {total_failures + total_errors} tests failed. Please review and fix issues.")
    
    return total_failures + total_errors == 0

if __name__ == "__main__":
    # Set up test environment
    os.environ["TESTING"] = "true"
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 