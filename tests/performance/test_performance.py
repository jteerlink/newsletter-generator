"""Performance tests for the newsletter generator."""

import pytest
import time
import psutil
import os
from unittest.mock import Mock, patch
from src.agents.agents import ResearchAgent, WriterAgent, EditorAgent, Task, EnhancedCrew
from src.core.core import query_llm

class TestAgentPerformance:
    """Test agent performance metrics."""
    
    def test_agent_execution_time(self):
        """Test agent execution time performance."""
        agent = ResearchAgent()
        
        start_time = time.time()
        with patch('src.agents.agents.query_llm') as mock_llm:
            mock_llm.return_value = "Research completed"
            result = agent.execute_task("Test research task")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertion: agent should execute within reasonable time
        assert execution_time < 5.0, f"Agent execution took {execution_time:.2f}s, expected < 5.0s"
        assert "Research completed" in result
    
    def test_agent_memory_usage(self):
        """Test agent memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and execute multiple agents
        agents = []
        for i in range(5):
            agent = ResearchAgent()
            with patch('src.agents.agents.query_llm') as mock_llm:
                mock_llm.return_value = f"Research {i} completed"
                agent.execute_task(f"Test task {i}")
            agents.append(agent)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"
    
    def test_concurrent_agent_execution(self):
        """Test concurrent agent execution performance."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def execute_agent(agent_id):
            agent = ResearchAgent()
            with patch('src.agents.agents.query_llm') as mock_llm:
                mock_llm.return_value = f"Agent {agent_id} completed"
                result = agent.execute_task(f"Task {agent_id}")
                results.put((agent_id, result))
        
        # Execute multiple agents concurrently
        threads = []
        start_time = time.time()
        
        for i in range(3):
            thread = threading.Thread(target=execute_agent, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        concurrent_execution_time = end_time - start_time
        
        # Verify all results
        collected_results = []
        while not results.empty():
            collected_results.append(results.get())
        
        assert len(collected_results) == 3
        assert concurrent_execution_time < 10.0, f"Concurrent execution took {concurrent_execution_time:.2f}s"

class TestWorkflowPerformance:
    """Test complete workflow performance."""
    
    def test_newsletter_workflow_performance(self):
        """Test complete newsletter workflow performance."""
        # Create agents
        research_agent = ResearchAgent()
        writer_agent = WriterAgent()
        editor_agent = EditorAgent()
        
        # Create tasks
        research_task = Task("Research AI trends", research_agent)
        writing_task = Task("Write newsletter", writer_agent)
        editing_task = Task("Edit newsletter", editor_agent)
        
        # Create crew
        crew = EnhancedCrew(
            [research_agent, writer_agent, editor_agent],
            [research_task, writing_task, editing_task]
        )
        
        # Execute workflow and measure performance
        start_time = time.time()
        with patch('src.agents.agents.query_llm') as mock_llm:
            mock_llm.side_effect = [
                "Research findings",
                "Written content",
                "Edited content"
            ]
            result = crew.kickoff()
        end_time = time.time()
        
        workflow_time = end_time - start_time
        
        # Performance assertion: complete workflow should complete within reasonable time
        assert workflow_time < 15.0, f"Workflow took {workflow_time:.2f}s, expected < 15.0s"
        assert "Research findings" in result
        assert "Written content" in result
        assert "Edited content" in result
    
    def test_workflow_scalability(self):
        """Test workflow scalability with multiple iterations."""
        # Test how the system performs under repeated load
        execution_times = []
        
        for iteration in range(3):
            research_agent = ResearchAgent()
            writer_agent = WriterAgent()
            
            research_task = Task("Research task", research_agent)
            writing_task = Task("Writing task", writer_agent)
            
            crew = EnhancedCrew([research_agent, writer_agent], [research_task, writing_task])
            
            start_time = time.time()
            with patch('src.agents.agents.query_llm') as mock_llm:
                mock_llm.side_effect = ["Research done", "Writing done"]
                crew.kickoff()
            end_time = time.time()
            
            execution_times.append(end_time - start_time)
        
        # Check for performance degradation
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        # Performance should remain consistent (no more than 50% degradation)
        assert max_time < avg_time * 1.5, f"Performance degraded: max={max_time:.2f}s, avg={avg_time:.2f}s"

class TestLLMPerformance:
    """Test LLM query performance."""
    
    def test_llm_query_response_time(self):
        """Test LLM query response time."""
        start_time = time.time()
        with patch('src.core.core.ollama') as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
            result = query_llm("Test prompt")
        end_time = time.time()
        
        query_time = end_time - start_time
        
        # LLM queries should complete within reasonable time
        assert query_time < 3.0, f"LLM query took {query_time:.2f}s, expected < 3.0s"
        assert result == "Test response"
    
    def test_llm_query_throughput(self):
        """Test LLM query throughput."""
        query_times = []
        
        with patch('src.core.core.ollama') as mock_ollama:
            mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
            
            for i in range(5):
                start_time = time.time()
                query_llm(f"Test prompt {i}")
                end_time = time.time()
                query_times.append(end_time - start_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        
        # Average query time should be reasonable
        assert avg_query_time < 2.0, f"Average query time: {avg_query_time:.2f}s"

class TestMemoryPerformance:
    """Test memory usage patterns."""
    
    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform operations that should use memory
        agents = []
        for i in range(10):
            agent = ResearchAgent()
            with patch('src.agents.agents.query_llm') as mock_llm:
                mock_llm.return_value = f"Result {i}"
                agent.execute_task(f"Task {i}")
            agents.append(agent)
        
        # Clear references to trigger garbage collection
        del agents
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_difference = abs(final_memory - initial_memory)
        
        # Memory should be cleaned up (difference should be small)
        assert memory_difference < 50 * 1024 * 1024, f"Memory not cleaned up: {memory_difference / 1024 / 1024:.2f}MB difference"

class TestPerformanceBenchmarks:
    """Performance benchmarks for the system."""
    
    def test_newsletter_generation_benchmark(self):
        """Benchmark complete newsletter generation."""
        # This test establishes baseline performance metrics
        # for newsletter generation
        pass
    
    def test_agent_coordination_benchmark(self):
        """Benchmark agent coordination overhead."""
        # This test measures the overhead of agent coordination
        # and communication
        pass
    
    def test_tool_execution_benchmark(self):
        """Benchmark tool execution performance."""
        # This test measures the performance of various tools
        # used by agents
        pass

class TestLoadTesting:
    """Load testing for the system."""
    
    def test_concurrent_workflow_execution(self):
        """Test system behavior under concurrent workflow load."""
        # This test would simulate multiple users generating
        # newsletters simultaneously
        pass
    
    def test_large_content_processing(self):
        """Test system performance with large content volumes."""
        # This test would verify system performance when processing
        # large amounts of content
        pass
    
    def test_extended_operation_stability(self):
        """Test system stability during extended operations."""
        # This test would verify system stability during long-running
        # operations
        pass 