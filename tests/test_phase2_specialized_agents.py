"""
Comprehensive Tests for Phase 2 Specialized Agents

Tests the specialized agents implemented in Phase 2:
- TechnicalAccuracyAgent
- ReadabilityAgent  
- ContinuityManagerAgent
- AgentCoordinator
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.technical_accuracy_agent import TechnicalAccuracyAgent, TechnicalClaim, CodeBlock
from src.agents.readability_agent import ReadabilityAgent, ReadabilityMetrics
from src.agents.continuity_manager_agent import ContinuityManagerAgent
from src.core.agent_coordinator import AgentCoordinator, AgentExecutionSpec
from src.agents.base_agent import (
    ProcessingContext, ProcessingResult, ProcessingMode, 
    AgentConfiguration, AgentStatus
)


class TestTechnicalAccuracyAgent:
    """Test Technical Accuracy Agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = TechnicalAccuracyAgent()
        assert agent.name == "TechnicalAccuracy"
        assert agent.status == AgentStatus.HEALTHY
        assert hasattr(agent, 'validation_rules')
        assert hasattr(agent, 'technical_terms')
    
    def test_code_block_extraction(self):
        """Test code block extraction from content."""
        agent = TechnicalAccuracyAgent()
        content = """
        Here's some Python code:
        
        ```python
        def hello():
            print("Hello World")
        ```
        
        And some inline code: `print("test")`
        """
        
        blocks = agent._extract_code_blocks(content)
        assert len(blocks) >= 1
        assert any('def hello' in block.content for block in blocks)
        assert any(block.language == 'python' for block in blocks)
    
    def test_python_code_validation(self):
        """Test Python code syntax validation."""
        agent = TechnicalAccuracyAgent()
        
        # Valid Python code
        valid_errors = agent._validate_python_code("def test(): pass")
        assert len(valid_errors) == 0
        
        # Invalid Python code
        invalid_errors = agent._validate_python_code("def test( invalid syntax")
        assert len(invalid_errors) > 0
        assert "syntax error" in invalid_errors[0].lower()
    
    def test_technical_claims_extraction(self):
        """Test extraction of technical claims."""
        agent = TechnicalAccuracyAgent()
        content = """
        Machine learning can improve accuracy by up to 95%.
        Research shows that neural networks are effective.
        According to studies, this algorithm achieves better performance.
        """
        
        claims = agent._extract_technical_claims(content, 'analysis')
        assert len(claims) >= 2
        assert any('95%' in claim.claim for claim in claims)
        assert any('research shows' in claim.claim.lower() for claim in claims)
    
    def test_claim_plausibility_assessment(self):
        """Test plausibility assessment of technical claims."""
        agent = TechnicalAccuracyAgent()
        
        # Plausible claim
        plausible_score = agent._assess_claim_plausibility("Research shows that this approach works well")
        assert plausible_score > 0.6
        
        # Questionable claim
        questionable_score = agent._assess_claim_plausibility("This revolutionary method never fails")
        assert questionable_score < 0.5
    
    def test_processing_with_code(self):
        """Test processing content with code blocks."""
        agent = TechnicalAccuracyAgent()
        content = """
        # Machine Learning Example
        
        This algorithm can achieve 99% accuracy:
        
        ```python
        import numpy as np
        
        def train_model(data):
            model = LinearRegression()
            model.fit(data)
            return model
        ```
        
        The implementation utilizes advanced techniques.
        """
        
        context = ProcessingContext(
            content=content,
            processing_mode=ProcessingMode.FULL
        )
        
        result = agent.process(context)
        assert result.success
        assert result.quality_score is not None
        assert result.confidence_score is not None
        assert 'code_blocks_found' in result.metadata
        assert 'technical_claims_analyzed' in result.metadata
    
    def test_fallback_processing(self):
        """Test fallback processing mode."""
        agent = TechnicalAccuracyAgent()
        content = """
        ```python
        def test():
            pass
        ```
        """
        
        context = ProcessingContext(
            content=content,
            processing_mode=ProcessingMode.FALLBACK
        )
        
        result = agent._process_fallback(context)
        assert result.success
        assert result.processed_content == content
        assert "Fallback mode" in result.suggestions[0]


class TestReadabilityAgent:
    """Test Readability Agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ReadabilityAgent()
        assert agent.name == "Readability"
        assert agent.status == AgentStatus.HEALTHY
    
    def test_syllable_counting(self):
        """Test syllable counting functionality."""
        agent = ReadabilityAgent()
        
        # Test known syllable counts
        assert agent._count_syllables("cat") >= 1
        assert agent._count_syllables("computer") >= 2
        assert agent._count_syllables("algorithm") >= 3
    
    def test_readability_metrics_calculation(self):
        """Test readability metrics calculation."""
        agent = ReadabilityAgent()
        
        # Simple readable text
        simple_text = "This is a simple sentence. It is easy to read."
        metrics = agent._compute_readability_metrics(simple_text)
        
        assert isinstance(metrics, ReadabilityMetrics)
        assert metrics.word_count > 0
        assert metrics.sentence_count > 0
        assert metrics.avg_sentence_length > 0
        assert 0 <= metrics.normalized_flesch <= 1
        
        # Complex text
        complex_text = """
        The utilization of sophisticated methodologies and comprehensive
        algorithmic implementations necessitates extraordinary consideration
        of multidimensional parameters affecting computational efficiency.
        """
        complex_metrics = agent._compute_readability_metrics(complex_text)
        assert complex_metrics.avg_sentence_length > metrics.avg_sentence_length
    
    def test_complex_words_detection(self):
        """Test complex words detection."""
        agent = ReadabilityAgent()
        
        text = "The implementation demonstrates optimization capabilities"
        complex_words = agent._find_complex_words(text)
        
        assert 'implementation' in complex_words
        assert 'demonstrates' in complex_words
        assert 'optimization' in complex_words
    
    def test_content_simplification(self):
        """Test light content simplification."""
        agent = ReadabilityAgent()
        
        original = "We will utilize this methodology to facilitate optimization."
        simplified = agent._lightly_simplify_content(original)
        
        assert "use" in simplified  # "utilize" should be replaced
        assert "help" in simplified  # "facilitate" should be replaced
    
    def test_mobile_readability_assessment(self):
        """Test mobile readability assessment."""
        agent = ReadabilityAgent()
        
        # Text with long paragraphs
        long_text = " ".join(["This is a very long paragraph."] * 30)
        warnings, suggestions = agent._assess_mobile_readability(long_text)
        
        assert any("mobile" in warning.lower() for warning in warnings)
        assert any("paragraph" in suggestion.lower() for suggestion in suggestions)
    
    def test_processing_full_mode(self):
        """Test processing in full mode."""
        agent = ReadabilityAgent()
        content = """
        The comprehensive utilization of sophisticated algorithms demonstrates
        extraordinary capabilities for optimizing performance metrics across
        multidimensional parameter spaces requiring extensive computational
        resources and methodological considerations.
        """
        
        context = ProcessingContext(
            content=content,
            processing_mode=ProcessingMode.FULL,
            audience="business executives"
        )
        
        result = agent.process(context)
        assert result.success
        assert result.quality_score is not None
        assert 'readability_metrics' in result.metadata
        assert len(result.suggestions) > 0


class TestContinuityManagerAgent:
    """Test Continuity Manager Agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = ContinuityManagerAgent()
        assert agent.name == "ContinuityManager"
        assert agent.status == AgentStatus.HEALTHY
        assert hasattr(agent, 'validator')
    
    def test_section_order_proposal(self):
        """Test section order proposal logic."""
        agent = ContinuityManagerAgent()
        
        from src.core.section_aware_prompts import SectionType
        
        # Test reordering with conclusion first
        current = [SectionType.CONCLUSION, SectionType.INTRODUCTION, SectionType.ANALYSIS]
        proposed = agent._propose_section_order(current)
        
        assert proposed[0] == SectionType.INTRODUCTION
        assert proposed[-1] == SectionType.CONCLUSION
    
    def test_naive_section_splitting(self):
        """Test naive section splitting from content."""
        agent = ContinuityManagerAgent()
        
        content = """
        # Introduction
        This is the introduction.
        
        ## Analysis
        This is the analysis section.
        
        ## Conclusion
        This is the conclusion.
        """
        
        sections = agent._naive_split_into_sections(content)
        assert len(sections) >= 2
        assert any('introduction' in section.lower() for section in sections.values())
    
    def test_processing_with_sections(self):
        """Test processing with provided sections."""
        agent = ContinuityManagerAgent()
        
        content = "Combined content of all sections"
        sections = {
            'introduction': 'This is the introduction section.',
            'analysis': 'This is the analysis section with technical details.',
            'conclusion': 'This is the conclusion section.'
        }
        
        context = ProcessingContext(
            content=content,
            metadata={'sections': sections}
        )
        
        result = agent.process(context)
        assert result.success
        assert result.quality_score is not None
        assert 'continuity_report' in result.metadata
        assert 'proposed_order' in result.metadata


class TestAgentCoordinator:
    """Test Agent Coordinator functionality."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly."""
        coordinator = AgentCoordinator()
        assert isinstance(coordinator.agents, dict)
        assert len(coordinator.agents) == 0
    
    def test_agent_registration(self):
        """Test agent registration."""
        coordinator = AgentCoordinator()
        agent = ReadabilityAgent()
        
        coordinator.register('readability', agent)
        assert 'readability' in coordinator.agents
        assert coordinator.agents['readability'] == agent
    
    def test_pipeline_execution(self):
        """Test pipeline execution with multiple agents."""
        # Initialize agents
        tech_agent = TechnicalAccuracyAgent()
        read_agent = ReadabilityAgent()
        
        coordinator = AgentCoordinator({
            'technical': tech_agent,
            'readability': read_agent
        })
        
        # Create execution specs
        specs = [
            AgentExecutionSpec('technical', tech_agent, []),
            AgentExecutionSpec('readability', read_agent, ['technical'])
        ]
        
        content = """
        # Machine Learning
        
        Machine learning can achieve 95% accuracy.
        
        ```python
        def train():
            pass
        ```
        """
        
        context = ProcessingContext(content=content)
        results = coordinator.execute_pipeline(specs, context)
        
        assert 'technical' in results
        assert 'readability' in results
        assert '__errors__' in results
        assert '__metrics__' in results
        assert results['__metrics__']['agents_executed'] == 2
    
    def test_parallel_best_effort_execution(self):
        """Test parallel best effort execution."""
        tech_agent = TechnicalAccuracyAgent()
        read_agent = ReadabilityAgent()
        
        coordinator = AgentCoordinator()
        
        specs = [
            AgentExecutionSpec('technical', tech_agent, []),
            AgentExecutionSpec('readability', read_agent, [])
        ]
        
        content = "Test content for parallel processing."
        context = ProcessingContext(content=content)
        
        results = coordinator.execute_parallel_best_effort(specs, context)
        
        assert 'technical' in results
        assert 'readability' in results
        assert 'final_content' in results
    
    def test_error_handling_in_coordination(self):
        """Test error handling during agent coordination."""
        # Create a mock agent that will fail
        class FailingAgent(TechnicalAccuracyAgent):
            def process(self, context):
                raise Exception("Simulated failure")
        
        failing_agent = FailingAgent()
        read_agent = ReadabilityAgent()
        
        coordinator = AgentCoordinator()
        
        specs = [
            AgentExecutionSpec('failing', failing_agent, []),
            AgentExecutionSpec('readability', read_agent, ['failing'])
        ]
        
        context = ProcessingContext(content="Test content")
        results = coordinator.execute_pipeline(specs, context)
        
        assert len(results['__errors__']) > 0
        assert any(error['agent'] == 'failing' for error in results['__errors__'])
        # Readability should be skipped due to failed dependency
        assert any(error['agent'] == 'readability' for error in results['__errors__'])


class TestPhase2Integration:
    """Test integration of all Phase 2 components."""
    
    def test_end_to_end_multi_agent_processing(self):
        """Test complete multi-agent processing workflow."""
        # Initialize all agents
        tech_agent = TechnicalAccuracyAgent()
        read_agent = ReadabilityAgent()
        cont_agent = ContinuityManagerAgent()
        
        coordinator = AgentCoordinator()
        
        # Create execution pipeline
        specs = [
            AgentExecutionSpec('technical_accuracy', tech_agent, []),
            AgentExecutionSpec('readability', read_agent, ['technical_accuracy']),
            AgentExecutionSpec('continuity_manager', cont_agent, ['readability'])
        ]
        
        # Test content with multiple aspects
        content = """
        # Machine Learning Tutorial
        
        Machine learning algorithms can achieve up to 99% accuracy on certain datasets.
        This comprehensive utilization of sophisticated methodologies demonstrates
        extraordinary capabilities for optimizing performance metrics.
        
        ```python
        import numpy as np
        
        def train_model(X, y):
            model = LinearRegression()
            model.fit(X, y)
            return model
        ```
        
        ## Implementation Details
        
        The implementation facilitates optimal performance through advanced techniques.
        This approach demonstrates significant improvements in computational efficiency.
        
        ## Conclusion
        
        This tutorial shows how machine learning can revolutionize data analysis.
        """
        
        # Add sections for continuity manager
        sections = {
            'introduction': content.split('##')[0],
            'analysis': '## Implementation Details' + content.split('## Implementation Details')[1].split('## Conclusion')[0],
            'conclusion': '## Conclusion' + content.split('## Conclusion')[1]
        }
        
        context = ProcessingContext(
            content=content,
            processing_mode=ProcessingMode.FULL,
            metadata={'sections': sections}
        )
        
        results = coordinator.execute_pipeline(specs, context)
        
        # Verify all agents executed
        assert 'technical_accuracy' in results
        assert 'readability' in results
        assert 'continuity_manager' in results
        
        # Verify metrics collection
        assert '__metrics__' in results
        assert results['__metrics__']['agents_executed'] == 3
        
        # Verify each agent produced meaningful results
        tech_result = results['technical_accuracy']
        assert tech_result['success']
        assert 'code_blocks_found' in tech_result['metadata']
        
        read_result = results['readability']
        assert read_result['success']
        assert 'readability_metrics' in read_result['metadata']
        
        cont_result = results['continuity_manager']
        assert cont_result['success']
        assert 'continuity_report' in cont_result['metadata']
    
    def test_performance_metrics_collection(self):
        """Test that performance metrics are properly collected."""
        agent = ReadabilityAgent()
        
        context = ProcessingContext(
            content="Short test content for performance measurement.",
            processing_mode=ProcessingMode.FAST
        )
        
        result = agent.process(context)
        
        assert result.processing_time_ms > 0
        assert result.success
        assert result.quality_score is not None
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality in base agent."""
        agent = TechnicalAccuracyAgent()
        
        # Get initial circuit breaker state
        initial_state = agent.circuit_breaker.state
        assert initial_state == "closed"
        
        # Check health status includes circuit breaker info
        health = agent.get_health_status()
        assert 'circuit_breaker_state' in health['metrics']


def run_phase2_comprehensive_tests():
    """Run comprehensive Phase 2 validation."""
    print("Running Phase 2 Comprehensive Tests...")
    print("=" * 50)
    
    # Test individual agents
    print("✓ Testing TechnicalAccuracyAgent...")
    tech_agent = TechnicalAccuracyAgent()
    assert tech_agent.name == "TechnicalAccuracy"
    
    print("✓ Testing ReadabilityAgent...")
    read_agent = ReadabilityAgent()
    assert read_agent.name == "Readability"
    
    print("✓ Testing ContinuityManagerAgent...")
    cont_agent = ContinuityManagerAgent()
    assert cont_agent.name == "ContinuityManager"
    
    print("✓ Testing AgentCoordinator...")
    coordinator = AgentCoordinator()
    coordinator.register('test', read_agent)
    assert 'test' in coordinator.agents
    
    # Test integration
    print("✓ Testing multi-agent coordination...")
    specs = [AgentExecutionSpec('readability', read_agent, [])]
    context = ProcessingContext(content="Test content")
    results = coordinator.execute_pipeline(specs, context)
    assert 'readability' in results
    
    print("✓ Phase 2 comprehensive validation complete!")
    print("All specialized agents and coordination framework working correctly.")


if __name__ == "__main__":
    run_phase2_comprehensive_tests()