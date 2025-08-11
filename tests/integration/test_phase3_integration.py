"""
Integration Tests for Phase 3: Code Generation System

This module provides comprehensive tests for the Phase 3 code generation system
integration with the newsletter workflow orchestrator.
"""

import asyncio
import pytest
import tempfile
import time
from unittest.mock import Mock, patch

from src.core.workflow_orchestrator import WorkflowOrchestrator, WorkflowResult
from src.agents.writing import WriterAgent
from src.tools.syntax_validator import SyntaxValidator, ValidationLevel
from src.tools.code_executor import SafeCodeExecutor, ExecutionConfig, SecurityLevel
from src.templates.code_templates import template_library, Framework, TemplateCategory
from src.core.template_manager import AIMLTemplateManager, NewsletterType


class TestPhase3Integration:
    """Test suite for Phase 3 code generation integration"""
    
    def setup_method(self):
        """Set up test environment"""
        self.orchestrator = WorkflowOrchestrator()
        self.writer_agent = WriterAgent()
        self.syntax_validator = SyntaxValidator(ValidationLevel.STANDARD)
        self.code_executor = SafeCodeExecutor()
        self.template_manager = AIMLTemplateManager()
    
    def test_workflow_orchestrator_initialization(self):
        """Test that workflow orchestrator properly initializes Phase 3 components"""
        assert hasattr(self.orchestrator, 'template_manager')
        assert hasattr(self.orchestrator, 'code_generation_metrics')
        assert isinstance(self.orchestrator.template_manager, AIMLTemplateManager)
        
        # Check initial metrics
        assert self.orchestrator.code_generation_metrics['examples_generated'] == 0
        assert self.orchestrator.code_generation_metrics['validation_score'] == 0.0
        assert self.orchestrator.code_generation_metrics['frameworks_used'] == []
    
    def test_technical_topic_detection(self):
        """Test technical topic detection in workflow orchestrator"""
        technical_topics = [
            "Introduction to PyTorch Neural Networks",
            "Machine Learning with TensorFlow",
            "Deep Learning Architecture Design",
            "Python API Development Best Practices"
        ]
        
        non_technical_topics = [
            "Market Trends in Technology",
            "Business Strategy for AI Companies",
            "Industry News and Updates"
        ]
        
        for topic in technical_topics:
            assert self.orchestrator._is_technical_topic(topic), f"Should detect '{topic}' as technical"
        
        for topic in non_technical_topics:
            assert not self.orchestrator._is_technical_topic(topic), f"Should not detect '{topic}' as technical"
    
    def test_code_generation_metrics_tracking(self):
        """Test code generation metrics tracking"""
        # Simulate code examples
        sample_code_examples = [
            "import torch\nmodel = torch.nn.Linear(10, 1)",
            "import pandas as pd\ndf = pd.DataFrame()"
        ]
        
        initial_count = self.orchestrator.code_generation_metrics['examples_generated']
        self.orchestrator._update_code_generation_metrics(sample_code_examples)
        
        # Check metrics were updated
        assert self.orchestrator.code_generation_metrics['examples_generated'] == initial_count + 2
        assert 'pytorch' in self.orchestrator.code_generation_metrics['frameworks_used']
        assert 'pandas' in self.orchestrator.code_generation_metrics['frameworks_used']
    
    def test_writer_agent_code_generation(self):
        """Test writer agent code generation capabilities"""
        topic = "PyTorch Neural Network Implementation"
        
        # Test code example generation
        code_examples = self.writer_agent.generate_code_examples(
            topic=topic,
            framework="pytorch",
            complexity="beginner",
            count=2
        )
        
        assert len(code_examples) >= 1, "Should generate at least one code example"
        
        # Verify code examples contain expected elements
        for example in code_examples:
            assert isinstance(example, str)
            assert len(example) > 50, "Code example should be substantial"
    
    def test_technical_content_generation(self):
        """Test technical content generation with code integration"""
        topic = "Introduction to TensorFlow for Beginners"
        context = "Educational newsletter for AI practitioners"
        
        content = self.writer_agent.generate_technical_content_with_code(
            topic=topic,
            context=context,
            include_code=True
        )
        
        # Verify content structure
        assert len(content) > 500, "Technical content should be substantial"
        assert "```python" in content or "```" in content, "Should contain code blocks"
        assert "tensorflow" in content.lower() or "tf." in content, "Should reference TensorFlow"
    
    def test_code_validation_integration(self):
        """Test code validation integration"""
        # Test valid code
        valid_code = """
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
"""
        
        validation_result = self.writer_agent.validate_and_test_code(valid_code, "pytorch")
        
        assert 'validation' in validation_result
        assert 'execution' in validation_result
        assert validation_result['validation']['syntax_score'] > 0.8
    
    def test_syntax_validator_functionality(self):
        """Test syntax validator comprehensive functionality"""
        # Test syntactically correct code
        correct_code = """
import numpy as np

def calculate_mean(data):
    \"\"\"Calculate the mean of a dataset.\"\"\"
    return np.mean(data)

# Example usage
sample_data = np.array([1, 2, 3, 4, 5])
result = calculate_mean(sample_data)
print(f"Mean: {result}")
"""
        
        result = self.syntax_validator.validate(correct_code)
        
        assert result.is_valid
        assert result.syntax_score == 1.0
        assert result.has_imports
        assert result.has_docstrings
        assert result.overall_score > 0.8
    
    def test_code_executor_safety(self):
        """Test code executor safety mechanisms"""
        # Test safe code execution
        safe_code = """
import math
result = math.sqrt(16)
print(f"Square root of 16: {result}")
"""
        
        execution_result = self.code_executor.execute(safe_code)
        
        assert execution_result.status.value == "success"
        assert "4.0" in execution_result.stdout
        assert execution_result.execution_time < 5.0
    
    def test_template_library_functionality(self):
        """Test code template library functionality"""
        # Test template retrieval
        pytorch_template = template_library.get_template(
            Framework.PYTORCH,
            TemplateCategory.BASIC_EXAMPLE,
            complexity="beginner"
        )
        
        assert pytorch_template is not None
        assert pytorch_template.framework == Framework.PYTORCH
        assert len(pytorch_template.code) > 100
        assert pytorch_template.dependencies
        
        # Test template search
        search_results = template_library.search_templates("neural network")
        assert len(search_results) > 0
        
        # Test framework listing
        frameworks = template_library.list_frameworks()
        assert len(frameworks) > 0
        assert any('pytorch' in fw for fw in frameworks)
    
    def test_template_manager_integration(self):
        """Test template manager integration with workflow"""
        topic = "Deep Learning with PyTorch"
        
        # Test template suggestion
        suggested_type = self.template_manager.suggest_template(topic)
        assert suggested_type == NewsletterType.TECHNICAL_DEEP_DIVE
        
        # Test template retrieval
        template = self.template_manager.get_template(suggested_type)
        assert template is not None
        assert template.type == NewsletterType.TECHNICAL_DEEP_DIVE
        
        # Test enhanced template generation
        enhanced_template = self.template_manager.enhance_template_with_code_examples(template, topic)
        assert enhanced_template.name.startswith("Enhanced")
        
        # Check that code generation requirements were added
        code_guidelines_found = False
        for section in enhanced_template.sections:
            if any("code example" in guideline.lower() for guideline in section.content_guidelines):
                code_guidelines_found = True
                break
        
        assert code_guidelines_found, "Enhanced template should include code generation guidelines"
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_code_generation(self):
        """Test complete end-to-end workflow with code generation"""
        topic = "Building Neural Networks with PyTorch"
        
        # Execute workflow
        result = self.orchestrator.execute_newsletter_generation(
            topic=topic,
            context_id="default",
            output_format="markdown"
        )
        
        # Verify workflow completion
        assert isinstance(result, WorkflowResult)
        assert result.status in ['completed', 'partial']
        assert len(result.final_content) > 1000
        
        # Verify code generation metrics were tracked
        assert result.code_generation_metrics is not None
        assert 'examples_generated' in result.code_generation_metrics
        assert 'frameworks_used' in result.code_generation_metrics
        
        # For technical topics, should have generated code examples
        if self.orchestrator._is_technical_topic(topic):
            assert result.code_generation_metrics['examples_generated'] > 0
    
    def test_performance_metrics(self):
        """Test performance metrics for code generation system"""
        topic = "Machine Learning Pipeline with scikit-learn"
        
        start_time = time.time()
        
        # Generate code examples
        code_examples = self.writer_agent.generate_code_examples(
            topic=topic,
            framework="sklearn",
            count=3
        )
        
        generation_time = time.time() - start_time
        
        # Performance assertions
        assert generation_time < 30.0, "Code generation should complete within 30 seconds"
        assert len(code_examples) > 0, "Should generate at least one example"
        
        # Test validation performance
        if code_examples:
            start_validation = time.time()
            for example in code_examples[:1]:  # Test first example
                validation_result = self.writer_agent.validate_and_test_code(example, "sklearn")
                assert 'validation' in validation_result
            
            validation_time = time.time() - start_validation
            assert validation_time < 10.0, "Code validation should complete within 10 seconds"
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms"""
        # Test with invalid framework
        try:
            code_examples = self.writer_agent.generate_code_examples(
                topic="Invalid Framework Test",
                framework="nonexistent_framework",
                count=1
            )
            # Should handle gracefully and return fallback
            assert isinstance(code_examples, list)
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Unexpected exception: {e}")
        
        # Test with empty topic
        try:
            content = self.writer_agent.generate_technical_content_with_code(
                topic="",
                context="",
                include_code=True
            )
            assert isinstance(content, str)
        except Exception as e:
            pytest.fail(f"Should handle empty topic gracefully: {e}")
    
    def test_security_compliance(self):
        """Test security compliance of generated code"""
        dangerous_patterns = [
            "eval(", "exec(", "os.system(", "subprocess.call(",
            "__import__", "compile(", "open("
        ]
        
        topic = "Data Processing with Python"
        code_examples = self.writer_agent.generate_code_examples(
            topic=topic,
            count=2
        )
        
        for example in code_examples:
            for pattern in dangerous_patterns:
                assert pattern not in example, f"Generated code should not contain dangerous pattern: {pattern}"
    
    def test_content_quality_validation(self):
        """Test content quality validation for technical newsletters"""
        topic = "Advanced TensorFlow Techniques"
        
        content = self.writer_agent.generate_technical_content_with_code(
            topic=topic,
            include_code=True
        )
        
        # Quality checks
        assert len(content.split()) > 500, "Content should be substantial"
        assert content.count('```') >= 2, "Should have properly formatted code blocks"
        assert any(keyword in content.lower() for keyword in ['tensorflow', 'tf.', 'keras']), "Should reference relevant frameworks"
        
        # Structure checks
        assert '#' in content, "Should have proper heading structure"
        assert '##' in content, "Should have section headings"
    
    def test_framework_coverage(self):
        """Test coverage of different AI/ML frameworks"""
        frameworks_to_test = ['pytorch', 'tensorflow', 'sklearn', 'pandas', 'numpy']
        
        for framework in frameworks_to_test:
            topic = f"Getting Started with {framework.title()}"
            
            code_examples = self.writer_agent.generate_code_examples(
                topic=topic,
                framework=framework,
                count=1
            )
            
            assert len(code_examples) > 0, f"Should generate examples for {framework}"
            
            # Verify framework-specific imports
            example_text = ' '.join(code_examples).lower()
            if framework == 'pytorch':
                assert 'torch' in example_text
            elif framework == 'tensorflow':
                assert 'tensorflow' in example_text or 'tf.' in example_text
            elif framework == 'sklearn':
                assert 'sklearn' in example_text
            elif framework == 'pandas':
                assert 'pandas' in example_text or 'pd.' in example_text
            elif framework == 'numpy':
                assert 'numpy' in example_text or 'np.' in example_text


class TestPhase3PerformanceMetrics:
    """Performance-focused tests for Phase 3 system"""
    
    def setup_method(self):
        """Set up performance test environment"""
        self.orchestrator = WorkflowOrchestrator()
        self.performance_metrics = {}
    
    def test_concurrent_code_generation_performance(self):
        """Test performance under concurrent code generation load"""
        topics = [
            "PyTorch Neural Networks",
            "TensorFlow Image Classification", 
            "Scikit-learn Model Selection",
            "Pandas Data Preprocessing",
            "NumPy Matrix Operations"
        ]
        
        start_time = time.time()
        
        # Simulate concurrent requests
        results = []
        for topic in topics:
            result = self.orchestrator.execute_newsletter_generation(
                topic=topic,
                context_id="default"
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 300.0, "Concurrent generation should complete within 5 minutes"
        assert len(results) == len(topics)
        assert all(r.status in ['completed', 'partial'] for r in results)
        
        # Track performance metrics
        self.performance_metrics['concurrent_generation_time'] = total_time
        self.performance_metrics['avg_time_per_newsletter'] = total_time / len(topics)
    
    def test_memory_usage_optimization(self):
        """Test memory usage during code generation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple code examples
        for i in range(10):
            topic = f"Test Topic {i}"
            self.orchestrator.execute_newsletter_generation(
                topic=topic,
                context_id="default"
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase dramatically
        assert memory_increase < 500, f"Memory increase should be under 500MB, got {memory_increase}MB"
        
        self.performance_metrics['memory_increase_mb'] = memory_increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
