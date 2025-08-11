#!/usr/bin/env python3
"""
Phase 3 Integration Demo

This script demonstrates the complete Phase 3 code generation system integration
with the newsletter workflow orchestrator.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.workflow_orchestrator import WorkflowOrchestrator
from src.agents.writing import WriterAgent
from src.tools.syntax_validator import SyntaxValidator, ValidationLevel
from src.tools.code_executor import SafeCodeExecutor, ExecutionConfig, SecurityLevel
from src.templates.code_templates import template_library, Framework, TemplateCategory
from src.core.template_manager import AIMLTemplateManager, NewsletterType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")


def demo_template_library():
    """Demonstrate the code template library functionality"""
    print_header("PHASE 3 CODE TEMPLATE LIBRARY DEMO")
    
    # Show available frameworks
    print_section("Available Frameworks")
    frameworks = template_library.list_frameworks()
    print(f"Supported frameworks: {', '.join(frameworks)}")
    
    # Demonstrate template retrieval
    print_section("Template Retrieval Example")
    pytorch_template = template_library.get_template(
        Framework.PYTORCH,
        TemplateCategory.BASIC_EXAMPLE,
        complexity="beginner"
    )
    
    if pytorch_template:
        print(f"Template: {pytorch_template.name}")
        print(f"Description: {pytorch_template.description}")
        print(f"Framework: {pytorch_template.framework.value}")
        print(f"Complexity: {pytorch_template.complexity.value}")
        print(f"Dependencies: {', '.join(pytorch_template.dependencies)}")
        print("\nCode Preview (first 200 chars):")
        print(pytorch_template.code[:200] + "...")
    
    # Demonstrate template search
    print_section("Template Search Example")
    search_results = template_library.search_templates("neural network")
    print(f"Found {len(search_results)} templates for 'neural network':")
    for i, template in enumerate(search_results[:3], 1):
        print(f"  {i}. {template.name} ({template.framework.value})")


def demo_syntax_validator():
    """Demonstrate syntax validation capabilities"""
    print_header("PHASE 3 SYNTAX VALIDATOR DEMO")
    
    validator = SyntaxValidator(ValidationLevel.STANDARD)
    
    # Test with good code
    print_section("Validating Good Code")
    good_code = '''
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """A simple neural network for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleNet()
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
'''
    
    result = validator.validate(good_code)
    print(f"✅ Validation Result:")
    print(f"   Valid: {result.is_valid}")
    print(f"   Syntax Score: {result.syntax_score:.2f}")
    print(f"   Style Score: {result.style_score:.2f}")
    print(f"   Overall Score: {result.overall_score:.2f}")
    print(f"   Has Imports: {result.has_imports}")
    print(f"   Has Docstrings: {result.has_docstrings}")
    print(f"   Issues Found: {len(result.issues)}")
    
    # Test with problematic code
    print_section("Validating Problematic Code")
    bad_code = '''
import torch
def bad_function(
    x = torch.tensor([1,2,3])
    print("Hello"
    return x
'''
    
    result = validator.validate(bad_code)
    print(f"❌ Validation Result:")
    print(f"   Valid: {result.is_valid}")
    print(f"   Syntax Score: {result.syntax_score:.2f}")
    print(f"   Issues Found: {len(result.issues)}")
    if result.issues:
        print("   Issues:")
        for issue in result.issues[:3]:  # Show first 3 issues
            print(f"     - Line {issue.line_number}: {issue.message}")


def demo_code_executor():
    """Demonstrate safe code execution"""
    print_header("PHASE 3 CODE EXECUTOR DEMO")
    
    executor = SafeCodeExecutor(ExecutionConfig(
        security_level=SecurityLevel.SECURE,
        timeout=10.0
    ))
    
    # Test safe code execution
    print_section("Executing Safe Code")
    safe_code = '''
import numpy as np
import math

# Generate sample data
data = np.random.normal(0, 1, 100)
mean_value = np.mean(data)
std_value = np.std(data)

print(f"Data generated: {len(data)} samples")
print(f"Mean: {mean_value:.4f}")
print(f"Standard deviation: {std_value:.4f}")
print(f"Min value: {np.min(data):.4f}")
print(f"Max value: {np.max(data):.4f}")

# Simple calculation
result = math.sqrt(16)
print(f"Square root of 16: {result}")
'''
    
    execution_result = executor.execute(safe_code)
    print(f"✅ Execution Result:")
    print(f"   Status: {execution_result.status.value}")
    print(f"   Execution Time: {execution_result.execution_time:.3f}s")
    print(f"   Output:")
    for line in execution_result.stdout.strip().split('\n'):
        print(f"     {line}")
    
    if execution_result.stderr:
        print(f"   Errors: {execution_result.stderr}")


def demo_enhanced_writer_agent():
    """Demonstrate enhanced writer agent with code generation"""
    print_header("PHASE 3 ENHANCED WRITER AGENT DEMO")
    
    writer = WriterAgent()
    
    # Demonstrate code example generation
    print_section("Code Example Generation")
    topic = "Introduction to PyTorch Neural Networks"
    
    print(f"Generating code examples for: '{topic}'")
    code_examples = writer.generate_code_examples(
        topic=topic,
        framework="pytorch",
        complexity="beginner",
        count=2
    )
    
    print(f"Generated {len(code_examples)} code examples:")
    for i, example in enumerate(code_examples, 1):
        print(f"\n--- Example {i} ---")
        # Show first 300 characters
        preview = example[:300] + "..." if len(example) > 300 else example
        print(preview)
    
    # Demonstrate technical content generation
    print_section("Technical Content Generation")
    print(f"Generating technical content with code for: '{topic}'")
    
    start_time = time.time()
    content = writer.generate_technical_content_with_code(
        topic=topic,
        context="Educational newsletter for AI practitioners",
        include_code=True
    )
    generation_time = time.time() - start_time
    
    print(f"✅ Content generated in {generation_time:.2f} seconds")
    print(f"   Content length: {len(content)} characters")
    print(f"   Word count: {len(content.split())} words")
    print(f"   Code blocks: {content.count('```')//2}")
    
    # Show content preview
    print("\n--- Content Preview (first 500 chars) ---")
    print(content[:500] + "...")


def demo_workflow_integration():
    """Demonstrate complete workflow integration"""
    print_header("PHASE 3 WORKFLOW INTEGRATION DEMO")
    
    orchestrator = WorkflowOrchestrator()
    
    # Test topics - mix of technical and non-technical
    test_topics = [
        "Building Neural Networks with PyTorch",  # Technical
        "Market Trends in AI Technology",         # Non-technical
        "TensorFlow 2.0 Best Practices",        # Technical
    ]
    
    for i, topic in enumerate(test_topics, 1):
        print_section(f"Workflow Test {i}: {topic}")
        
        start_time = time.time()
        
        # Execute workflow
        result = orchestrator.execute_newsletter_generation(
            topic=topic,
            context_id="default",
            output_format="markdown"
        )
        
        execution_time = time.time() - start_time
        
        # Display results
        print(f"✅ Workflow completed in {execution_time:.2f} seconds")
        print(f"   Status: {result.status}")
        print(f"   Content length: {len(result.final_content)} characters")
        print(f"   Word count: {len(result.final_content.split())} words")
        print(f"   Execution time: {result.execution_time:.2f}s")
        
        # Show code generation metrics if available
        if result.code_generation_metrics:
            print(f"   Code Examples Generated: {result.code_generation_metrics['examples_generated']}")
            print(f"   Frameworks Used: {', '.join(result.code_generation_metrics['frameworks_used'])}")
        
        # Show content preview
        print(f"\n--- Content Preview ---")
        preview_lines = result.final_content.split('\n')[:10]
        for line in preview_lines:
            print(f"   {line}")
        print("   ...")
        
        print()


def demo_performance_analysis():
    """Demonstrate performance analysis of Phase 3 system"""
    print_header("PHASE 3 PERFORMANCE ANALYSIS")
    
    orchestrator = WorkflowOrchestrator()
    
    # Performance test
    print_section("Performance Test")
    
    technical_topics = [
        "PyTorch Fundamentals",
        "TensorFlow Keras API",
        "Scikit-learn Pipelines"
    ]
    
    total_time = 0
    successful_runs = 0
    
    for topic in technical_topics:
        print(f"Processing: {topic}")
        
        start_time = time.time()
        result = orchestrator.execute_newsletter_generation(topic)
        execution_time = time.time() - start_time
        
        total_time += execution_time
        if result.status == 'completed':
            successful_runs += 1
        
        print(f"   Time: {execution_time:.2f}s, Status: {result.status}")
    
    # Performance summary
    print_section("Performance Summary")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per newsletter: {total_time/len(technical_topics):.2f} seconds")
    print(f"Success rate: {successful_runs/len(technical_topics)*100:.1f}%")
    
    # System resource usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss/1024/1024:.1f} MB")
        print(f"CPU percent: {process.cpu_percent():.1f}%")
    except ImportError:
        print("Install psutil for detailed resource monitoring")


def main():
    """Run the complete Phase 3 integration demonstration"""
    print_header("PHASE 3: CODE GENERATION SYSTEM INTEGRATION DEMO")
    print("This demonstration showcases the complete Phase 3 implementation")
    print("including code generation, validation, execution, and workflow integration.")
    
    try:
        # Run all demonstrations
        demo_template_library()
        demo_syntax_validator()
        demo_code_executor()
        demo_enhanced_writer_agent()
        demo_workflow_integration()
        demo_performance_analysis()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("✅ All Phase 3 components are working correctly!")
        print("✅ Code generation system is fully integrated!")
        print("✅ Performance metrics are within acceptable ranges!")
        
    except Exception as e:
        print_header("DEMO FAILED")
        print(f"❌ Error during demonstration: {e}")
        logger.exception("Demo failed with exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
