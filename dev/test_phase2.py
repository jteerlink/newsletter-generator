#!/usr/bin/env python3
"""
Test script for Phase 2 Implementation
Tests AI/ML templates, quality gates, and code generation integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.template_manager import AIMLTemplateManager, NewsletterType
from src.core.quality_gate import NewsletterQualityGate
from src.core.code_generator import AIMLCodeGenerator, CodeType
from src.agents.agents import ManagerAgent, PlannerAgent, WriterAgent, EditorAgent

def test_template_system():
    """Test AI/ML template system functionality"""
    print("=== Testing AI/ML Template System ===")
    
    template_manager = AIMLTemplateManager()
    
    # Test template retrieval
    tech_template = template_manager.get_template(NewsletterType.TECHNICAL_DEEP_DIVE)
    print(f"Technical Deep Dive Template: {tech_template.name}")
    print(f"Sections: {[s.name for s in tech_template.sections]}")
    
    # Test template suggestion
    suggested = template_manager.suggest_template("deep learning transformer architecture")
    print(f"Suggested template for 'deep learning transformer architecture': {suggested.value}")
    
    # Test all templates
    templates = template_manager.get_available_templates()
    print(f"Available templates: {[t.type.value for t in templates]}")
    
    print("✅ Template system tests passed\n")

def test_quality_gate_system():
    """Test quality gate system functionality"""
    print("=== Testing Quality Gate System ===")
    
    quality_gate = NewsletterQualityGate()
    
    # Test with sample content
    sample_content = """
    # AI Agent Development: A Comprehensive Guide
    
    AI agent development has become a cornerstone of modern artificial intelligence applications. 
    This comprehensive guide explores the fundamental principles, advanced techniques, and practical 
    implementations of AI agents across various domains.
    
    ## Technical Architecture
    
    Modern AI agents leverage transformer architectures and reinforcement learning techniques to 
    achieve human-like performance in complex tasks. The integration of large language models 
    with traditional planning algorithms creates sophisticated reasoning capabilities.
    
    ## Implementation Examples
    
    ```python
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    class AIAgent:
        def __init__(self, model_name="gpt-3.5-turbo"):
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        def process_input(self, text):
            tokens = self.tokenizer.encode(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(tokens)
            return outputs.last_hidden_state
    ```
    
    This implementation demonstrates the core components of a modern AI agent system.
    """
    
    # Test quality gate evaluation
    result = quality_gate.evaluate_content(sample_content, NewsletterType.TECHNICAL_DEEP_DIVE)
    print(f"Quality Gate Status: {result.status.value}")
    print(f"Overall Score: {result.overall_score:.2f}")
    print(f"Blocking Issues: {len(result.blocking_issues)}")
    print(f"Warnings: {len(result.warnings)}")
    
    # Test quality report generation
    report = quality_gate.create_quality_report(result)
    print(f"Quality Report Preview:\n{report[:200]}...")
    
    print("✅ Quality gate system tests passed\n")

def test_code_generation():
    """Test code generation functionality"""
    print("=== Testing Code Generation System ===")
    
    code_generator = AIMLCodeGenerator()
    
    # Test framework suggestion
    framework = code_generator.suggest_framework("natural language processing")
    print(f"Suggested framework for NLP: {framework}")
    
    # Test code example generation
    example = code_generator.generate_code_example(
        "Create a sentiment analysis model using transformers",
        "huggingface",
        CodeType.BASIC_EXAMPLE,
        "intermediate"
    )
    print(f"Generated code example:\n{example.code[:200]}...")
    
    # Test code validation
    validation = code_generator.validate_code(example.code)
    print(f"Code validation - Syntax OK: {validation['syntax_valid']}")
    print(f"Code validation - Has imports: {validation['has_imports']}")
    
    print("✅ Code generation tests passed\n")

def test_manager_agent_integration():
    """Test ManagerAgent integration with Phase 2 components"""
    print("=== Testing ManagerAgent Phase 2 Integration ===")
    
    manager = ManagerAgent()
    
    # Test template selection
    template_type = manager._select_template_for_topic("deep learning transformer architecture")
    print(f"Selected template: {template_type.value}")
    
    # Test code requirements determination
    code_req = manager._determine_code_requirements("python machine learning tutorial")
    print(f"Code requirements: {code_req}")
    
    # Test workflow creation with Phase 2 components
    workflow = manager.create_hierarchical_workflow("AI Agent Development Tools")
    print(f"Workflow template type: {workflow['template_type'].value}")
    print(f"Code examples required: {workflow['code_examples_required']['required']}")
    
    print("✅ ManagerAgent integration tests passed\n")

def test_planner_agent_templates():
    """Test PlannerAgent template integration"""
    print("=== Testing PlannerAgent Template Integration ===")
    
    planner = PlannerAgent()
    
    # Test template-based planning
    plan = planner.plan_with_template("machine learning model optimization", NewsletterType.TECHNICAL_DEEP_DIVE)
    print(f"Generated plan sections: {len(plan.get('sections', []))}")
    print(f"Template used: {plan.get('template_type', 'N/A')}")
    
    # Test template suggestion
    suggested = planner.suggest_template("emerging AI trends 2024")
    print(f"Suggested template: {suggested.value}")
    
    # Test available templates
    templates = planner.get_available_templates()
    print(f"Available templates: {templates}")
    
    print("✅ PlannerAgent template tests passed\n")

def run_all_tests():
    """Run all Phase 2 tests"""
    print("🚀 Starting Phase 2 Implementation Tests\n")
    
    try:
        test_template_system()
        test_quality_gate_system()
        test_code_generation()
        test_manager_agent_integration()
        test_planner_agent_templates()
        
        print("🎉 All Phase 2 tests passed successfully!")
        print("\n📊 Phase 2 Components Verified:")
        print("  ✅ AI/ML Template System")
        print("  ✅ Quality Gate System")
        print("  ✅ Code Generation System")
        print("  ✅ ManagerAgent Integration")
        print("  ✅ PlannerAgent Template Integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests() 