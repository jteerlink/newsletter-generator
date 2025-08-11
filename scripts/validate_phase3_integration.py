#!/usr/bin/env python3
"""
Phase 3 Integration Validation Script

Quick validation script to ensure Phase 3 code generation system
is properly integrated and functional.
"""

import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all Phase 3 components can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from src.tools.syntax_validator import SyntaxValidator
        from src.tools.code_executor import SafeCodeExecutor  
        from src.templates.code_templates import template_library
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        from src.agents.writing import WriterAgent
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of Phase 3 components"""
    print("🔍 Testing basic functionality...")
    
    try:
        # Test syntax validator
        from src.tools.syntax_validator import SyntaxValidator, ValidationLevel
        validator = SyntaxValidator(ValidationLevel.STANDARD)
        result = validator.validate("print('Hello, World!')")
        assert result.is_valid, "Simple code should be valid"
        
        # Test template library
        from src.templates.code_templates import template_library, Framework, TemplateCategory
        frameworks = template_library.list_available_templates()
        assert len(frameworks) > 0, "Should have available templates"
        
        # Test workflow orchestrator initialization
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        orchestrator = WorkflowOrchestrator()
        assert hasattr(orchestrator, 'template_manager'), "Should have template manager"
        assert hasattr(orchestrator, 'code_generation_metrics'), "Should have code metrics"
        
        print("✅ Basic functionality tests passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_code_generation():
    """Test code generation capabilities"""
    print("🔍 Testing code generation...")
    
    try:
        from src.agents.writing import WriterAgent
        
        writer = WriterAgent()
        
        # Test code example generation
        examples = writer.generate_code_examples(
            topic="Simple Python Example",
            count=1
        )
        
        assert len(examples) > 0, "Should generate at least one example"
        assert isinstance(examples[0], str), "Example should be a string"
        assert len(examples[0]) > 50, "Example should be substantial"
        
        print("✅ Code generation test passed")
        return True
    except Exception as e:
        print(f"❌ Code generation test failed: {e}")
        return False


def test_workflow_integration():
    """Test workflow integration"""
    print("🔍 Testing workflow integration...")
    
    try:
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator()
        
        # Test technical topic detection
        is_technical = orchestrator._is_technical_topic("PyTorch Neural Networks")
        assert is_technical, "Should detect technical topics"
        
        is_not_technical = orchestrator._is_technical_topic("Business Strategy")
        assert not is_not_technical, "Should not detect non-technical topics"
        
        print("✅ Workflow integration test passed")
        return True
    except Exception as e:
        print(f"❌ Workflow integration test failed: {e}")
        return False


def test_end_to_end_simple():
    """Test simple end-to-end workflow"""
    print("🔍 Testing simple end-to-end workflow...")
    
    try:
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator()
        
        # Test with a simple topic
        result = orchestrator.execute_newsletter_generation(
            topic="Python Programming Basics",
            context_id="default"
        )
        
        assert result is not None, "Should return a result"
        assert hasattr(result, 'final_content'), "Should have final content"
        assert hasattr(result, 'code_generation_metrics'), "Should have code metrics"
        assert result.status in ['completed', 'partial', 'failed'], "Should have valid status"
        
        print("✅ End-to-end test passed")
        print(f"   Status: {result.status}")
        print(f"   Content length: {len(result.final_content)} chars")
        if result.code_generation_metrics:
            print(f"   Code examples: {result.code_generation_metrics['examples_generated']}")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("  PHASE 3 INTEGRATION VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_code_generation,
        test_workflow_integration,
        test_end_to_end_simple
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 3 integration is successful!")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the integration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
