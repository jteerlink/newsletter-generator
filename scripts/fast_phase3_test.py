#!/usr/bin/env python3
"""
Fast Phase 3 Test with NVIDIA

Streamlined test focusing on core Phase 3 functionality using NVIDIA for speed.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_nvidia_setup():
    """Check if NVIDIA API key is configured"""
    nvidia_key = os.getenv('NVIDIA_API_KEY')
    if not nvidia_key:
        print("⚠️  NVIDIA API key not found in environment")
        print("   Please set NVIDIA_API_KEY environment variable for faster testing")
        return False
    else:
        print("✅ NVIDIA API key configured")
        return True


def test_core_components():
    """Test core Phase 3 components quickly"""
    print("🔍 Testing core Phase 3 components...")
    
    try:
        # Test imports
        from src.tools.syntax_validator import SyntaxValidator, ValidationLevel
        from src.tools.code_executor import SafeCodeExecutor, ExecutionConfig, SecurityLevel
        from src.templates.code_templates import template_library, Framework, TemplateCategory
        from src.agents.writing import WriterAgent
        
        # Quick syntax validation test
        validator = SyntaxValidator(ValidationLevel.BASIC)  # Use basic level for speed
        result = validator.validate("import torch\nprint('Hello PyTorch')")
        assert result.is_valid, "Basic code should validate"
        
        # Quick code execution test
        executor = SafeCodeExecutor(ExecutionConfig(
            security_level=SecurityLevel.SECURE,
            timeout=5.0  # Short timeout for speed
        ))
        exec_result = executor.execute("print('Code execution test')")
        assert exec_result.status.value == "success", "Basic execution should work"
        
        # Quick template test
        pytorch_template = template_library.get_template(
            Framework.PYTORCH,
            TemplateCategory.BASIC_EXAMPLE
        )
        assert pytorch_template is not None, "Should have PyTorch template"
        
        print("✅ Core components test passed")
        return True
        
    except Exception as e:
        print(f"❌ Core components test failed: {e}")
        return False


def test_code_generation_fast():
    """Test code generation with minimal examples"""
    print("🔍 Testing fast code generation...")
    
    try:
        from src.agents.writing import WriterAgent
        
        writer = WriterAgent()
        
        # Generate a single, simple code example
        examples = writer.generate_code_examples(
            topic="Simple PyTorch tensor",
            framework="pytorch",
            complexity="beginner",
            count=1  # Just one example for speed
        )
        
        assert len(examples) > 0, "Should generate at least one example"
        assert "torch" in examples[0].lower(), "Should contain PyTorch content"
        
        print("✅ Fast code generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Fast code generation test failed: {e}")
        return False


def test_nvidia_integration():
    """Test NVIDIA integration specifically"""
    print("🔍 Testing NVIDIA integration...")
    
    try:
        from src.core.core import query_llm
        
        # Simple NVIDIA query test
        start_time = time.time()
        response = query_llm("Generate a simple Python print statement for 'Hello AI'")
        query_time = time.time() - start_time
        
        assert len(response) > 10, "Should get substantial response"
        print(f"✅ NVIDIA query completed in {query_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ NVIDIA integration test failed: {e}")
        return False


def test_workflow_technical_detection():
    """Test technical topic detection in workflow"""
    print("🔍 Testing workflow technical detection...")
    
    try:
        from src.core.workflow_orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator()
        
        # Test technical detection
        technical_topics = [
            "PyTorch Neural Networks",
            "TensorFlow Image Classification",
            "Python Machine Learning"
        ]
        
        non_technical_topics = [
            "Business Strategy",
            "Market Analysis", 
            "Company News"
        ]
        
        for topic in technical_topics:
            assert orchestrator._is_technical_topic(topic), f"Should detect '{topic}' as technical"
        
        for topic in non_technical_topics:
            assert not orchestrator._is_technical_topic(topic), f"Should not detect '{topic}' as technical"
        
        print("✅ Technical detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Technical detection test failed: {e}")
        return False


def test_template_manager_integration():
    """Test template manager integration"""
    print("🔍 Testing template manager integration...")
    
    try:
        from src.core.template_manager import AIMLTemplateManager, NewsletterType
        
        manager = AIMLTemplateManager()
        
        # Test template suggestion
        suggested = manager.suggest_template("PyTorch Deep Learning")
        assert suggested == NewsletterType.TECHNICAL_DEEP_DIVE, "Should suggest technical template"
        
        # Test template retrieval
        template = manager.get_template(suggested)
        assert template is not None, "Should retrieve template"
        
        # Test framework suggestion
        frameworks = manager.suggest_code_frameworks_for_topic("neural networks")
        assert len(frameworks) > 0, "Should suggest frameworks"
        assert any("pytorch" in fw for fw in frameworks), "Should suggest PyTorch"
        
        print("✅ Template manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ Template manager test failed: {e}")
        return False


def main():
    """Run fast Phase 3 tests with NVIDIA"""
    print("=" * 60)
    print("  FAST PHASE 3 TEST WITH NVIDIA")
    print("=" * 60)
    
    # Check NVIDIA setup
    nvidia_available = check_nvidia_setup()
    if not nvidia_available:
        print("\n💡 To use NVIDIA for faster testing:")
        print("   export NVIDIA_API_KEY='your-api-key-here'")
        print("   Continuing with available provider...\n")
    
    tests = [
        test_core_components,
        test_code_generation_fast,
        test_workflow_technical_detection,
        test_template_manager_integration
    ]
    
    # Only test NVIDIA if available
    if nvidia_available:
        tests.append(test_nvidia_integration)
    
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
    print("  FAST TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🚀 ALL FAST TESTS PASSED! Phase 3 is ready!")
        if nvidia_available:
            print("✅ NVIDIA integration confirmed working")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
