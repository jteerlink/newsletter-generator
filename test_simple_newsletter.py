#!/usr/bin/env python3
"""
Simple test script to debug newsletter generation without search tools
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_daily_quick_pipeline():
    """Test the daily quick pipeline"""
    try:
        from agents.daily_quick_pipeline import DailyQuickPipeline
        print("✅ DailyQuickPipeline import successful")
        
        # Create pipeline instance
        pipeline = DailyQuickPipeline()
        print("✅ Pipeline instance created")
        
        # Test with mock content
        result = pipeline.generate_daily_newsletter()
        print(f"✅ Newsletter generation result: {len(str(result))} characters")
        print(f"Result preview: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in daily quick pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deep_dive_pipeline():
    """Test the deep dive pipeline"""
    try:
        from agents.hybrid_workflow_manager import HybridWorkflowManager, ContentRequest, ContentPipelineType
        print("✅ HybridWorkflowManager import successful")
        
        # Create workflow manager
        manager = HybridWorkflowManager()
        print("✅ Workflow manager created")
        
        # Create a content request
        from datetime import datetime, timedelta
        request = ContentRequest(
            topic="AI News",
            content_pillar="news_breakthroughs",
            target_audience="developers",
            word_count_target=1000,
            deadline=datetime.now() + timedelta(hours=1)
        )
        
        # Test routing
        result = manager.route_content_workflow(request)
        print(f"✅ Workflow routing result: {len(str(result))} characters")
        print(f"Result preview: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in deep dive pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_llm():
    """Test core LLM functionality"""
    try:
        from core.core import query_llm
        print("✅ Core LLM import successful")
        
        # Test simple query
        result = query_llm("What is AI? Answer in one sentence.")
        print(f"✅ LLM query result: {len(str(result))} characters")
        print(f"Result: {str(result)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in core LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Newsletter Generation System (Simple)")
    print("=" * 50)
    
    # Test core LLM first
    print("\n1. Testing Core LLM...")
    llm_ok = test_core_llm()
    
    # Test daily quick pipeline
    print("\n2. Testing Daily Quick Pipeline...")
    daily_ok = test_daily_quick_pipeline()
    
    # Test deep dive pipeline
    print("\n3. Testing Deep Dive Pipeline...")
    deep_ok = test_deep_dive_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Core LLM: {'✅ PASS' if llm_ok else '❌ FAIL'}")
    print(f"Daily Quick: {'✅ PASS' if daily_ok else '❌ FAIL'}")
    print(f"Deep Dive: {'✅ PASS' if deep_ok else '❌ FAIL'}")
    
    if all([llm_ok, daily_ok, deep_ok]):
        print("\n🎉 All tests passed! Newsletter generation should work.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 