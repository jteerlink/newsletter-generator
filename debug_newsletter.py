#!/usr/bin/env python3
"""
Debug script to test newsletter generation and see the exact output structure
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_daily_quick_pipeline():
    """Debug the daily quick pipeline output"""
    try:
        from agents.daily_quick_pipeline import DailyQuickPipeline
        print("✅ DailyQuickPipeline import successful")
        
        # Create pipeline instance
        pipeline = DailyQuickPipeline()
        print("✅ Pipeline instance created")
        
        # Test with mock content
        result = pipeline.generate_daily_newsletter()
        print(f"✅ Newsletter generation result type: {type(result)}")
        print(f"✅ Newsletter generation result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"✅ Newsletter generation result length: {len(str(result))} characters")
        
        # Pretty print the result
        print("\n📋 RESULT STRUCTURE:")
        print(json.dumps(result, indent=2, default=str)[:1000] + "..." if len(str(result)) > 1000 else json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"❌ Error in daily quick pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_deep_dive_pipeline():
    """Debug the deep dive pipeline output"""
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
        print(f"✅ Workflow routing result type: {type(result)}")
        print(f"✅ Workflow routing result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"✅ Workflow routing result length: {len(str(result))} characters")
        
        # Pretty print the result
        print("\n📋 RESULT STRUCTURE:")
        print(json.dumps(result, indent=2, default=str)[:1000] + "..." if len(str(result)) > 1000 else json.dumps(result, indent=2, default=str))
        
        return result
        
    except Exception as e:
        print(f"❌ Error in deep dive pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run debug tests"""
    print("🔍 Debugging Newsletter Generation Output Structure")
    print("=" * 60)
    
    # Test daily quick pipeline
    print("\n1. Testing Daily Quick Pipeline Output...")
    daily_result = debug_daily_quick_pipeline()
    
    # Test deep dive pipeline
    print("\n2. Testing Deep Dive Pipeline Output...")
    deep_result = debug_deep_dive_pipeline()
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 ANALYSIS:")
    
    if daily_result:
        print(f"Daily Quick - Type: {type(daily_result)}")
        if isinstance(daily_result, dict):
            print(f"Daily Quick - Keys: {list(daily_result.keys())}")
            if 'newsletter_content' in daily_result:
                print(f"Daily Quick - Has newsletter_content: ✅")
            else:
                print(f"Daily Quick - Has newsletter_content: ❌")
    
    if deep_result:
        print(f"Deep Dive - Type: {type(deep_result)}")
        if isinstance(deep_result, dict):
            print(f"Deep Dive - Keys: {list(deep_result.keys())}")
            if 'result' in deep_result:
                print(f"Deep Dive - Has result: ✅")
            else:
                print(f"Deep Dive - Has result: ❌")

if __name__ == "__main__":
    main() 