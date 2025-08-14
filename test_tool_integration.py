#!/usr/bin/env python3
"""
Test script for tool-augmented newsletter generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.enhanced_generation import execute_tool_augmented_generation

def test_tool_augmented_generation():
    """Test the tool-augmented generation directly"""
    print("🧪 Testing Tool-Augmented Newsletter Generation")
    print("=" * 50)
    
    # Initialize tool usage metrics
    tool_usage_metrics = {
        'vector_queries': 0,
        'web_searches': 0,
        'verified_claims': [],
        'search_providers': [],
        'tool_integration_score': 0.0
    }
    
    try:
        # Test with a simple topic
        result = execute_tool_augmented_generation(
            topic="Machine Learning Model Deployment",
            audience="AI/ML Engineers",
            tool_usage_metrics=tool_usage_metrics
        )
        
        print("\n✅ Tool-Augmented Generation Results:")
        print(f"Success: {result.get('success')}")
        print(f"Output file: {result.get('output_file')}")
        print(f"Execution time: {result.get('execution_time', 0):.2f}s")
        
        # Print tool usage metrics
        tool_usage = result.get('tool_usage', {})
        print(f"\n📊 Tool Usage Analytics:")
        print(f"  Vector Queries: {tool_usage.get('vector_queries', 0)}")
        print(f"  Web Searches: {tool_usage.get('web_searches', 0)}")
        print(f"  Verified Claims: {len(tool_usage.get('verified_claims', []))}")
        print(f"  Search Providers: {tool_usage.get('search_providers', [])}")
        print(f"  Tool Integration Score: {tool_usage.get('tool_integration_score', 0.0):.1%}")
        
        # Check if content has tool usage indicators
        content = result.get('content', '')
        has_analytics_footer = 'Newsletter Generation Analytics' in content
        print(f"\n🔍 Content Analysis:")
        print(f"  Content length: {len(content):,} characters")
        print(f"  Has analytics footer: {has_analytics_footer}")
        
        if has_analytics_footer:
            print("\n🎯 SUCCESS: Tool-augmented generation is working!")
            print("The newsletter includes tool usage analytics.")
        else:
            print("\n⚠️  WARNING: Analytics footer not found in content.")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tool_augmented_generation()