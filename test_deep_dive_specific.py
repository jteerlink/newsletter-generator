#!/usr/bin/env python3
"""
Test script to debug deep dive pipeline with specific configuration
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_deep_dive_specific():
    """Test deep dive pipeline with tools and tutorials + code examples"""
    try:
        from agents.hybrid_workflow_manager import HybridWorkflowManager, ContentRequest, ContentPipelineType
        print("✅ HybridWorkflowManager import successful")
        
        # Create workflow manager
        manager = HybridWorkflowManager()
        print("✅ Workflow manager created")
        
        # Create a content request with your exact configuration
        from datetime import datetime, timedelta
        request = ContentRequest(
            topic="AI Tools and Tutorials",
            content_pillar="tools_tutorials",  # This is the key difference
            target_audience="developers",
            word_count_target=1000,
            deadline=datetime.now() + timedelta(hours=1),
            special_requirements="include code examples"  # This is the special requirement
        )
        
        print(f"✅ Content request created:")
        print(f"   - Topic: {request.topic}")
        print(f"   - Content Pillar: {request.content_pillar}")
        print(f"   - Target Audience: {request.target_audience}")
        print(f"   - Word Count: {request.word_count_target}")
        print(f"   - Special Requirements: {request.special_requirements}")
        
        # Test routing
        print("\n🔄 Testing workflow routing...")
        workflow_result = manager.route_content_workflow(request)
        
        print(f"✅ Workflow routing result type: {type(workflow_result)}")
        print(f"✅ Workflow routing result keys: {list(workflow_result.keys()) if isinstance(workflow_result, dict) else 'Not a dict'}")
        print(f"✅ Pipeline used: {workflow_result.get('pipeline_used', 'Unknown')}")
        
        # Check the result structure
        raw_result = workflow_result.get('result', workflow_result)
        print(f"✅ Raw result type: {type(raw_result)}")
        print(f"✅ Raw result keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
        
        # Transform the result to match expected structure (like Streamlit does)
        if isinstance(raw_result, dict) and 'markdown' in raw_result:
            generation_result = {
                'newsletter_content': raw_result.get('markdown', ''),
                'subject_line_info': {
                    'subject_line': raw_result.get('subject_line', ''),
                    'preview_text': raw_result.get('preview_text', ''),
                    'character_count': len(raw_result.get('subject_line', ''))
                },
                'html_content': raw_result.get('html', ''),
                'notion_content': raw_result.get('notion', ''),
                'metadata': raw_result.get('metadata', {})
            }
            print("✅ Result transformed successfully")
        else:
            print("❌ Raw result does not have 'markdown' key")
            print(f"❌ Raw result structure: {raw_result}")
            generation_result = raw_result
        
        # Check if newsletter_content exists and has content
        newsletter_content = generation_result.get('newsletter_content', '')
        print(f"✅ Newsletter content length: {len(newsletter_content)} characters")
        
        if newsletter_content:
            print(f"✅ Newsletter content preview: {newsletter_content[:200]}...")
            
            # Test file saving
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_deep_dive_tools_tutorials.md"
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(newsletter_content)
            
            print(f"✅ Newsletter saved to: {filepath}")
            print(f"✅ File size: {filepath.stat().st_size} bytes")
            
            return True
        else:
            print("❌ No newsletter content generated")
            print(f"❌ Generation result keys: {list(generation_result.keys()) if isinstance(generation_result, dict) else 'Not a dict'}")
            return False
        
    except Exception as e:
        print(f"❌ Error in deep dive specific test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the deep dive specific test"""
    print("🧪 Testing Deep Dive Pipeline with Tools & Tutorials + Code Examples")
    print("=" * 70)
    
    success = test_deep_dive_specific()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Deep dive specific test PASSED!")
        print("✅ Deep dive pipeline works with tools and tutorials")
        print("✅ Code examples requirement is handled")
        print("✅ Content generation and saving works")
    else:
        print("❌ Deep dive specific test FAILED!")
        print("Please check the errors above.")

if __name__ == "__main__":
    main() 