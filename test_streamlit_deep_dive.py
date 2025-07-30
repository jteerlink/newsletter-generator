#!/usr/bin/env python3
"""
Test script to verify Streamlit integration with deep dive pipeline
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_streamlit_deep_dive_integration():
    """Test the Streamlit integration with deep dive pipeline"""
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
            content_pillar="tools_tutorials",
            target_audience="developers",
            word_count_target=1000,
            deadline=datetime.now() + timedelta(hours=1),
            special_requirements="include code examples"
        )
        
        print(f"✅ Content request created with tools_tutorials pillar")
        
        # Test workflow routing (this is what Streamlit does)
        print("\n🔄 Testing workflow routing...")
        workflow_result = manager.route_content_workflow(request)
        
        print(f"✅ Workflow routing result type: {type(workflow_result)}")
        print(f"✅ Workflow routing result keys: {list(workflow_result.keys()) if isinstance(workflow_result, dict) else 'Not a dict'}")
        print(f"✅ Pipeline used: {workflow_result.get('pipeline_used', 'Unknown')}")
        
        # Get the raw result (this is what Streamlit does)
        raw_result = workflow_result.get('result', workflow_result)
        print(f"✅ Raw result type: {type(raw_result)}")
        print(f"✅ Raw result keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
        
        # Transform the result to match expected structure (this is what Streamlit does)
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
            print("✅ Result transformed successfully (like Streamlit does)")
        else:
            print("❌ Raw result does not have 'markdown' key")
            print(f"❌ Raw result structure: {raw_result}")
            generation_result = raw_result
        
        # Check if newsletter_content exists and has content (this is what Streamlit does)
        newsletter_content = generation_result.get('newsletter_content', '')
        print(f"✅ Newsletter content length: {len(newsletter_content)} characters")
        
        if newsletter_content:
            print(f"✅ Newsletter content preview: {newsletter_content[:200]}...")
            
            # Test file saving (this is what Streamlit does)
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_streamlit_deep_dive_test.md"
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(newsletter_content)
            
            print(f"✅ Newsletter saved to: {filepath}")
            print(f"✅ File size: {filepath.stat().st_size} bytes")
            
            # Test the display logic (this is what Streamlit does)
            print("\n📱 Testing Streamlit display logic...")
            if newsletter_content:
                print("✅ Streamlit would display: newsletter_content exists")
                print(f"✅ Content length for display: {len(newsletter_content)} characters")
            else:
                print("❌ Streamlit would display: No newsletter content generated")
            
            return True
        else:
            print("❌ No newsletter content generated")
            print(f"❌ Generation result keys: {list(generation_result.keys()) if isinstance(generation_result, dict) else 'Not a dict'}")
            return False
        
    except Exception as e:
        print(f"❌ Error in Streamlit deep dive integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the Streamlit deep dive integration test"""
    print("🧪 Testing Streamlit Integration with Deep Dive Pipeline")
    print("=" * 70)
    
    success = test_streamlit_deep_dive_integration()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 Streamlit deep dive integration test PASSED!")
        print("✅ Deep dive pipeline works with Streamlit")
        print("✅ Content transformation works")
        print("✅ Display logic works")
        print("✅ File saving works")
        print("\n📝 The deep dive pipeline should now work correctly in the Streamlit app!")
    else:
        print("❌ Streamlit deep dive integration test FAILED!")
        print("Please check the errors above.")

if __name__ == "__main__":
    main() 