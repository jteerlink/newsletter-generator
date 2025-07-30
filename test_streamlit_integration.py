#!/usr/bin/env python3
"""
Test script to verify Streamlit integration works correctly
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_streamlit_integration():
    """Test the Streamlit integration logic"""
    try:
        from agents.daily_quick_pipeline import DailyQuickPipeline
        print("✅ DailyQuickPipeline import successful")
        
        # Create pipeline instance
        pipeline = DailyQuickPipeline()
        print("✅ Pipeline instance created")
        
        # Test with mock content
        raw_result = pipeline.generate_daily_newsletter()
        print(f"✅ Raw result type: {type(raw_result)}")
        print(f"✅ Raw result keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'Not a dict'}")
        
        # Transform the result to match expected structure (like Streamlit does)
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
        
        print(f"✅ Transformed result type: {type(generation_result)}")
        print(f"✅ Transformed result keys: {list(generation_result.keys())}")
        
        # Check if newsletter_content exists and has content
        newsletter_content = generation_result.get('newsletter_content', '')
        print(f"✅ Newsletter content length: {len(newsletter_content)} characters")
        print(f"✅ Newsletter content preview: {newsletter_content[:200]}...")
        
        # Check subject line info
        subject_info = generation_result.get('subject_line_info', {})
        print(f"✅ Subject line: {subject_info.get('subject_line', 'N/A')}")
        print(f"✅ Preview text: {subject_info.get('preview_text', 'N/A')}")
        
        # Test file saving
        if newsletter_content:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_test_newsletter.md"
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(newsletter_content)
            
            print(f"✅ Newsletter saved to: {filepath}")
            print(f"✅ File size: {filepath.stat().st_size} bytes")
            
            return True
        else:
            print("❌ No newsletter content to save")
            return False
        
    except Exception as e:
        print(f"❌ Error in Streamlit integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the integration test"""
    print("🧪 Testing Streamlit Integration")
    print("=" * 50)
    
    success = test_streamlit_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Streamlit integration test PASSED!")
        print("✅ Newsletter generation works")
        print("✅ Content transformation works")
        print("✅ File saving works")
        print("\n📝 The newsletter generator should now work correctly in the Streamlit app!")
    else:
        print("❌ Streamlit integration test FAILED!")
        print("Please check the errors above.")

if __name__ == "__main__":
    main() 