#!/usr/bin/env python3
"""
Minimal test to bypass agent recursion issues
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_minimal_newsletter_generation():
    """Test minimal newsletter generation without agents"""
    try:
        from src.core.core import query_llm
        print("✅ LLM query import successful")
        
        topic = "AI and Machine Learning Trends"
        print(f"✅ Testing with topic: {topic}")
        
        # Create a simple prompt for newsletter generation
        prompt = f"""
        Write a comprehensive newsletter about {topic}.
        
        The newsletter should include:
        1. An engaging introduction
        2. Key developments and trends
        3. Technical insights and analysis
        4. Practical implications
        5. Future outlook
        
        Make it informative, well-structured, and suitable for technical professionals.
        """
        
        print("✅ Generating newsletter content...")
        content = query_llm(prompt)
        
        if content and len(content) > 100:
            print("✅ Newsletter generation successful")
            print(f"✅ Content length: {len(content)} characters")
            print(f"✅ Content preview: {content[:200]}...")
            
            # Save to file
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_minimal_newsletter.md"
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✅ Newsletter saved to: {filepath}")
            print(f"✅ File size: {filepath.stat().st_size} bytes")
            
            return True
        else:
            print("❌ Newsletter generation failed - insufficient content")
            return False
        
    except Exception as e:
        print(f"❌ Error in minimal newsletter generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the minimal test"""
    print("🧪 Testing Minimal Newsletter Generation")
    print("=" * 50)
    
    success = test_minimal_newsletter_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Minimal newsletter generation PASSED!")
        print("✅ Basic functionality works")
        print("✅ Content generation works")
        print("✅ File saving works")
    else:
        print("❌ Minimal newsletter generation FAILED!")

if __name__ == "__main__":
    main() 