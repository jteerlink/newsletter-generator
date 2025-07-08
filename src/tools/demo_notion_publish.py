#!/usr/bin/env python3
"""
Demonstration script for publishing newsletters to Notion using MCP tools.

This script shows how to use the Notion MCP tools to publish newsletters
from the newsletter generator system to Notion pages.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add the project root directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.notion_integration import NotionNewsletterPublisher

def demo_prepare_newsletter_for_notion():
    """
    Demonstrate how to prepare a newsletter for Notion publication.
    """
    print("🚀 Newsletter to Notion Publishing Demo")
    print("="*50)
    
    # Check for available newsletter files
    output_dir = Path("output")
    newsletter_files = list(output_dir.glob("*.md"))
    
    if not newsletter_files:
        print("❌ No newsletter files found in output directory")
        return
    
    # Get the most recent newsletter
    latest_file = max(newsletter_files, key=lambda f: f.stat().st_mtime)
    print(f"📰 Latest newsletter: {latest_file.name}")
    
    # Create publisher
    publisher = NotionNewsletterPublisher()
    
    # Prepare the newsletter for publication
    result = publisher.publish_newsletter_file(str(latest_file))
    
    if result["success"]:
        print("✅ Newsletter prepared successfully!")
        print(f"📋 Title: {result['page_data']['properties']['title']}")
        print(f"📄 Content length: {len(result['page_data']['content'])} characters")
        
        # Show the structure
        print("\\n📊 Page structure:")
        print(json.dumps(result["page_data"]["properties"], indent=2))
        
        return result["page_data"]
    else:
        print(f"❌ Error: {result['error']}")
        return None

def demo_create_notion_page(page_data: Dict[str, Any], parent_page_id: Optional[str] = None):
    """
    Demonstrate how to create a Notion page using MCP tools.
    
    This function shows the exact structure needed for the mcp_Notion_create-pages call.
    """
    print("\\n🔧 Creating Notion Page")
    print("="*30)
    
    # Prepare the pages array for the MCP tool
    pages_data = [page_data]
    
    # Add parent if provided
    parent_info = None
    if parent_page_id:
        parent_info = {"page_id": parent_page_id}
    
    # This is the exact structure for the MCP tool call
    mcp_call_data = {
        "pages": pages_data
    }
    
    if parent_info:
        mcp_call_data["parent"] = parent_info
    
    print("📋 MCP Tool Call Structure:")
    print(json.dumps(mcp_call_data, indent=2))
    
    print("\\n🎯 To create this page in Notion, use the following MCP tool call:")
    print("mcp_Notion_create-pages")
    print("Parameters:")
    print(json.dumps(mcp_call_data, indent=2))
    
    return mcp_call_data

def demo_full_workflow():
    """
    Demonstrate the complete workflow from newsletter to Notion page.
    """
    print("🎪 Complete Newsletter to Notion Workflow")
    print("="*50)
    
    # Step 1: Prepare newsletter
    print("\\n👉 Step 1: Preparing newsletter...")
    page_data = demo_prepare_newsletter_for_notion()
    
    if not page_data:
        print("❌ Cannot continue - newsletter preparation failed")
        return
    
    # Step 2: Show how to create the page
    print("\\n👉 Step 2: Creating Notion page...")
    mcp_call_data = demo_create_notion_page(page_data)
    
    # Step 3: Instructions for actual publication
    print("\\n👉 Step 3: Actual publication instructions")
    print("="*40)
    print("""
    To actually publish this newsletter to Notion:
    
    1. Copy the MCP tool call data above
    2. In Cursor, use the Notion MCP tools
    3. Call mcp_Notion_create-pages with the provided parameters
    
    Example in Cursor:
    - Open the MCP tools panel
    - Select mcp_Notion_create-pages
    - Paste the parameters from above
    - Execute the tool call
    
    The newsletter will be created as a new page in your Notion workspace!
    """)
    
    return mcp_call_data

def demo_with_parent_page():
    """
    Demonstrate publishing with a specific parent page.
    """
    print("\\n🏠 Publishing with Parent Page")
    print("="*40)
    
    # This would typically be provided by the user
    parent_page_id = "your-parent-page-id-here"
    
    print(f"🔗 Parent page ID: {parent_page_id}")
    
    # Prepare newsletter
    page_data = demo_prepare_newsletter_for_notion()
    
    if page_data:
        # Create with parent
        mcp_call_data = demo_create_notion_page(page_data, parent_page_id)
        
        print("\\n📌 Note: Replace 'your-parent-page-id-here' with actual page ID")
        print("   You can find page IDs by:")
        print("   1. Using mcp_Notion_search to find your parent page")
        print("   2. Copying the page ID from the URL")
        
        return mcp_call_data
    
    return None

def main():
    """
    Main function to run the demonstration.
    """
    print("🎨 Notion Newsletter Publishing Demonstration")
    print("="*60)
    
    # Run the full workflow demonstration
    demo_full_workflow()
    
    # Show how to use with parent page
    demo_with_parent_page()
    
    print("\\n✨ Demo completed!")
    print("Ready to publish your newsletters to Notion using MCP tools! 🚀")

if __name__ == "__main__":
    main() 