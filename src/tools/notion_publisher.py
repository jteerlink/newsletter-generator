#!/usr/bin/env python3

"""
Notion Newsletter Publisher CLI Tool

This tool publishes prepared newsletters to Notion by reading the JSON preparation files
and calling the appropriate MCP tools. It's designed to be run by the AI assistant.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def load_prepared_newsletter(json_file: str) -> Dict[str, Any]:
    """Load a prepared newsletter from JSON file.
    
    Args:
        json_file: Path to the JSON preparation file
        
    Returns:
        Dictionary containing newsletter data
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading prepared newsletter: {e}")
        return {}

def publish_newsletter_to_notion(newsletter_data: Dict[str, Any]) -> str:
    """Publish a newsletter to Notion using the prepared data.
    
    This function prepares the data in the format expected by MCP tools.
    The actual MCP tool call must be made by the AI assistant.
    
    Args:
        newsletter_data: Dictionary containing newsletter data
        
    Returns:
        Message about the publication status
    """
    title = newsletter_data.get('title', 'Untitled Newsletter')
    content = newsletter_data.get('content', '')
    parent_page_id = newsletter_data.get('parent_page_id')
    
    print(f"📋 Publishing newsletter: {title}")
    print(f"📄 Content length: {len(content)} characters")
    print(f"📁 Parent page ID: {parent_page_id}")
    
    # Prepare the MCP tool call data
    mcp_data = {
        "pages": [{
            "properties": {
                "title": title
            },
            "content": content
        }]
    }
    
    # Add parent if specified
    if parent_page_id:
        mcp_data["parent"] = {
            "page_id": parent_page_id
        }
    
    print("\n🚀 MCP Tool Call Data:")
    print("=" * 50)
    print(json.dumps(mcp_data, indent=2))
    print("=" * 50)
    
    return f"Newsletter '{title}' prepared for MCP publishing"

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Publish prepared newsletters to Notion using MCP tools"
    )
    
    parser.add_argument(
        "json_file",
        help="Path to the JSON preparation file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be published without actually publishing"
    )
    
    args = parser.parse_args()
    
    # Load the prepared newsletter
    newsletter_data = load_prepared_newsletter(args.json_file)
    
    if not newsletter_data:
        print("❌ Failed to load newsletter data")
        sys.exit(1)
    
    # Publish the newsletter
    if args.dry_run:
        print("🧪 DRY RUN MODE - No actual publishing will occur")
    
    result = publish_newsletter_to_notion(newsletter_data)
    print(f"\n✅ {result}")
    
    if not args.dry_run:
        print("\n📝 To actually publish, use the MCP tool call data above")
        print("   The AI assistant should call mcp_Notion_create-pages with this data")

if __name__ == "__main__":
    main() 