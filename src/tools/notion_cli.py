#!/usr/bin/env python3
"""
CLI tool for publishing newsletters to Notion using MCP tools.

This script demonstrates how to integrate with the Notion MCP tools
to publish generated newsletters as Notion pages.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from tools.notion_integration import NotionNewsletterPublisher

def main():
    """Main CLI interface for Notion newsletter publishing."""
    parser = argparse.ArgumentParser(
        description="Publish newsletters to Notion using MCP tools"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a specific newsletter file to publish"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Publish all newsletters in the output directory"
    )
    
    parser.add_argument(
        "--latest", "-l",
        action="store_true",
        help="Publish the latest newsletter"
    )
    
    parser.add_argument(
        "--parent-page-id", "-p",
        type=str,
        help="ID of the parent Notion page where newsletters will be created"
    )
    
    parser.add_argument(
        "--title", "-t",
        type=str,
        help="Custom title for the newsletter page"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Directory containing newsletter files (default: output)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Prepare the page data without actually creating the page"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.file, args.all, args.latest]):
        print("Error: Must specify either --file, --all, or --latest")
        parser.print_help()
        sys.exit(1)
    
    # Create publisher
    publisher = NotionNewsletterPublisher(parent_page_id=args.parent_page_id)
    
    try:
        if args.file:
            # Publish a specific file
            result = publisher.publish_newsletter_file(args.file, args.title)
            print_result(result, args.dry_run)
            
        elif args.latest:
            # Publish the latest newsletter
            from tools.notion_integration import publish_latest_newsletter
            result = publish_latest_newsletter(args.parent_page_id)
            print_result(result, args.dry_run)
            
        elif args.all:
            # Publish all newsletters
            results = publisher.batch_publish_newsletters(args.output_dir)
            for i, result in enumerate(results, 1):
                print(f"\\nResult {i}:")
                print_result(result, args.dry_run)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def print_result(result: Dict[str, Any], dry_run: bool = False):
    """Print the result of a publication attempt."""
    if result["success"]:
        print("✅ Success!")
        print(f"Message: {result['message']}")
        
        if dry_run:
            print("\\n📋 Page data prepared (dry run):")
            if "page_data" in result:
                print(json.dumps(result["page_data"], indent=2))
        else:
            print("\\n📝 To actually create the page in Notion, you would need to:")
            print("1. Use the Notion MCP tools in Cursor")
            print("2. Call mcp_Notion_create-pages with the prepared data")
            
            if "page_data" in result:
                print("\\n📋 Prepared page data:")
                print(json.dumps(result["page_data"], indent=2))
    else:
        print("❌ Failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Message: {result.get('message', 'No message')}")

def demo_publish_with_mcp_tools():
    """
    Demonstration of how to use the MCP tools to actually publish to Notion.
    
    This function shows the pattern for using the MCP tools that are available
    in Cursor to create pages. The actual MCP tool calls would need to be made
    from within Cursor's environment.
    """
    print("""
    📚 How to use MCP tools to publish newsletters to Notion:
    
    1. First, prepare your newsletter data using the NotionNewsletterPublisher
    2. Then use the mcp_Notion_create-pages tool in Cursor
    
    Example workflow:
    
    # Step 1: Prepare newsletter data
    from tools.notion_integration import NotionNewsletterPublisher
    
    publisher = NotionNewsletterPublisher(parent_page_id="your-parent-page-id")
    result = publisher.publish_newsletter_file("output/newsletter.md")
    
    # Step 2: Use MCP tools in Cursor to create the page
    # This would be done through the MCP tools interface:
    # mcp_Notion_create-pages with the page_data from the result
    
    The page_data structure includes:
    - properties: {"title": "Newsletter Title"}
    - content: "Newsletter content in Notion markdown format"
    - parent: {"page_id": "parent-page-id"} (optional)
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo_publish_with_mcp_tools()
    else:
        main() 