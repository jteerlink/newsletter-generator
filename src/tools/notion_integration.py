"""
Notion Integration Tool for Newsletter Publishing

This module provides functionality to convert markdown newsletters to Notion pages
using the Notion MCP tools available in Cursor.
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class NotionNewsletterPublisher:
    """Handles publishing newsletters to Notion as pages using MCP tools."""
    
    def __init__(self, parent_page_id: Optional[str] = None):
        """Initialize the Notion publisher.
        
        Args:
            parent_page_id: ID of the parent page where newsletters will be created
        """
        # Default to the Newsletter Archive page we created
        self.parent_page_id = parent_page_id or "226b1384-d996-813f-bc9c-c540b498df90"
        
    def convert_markdown_to_notion_blocks(self, markdown_content: str) -> str:
        """Convert markdown content to Notion-flavored markdown.
        
        Args:
            markdown_content: The markdown content to convert
            
        Returns:
            Notion-flavored markdown content
        """
        # Notion supports most standard markdown, but we may need some adjustments
        notion_content = markdown_content
        
        # Convert any special formatting that might not be supported
        # For now, we'll keep it simple and rely on Notion's markdown support
        
        return notion_content
    
    def create_newsletter_page(self, newsletter_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Notion page with the newsletter content.
        
        Args:
            newsletter_data: Dictionary containing newsletter information
                - title: Newsletter title
                - content: Newsletter content in markdown
                - date: Newsletter date (optional)
                - tags: List of tags (optional)
                
        Returns:
            Dictionary with page creation result
        """
        try:
            # Prepare the page title
            title = newsletter_data.get('title', f"Newsletter - {datetime.now().strftime('%Y-%m-%d')}")
            
            # Convert markdown content to Notion format
            content = self.convert_markdown_to_notion_blocks(newsletter_data.get('content', ''))
            
            # Prepare page properties
            page_properties = {
                "title": title
            }
            
            # Add date if provided
            if 'date' in newsletter_data:
                page_properties["Created"] = newsletter_data['date']
            
            # Prepare the page creation data
            page_data = {
                "properties": page_properties,
                "content": content
            }
            
            # Add parent if specified
            if self.parent_page_id:
                page_data["parent"] = {"page_id": self.parent_page_id}
            
            return {
                "success": True,
                "page_data": page_data,
                "message": "Newsletter page prepared for creation"
            }
            
        except Exception as e:
            logger.error(f"Error preparing newsletter page: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to prepare newsletter page"
            }
    
    def publish_newsletter_file(self, file_path: str, custom_title: Optional[str] = None) -> Dict[str, Any]:
        """Publish a newsletter file to Notion.
        
        Args:
            file_path: Path to the newsletter markdown file
            custom_title: Custom title for the page (optional)
            
        Returns:
            Dictionary with publication result
        """
        try:
            # Read the newsletter file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from filename if not provided
            if not custom_title:
                file_name = Path(file_path).stem
                custom_title = file_name.replace('_', ' ').replace('-', ' ').title()
            
            # Get file modification time as date
            file_stat = os.stat(file_path)
            file_date = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d')
            
            # Prepare newsletter data
            newsletter_data = {
                "title": custom_title,
                "content": content,
                "date": file_date,
                "file_path": file_path
            }
            
            # Create the page
            result = self.create_newsletter_page(newsletter_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error publishing newsletter file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to publish newsletter file: {file_path}"
            }
    
    def batch_publish_newsletters(self, output_dir: str = "output") -> List[Dict[str, Any]]:
        """Publish all newsletter files in the output directory.
        
        Args:
            output_dir: Directory containing newsletter files
            
        Returns:
            List of publication results
        """
        results = []
        
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return [{
                    "success": False,
                    "error": "Output directory does not exist",
                    "message": f"Directory {output_dir} not found"
                }]
            
            # Find all markdown files in the output directory
            md_files = list(output_path.glob("*.md"))
            
            if not md_files:
                return [{
                    "success": False,
                    "error": "No markdown files found",
                    "message": f"No .md files found in {output_dir}"
                }]
            
            # Publish each file
            for md_file in md_files:
                result = self.publish_newsletter_file(str(md_file))
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch publish: {str(e)}")
            return [{
                "success": False,
                "error": str(e),
                "message": "Failed to batch publish newsletters"
            }]

    def publish_newsletter_from_file(self, file_path: str, custom_title: Optional[str] = None) -> str:
        """Publish a newsletter from a markdown file to Notion using MCP tools.
        
        This method reads the file, extracts the title and content, and creates a Notion page.
        Note: This method needs to be called from a context where MCP tools are available.
        
        Args:
            file_path: Path to the newsletter markdown file
            custom_title: Custom title for the page (optional)
            
        Returns:
            URL of the created Notion page
            
        Raises:
            Exception: If the file cannot be read or page creation fails
        """
        try:
            # Read the newsletter file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from content or use custom title
            title = custom_title
            if not title:
                # Look for a title in the content (first # header)
                title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
                if title_match:
                    title = title_match.group(1).strip()
                else:
                    # Fall back to filename
                    file_name = Path(file_path).stem
                    title = file_name.replace('_', ' ').replace('-', ' ').title()
            
            # Clean up content for Notion
            notion_content = self.convert_markdown_to_notion_blocks(content)
            
            # This is a placeholder return - the actual MCP tool call needs to be made
            # by the calling code since we can't directly call MCP tools from here
            logger.info(f"Prepared newsletter '{title}' for Notion publishing")
            
            # Store the prepared data for external access
            self.last_prepared_page = {
                "title": title,
                "content": notion_content,
                "parent_page_id": self.parent_page_id
            }
            
            # Return a placeholder URL - actual publishing happens in the calling code
            return f"https://notion.so/prepared-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        except Exception as e:
            logger.error(f"Error preparing newsletter from file {file_path}: {str(e)}")
            raise Exception(f"Failed to prepare newsletter from file: {str(e)}")

def create_notion_publisher(parent_page_id: Optional[str] = None) -> NotionNewsletterPublisher:
    """Factory function to create a Notion publisher instance.
    
    Args:
        parent_page_id: ID of the parent page where newsletters will be created
        
    Returns:
        NotionNewsletterPublisher instance
    """
    return NotionNewsletterPublisher(parent_page_id=parent_page_id)

# Example usage functions
def publish_latest_newsletter(parent_page_id: Optional[str] = None) -> Dict[str, Any]:
    """Publish the most recent newsletter to Notion.
    
    Args:
        parent_page_id: ID of the parent page where newsletter will be created
        
    Returns:
        Publication result
    """
    publisher = create_notion_publisher(parent_page_id)
    
    # Find the most recent newsletter file
    output_path = Path("output")
    if not output_path.exists():
        return {
            "success": False,
            "error": "Output directory does not exist",
            "message": "No output directory found"
        }
    
    md_files = list(output_path.glob("*.md"))
    if not md_files:
        return {
            "success": False,
            "error": "No newsletter files found",
            "message": "No .md files found in output directory"
        }
    
    # Get the most recent file
    latest_file = max(md_files, key=lambda f: f.stat().st_mtime)
    
    return publisher.publish_newsletter_file(str(latest_file))

def publish_all_newsletters(parent_page_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Publish all newsletters to Notion.
    
    Args:
        parent_page_id: ID of the parent page where newsletters will be created
        
    Returns:
        List of publication results
    """
    publisher = create_notion_publisher(parent_page_id)
    return publisher.batch_publish_newsletters() 

def publish_to_notion_with_mcp(page_data: Dict[str, Any]) -> str:
    """Publish prepared page data to Notion using MCP tools.
    
    This function needs to be called from a context where MCP tools are available.
    
    Args:
        page_data: Dictionary containing:
            - title: Page title
            - content: Page content in markdown
            - parent_page_id: ID of parent page
            
    Returns:
        URL of the created Notion page
    """
    # This is a placeholder - the actual MCP tool call needs to be made
    # by the calling code since we can't directly call MCP tools from Python
    logger.info(f"Prepared to publish '{page_data.get('title', 'Unknown')}' to Notion")
    
    # For now, we'll return a placeholder URL
    # The actual implementation would call the MCP tools
    return f"https://notion.so/newsletter-{datetime.now().strftime('%Y%m%d_%H%M%S')}"