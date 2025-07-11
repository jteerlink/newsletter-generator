"""
Phase 3: Content Format Optimization Implementation

Handles mobile-first responsive design, format adaptation, and multi-platform content optimization.
Optimizes content for 60% mobile readership with adaptive layouts and progressive enhancement.

Based on hybrid_newsletter_system_plan.md requirements for Phase 3.
"""

from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from core.core import query_llm

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Target device types for content optimization"""
    MOBILE = "mobile"
    TABLET = "tablet"  
    DESKTOP = "desktop"
    EMAIL_CLIENT = "email_client"

class ContentFormat(Enum):
    """Content output formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    EMAIL_HTML = "email_html"
    NOTION = "notion"
    PLAIN_TEXT = "plain_text"
    JSON = "json"

@dataclass
class FormatOptimizationConfig:
    """Configuration for format optimization"""
    target_device: DeviceType
    output_format: ContentFormat
    max_line_length: int = 80
    enable_progressive_enhancement: bool = True
    optimize_images: bool = True
    mobile_first: bool = True
    accessibility_level: str = "AA"  # WCAG compliance level

@dataclass
class ContentBlock:
    """Individual content block for mobile optimization"""
    content_type: str  # heading, paragraph, list, code, quote, etc.
    content: str
    priority: int = 1  # 1=high, 2=medium, 3=low for mobile stacking
    mobile_optimized: bool = False
    word_count: int = 0

class MobileFirstOptimizer:
    """Optimizes content layout for mobile-first approach"""
    
    def __init__(self):
        self.mobile_breakpoint = 480  # px
        self.tablet_breakpoint = 768  # px
        
    def optimize_for_mobile(self, content: str) -> Dict[str, Any]:
        """Apply mobile-first optimization strategies"""
        
        # Parse content into blocks
        content_blocks = self._parse_content_blocks(content)
        
        # Apply mobile optimizations
        optimized_blocks = []
        for block in content_blocks:
            optimized_block = self._optimize_block_for_mobile(block)
            optimized_blocks.append(optimized_block)
        
        # Reorder for mobile priority
        mobile_ordered = self._apply_mobile_priority_order(optimized_blocks)
        
        # Generate responsive structure
        responsive_content = self._generate_responsive_structure(mobile_ordered)
        
        return {
            'mobile_optimized_content': responsive_content,
            'content_blocks': mobile_ordered,
            'mobile_reading_time': self._calculate_mobile_reading_time(mobile_ordered),
            'optimization_applied': True
        }
    
    def _parse_content_blocks(self, content: str) -> List[ContentBlock]:
        """Parse content into mobile-optimizable blocks"""
        blocks = []
        
        # Split by major sections (headings, paragraphs, lists, etc.)
        lines = content.split('\n')
        current_block = ""
        current_type = "paragraph"
        
        for line in lines:
            stripped_line = line.strip()
            
            if not stripped_line:
                if current_block.strip():
                    blocks.append(ContentBlock(
                        content_type=current_type,
                        content=current_block.strip(),
                        word_count=len(current_block.split())
                    ))
                    current_block = ""
                continue
            
            # Detect content type
            if stripped_line.startswith('#'):
                # Save previous block
                if current_block.strip():
                    blocks.append(ContentBlock(
                        content_type=current_type,
                        content=current_block.strip(),
                        word_count=len(current_block.split())
                    ))
                
                # Start new heading block
                current_type = "heading"
                current_block = stripped_line
                
                # Immediately save heading (single line)
                blocks.append(ContentBlock(
                    content_type=current_type,
                    content=current_block,
                    priority=1,  # Headings are high priority
                    word_count=len(current_block.split())
                ))
                current_block = ""
                current_type = "paragraph"
                
            elif stripped_line.startswith('*') or stripped_line.startswith('-'):
                if current_type != "list":
                    # Save previous block
                    if current_block.strip():
                        blocks.append(ContentBlock(
                            content_type=current_type,
                            content=current_block.strip(),
                            word_count=len(current_block.split())
                        ))
                    current_block = ""
                    current_type = "list"
                current_block += line + "\n"
                
            elif stripped_line.startswith('```'):
                if current_type != "code":
                    # Save previous block
                    if current_block.strip():
                        blocks.append(ContentBlock(
                            content_type=current_type,
                            content=current_block.strip(),
                            word_count=len(current_block.split())
                        ))
                    current_block = ""
                    current_type = "code"
                current_block += line + "\n"
                
            elif stripped_line.startswith('>'):
                if current_type != "quote":
                    # Save previous block
                    if current_block.strip():
                        blocks.append(ContentBlock(
                            content_type=current_type,
                            content=current_block.strip(),
                            word_count=len(current_block.split())
                        ))
                    current_block = ""
                    current_type = "quote"
                current_block += line + "\n"
                
            else:
                if current_type not in ["paragraph", "list", "code", "quote"]:
                    current_type = "paragraph"
                current_block += line + "\n"
        
        # Save final block
        if current_block.strip():
            blocks.append(ContentBlock(
                content_type=current_type,
                content=current_block.strip(),
                word_count=len(current_block.split())
            ))
        
        return blocks
    
    def _optimize_block_for_mobile(self, block: ContentBlock) -> ContentBlock:
        """Apply mobile-specific optimizations to a content block"""
        
        optimized_content = block.content
        
        # Optimize based on content type
        if block.content_type == "paragraph":
            # Break long paragraphs for mobile readability
            if block.word_count > 40:  # Mobile threshold
                optimized_content = self._break_long_paragraph(block.content)
                
        elif block.content_type == "list":
            # Optimize list formatting for mobile
            optimized_content = self._optimize_list_for_mobile(block.content)
            
        elif block.content_type == "code":
            # Ensure code blocks are mobile-scrollable
            optimized_content = self._optimize_code_for_mobile(block.content)
            
        elif block.content_type == "heading":
            # Optimize heading hierarchy for mobile
            optimized_content = self._optimize_heading_for_mobile(block.content)
        
        # Set mobile priority
        priority = self._calculate_mobile_priority(block)
        
        return ContentBlock(
            content_type=block.content_type,
            content=optimized_content,
            priority=priority,
            mobile_optimized=True,
            word_count=len(optimized_content.split())
        )
    
    def _break_long_paragraph(self, paragraph: str) -> str:
        """Break long paragraphs into mobile-friendly chunks"""
        sentences = re.split(r'[.!?]+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk
            test_chunk = (current_chunk + " " + sentence).strip()
            
            # If chunk gets too long, start new chunk
            if len(test_chunk.split()) > 25:  # Mobile sentence limit
                if current_chunk:
                    chunks.append(current_chunk.strip() + ".")
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip() + ".")
        
        return "\n\n".join(chunks)
    
    def _optimize_list_for_mobile(self, list_content: str) -> str:
        """Optimize list formatting for mobile devices"""
        lines = list_content.split('\n')
        optimized_lines = []
        
        for line in lines:
            if line.strip().startswith('*') or line.strip().startswith('-'):
                # Ensure proper spacing for mobile touch targets
                if len(line.strip()) > 60:  # Long list item
                    # Break long list items
                    item_text = line.strip()[1:].strip()  # Remove bullet
                    if len(item_text) > 50:
                        wrapped_item = self._wrap_list_item(item_text)
                        optimized_lines.append(f"* {wrapped_item}")
                    else:
                        optimized_lines.append(line)
                else:
                    optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _wrap_list_item(self, item_text: str) -> str:
        """Wrap long list items for mobile readability"""
        if len(item_text) <= 50:
            return item_text
            
        # Find good break point
        words = item_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            if len(test_line) > 50 and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
            
        return '\n  '.join(lines)  # Indent continuation lines
    
    def _optimize_code_for_mobile(self, code_content: str) -> str:
        """Optimize code blocks for mobile viewing"""
        lines = code_content.split('\n')
        
        # Add mobile scroll indicators for wide code
        optimized_lines = []
        for line in lines:
            if len(line) > 50:  # Mobile code width threshold
                # Add horizontal scroll hint
                if line.strip().startswith('```'):
                    optimized_lines.append(line + "  // ← scroll horizontally →")
                else:
                    optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def _optimize_heading_for_mobile(self, heading: str) -> str:
        """Optimize heading formatting for mobile hierarchy"""
        # Ensure proper heading hierarchy for mobile screen readers
        if heading.startswith('####'):
            # Convert H4+ to H3 for mobile simplicity
            return heading.replace('####', '###', 1)
        return heading
    
    def _calculate_mobile_priority(self, block: ContentBlock) -> int:
        """Calculate mobile display priority for block"""
        if block.content_type == "heading":
            return 1  # High priority
        elif block.content_type == "paragraph" and block.word_count <= 30:
            return 1  # Short, important content
        elif block.content_type == "list":
            return 2  # Medium priority
        elif block.content_type == "code":
            return 3  # Lower priority on mobile
        else:
            return 2  # Default medium priority
    
    def _apply_mobile_priority_order(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """Reorder content blocks based on mobile reading priorities"""
        # Keep original order but group by priority for later responsive stacking
        high_priority = [b for b in blocks if b.priority == 1]
        medium_priority = [b for b in blocks if b.priority == 2]  
        low_priority = [b for b in blocks if b.priority == 3]
        
        # For mobile: high priority content first, then medium, then low
        # But maintain reading flow within each priority level
        return high_priority + medium_priority + low_priority
    
    def _generate_responsive_structure(self, blocks: List[ContentBlock]) -> str:
        """Generate responsive content structure"""
        mobile_content = []
        
        for block in blocks:
            # Add mobile-optimized content with proper spacing
            mobile_content.append(block.content)
            
            # Add appropriate spacing between blocks for mobile
            if block.content_type == "heading":
                mobile_content.append("")  # Space after headings
            elif block.content_type == "paragraph":
                mobile_content.append("")  # Space between paragraphs
            # Lists and code get minimal spacing
        
        return '\n'.join(mobile_content)
    
    def _calculate_mobile_reading_time(self, blocks: List[ContentBlock]) -> int:
        """Calculate reading time optimized for mobile consumption"""
        total_words = sum(block.word_count for block in blocks)
        
        # Mobile reading speed is slower: ~150 WPM vs desktop ~200 WPM
        mobile_wpm = 150
        reading_time = max(1, total_words // mobile_wpm)
        
        return reading_time

class ResponsiveHTMLGenerator:
    """Generates responsive HTML optimized for multiple devices"""
    
    def __init__(self):
        self.css_framework = "mobile-first"
        
    def generate_responsive_html(self, content: str, config: FormatOptimizationConfig) -> str:
        """Generate responsive HTML from optimized content"""
        
        # Parse content blocks
        optimizer = MobileFirstOptimizer()
        optimization_result = optimizer.optimize_for_mobile(content)
        
        # Generate HTML structure
        html_template = self._get_responsive_template(config)
        
        # Convert content blocks to HTML
        content_html = self._convert_blocks_to_html(
            optimization_result['content_blocks'], 
            config
        )
        
        # Apply responsive CSS
        responsive_css = self._generate_responsive_css(config)
        
        # Assemble final HTML
        final_html = html_template.format(
            title="Newsletter Content",
            responsive_css=responsive_css,
            content_html=content_html,
            reading_time=optimization_result['mobile_reading_time']
        )
        
        return final_html
    
    def _get_responsive_template(self, config: FormatOptimizationConfig) -> str:
        """Get base responsive HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="format-detection" content="telephone=no">
    <title>{title}</title>
    <style>
        {responsive_css}
    </style>
</head>
<body>
    <div class="newsletter-container">
        <div class="reading-time">📱 {reading_time} min read (mobile optimized)</div>
        <main class="content">
            {content_html}
        </main>
    </div>
</body>
</html>'''
    
    def _convert_blocks_to_html(self, blocks: List[ContentBlock], config: FormatOptimizationConfig) -> str:
        """Convert content blocks to responsive HTML"""
        html_parts = []
        
        for block in blocks:
            if block.content_type == "heading":
                html_parts.append(self._format_responsive_heading(block))
            elif block.content_type == "paragraph":
                html_parts.append(self._format_responsive_paragraph(block))
            elif block.content_type == "list":
                html_parts.append(self._format_responsive_list(block))
            elif block.content_type == "code":
                html_parts.append(self._format_responsive_code(block))
            elif block.content_type == "quote":
                html_parts.append(self._format_responsive_quote(block))
        
        return '\n'.join(html_parts)
    
    def _format_responsive_heading(self, block: ContentBlock) -> str:
        """Format heading with responsive typography"""
        heading_level = len(block.content.split()[0]) if block.content.startswith('#') else 2
        heading_text = block.content.lstrip('#').strip()
        
        return f'''<h{heading_level} class="responsive-heading priority-{block.priority}">
            {heading_text}
        </h{heading_level}>'''
    
    def _format_responsive_paragraph(self, block: ContentBlock) -> str:
        """Format paragraph with mobile-optimized typography"""
        return f'''<p class="responsive-paragraph priority-{block.priority}">
            {block.content}
        </p>'''
    
    def _format_responsive_list(self, block: ContentBlock) -> str:
        """Format list with touch-friendly spacing"""
        list_items = []
        for line in block.content.split('\n'):
            if line.strip().startswith('*') or line.strip().startswith('-'):
                item_text = line.strip()[1:].strip()
                list_items.append(f'<li class="responsive-list-item">{item_text}</li>')
        
        return f'''<ul class="responsive-list priority-{block.priority}">
            {chr(10).join(list_items)}
        </ul>'''
    
    def _format_responsive_code(self, block: ContentBlock) -> str:
        """Format code with horizontal scroll support"""
        code_content = block.content.strip('`')
        
        return f'''<div class="responsive-code-container priority-{block.priority}">
            <pre><code class="responsive-code">{code_content}</code></pre>
            <div class="mobile-scroll-hint">← Scroll horizontally →</div>
        </div>'''
    
    def _format_responsive_quote(self, block: ContentBlock) -> str:
        """Format quote with mobile-friendly styling"""
        quote_text = block.content.lstrip('>').strip()
        
        return f'''<blockquote class="responsive-quote priority-{block.priority}">
            {quote_text}
        </blockquote>'''
    
    def _generate_responsive_css(self, config: FormatOptimizationConfig) -> str:
        """Generate mobile-first responsive CSS"""
        return '''
        /* Mobile-first responsive styles */
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 16px;
            font-size: 16px;
            color: #333;
            background: #fff;
        }
        
        .newsletter-container {
            max-width: 100%;
            margin: 0 auto;
        }
        
        .reading-time {
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .responsive-heading {
            font-weight: 600;
            margin: 24px 0 12px 0;
            line-height: 1.3;
        }
        
        .responsive-paragraph {
            margin-bottom: 16px;
            line-height: 1.7;
        }
        
        .responsive-list {
            padding-left: 20px;
            margin-bottom: 16px;
        }
        
        .responsive-list-item {
            margin-bottom: 8px;
            line-height: 1.6;
        }
        
        .responsive-code-container {
            background: #f8f8f8;
            border-radius: 6px;
            margin: 16px 0;
            overflow-x: auto;
        }
        
        .responsive-code {
            padding: 16px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            white-space: pre;
            overflow-x: auto;
        }
        
        .mobile-scroll-hint {
            text-align: center;
            font-size: 12px;
            color: #666;
            padding: 8px;
            background: #f0f0f0;
        }
        
        .responsive-quote {
            border-left: 4px solid #007AFF;
            padding-left: 16px;
            margin: 16px 0;
            font-style: italic;
            color: #555;
        }
        
        /* Tablet styles */
        @media (min-width: 768px) {
            body {
                padding: 24px;
                font-size: 17px;
            }
            
            .newsletter-container {
                max-width: 680px;
            }
            
            .responsive-heading {
                margin: 32px 0 16px 0;
            }
        }
        
        /* Desktop styles */
        @media (min-width: 1024px) {
            body {
                padding: 32px;
                font-size: 18px;
            }
            
            .newsletter-container {
                max-width: 760px;
            }
            
            .reading-time {
                position: sticky;
                top: 0;
                z-index: 100;
            }
        }
        
        /* Priority-based mobile stacking */
        @media (max-width: 767px) {
            .priority-3 {
                margin-top: 24px;
                padding-top: 16px;
                border-top: 1px solid #eee;
            }
        }
        '''

class MultiPlatformAdapter:
    """Adapts content for different platforms and formats"""
    
    def __init__(self):
        self.format_adapters = {
            ContentFormat.MARKDOWN: self._adapt_to_markdown,
            ContentFormat.HTML: self._adapt_to_html, 
            ContentFormat.EMAIL_HTML: self._adapt_to_email_html,
            ContentFormat.NOTION: self._adapt_to_notion,
            ContentFormat.PLAIN_TEXT: self._adapt_to_plain_text
        }
    
    def adapt_content(self, content: str, target_format: ContentFormat, 
                     device_config: FormatOptimizationConfig) -> Dict[str, Any]:
        """Adapt content to target format and device configuration"""
        
        logger.info(f"Adapting content for {target_format.value} on {device_config.target_device.value}")
        
        # Apply mobile optimization first
        mobile_optimizer = MobileFirstOptimizer()
        mobile_result = mobile_optimizer.optimize_for_mobile(content)
        
        # Apply format-specific adaptation
        adapter_func = self.format_adapters.get(target_format)
        if not adapter_func:
            raise ValueError(f"Unsupported format: {target_format}")
        
        adapted_content = adapter_func(mobile_result['mobile_optimized_content'], device_config)
        
        return {
            'adapted_content': adapted_content,
            'target_format': target_format.value,
            'target_device': device_config.target_device.value,
            'mobile_optimized': True,
            'reading_time': mobile_result['mobile_reading_time'],
            'optimization_config': device_config
        }
    
    def _adapt_to_markdown(self, content: str, config: FormatOptimizationConfig) -> str:
        """Adapt content to mobile-optimized Markdown"""
        # Markdown is already mobile-friendly, just ensure proper formatting
        lines = content.split('\n')
        adapted_lines = []
        
        for line in lines:
            # Ensure mobile-friendly line lengths
            if len(line) > config.max_line_length and not line.startswith('#'):
                # Wrap long lines for mobile readability
                wrapped = self._wrap_line_for_mobile(line, config.max_line_length)
                adapted_lines.extend(wrapped)
            else:
                adapted_lines.append(line)
        
        return '\n'.join(adapted_lines)
    
    def _wrap_line_for_mobile(self, line: str, max_length: int) -> List[str]:
        """Wrap long lines for mobile readability"""
        if len(line) <= max_length:
            return [line]
        
        words = line.split()
        wrapped_lines = []
        current_line = ""
        
        for word in words:
            test_line = (current_line + " " + word).strip()
            if len(test_line) > max_length and current_line:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            wrapped_lines.append(current_line)
        
        return wrapped_lines
    
    def _adapt_to_html(self, content: str, config: FormatOptimizationConfig) -> str:
        """Adapt content to responsive HTML"""
        html_generator = ResponsiveHTMLGenerator()
        return html_generator.generate_responsive_html(content, config)
    
    def _adapt_to_email_html(self, content: str, config: FormatOptimizationConfig) -> str:
        """Adapt content to email-compatible HTML"""
        # Email HTML has more restrictions - inline styles, limited CSS support
        html_generator = ResponsiveHTMLGenerator()
        base_html = html_generator.generate_responsive_html(content, config)
        
        # Convert to email-compatible format
        email_html = self._convert_to_email_html(base_html)
        return email_html
    
    def _convert_to_email_html(self, html: str) -> str:
        """Convert responsive HTML to email client compatible format"""
        # Inline all styles for email compatibility
        # This is a simplified version - in production, use tools like premailer
        
        inline_styles = {
            'body': 'font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 16px; font-size: 16px; color: #333;',
            'h1': 'font-weight: 600; margin: 24px 0 12px 0; line-height: 1.3; font-size: 24px;',
            'h2': 'font-weight: 600; margin: 24px 0 12px 0; line-height: 1.3; font-size: 20px;',
            'h3': 'font-weight: 600; margin: 24px 0 12px 0; line-height: 1.3; font-size: 18px;',
            'p': 'margin-bottom: 16px; line-height: 1.7;',
            'ul': 'padding-left: 20px; margin-bottom: 16px;',
            'li': 'margin-bottom: 8px; line-height: 1.6;',
            'pre': 'background: #f8f8f8; padding: 16px; border-radius: 6px; overflow-x: auto;',
            'blockquote': 'border-left: 4px solid #007AFF; padding-left: 16px; margin: 16px 0; font-style: italic; color: #555;'
        }
        
        # Simple inline style injection
        for tag, style in inline_styles.items():
            html = html.replace(f'<{tag}>', f'<{tag} style="{style}">')
        
        return html
    
    def _adapt_to_notion(self, content: str, config: FormatOptimizationConfig) -> str:
        """Adapt content to Notion format"""
        # Convert to Notion-compatible format
        lines = content.split('\n')
        notion_lines = []
        
        for line in lines:
            if line.startswith('#'):
                # Convert headings to Notion format
                level = len(line.split()[0])
                text = line.lstrip('#').strip()
                notion_lines.append(f"{'#' * min(level, 3)} {text}")
            elif line.startswith('*') or line.startswith('-'):
                # Convert to Notion bullet format
                text = line.lstrip('*-').strip()
                notion_lines.append(f"• {text}")
            elif line.startswith('```'):
                # Convert code blocks
                notion_lines.append("```")
            else:
                notion_lines.append(line)
        
        return '\n'.join(notion_lines)
    
    def _adapt_to_plain_text(self, content: str, config: FormatOptimizationConfig) -> str:
        """Adapt content to plain text format"""
        # Remove all markdown formatting for plain text
        plain_text = content
        
        # Remove markdown syntax
        plain_text = re.sub(r'#+\s*', '', plain_text)  # Remove heading markers
        plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_text)  # Remove bold
        plain_text = re.sub(r'\*(.*?)\*', r'\1', plain_text)  # Remove italic
        plain_text = re.sub(r'`(.*?)`', r'\1', plain_text)  # Remove inline code
        plain_text = re.sub(r'```.*?```', '[CODE BLOCK]', plain_text, flags=re.DOTALL)  # Replace code blocks
        
        # Clean up extra whitespace
        plain_text = re.sub(r'\n\s*\n', '\n\n', plain_text)  # Clean multiple newlines
        
        return plain_text.strip()

class ContentFormatOptimizer:
    """Main coordinator for content format optimization"""
    
    def __init__(self):
        self.mobile_optimizer = MobileFirstOptimizer()
        self.html_generator = ResponsiveHTMLGenerator()
        self.platform_adapter = MultiPlatformAdapter()
        
    def optimize_content_for_all_formats(self, content: str, 
                                       target_devices: List[DeviceType] = None,
                                       target_formats: List[ContentFormat] = None) -> Dict[str, Any]:
        """Optimize content for all specified devices and formats"""
        
        if target_devices is None:
            target_devices = [DeviceType.MOBILE, DeviceType.TABLET, DeviceType.DESKTOP]
            
        if target_formats is None:
            target_formats = [ContentFormat.MARKDOWN, ContentFormat.HTML, ContentFormat.EMAIL_HTML]
        
        logger.info(f"Optimizing content for {len(target_devices)} devices and {len(target_formats)} formats")
        
        optimization_results = {}
        
        for device in target_devices:
            device_results = {}
            
            for format_type in target_formats:
                config = FormatOptimizationConfig(
                    target_device=device,
                    output_format=format_type,
                    mobile_first=True,
                    optimize_images=True
                )
                
                # Adapt content for this device/format combination
                adaptation_result = self.platform_adapter.adapt_content(content, format_type, config)
                
                device_results[format_type.value] = adaptation_result
            
            optimization_results[device.value] = device_results
        
        # Generate summary metrics
        summary = self._generate_optimization_summary(optimization_results)
        
        return {
            'optimized_content': optimization_results,
            'summary': summary,
            'mobile_first_applied': True,
            'responsive_design': True
        }
    
    def _generate_optimization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of optimization results"""
        
        total_adaptations = sum(len(device_results) for device_results in results.values())
        
        # Calculate average reading times by device
        device_reading_times = {}
        for device, device_results in results.items():
            times = [result['reading_time'] for result in device_results.values() if 'reading_time' in result]
            if times:
                device_reading_times[device] = sum(times) / len(times)
        
        return {
            'total_adaptations': total_adaptations,
            'devices_optimized': list(results.keys()),
            'formats_generated': list(set(
                format_name for device_results in results.values() 
                for format_name in device_results.keys()
            )),
            'average_reading_times': device_reading_times,
            'mobile_priority_applied': True,
            'responsive_breakpoints': ['480px', '768px', '1024px']
        } 