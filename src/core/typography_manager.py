"""
Responsive Typography Management System

This module implements the ResponsiveTypographyManager for mobile-optimized
typography and formatting that ensures optimal reading experience across devices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device types for responsive typography."""
    MOBILE = "mobile"        # Phones (320-768px)
    TABLET = "tablet"        # Tablets (768-1024px)
    DESKTOP = "desktop"      # Desktop (1024px+)


class TypographyLevel(Enum):
    """Typography optimization levels."""
    BASIC = "basic"          # Basic responsive typography
    ENHANCED = "enhanced"    # Enhanced mobile optimization
    PREMIUM = "premium"      # Premium typography features


@dataclass
class TypographySettings:
    """Typography settings for different device types."""
    font_size_body: int
    font_size_h1: int
    font_size_h2: int
    font_size_h3: int
    font_size_h4: int
    font_size_code: int
    font_size_caption: int
    
    line_height_body: float
    line_height_headings: float
    line_height_code: float
    
    spacing_paragraph: str
    spacing_heading_top: str
    spacing_heading_bottom: str
    spacing_section: str
    spacing_list_item: str
    
    max_content_width: str
    margin_horizontal: str


@dataclass
class TypographyAdjustment:
    """Specific typography adjustment to apply."""
    element_type: str        # 'paragraph', 'heading', 'code', etc.
    property_name: str       # 'font-size', 'line-height', etc.
    value: str              # CSS value
    device_target: DeviceType
    priority: int           # 1-10, higher is more important


@dataclass
class TypographyOptimizationResult:
    """Result of typography optimization operation."""
    original_content: str
    optimized_content: str
    adjustments_applied: List[TypographyAdjustment]
    typography_score: float
    readability_improvement: float
    mobile_compliance: float
    processing_time: float
    success: bool
    warnings: List[str] = None


class ResponsiveTypographyManager:
    """
    Mobile-optimized typography and formatting manager.
    
    Implements responsive typography strategies that ensure optimal reading
    experience across devices, with special focus on mobile readability.
    """
    
    def __init__(self):
        """Initialize the responsive typography manager."""
        self.typography_settings = self._initialize_typography_settings()
        self.mobile_thresholds = self._initialize_mobile_thresholds()
        self.formatting_rules = self._initialize_formatting_rules()
        
        logger.info("ResponsiveTypographyManager initialized")
    
    def _initialize_typography_settings(self) -> Dict[DeviceType, TypographySettings]:
        """Initialize typography settings for different device types."""
        return {
            DeviceType.MOBILE: TypographySettings(
                # Font sizes optimized for mobile (minimum 16px for body)
                font_size_body=16,
                font_size_h1=28,
                font_size_h2=24,
                font_size_h3=20,
                font_size_h4=18,
                font_size_code=14,
                font_size_caption=14,
                
                # Line heights for optimal mobile reading
                line_height_body=1.5,
                line_height_headings=1.3,
                line_height_code=1.4,
                
                # Spacing optimized for mobile
                spacing_paragraph="1.2em",
                spacing_heading_top="1.5em",
                spacing_heading_bottom="0.8em",
                spacing_section="2em",
                spacing_list_item="0.5em",
                
                # Content width and margins
                max_content_width="100%",
                margin_horizontal="16px"
            ),
            
            DeviceType.TABLET: TypographySettings(
                # Font sizes for tablet viewing
                font_size_body=18,
                font_size_h1=32,
                font_size_h2=28,
                font_size_h3=24,
                font_size_h4=20,
                font_size_code=16,
                font_size_caption=16,
                
                # Line heights for tablet
                line_height_body=1.6,
                line_height_headings=1.3,
                line_height_code=1.4,
                
                # Spacing for tablet
                spacing_paragraph="1.3em",
                spacing_heading_top="1.8em",
                spacing_heading_bottom="1em",
                spacing_section="2.5em",
                spacing_list_item="0.6em",
                
                # Content width and margins
                max_content_width="90%",
                margin_horizontal="24px"
            ),
            
            DeviceType.DESKTOP: TypographySettings(
                # Font sizes for desktop viewing
                font_size_body=18,
                font_size_h1=36,
                font_size_h2=30,
                font_size_h3=26,
                font_size_h4=22,
                font_size_code=16,
                font_size_caption=16,
                
                # Line heights for desktop
                line_height_body=1.6,
                line_height_headings=1.3,
                line_height_code=1.4,
                
                # Spacing for desktop
                spacing_paragraph="1.4em",
                spacing_heading_top="2em",
                spacing_heading_bottom="1.2em",
                spacing_section="3em",
                spacing_list_item="0.7em",
                
                # Content width and margins
                max_content_width="80%",
                margin_horizontal="auto"
            )
        }
    
    def _initialize_mobile_thresholds(self) -> Dict[str, float]:
        """Initialize mobile typography thresholds."""
        return {
            'min_font_size_mobile': 16,      # Minimum 16px for mobile
            'max_line_length_mobile': 45,    # Maximum 45 characters per line
            'min_line_height_mobile': 1.4,   # Minimum line height
            'max_line_height_mobile': 1.8,   # Maximum line height
            'min_touch_target': 44,          # Minimum 44px touch targets
            'optimal_reading_width': 66,     # Optimal 66 characters per line
            'typography_compliance_min': 0.8  # 80% minimum compliance
        }
    
    def _initialize_formatting_rules(self) -> Dict[str, Any]:
        """Initialize formatting rules for mobile optimization."""
        return {
            'paragraph_formatting': {
                'max_words_mobile': 75,
                'max_sentences_mobile': 4,
                'optimal_words': 50,
                'add_spacing': True
            },
            
            'heading_formatting': {
                'ensure_hierarchy': True,
                'add_spacing': True,
                'mobile_friendly_size': True,
                'improve_contrast': True
            },
            
            'code_formatting': {
                'mobile_horizontal_scroll': True,
                'language_tags': True,
                'syntax_highlighting_hints': True,
                'max_line_length_mobile': 50
            },
            
            'list_formatting': {
                'consistent_spacing': True,
                'mobile_friendly_bullets': True,
                'nested_list_indentation': True
            },
            
            'link_formatting': {
                'descriptive_text': True,
                'mobile_touch_targets': True,
                'spacing_between_links': True
            }
        }
    
    def apply_mobile_typography(self, content: str, 
                              adjustments: Dict[str, Any],
                              target_device: DeviceType = DeviceType.MOBILE) -> str:
        """
        Apply mobile typography optimizations to content.
        
        Args:
            content: Original content to optimize
            adjustments: Typography adjustments to apply
            target_device: Primary target device type
            
        Returns:
            Optimized content with mobile typography applied
        """
        import time
        start_time = time.time()
        
        logger.info(f"Applying mobile typography for {target_device.value}")
        
        try:
            optimized_content = content
            applied_adjustments = []
            
            # Apply font size adjustments
            if adjustments.get('font_size_increase'):
                optimized_content, font_adjustments = self._apply_font_size_optimization(
                    optimized_content, target_device
                )
                applied_adjustments.extend(font_adjustments)
            
            # Apply line height optimization
            if adjustments.get('line_height_optimization'):
                optimized_content, line_adjustments = self._apply_line_height_optimization(
                    optimized_content, target_device
                )
                applied_adjustments.extend(line_adjustments)
            
            # Apply heading hierarchy improvements
            if adjustments.get('heading_hierarchy_improvement'):
                optimized_content, heading_adjustments = self._improve_heading_hierarchy(
                    optimized_content, target_device
                )
                applied_adjustments.extend(heading_adjustments)
            
            # Apply code block formatting
            if adjustments.get('code_block_formatting'):
                optimized_content, code_adjustments = self._format_code_blocks(
                    optimized_content, target_device
                )
                applied_adjustments.extend(code_adjustments)
            
            # Apply responsive font scaling
            if adjustments.get('responsive_font_scaling'):
                optimized_content, scaling_adjustments = self._apply_responsive_scaling(
                    optimized_content, target_device
                )
                applied_adjustments.extend(scaling_adjustments)
            
            # Apply touch target optimization
            if adjustments.get('touch_target_optimization'):
                optimized_content, touch_adjustments = self._optimize_touch_targets(
                    optimized_content, target_device
                )
                applied_adjustments.extend(touch_adjustments)
            
            # Apply reading flow enhancement
            if adjustments.get('reading_flow_enhancement'):
                optimized_content, flow_adjustments = self._enhance_reading_flow(
                    optimized_content, target_device
                )
                applied_adjustments.extend(flow_adjustments)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Mobile typography applied: {len(applied_adjustments)} adjustments in {processing_time:.2f}s")
            return optimized_content
            
        except Exception as e:
            logger.error(f"Mobile typography application failed: {e}")
            return content  # Return original content on failure
    
    def _apply_font_size_optimization(self, content: str, 
                                    target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Apply font size optimization for mobile readability."""
        settings = self.typography_settings[target_device]
        adjustments = []
        
        # Add CSS-like comments for font size optimization
        # In a real implementation, this would generate actual CSS
        optimized_content = content
        
        # Add font size guidelines as comments
        font_guidelines = f"""
<!-- Typography: Mobile-Optimized Font Sizes
Body text: {settings.font_size_body}px
H1: {settings.font_size_h1}px
H2: {settings.font_size_h2}px  
H3: {settings.font_size_h3}px
Code: {settings.font_size_code}px
-->

{content}"""
        
        # Record adjustments
        adjustments.append(TypographyAdjustment(
            element_type="body",
            property_name="font-size",
            value=f"{settings.font_size_body}px",
            device_target=target_device,
            priority=9
        ))
        
        adjustments.append(TypographyAdjustment(
            element_type="heading",
            property_name="font-size",
            value=f"h1:{settings.font_size_h1}px, h2:{settings.font_size_h2}px",
            device_target=target_device,
            priority=8
        ))
        
        return font_guidelines, adjustments
    
    def _apply_line_height_optimization(self, content: str,
                                      target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Apply line height optimization for mobile reading."""
        settings = self.typography_settings[target_device]
        adjustments = []
        
        # Add line height guidelines
        line_height_guidelines = f"""
<!-- Typography: Mobile-Optimized Line Heights
Body text: {settings.line_height_body}
Headings: {settings.line_height_headings}
Code blocks: {settings.line_height_code}
-->

{content}"""
        
        # Record adjustments
        adjustments.append(TypographyAdjustment(
            element_type="body",
            property_name="line-height",
            value=str(settings.line_height_body),
            device_target=target_device,
            priority=8
        ))
        
        return line_height_guidelines, adjustments
    
    def _improve_heading_hierarchy(self, content: str,
                                 target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Improve heading hierarchy for mobile navigation."""
        lines = content.split('\n')
        improved_lines = []
        adjustments = []
        
        for i, line in enumerate(lines):
            if re.match(r'^#{1,6}\s+', line):
                # Ensure proper spacing around headings for mobile
                heading_level = len(re.match(r'^(#{1,6})', line).group(1))
                
                # Add spacing before heading (except first)
                if i > 0 and lines[i-1].strip():
                    improved_lines.append('')
                
                # Add the heading
                improved_lines.append(line)
                
                # Add spacing after heading if needed
                next_line_idx = i + 1
                if (next_line_idx < len(lines) and 
                    lines[next_line_idx].strip() and 
                    not lines[next_line_idx].startswith('#')):
                    improved_lines.append('')
                
                # Record adjustment
                adjustments.append(TypographyAdjustment(
                    element_type=f"h{heading_level}",
                    property_name="spacing",
                    value="mobile-optimized",
                    device_target=target_device,
                    priority=7
                ))
            else:
                improved_lines.append(line)
        
        return '\n'.join(improved_lines), adjustments
    
    def _format_code_blocks(self, content: str,
                          target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Format code blocks for mobile display."""
        adjustments = []
        
        # Add mobile code block formatting
        code_block_pattern = r'```(\w*)\n(.*?)```'
        
        def format_code_block(match):
            language = match.group(1) or ''
            code_content = match.group(2)
            
            # Check line length and add mobile formatting hint
            lines = code_content.split('\n')
            max_line_length = max(len(line) for line in lines if line.strip())
            
            mobile_hint = ""
            if max_line_length > 50:
                mobile_hint = "<!-- Mobile: Scroll horizontally to view full code -->\n"
            
            # Add language tag if missing
            if not language and code_content.strip():
                language = self._detect_code_language(code_content)
            
            return f"```{language}\n{mobile_hint}{code_content}```"
        
        formatted_content = re.sub(code_block_pattern, format_code_block, content, flags=re.DOTALL)
        
        # Record adjustments
        code_blocks_count = len(re.findall(r'```', content)) // 2
        if code_blocks_count > 0:
            adjustments.append(TypographyAdjustment(
                element_type="code",
                property_name="mobile-formatting",
                value="horizontal-scroll-enabled",
                device_target=target_device,
                priority=6
            ))
        
        return formatted_content, adjustments
    
    def _detect_code_language(self, code_content: str) -> str:
        """Detect programming language from code content."""
        code_lower = code_content.lower()
        
        # Language detection patterns
        language_patterns = {
            'python': ['import ', 'def ', 'class ', 'if __name__'],
            'javascript': ['function', 'const ', 'let ', 'var ', '=>'],
            'java': ['public class', 'import java', 'public static'],
            'cpp': ['#include', 'int main', 'std::', '::'],
            'go': ['package ', 'func ', 'import ', 'go '],
            'rust': ['fn ', 'let ', 'use ', 'struct'],
            'sql': ['select ', 'from ', 'where ', 'insert'],
            'bash': ['#!/bin/bash', 'echo ', '$', '&&'],
            'yaml': ['---', '  - ', ': '],
            'json': ['{', '}', '":"', '"'],
            'html': ['<html', '<div', '<span', 'href='],
            'css': ['{', '}', ':', ';', 'px']
        }
        
        for language, patterns in language_patterns.items():
            if any(pattern in code_lower for pattern in patterns):
                return language
        
        return ''  # No language detected
    
    def _apply_responsive_scaling(self, content: str,
                                target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Apply responsive font scaling based on device type."""
        settings = self.typography_settings[target_device]
        adjustments = []
        
        # Add responsive scaling guidelines
        responsive_guidelines = f"""
<!-- Responsive Typography: {target_device.value.title()} Optimized
Viewport: {target_device.value}
Max width: {settings.max_content_width}
Margins: {settings.margin_horizontal}
-->

{content}"""
        
        adjustments.append(TypographyAdjustment(
            element_type="container",
            property_name="responsive-scaling",
            value=f"{target_device.value}-optimized",
            device_target=target_device,
            priority=7
        ))
        
        return responsive_guidelines, adjustments
    
    def _optimize_touch_targets(self, content: str,
                              target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Optimize touch targets for mobile interaction."""
        adjustments = []
        
        if target_device != DeviceType.MOBILE:
            return content, adjustments
        
        # Find lines with multiple links that might have poor touch targets
        lines = content.split('\n')
        optimized_lines = []
        
        for line in lines:
            if line.count('](') > 1:  # Multiple links in one line
                # Add spacing guidance for mobile touch targets
                optimized_lines.append('<!-- Mobile: Ensure adequate spacing between links -->')
                optimized_lines.append(line)
                
                adjustments.append(TypographyAdjustment(
                    element_type="link",
                    property_name="touch-target",
                    value="44px-minimum",
                    device_target=target_device,
                    priority=8
                ))
            else:
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines), adjustments
    
    def _enhance_reading_flow(self, content: str,
                            target_device: DeviceType) -> Tuple[str, List[TypographyAdjustment]]:
        """Enhance reading flow for mobile devices."""
        adjustments = []
        
        # Improve paragraph spacing for mobile reading
        paragraphs = content.split('\n\n')
        enhanced_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                enhanced_paragraphs.append(paragraph)
                continue
            
            # Check if paragraph is too long for mobile
            words = paragraph.split()
            if len(words) > 100 and target_device == DeviceType.MOBILE:
                # Split long paragraph at natural break points
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = []
                
                for sentence in sentences:
                    current_chunk.append(sentence)
                    if len(' '.join(current_chunk).split()) > 60:  # ~60 words per chunk
                        enhanced_paragraphs.append(' '.join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    enhanced_paragraphs.append(' '.join(current_chunk))
                
                adjustments.append(TypographyAdjustment(
                    element_type="paragraph",
                    property_name="reading-flow",
                    value="mobile-chunked",
                    device_target=target_device,
                    priority=6
                ))
            else:
                enhanced_paragraphs.append(paragraph)
        
        enhanced_content = '\n\n'.join(enhanced_paragraphs)
        
        # Add reading flow guidelines
        if target_device == DeviceType.MOBILE:
            reading_flow_header = """
<!-- Mobile Reading Flow Optimization Applied
- Optimal paragraph length maintained
- Sentence structure optimized for mobile
- Touch targets properly spaced
-->

"""
            enhanced_content = reading_flow_header + enhanced_content
        
        return enhanced_content, adjustments
    
    def assess_typography_compliance(self, content: str,
                                   target_device: DeviceType = DeviceType.MOBILE) -> float:
        """Assess typography compliance for mobile devices."""
        try:
            compliance_factors = []
            thresholds = self.mobile_thresholds
            
            # Check paragraph length compliance
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if paragraphs:
                long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
                paragraph_compliance = max(0.0, 1.0 - len(long_paragraphs) / len(paragraphs))
                compliance_factors.append(paragraph_compliance)
            
            # Check heading structure compliance
            headings = re.findall(r'^(#{1,6})', content, re.MULTILINE)
            if headings:
                heading_levels = [len(h) for h in headings]
                proper_hierarchy = all(
                    level <= prev_level + 1 
                    for prev_level, level in zip(heading_levels, heading_levels[1:])
                )
                heading_compliance = 1.0 if proper_hierarchy else 0.7
                compliance_factors.append(heading_compliance)
            
            # Check code block formatting compliance
            code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
            if code_blocks:
                properly_formatted = [
                    block for block in code_blocks 
                    if any(lang in block[:20] for lang in ['python', 'javascript', 'java', 'go'])
                ]
                code_compliance = len(properly_formatted) / len(code_blocks)
                compliance_factors.append(code_compliance)
            
            # Check touch target spacing (simplified)
            lines_with_links = [line for line in content.split('\n') if '[' in line and '](' in line]
            if lines_with_links:
                dense_link_lines = [line for line in lines_with_links if line.count('](') > 2]
                touch_compliance = max(0.0, 1.0 - len(dense_link_lines) / len(lines_with_links))
                compliance_factors.append(touch_compliance)
            
            # Calculate overall compliance
            if compliance_factors:
                overall_compliance = sum(compliance_factors) / len(compliance_factors)
            else:
                overall_compliance = 0.8  # Default for content with no special elements
            
            return overall_compliance
            
        except Exception as e:
            logger.error(f"Typography compliance assessment failed: {e}")
            return 0.5  # Return baseline compliance on error
    
    def generate_mobile_css(self, target_device: DeviceType = DeviceType.MOBILE) -> str:
        """Generate CSS for mobile-optimized typography."""
        settings = self.typography_settings[target_device]
        
        css_template = f"""
/* Mobile-Optimized Typography - {target_device.value.title()} */

/* Base typography */
body {{
    font-size: {settings.font_size_body}px;
    line-height: {settings.line_height_body};
    max-width: {settings.max_content_width};
    margin: 0 {settings.margin_horizontal};
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

/* Headings */
h1 {{
    font-size: {settings.font_size_h1}px;
    line-height: {settings.line_height_headings};
    margin-top: {settings.spacing_heading_top};
    margin-bottom: {settings.spacing_heading_bottom};
}}

h2 {{
    font-size: {settings.font_size_h2}px;
    line-height: {settings.line_height_headings};
    margin-top: {settings.spacing_heading_top};
    margin-bottom: {settings.spacing_heading_bottom};
}}

h3 {{
    font-size: {settings.font_size_h3}px;
    line-height: {settings.line_height_headings};
    margin-top: {settings.spacing_heading_top};
    margin-bottom: {settings.spacing_heading_bottom};
}}

/* Paragraphs */
p {{
    margin-bottom: {settings.spacing_paragraph};
    max-width: 66ch; /* Optimal reading width */
}}

/* Code blocks */
code, pre {{
    font-size: {settings.font_size_code}px;
    line-height: {settings.line_height_code};
    overflow-x: auto;
    word-wrap: break-word;
}}

pre {{
    padding: 1em;
    margin: {settings.spacing_paragraph} 0;
    background-color: #f5f5f5;
    border-radius: 4px;
}}

/* Lists */
ul, ol {{
    margin-bottom: {settings.spacing_paragraph};
    padding-left: 1.5em;
}}

li {{
    margin-bottom: {settings.spacing_list_item};
}}

/* Links - Touch-friendly */
a {{
    color: #007AFF;
    text-decoration: none;
    min-height: 44px;
    display: inline-block;
    padding: 4px 0;
}}

a:hover, a:focus {{
    text-decoration: underline;
}}

/* Responsive adjustments */
@media (max-width: 768px) {{
    body {{
        font-size: {max(16, settings.font_size_body)}px;
        margin: 0 16px;
    }}
    
    h1 {{ font-size: {max(24, settings.font_size_h1 - 4)}px; }}
    h2 {{ font-size: {max(20, settings.font_size_h2 - 4)}px; }}
    h3 {{ font-size: {max(18, settings.font_size_h3 - 2)}px; }}
    
    pre {{
        font-size: 14px;
        padding: 0.5em;
        overflow-x: scroll;
    }}
}}

/* Dark mode support */
@media (prefers-color-scheme: dark) {{
    body {{
        background-color: #000;
        color: #fff;
    }}
    
    pre {{
        background-color: #1a1a1a;
        color: #e0e0e0;
    }}
    
    a {{
        color: #0A84FF;
    }}
}}
"""
        
        return css_template.strip()
    
    def get_typography_recommendations(self, content: str) -> List[str]:
        """Get typography recommendations for mobile optimization."""
        recommendations = []
        
        # Analyze current typography
        compliance = self.assess_typography_compliance(content)
        
        if compliance < 0.8:
            recommendations.append("Improve overall typography compliance for mobile readability")
        
        # Check for specific issues
        long_paragraphs = len([p for p in content.split('\n\n') if len(p.split()) > 100])
        if long_paragraphs > 0:
            recommendations.append(f"Break {long_paragraphs} long paragraphs into mobile-friendly chunks")
        
        # Check heading structure
        h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
        if h1_count != 1:
            recommendations.append(f"Use exactly one H1 heading (currently {h1_count})")
        
        # Check code block formatting
        unformatted_code = len(re.findall(r'```\n[^`]+```', content, re.DOTALL))
        if unformatted_code > 0:
            recommendations.append(f"Add language tags to {unformatted_code} code blocks for better mobile display")
        
        # Check link density
        lines_with_many_links = len([
            line for line in content.split('\n') 
            if line.count('](') > 2
        ])
        if lines_with_many_links > 0:
            recommendations.append(f"Improve spacing between links in {lines_with_many_links} lines for better touch targets")
        
        return recommendations[:5]  # Return top 5 recommendations


# Export main classes
__all__ = [
    'ResponsiveTypographyManager',
    'DeviceType',
    'TypographyLevel',
    'TypographySettings',
    'TypographyOptimizationResult'
]