"""
Mobile-First Content Optimization System

This module implements the MobileContentOptimizer for comprehensive mobile optimization
to achieve 95%+ mobile readability targets while maintaining professional quality
for technical content.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.advanced_quality_gates import ConfigurableQualityGate, QualityDimension
from core.readability_analyzer import MobileReadabilityAnalyzer
from core.template_manager import NewsletterTemplate, NewsletterType
from core.typography_manager import ResponsiveTypographyManager

logger = logging.getLogger(__name__)


class MobileOptimizationLevel(Enum):
    """Mobile optimization levels with different intensity."""
    BASIC = "basic"          # Basic mobile compatibility
    STANDARD = "standard"    # Standard mobile optimization
    ENHANCED = "enhanced"    # Enhanced mobile experience
    PREMIUM = "premium"      # Premium mobile optimization


class ContentStructureType(Enum):
    """Types of content structure for mobile optimization."""
    TECHNICAL_ARTICLE = "technical_article"
    TUTORIAL_GUIDE = "tutorial_guide"
    NEWS_UPDATE = "news_update"
    ANALYSIS_REPORT = "analysis_report"
    REFERENCE_DOCUMENTATION = "reference_documentation"


@dataclass
class MobileOptimizationPlan:
    """Detailed plan for mobile content optimization."""
    content_type: ContentStructureType
    optimization_level: MobileOptimizationLevel
    target_readability_score: float
    typography_adjustments: Dict[str, Any]
    structure_modifications: List[str]
    performance_targets: Dict[str, Any]
    accessibility_requirements: Dict[str, Any]
    estimated_processing_time: float


@dataclass
class MobileOptimizationMetrics:
    """Metrics for mobile optimization assessment."""
    readability_score: float
    typography_compliance: float
    structure_optimization: float
    performance_score: float
    accessibility_score: float
    mobile_friendliness: float
    touch_target_compliance: float
    load_time_estimate: float


@dataclass
class MobileOptimizationResult:
    """Result of mobile content optimization operation."""
    original_content: str
    optimized_content: str
    optimization_plan: MobileOptimizationPlan
    metrics: MobileOptimizationMetrics
    improvements_applied: List[str]
    processing_time: float
    success: bool
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class MobileContentOptimizer:
    """
    Comprehensive mobile-first content optimization system.
    
    Implements mobile optimization to exceed 95% mobile readability targets
    while maintaining professional quality for technical content.
    """
    
    def __init__(self, quality_gate: Optional[ConfigurableQualityGate] = None):
        """Initialize the mobile content optimizer."""
        self.quality_gate = quality_gate or ConfigurableQualityGate("enforcing")
        
        # Initialize mobile optimization components
        self.readability_analyzer = None  # Lazy initialization
        self.typography_manager = None    # Lazy initialization
        
        # Mobile optimization configuration
        self.optimization_config = self._initialize_optimization_config()
        self.mobile_thresholds = self._initialize_mobile_thresholds()
        
        logger.info("MobileContentOptimizer initialized")
    
    def _initialize_optimization_config(self) -> Dict[str, Any]:
        """Initialize mobile optimization configuration."""
        return {
            'mobile_viewport_width': 375,  # iPhone standard width
            'tablet_viewport_width': 768,   # iPad standard width
            'desktop_viewport_width': 1024, # Desktop standard width
            
            'font_size_mobile': {
                'body': 16,           # 16px minimum for mobile
                'h1': 28,            # 28px for main headings
                'h2': 24,            # 24px for section headings
                'h3': 20,            # 20px for subsection headings
                'code': 14,          # 14px for code blocks
                'caption': 14        # 14px for captions
            },
            
            'line_height_mobile': {
                'body': 1.5,         # 1.5 for optimal reading
                'headings': 1.3,     # 1.3 for headings
                'code': 1.4          # 1.4 for code blocks
            },
            
            'spacing_mobile': {
                'paragraph_bottom': '1.2em',
                'heading_top': '1.5em',
                'heading_bottom': '0.8em',
                'section_spacing': '2em',
                'code_block_margin': '1.5em'
            },
            
            'content_width': {
                'mobile_max': '100%',
                'tablet_max': '90%',
                'desktop_max': '80%'
            }
        }
    
    def _initialize_mobile_thresholds(self) -> Dict[str, float]:
        """Initialize mobile optimization thresholds."""
        return {
            'readability_minimum': 0.85,      # 85% minimum readability
            'readability_target': 0.95,       # 95% target readability
            'typography_compliance': 0.90,    # 90% typography compliance
            'structure_optimization': 0.90,   # 90% structure optimization
            'performance_target': 0.85,       # 85% performance score
            'accessibility_minimum': 0.80,    # 80% accessibility compliance
            'touch_target_minimum': 0.85      # 85% touch target compliance
        }
    
    def optimize_for_mobile(self, content: str, template_type: str, 
                          optimization_level: MobileOptimizationLevel = MobileOptimizationLevel.ENHANCED,
                          metadata: Optional[Dict[str, Any]] = None) -> MobileOptimizationResult:
        """
        Optimize content for mobile devices with comprehensive enhancements.
        
        Args:
            content: Original content to optimize
            template_type: Newsletter template type
            optimization_level: Level of mobile optimization to apply
            metadata: Additional context and configuration
            
        Returns:
            MobileOptimizationResult with optimization details and metrics
        """
        start_time = time.time()
        metadata = metadata or {}
        
        logger.info(f"Starting mobile optimization: {optimization_level.value} level")
        
        try:
            # Step 1: Analyze content structure and mobile readiness
            content_analysis = self._analyze_mobile_readiness(content, template_type)
            
            # Step 2: Create mobile optimization plan
            optimization_plan = self._create_optimization_plan(
                content_analysis, optimization_level, template_type, metadata
            )
            
            # Step 3: Apply mobile optimizations
            optimized_content = self._apply_mobile_optimizations(
                content, optimization_plan, metadata
            )
            
            # Step 4: Validate mobile optimization quality
            optimization_metrics = self._assess_mobile_optimization(
                optimized_content, optimization_plan
            )
            
            # Step 5: Generate improvement list
            improvements_applied = self._generate_improvement_list(optimization_plan)
            
            processing_time = time.time() - start_time
            
            # Check if optimization meets targets
            success = self._validate_optimization_targets(optimization_metrics)
            warnings = self._generate_optimization_warnings(optimization_metrics)
            
            result = MobileOptimizationResult(
                original_content=content,
                optimized_content=optimized_content,
                optimization_plan=optimization_plan,
                metrics=optimization_metrics,
                improvements_applied=improvements_applied,
                processing_time=processing_time,
                success=success,
                warnings=warnings
            )
            
            logger.info(f"Mobile optimization completed: {optimization_metrics.mobile_friendliness:.1%} mobile friendliness in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            processing_time = time.time() - start_time
            
            return MobileOptimizationResult(
                original_content=content,
                optimized_content=content,  # Return original on failure
                optimization_plan=MobileOptimizationPlan(
                    content_type=ContentStructureType.TECHNICAL_ARTICLE,
                    optimization_level=optimization_level,
                    target_readability_score=0.0,
                    typography_adjustments={},
                    structure_modifications=[],
                    performance_targets={},
                    accessibility_requirements={},
                    estimated_processing_time=0.0
                ),
                metrics=MobileOptimizationMetrics(
                    readability_score=0.0,
                    typography_compliance=0.0,
                    structure_optimization=0.0,
                    performance_score=0.0,
                    accessibility_score=0.0,
                    mobile_friendliness=0.0,
                    touch_target_compliance=0.0,
                    load_time_estimate=0.0
                ),
                improvements_applied=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _get_readability_analyzer(self) -> MobileReadabilityAnalyzer:
        """Get readability analyzer with lazy initialization."""
        if self.readability_analyzer is None:
            self.readability_analyzer = MobileReadabilityAnalyzer()
        return self.readability_analyzer
    
    def _get_typography_manager(self) -> ResponsiveTypographyManager:
        """Get typography manager with lazy initialization."""
        if self.typography_manager is None:
            self.typography_manager = ResponsiveTypographyManager()
        return self.typography_manager
    
    def _analyze_mobile_readiness(self, content: str, template_type: str) -> Dict[str, Any]:
        """Analyze content for mobile readiness and optimization opportunities."""
        analysis = {
            'content_type': self._classify_content_structure(content, template_type),
            'current_readability': self._assess_current_readability(content),
            'typography_issues': self._identify_typography_issues(content),
            'structure_issues': self._identify_structure_issues(content),
            'performance_issues': self._identify_performance_issues(content),
            'accessibility_issues': self._identify_accessibility_issues(content)
        }
        
        logger.debug(f"Mobile readiness analysis: {analysis['current_readability']:.2f} readability score")
        return analysis
    
    def _classify_content_structure(self, content: str, template_type: str) -> ContentStructureType:
        """Classify the content structure type for optimization strategy."""
        template_mapping = {
            'technical_deep_dive': ContentStructureType.TECHNICAL_ARTICLE,
            'tutorial_guide': ContentStructureType.TUTORIAL_GUIDE,
            'trend_analysis': ContentStructureType.ANALYSIS_REPORT,
            'research_summary': ContentStructureType.ANALYSIS_REPORT,
            'news_update': ContentStructureType.NEWS_UPDATE
        }
        
        return template_mapping.get(template_type, ContentStructureType.TECHNICAL_ARTICLE)
    
    def _assess_current_readability(self, content: str) -> float:
        """Assess current mobile readability score."""
        analyzer = self._get_readability_analyzer()
        readability_report = analyzer.analyze_mobile_readability(content)
        return readability_report.overall_mobile_score
    
    def _identify_typography_issues(self, content: str) -> List[str]:
        """Identify typography issues that affect mobile readability."""
        issues = []
        
        # Check for long lines
        lines = content.split('\n')
        for line in lines:
            if len(line) > 80 and not line.startswith('#'):  # Ignore headers
                issues.append("Long line length detected (>80 characters)")
                break
        
        # Check for small code blocks without proper formatting
        code_blocks = re.findall(r'`([^`\n]+)`', content)
        for block in code_blocks:
            if len(block) > 50:
                issues.append("Long inline code detected")
                break
        
        # Check for missing heading hierarchy
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        if len(headings) < 2:
            issues.append("Insufficient heading structure for mobile navigation")
        
        return list(set(issues))  # Remove duplicates
    
    def _identify_structure_issues(self, content: str) -> List[str]:
        """Identify content structure issues for mobile optimization."""
        issues = []
        
        # Check paragraph length
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        if long_paragraphs:
            issues.append(f"Long paragraphs detected ({len(long_paragraphs)} paragraphs >100 words)")
        
        # Check for missing lists where appropriate
        sentences = re.split(r'[.!?]+', content)
        list_indicators = ['first', 'second', 'third', 'also', 'additionally', 'furthermore']
        enumeration_sentences = [s for s in sentences if any(indicator in s.lower() for indicator in list_indicators)]
        if len(enumeration_sentences) > 3 and content.count('\n- ') < 3:
            issues.append("Content could benefit from bullet points or numbered lists")
        
        # Check for code block formatting
        if '```' in content:
            code_blocks = re.findall(r'```([^`]*?)```', content, re.DOTALL)
            for block in code_blocks:
                if len(block.split('\n')) > 20:
                    issues.append("Long code blocks may need mobile scrolling optimization")
                    break
        
        return issues
    
    def _identify_performance_issues(self, content: str) -> List[str]:
        """Identify performance issues that affect mobile experience."""
        issues = []
        
        # Check content length
        word_count = len(content.split())
        if word_count > 5000:
            issues.append(f"Very long content ({word_count} words) may impact mobile load times")
        
        # Check for embedded images or media
        image_refs = re.findall(r'!\[.*?\]\(.*?\)', content)
        if len(image_refs) > 10:
            issues.append(f"Many images detected ({len(image_refs)}) - optimize for mobile bandwidth")
        
        # Check for external links
        external_links = re.findall(r'\[.*?\]\(https?://.*?\)', content)
        if len(external_links) > 20:
            issues.append(f"Many external links ({len(external_links)}) may impact mobile navigation")
        
        return issues
    
    def _identify_accessibility_issues(self, content: str) -> List[str]:
        """Identify accessibility issues for mobile devices."""
        issues = []
        
        # Check for missing alt text in images
        images_without_alt = re.findall(r'!\[\s*\]\([^)]+\)', content)
        if images_without_alt:
            issues.append(f"Images missing alt text ({len(images_without_alt)} images)")
        
        # Check for poor link text
        poor_links = re.findall(r'\[(?:here|click here|read more|more)\]\([^)]+\)', content, re.IGNORECASE)
        if poor_links:
            issues.append(f"Poor link text detected ({len(poor_links)} instances)")
        
        # Check for missing heading structure
        h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
        if h1_count != 1:
            issues.append(f"Should have exactly one H1 heading (found {h1_count})")
        
        return issues
    
    def _create_optimization_plan(self, analysis: Dict[str, Any], 
                                optimization_level: MobileOptimizationLevel,
                                template_type: str, metadata: Dict[str, Any]) -> MobileOptimizationPlan:
        """Create detailed mobile optimization plan."""
        content_type = analysis['content_type']
        current_readability = analysis['current_readability']
        
        # Determine target readability based on optimization level
        target_mapping = {
            MobileOptimizationLevel.BASIC: 0.85,
            MobileOptimizationLevel.STANDARD: 0.90,
            MobileOptimizationLevel.ENHANCED: 0.95,
            MobileOptimizationLevel.PREMIUM: 0.98
        }
        target_readability = target_mapping[optimization_level]
        
        # Plan typography adjustments
        typography_adjustments = self._plan_typography_adjustments(
            analysis, optimization_level
        )
        
        # Plan structure modifications
        structure_modifications = self._plan_structure_modifications(
            analysis, content_type, optimization_level
        )
        
        # Set performance targets
        performance_targets = self._set_performance_targets(optimization_level)
        
        # Set accessibility requirements
        accessibility_requirements = self._set_accessibility_requirements(optimization_level)
        
        # Estimate processing time
        estimated_time = self._estimate_optimization_time(
            analysis, optimization_level, len(structure_modifications)
        )
        
        return MobileOptimizationPlan(
            content_type=content_type,
            optimization_level=optimization_level,
            target_readability_score=target_readability,
            typography_adjustments=typography_adjustments,
            structure_modifications=structure_modifications,
            performance_targets=performance_targets,
            accessibility_requirements=accessibility_requirements,
            estimated_processing_time=estimated_time
        )
    
    def _plan_typography_adjustments(self, analysis: Dict[str, Any], 
                                   optimization_level: MobileOptimizationLevel) -> Dict[str, Any]:
        """Plan typography adjustments for mobile optimization."""
        base_adjustments = {
            'font_size_increase': True,
            'line_height_optimization': True,
            'heading_hierarchy_improvement': True,
            'code_block_formatting': True
        }
        
        if optimization_level in [MobileOptimizationLevel.ENHANCED, MobileOptimizationLevel.PREMIUM]:
            base_adjustments.update({
                'responsive_font_scaling': True,
                'touch_target_optimization': True,
                'reading_flow_enhancement': True
            })
        
        if optimization_level == MobileOptimizationLevel.PREMIUM:
            base_adjustments.update({
                'advanced_typography_features': True,
                'custom_mobile_styles': True,
                'dark_mode_optimization': True
            })
        
        return base_adjustments
    
    def _plan_structure_modifications(self, analysis: Dict[str, Any],
                                    content_type: ContentStructureType,
                                    optimization_level: MobileOptimizationLevel) -> List[str]:
        """Plan content structure modifications for mobile."""
        modifications = []
        
        # Basic modifications for all levels
        modifications.extend([
            "paragraph_length_optimization",
            "heading_structure_improvement",
            "list_formatting_enhancement"
        ])
        
        # Content type specific modifications
        if content_type == ContentStructureType.TECHNICAL_ARTICLE:
            modifications.extend([
                "code_block_mobile_formatting",
                "technical_term_highlighting",
                "cross_reference_optimization"
            ])
        elif content_type == ContentStructureType.TUTORIAL_GUIDE:
            modifications.extend([
                "step_by_step_formatting",
                "progress_indicator_addition",
                "mobile_friendly_examples"
            ])
        
        # Enhanced level modifications
        if optimization_level in [MobileOptimizationLevel.ENHANCED, MobileOptimizationLevel.PREMIUM]:
            modifications.extend([
                "collapsible_sections",
                "mobile_navigation_aids",
                "touch_friendly_interactions"
            ])
        
        # Premium level modifications
        if optimization_level == MobileOptimizationLevel.PREMIUM:
            modifications.extend([
                "progressive_disclosure",
                "adaptive_content_loading",
                "gesture_based_navigation"
            ])
        
        return modifications
    
    def _set_performance_targets(self, optimization_level: MobileOptimizationLevel) -> Dict[str, Any]:
        """Set performance targets based on optimization level."""
        base_targets = {
            'mobile_load_time': 3.0,      # 3 seconds
            'content_delivery': 'optimized',
            'image_optimization': True
        }
        
        if optimization_level in [MobileOptimizationLevel.ENHANCED, MobileOptimizationLevel.PREMIUM]:
            base_targets.update({
                'mobile_load_time': 2.0,   # 2 seconds
                'lazy_loading': True,
                'compression_optimization': True
            })
        
        if optimization_level == MobileOptimizationLevel.PREMIUM:
            base_targets.update({
                'mobile_load_time': 1.5,   # 1.5 seconds
                'advanced_caching': True,
                'cdn_optimization': True
            })
        
        return base_targets
    
    def _set_accessibility_requirements(self, optimization_level: MobileOptimizationLevel) -> Dict[str, Any]:
        """Set accessibility requirements based on optimization level."""
        base_requirements = {
            'wcag_level': 'AA',
            'touch_target_size': '44px',
            'color_contrast_ratio': 4.5,
            'keyboard_navigation': True
        }
        
        if optimization_level in [MobileOptimizationLevel.ENHANCED, MobileOptimizationLevel.PREMIUM]:
            base_requirements.update({
                'screen_reader_optimization': True,
                'voice_navigation_support': True,
                'gesture_alternatives': True
            })
        
        if optimization_level == MobileOptimizationLevel.PREMIUM:
            base_requirements.update({
                'wcag_level': 'AAA',
                'advanced_accessibility_features': True,
                'personalization_support': True
            })
        
        return base_requirements
    
    def _estimate_optimization_time(self, analysis: Dict[str, Any],
                                  optimization_level: MobileOptimizationLevel,
                                  modification_count: int) -> float:
        """Estimate time required for mobile optimization."""
        base_time = 2.0  # 2 seconds base time
        
        # Add time based on content complexity
        content_complexity = len(analysis.get('typography_issues', [])) + \
                            len(analysis.get('structure_issues', [])) + \
                            len(analysis.get('accessibility_issues', []))
        
        complexity_time = content_complexity * 0.5
        
        # Add time based on optimization level
        level_multipliers = {
            MobileOptimizationLevel.BASIC: 1.0,
            MobileOptimizationLevel.STANDARD: 1.2,
            MobileOptimizationLevel.ENHANCED: 1.5,
            MobileOptimizationLevel.PREMIUM: 2.0
        }
        
        level_multiplier = level_multipliers[optimization_level]
        modification_time = modification_count * 0.3 * level_multiplier
        
        return base_time + complexity_time + modification_time
    
    def _apply_mobile_optimizations(self, content: str, plan: MobileOptimizationPlan,
                                  metadata: Dict[str, Any]) -> str:
        """Apply mobile optimizations based on the optimization plan."""
        optimized_content = content
        
        try:
            # Apply typography optimizations
            if plan.typography_adjustments:
                typography_manager = self._get_typography_manager()
                optimized_content = typography_manager.apply_mobile_typography(
                    optimized_content, plan.typography_adjustments
                )
            
            # Apply structure modifications
            for modification in plan.structure_modifications:
                optimized_content = self._apply_structure_modification(
                    optimized_content, modification, plan.content_type
                )
            
            # Apply accessibility improvements
            optimized_content = self._apply_accessibility_improvements(
                optimized_content, plan.accessibility_requirements
            )
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Failed to apply mobile optimizations: {e}")
            return content  # Return original content on failure
    
    def _apply_structure_modification(self, content: str, modification: str,
                                    content_type: ContentStructureType) -> str:
        """Apply a specific structure modification to content."""
        if modification == "paragraph_length_optimization":
            return self._optimize_paragraph_length(content)
        elif modification == "heading_structure_improvement":
            return self._improve_heading_structure(content)
        elif modification == "list_formatting_enhancement":
            return self._enhance_list_formatting(content)
        elif modification == "code_block_mobile_formatting":
            return self._format_code_blocks_for_mobile(content)
        elif modification == "collapsible_sections":
            return self._add_collapsible_sections(content)
        else:
            # For modifications not yet implemented, return content unchanged
            logger.debug(f"Structure modification '{modification}' not yet implemented")
            return content
    
    def _optimize_paragraph_length(self, content: str) -> str:
        """Optimize paragraph length for mobile readability."""
        paragraphs = content.split('\n\n')
        optimized_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                optimized_paragraphs.append(paragraph)
                continue
            
            # Split long paragraphs (>150 words) into smaller ones
            words = paragraph.split()
            if len(words) > 150:
                # Find natural break points (sentences)
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = []
                
                for sentence in sentences:
                    current_chunk.append(sentence)
                    if len(' '.join(current_chunk).split()) > 75:  # ~75 words per paragraph
                        optimized_paragraphs.append(' '.join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    optimized_paragraphs.append(' '.join(current_chunk))
            else:
                optimized_paragraphs.append(paragraph)
        
        return '\n\n'.join(optimized_paragraphs)
    
    def _improve_heading_structure(self, content: str) -> str:
        """Improve heading structure for mobile navigation."""
        lines = content.split('\n')
        improved_lines = []
        
        for line in lines:
            if re.match(r'^#{1,6}\s+', line):
                # Ensure proper spacing after headings for mobile
                improved_lines.append(line)
                # Add empty line after heading if not present
                next_line_idx = lines.index(line) + 1
                if (next_line_idx < len(lines) and 
                    lines[next_line_idx].strip() and 
                    not lines[next_line_idx].startswith('#')):
                    improved_lines.append('')
            else:
                improved_lines.append(line)
        
        return '\n'.join(improved_lines)
    
    def _enhance_list_formatting(self, content: str) -> str:
        """Enhance list formatting for mobile readability."""
        # Convert enumeration patterns to proper lists
        lines = content.split('\n')
        enhanced_lines = []
        in_list_conversion = False
        
        for i, line in enumerate(lines):
            # Look for enumeration patterns
            enumeration_pattern = r'^(\d+[\.\)]|\(?[a-z][\.\)]|first|second|third|also|additionally)'
            if re.match(enumeration_pattern, line.strip().lower()):
                if not in_list_conversion:
                    in_list_conversion = True
                    enhanced_lines.append('')  # Add spacing before list
                
                # Convert to bullet point
                cleaned_line = re.sub(r'^(\d+[\.\)]|\(?[a-z][\.\)])', '- ', line.strip())
                cleaned_line = re.sub(r'^(first|second|third|also|additionally)', '- ', cleaned_line, flags=re.IGNORECASE)
                enhanced_lines.append(cleaned_line)
            else:
                if in_list_conversion and line.strip():
                    in_list_conversion = False
                    enhanced_lines.append('')  # Add spacing after list
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _format_code_blocks_for_mobile(self, content: str) -> str:
        """Format code blocks for better mobile display."""
        # Add language tags to code blocks if missing
        code_block_pattern = r'```(\n)(.*?)```'
        
        def improve_code_block(match):
            code_content = match.group(2)
            
            # Detect language from content
            language = self._detect_code_language(code_content)
            
            # Add horizontal scroll hint for long lines
            lines = code_content.split('\n')
            max_line_length = max(len(line) for line in lines if line.strip())
            
            mobile_hint = ""
            if max_line_length > 50:
                mobile_hint = "<!-- Scroll horizontally to see full code -->\n"
            
            return f"```{language}\n{mobile_hint}{code_content}```"
        
        return re.sub(code_block_pattern, improve_code_block, content, flags=re.DOTALL)
    
    def _detect_code_language(self, code_content: str) -> str:
        """Detect programming language from code content."""
        code_lower = code_content.lower()
        
        if 'import ' in code_lower and 'def ' in code_lower:
            return 'python'
        elif 'function' in code_lower or 'const ' in code_lower:
            return 'javascript'
        elif 'public class' in code_lower or 'import java' in code_lower:
            return 'java'
        elif '#include' in code_lower or 'int main' in code_lower:
            return 'cpp'
        elif 'package ' in code_lower and 'func ' in code_lower:
            return 'go'
        else:
            return ''  # Default to no language tag
    
    def _add_collapsible_sections(self, content: str) -> str:
        """Add collapsible sections for better mobile navigation."""
        # This would typically add HTML details/summary tags
        # For now, we'll add mobile-friendly section markers
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Look for level 3+ headings that could be collapsible
            if re.match(r'^#{3,6}\s+', line):
                enhanced_lines.append(line)
                enhanced_lines.append('<!-- Mobile: Tap to expand/collapse -->')
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _apply_accessibility_improvements(self, content: str, 
                                        requirements: Dict[str, Any]) -> str:
        """Apply accessibility improvements for mobile devices."""
        improved_content = content
        
        # Improve link text
        poor_link_pattern = r'\[(here|click here|read more|more)\]\(([^)]+)\)'
        improved_content = re.sub(
            poor_link_pattern, 
            lambda m: f'[{self._generate_better_link_text(m.group(2))}]({m.group(2)})',
            improved_content,
            flags=re.IGNORECASE
        )
        
        # Add mobile-specific accessibility hints
        if requirements.get('screen_reader_optimization'):
            improved_content = self._add_screen_reader_hints(improved_content)
        
        return improved_content
    
    def _generate_better_link_text(self, url: str) -> str:
        """Generate better link text from URL."""
        # Extract domain name or path for better context
        if 'github.com' in url:
            return 'GitHub repository'
        elif 'docs.' in url:
            return 'documentation'
        elif 'arxiv.org' in url:
            return 'research paper'
        elif '.pdf' in url:
            return 'PDF document'
        else:
            return 'external link'
    
    def _add_screen_reader_hints(self, content: str) -> str:
        """Add screen reader optimization hints."""
        # Add navigation landmarks for screen readers
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            if re.match(r'^# ', line):
                enhanced_lines.append('<!-- Mobile Screen Reader: Main heading -->')
                enhanced_lines.append(line)
            elif re.match(r'^## ', line):
                enhanced_lines.append('<!-- Mobile Screen Reader: Section heading -->')
                enhanced_lines.append(line)
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _assess_mobile_optimization(self, content: str, 
                                  plan: MobileOptimizationPlan) -> MobileOptimizationMetrics:
        """Assess the quality of mobile optimization."""
        try:
            # Assess readability
            readability_analyzer = self._get_readability_analyzer()
            readability_report = readability_analyzer.analyze_mobile_readability(content)
            readability_score = readability_report.overall_mobile_score
            
            # Assess typography compliance
            typography_compliance = self._assess_typography_compliance(content)
            
            # Assess structure optimization
            structure_optimization = self._assess_structure_optimization(content, plan)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(content)
            
            # Calculate accessibility score
            accessibility_score = self._calculate_accessibility_score(content)
            
            # Calculate overall mobile friendliness
            mobile_friendliness = (
                readability_score * 0.3 +
                typography_compliance * 0.2 +
                structure_optimization * 0.2 +
                performance_score * 0.15 +
                accessibility_score * 0.15
            )
            
            # Touch target compliance (estimated)
            touch_target_compliance = self._assess_touch_target_compliance(content)
            
            # Load time estimate
            load_time_estimate = self._estimate_mobile_load_time(content)
            
            return MobileOptimizationMetrics(
                readability_score=readability_score,
                typography_compliance=typography_compliance,
                structure_optimization=structure_optimization,
                performance_score=performance_score,
                accessibility_score=accessibility_score,
                mobile_friendliness=mobile_friendliness,
                touch_target_compliance=touch_target_compliance,
                load_time_estimate=load_time_estimate
            )
            
        except Exception as e:
            logger.error(f"Mobile optimization assessment failed: {e}")
            # Return baseline metrics on failure
            return MobileOptimizationMetrics(
                readability_score=0.5,
                typography_compliance=0.5,
                structure_optimization=0.5,
                performance_score=0.5,
                accessibility_score=0.5,
                mobile_friendliness=0.5,
                touch_target_compliance=0.5,
                load_time_estimate=5.0
            )
    
    def _assess_typography_compliance(self, content: str) -> float:
        """Assess typography compliance for mobile devices."""
        compliance_factors = []
        
        # Check heading hierarchy
        headings = re.findall(r'^(#{1,6})', content, re.MULTILINE)
        if headings:
            heading_levels = [len(h) for h in headings]
            proper_hierarchy = all(
                level <= prev_level + 1 
                for prev_level, level in zip(heading_levels, heading_levels[1:])
            )
            compliance_factors.append(1.0 if proper_hierarchy else 0.7)
        
        # Check paragraph length
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        paragraph_score = max(0.0, 1.0 - len(long_paragraphs) / max(1, len(paragraphs)))
        compliance_factors.append(paragraph_score)
        
        # Check code block formatting
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        if code_blocks:
            formatted_blocks = [block for block in code_blocks if '```\n' not in block[:4]]
            code_score = len(formatted_blocks) / len(code_blocks)
            compliance_factors.append(code_score)
        
        return sum(compliance_factors) / max(1, len(compliance_factors))
    
    def _assess_structure_optimization(self, content: str, 
                                     plan: MobileOptimizationPlan) -> float:
        """Assess content structure optimization for mobile."""
        optimization_factors = []
        
        # Check for proper list usage
        list_patterns = content.count('- ') + content.count('* ') + len(re.findall(r'^\d+\.', content, re.MULTILINE))
        enumeration_sentences = len(re.findall(r'\b(?:first|second|third|also|additionally)\b', content, re.IGNORECASE))
        
        if enumeration_sentences > 0:
            list_usage_score = min(1.0, list_patterns / enumeration_sentences)
            optimization_factors.append(list_usage_score)
        
        # Check section length distribution
        sections = re.split(r'^##\s+', content, flags=re.MULTILINE)[1:]  # Skip intro
        if sections:
            section_lengths = [len(section.split()) for section in sections]
            avg_section_length = sum(section_lengths) / len(section_lengths)
            # Optimal section length for mobile: 200-400 words
            length_score = max(0.0, 1.0 - abs(avg_section_length - 300) / 200)
            optimization_factors.append(length_score)
        
        # Check for mobile-friendly formatting
        mobile_indicators = [
            '<!-- Mobile:' in content,
            '<!-- Scroll horizontally' in content,
            content.count('\n\n') / max(1, content.count('\n'))  # Good spacing ratio
        ]
        mobile_score = sum(1 for indicator in mobile_indicators if (isinstance(indicator, bool) and indicator) or (isinstance(indicator, float) and indicator > 0.1)) / len(mobile_indicators)
        optimization_factors.append(mobile_score)
        
        return sum(optimization_factors) / max(1, len(optimization_factors))
    
    def _calculate_performance_score(self, content: str) -> float:
        """Calculate performance score for mobile devices."""
        performance_factors = []
        
        # Content length factor
        word_count = len(content.split())
        length_score = max(0.0, 1.0 - max(0, word_count - 3000) / 5000)  # Penalty for >3000 words
        performance_factors.append(length_score)
        
        # Image optimization factor
        images = re.findall(r'!\[.*?\]\(.*?\)', content)
        image_score = max(0.0, 1.0 - max(0, len(images) - 5) / 10)  # Penalty for >5 images
        performance_factors.append(image_score)
        
        # External link factor
        external_links = re.findall(r'\[.*?\]\(https?://.*?\)', content)
        link_score = max(0.0, 1.0 - max(0, len(external_links) - 10) / 20)  # Penalty for >10 links
        performance_factors.append(link_score)
        
        return sum(performance_factors) / len(performance_factors)
    
    def _calculate_accessibility_score(self, content: str) -> float:
        """Calculate accessibility score for mobile devices."""
        accessibility_factors = []
        
        # Image alt text factor
        images = re.findall(r'!\[.*?\]\(.*?\)', content)
        images_with_alt = re.findall(r'!\[.+\]\(.*?\)', content)
        if images:
            alt_text_score = len(images_with_alt) / len(images)
            accessibility_factors.append(alt_text_score)
        
        # Link text quality factor
        links = re.findall(r'\[(.+?)\]\(.*?\)', content)
        poor_links = [link for link in links if link.lower() in ['here', 'click here', 'read more', 'more']]
        if links:
            link_quality_score = max(0.0, 1.0 - len(poor_links) / len(links))
            accessibility_factors.append(link_quality_score)
        
        # Heading structure factor
        h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
        heading_structure_score = 1.0 if h1_count == 1 else 0.5
        accessibility_factors.append(heading_structure_score)
        
        return sum(accessibility_factors) / max(1, len(accessibility_factors))
    
    def _assess_touch_target_compliance(self, content: str) -> float:
        """Assess touch target compliance for mobile devices."""
        # This is a simplified assessment for markdown content
        # In a real implementation, this would analyze rendered HTML
        
        # Check for proper link spacing
        lines_with_links = [line for line in content.split('\n') if '[' in line and '](' in line]
        dense_link_lines = [line for line in lines_with_links if line.count('](') > 3]
        
        if lines_with_links:
            touch_target_score = max(0.0, 1.0 - len(dense_link_lines) / len(lines_with_links))
        else:
            touch_target_score = 1.0
        
        return touch_target_score
    
    def _estimate_mobile_load_time(self, content: str) -> float:
        """Estimate mobile load time for content."""
        # Simplified estimation based on content characteristics
        base_time = 1.0  # 1 second base load time
        
        # Add time for content length
        word_count = len(content.split())
        content_time = word_count / 1000 * 0.5  # 0.5s per 1000 words
        
        # Add time for images
        images = re.findall(r'!\[.*?\]\(.*?\)', content)
        image_time = len(images) * 0.3  # 0.3s per image
        
        # Add time for external links
        external_links = re.findall(r'\[.*?\]\(https?://.*?\)', content)
        link_time = len(external_links) * 0.1  # 0.1s per external link
        
        total_time = base_time + content_time + image_time + link_time
        return min(10.0, total_time)  # Cap at 10 seconds
    
    def _generate_improvement_list(self, plan: MobileOptimizationPlan) -> List[str]:
        """Generate list of improvements applied during optimization."""
        improvements = []
        
        if plan.typography_adjustments.get('font_size_increase'):
            improvements.append("Optimized font sizes for mobile readability")
        
        if plan.typography_adjustments.get('line_height_optimization'):
            improvements.append("Improved line height for better reading flow")
        
        if "paragraph_length_optimization" in plan.structure_modifications:
            improvements.append("Optimized paragraph length for mobile scanning")
        
        if "heading_structure_improvement" in plan.structure_modifications:
            improvements.append("Enhanced heading structure for mobile navigation")
        
        if "list_formatting_enhancement" in plan.structure_modifications:
            improvements.append("Converted text patterns to mobile-friendly lists")
        
        if "code_block_mobile_formatting" in plan.structure_modifications:
            improvements.append("Optimized code blocks for mobile display")
        
        if "collapsible_sections" in plan.structure_modifications:
            improvements.append("Added collapsible sections for better mobile navigation")
        
        if plan.accessibility_requirements.get('screen_reader_optimization'):
            improvements.append("Enhanced screen reader compatibility")
        
        return improvements
    
    def _validate_optimization_targets(self, metrics: MobileOptimizationMetrics) -> bool:
        """Validate that optimization meets minimum targets."""
        thresholds = self.mobile_thresholds
        
        validation_criteria = [
            metrics.readability_score >= thresholds['readability_minimum'],
            metrics.typography_compliance >= thresholds['typography_compliance'],
            metrics.structure_optimization >= thresholds['structure_optimization'],
            metrics.accessibility_score >= thresholds['accessibility_minimum'],
            metrics.mobile_friendliness >= thresholds['readability_minimum']
        ]
        
        return all(validation_criteria)
    
    def _generate_optimization_warnings(self, metrics: MobileOptimizationMetrics) -> List[str]:
        """Generate warnings for optimization issues."""
        warnings = []
        thresholds = self.mobile_thresholds
        
        if metrics.readability_score < thresholds['readability_target']:
            warnings.append(f"Readability score ({metrics.readability_score:.1%}) below target ({thresholds['readability_target']:.1%})")
        
        if metrics.typography_compliance < thresholds['typography_compliance']:
            warnings.append(f"Typography compliance ({metrics.typography_compliance:.1%}) needs improvement")
        
        if metrics.performance_score < thresholds['performance_target']:
            warnings.append(f"Performance score ({metrics.performance_score:.1%}) may impact mobile experience")
        
        if metrics.load_time_estimate > 3.0:
            warnings.append(f"Estimated load time ({metrics.load_time_estimate:.1f}s) may be slow on mobile")
        
        return warnings


# Export main classes
__all__ = [
    'MobileContentOptimizer',
    'MobileOptimizationLevel',
    'ContentStructureType',
    'MobileOptimizationResult',
    'MobileOptimizationMetrics'
]