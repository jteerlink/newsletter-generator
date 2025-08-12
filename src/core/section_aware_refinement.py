"""
Section-Aware Refinement Loop for Newsletter Generation

This module implements the multi-pass section processing system as specified 
in Phase 1 FR1.2 of the Multi-Agent Enhancement PRD.

Features:
- Section-focused editing and refinement cycles
- Up to 3 refinement iterations per section
- Section boundary detection and preservation
- Overall narrative flow preservation
- Quality gates for section-level validation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .section_aware_prompts import SectionType, SectionAwarePromptManager

logger = logging.getLogger(__name__)


class RefinementPass(Enum):
    """Types of refinement passes."""
    STRUCTURE = "structure"
    CONTENT = "content"
    STYLE = "style"
    TECHNICAL = "technical"
    FINAL = "final"


@dataclass
class SectionBoundary:
    """Represents a section boundary within newsletter content."""
    section_type: SectionType
    start_index: int
    end_index: int
    title: str = ""
    confidence: float = 0.0


@dataclass
class SectionContent:
    """Represents content for a specific newsletter section."""
    section_type: SectionType
    content: str
    boundary: SectionBoundary
    quality_score: float = 0.0
    refinement_count: int = 0
    issues: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


@dataclass
class RefinementResult:
    """Result of a refinement operation."""
    refined_content: str
    quality_improvement: float
    issues_addressed: List[str]
    new_issues: List[str]
    processing_time: float
    pass_type: RefinementPass


class SectionBoundaryDetector:
    """Detects section boundaries within newsletter content."""

    def __init__(self):
        """Initialize section boundary detector."""
        self.section_patterns = {
            SectionType.INTRODUCTION: [
                r'(?i)^#+\s*(introduction|overview|welcome)',
                r'(?i)^#+\s*(getting started|beginning)',
                r'(?i)^(introduction|overview|welcome)[:.]',
            ],
            SectionType.NEWS: [
                r'(?i)^#+\s*(news|updates|announcements)',
                r'(?i)^#+\s*(latest|recent|breaking)',
                r'(?i)^(news|updates|latest)[:.]',
            ],
            SectionType.ANALYSIS: [
                r'(?i)^#+\s*(analysis|deep dive|examination)',
                r'(?i)^#+\s*(technical analysis|review)',
                r'(?i)^(analysis|review)[:.]',
            ],
            SectionType.TUTORIAL: [
                r'(?i)^#+\s*(tutorial|how.?to|guide)',
                r'(?i)^#+\s*(step.?by.?step|implementation)',
                r'(?i)^(tutorial|guide|how.?to)[:.]',
            ],
            SectionType.CONCLUSION: [
                r'(?i)^#+\s*(conclusion|summary|wrap.?up)',
                r'(?i)^#+\s*(final thoughts|takeaways)',
                r'(?i)^(conclusion|summary|finally)[:.]',
            ]
        }

    def detect_boundaries(self, content: str) -> List[SectionBoundary]:
        """
        Detect section boundaries in content.
        
        Args:
            content: Newsletter content to analyze
            
        Returns:
            List[SectionBoundary]: Detected section boundaries
        """
        boundaries = []
        lines = content.split('\n')
        current_position = 0
        
        for i, line in enumerate(lines):
            line_position = current_position
            current_position += len(line) + 1  # +1 for newline
            
            # Check for section headers
            section_type, confidence = self._match_section_pattern(line)
            
            if section_type != SectionType.GENERAL and confidence > 0.7:
                # Calculate end of previous section
                if boundaries:
                    boundaries[-1].end_index = line_position
                
                # Create new boundary
                boundary = SectionBoundary(
                    section_type=section_type,
                    start_index=line_position,
                    end_index=len(content),  # Will be updated by next boundary
                    title=line.strip(),
                    confidence=confidence
                )
                boundaries.append(boundary)
        
        # If no explicit boundaries found, treat as single analysis section
        if not boundaries:
            boundaries.append(SectionBoundary(
                section_type=SectionType.ANALYSIS,
                start_index=0,
                end_index=len(content),
                title="Main Content",
                confidence=0.5
            ))
        
        return boundaries

    def _match_section_pattern(self, line: str) -> Tuple[SectionType, float]:
        """Match line against section patterns."""
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line):
                    confidence = 0.9 if line.startswith('#') else 0.8
                    return section_type, confidence
        
        return SectionType.GENERAL, 0.0


class SectionAwareRefinementLoop:
    """
    Enhanced refinement loop with section-specific processing.
    
    Implements FR1.2 requirements for iterative refinement with 
    section-specific focus while maintaining narrative flow.
    """

    def __init__(self, max_iterations: int = 3):
        """
        Initialize section-aware refinement loop.
        
        Args:
            max_iterations: Maximum refinement iterations per section
        """
        self.max_iterations = max_iterations
        self.boundary_detector = SectionBoundaryDetector()
        self.prompt_manager = SectionAwarePromptManager()
        
        # Quality thresholds for each refinement pass
        self.quality_thresholds = {
            RefinementPass.STRUCTURE: 0.7,
            RefinementPass.CONTENT: 0.8,
            RefinementPass.STYLE: 0.85,
            RefinementPass.TECHNICAL: 0.9,
            RefinementPass.FINAL: 0.95
        }
        
        logger.info("Section-aware refinement loop initialized (max_iterations=%d)", 
                   max_iterations)

    def refine_newsletter(self, content: str, context: Dict[str, Any]) -> str:
        """
        Refine newsletter content using section-aware processing.
        
        Args:
            content: Original newsletter content
            context: Context information for refinement
            
        Returns:
            str: Refined newsletter content
        """
        try:
            logger.info("Starting section-aware refinement of %d character content", 
                       len(content))
            
            # 1. Detect section boundaries
            boundaries = self.boundary_detector.detect_boundaries(content)
            logger.info("Detected %d sections", len(boundaries))
            
            # 2. Extract sections
            sections = self._extract_sections(content, boundaries)
            
            # 3. Refine each section independently
            refined_sections = []
            for section in sections:
                refined_section = self._refine_section(section, context)
                refined_sections.append(refined_section)
            
            # 4. Reassemble content with flow validation
            refined_content = self._reassemble_content(refined_sections)
            
            # 5. Final narrative flow validation
            final_content = self._validate_narrative_flow(refined_content, context)
            
            logger.info("Section-aware refinement completed")
            return final_content
            
        except Exception as e:
            logger.error("Error in section-aware refinement: %s", e)
            return content  # Return original content on error

    def refine_by_section(self, content: str, section_type: SectionType, 
                         context: Dict[str, Any]) -> str:
        """
        Apply section-specific refinement logic.
        
        Args:
            content: Section content to refine
            section_type: Type of section being refined
            context: Context for refinement
            
        Returns:
            str: Refined section content
        """
        refinement_prompt = self._generate_refinement_prompt(
            content, section_type, context
        )
        
        # This would integrate with the LLM system
        # For now, return content with basic improvements
        return self._apply_basic_refinements(content, section_type)

    def validate_section_quality(self, content: str, section_type: SectionType) -> float:
        """
        Calculate section-specific quality metrics.
        
        Args:
            content: Section content to validate
            section_type: Type of section
            
        Returns:
            float: Quality score (0.0 to 1.0)
        """
        score = 0.0
        
        # Basic quality checks
        if len(content.strip()) > 50:
            score += 0.2
        
        # Section-specific quality checks
        if section_type == SectionType.INTRODUCTION:
            score += self._validate_introduction_quality(content)
        elif section_type == SectionType.ANALYSIS:
            score += self._validate_analysis_quality(content)
        elif section_type == SectionType.TUTORIAL:
            score += self._validate_tutorial_quality(content)
        elif section_type == SectionType.NEWS:
            score += self._validate_news_quality(content)
        elif section_type == SectionType.CONCLUSION:
            score += self._validate_conclusion_quality(content)
        
        return min(1.0, score)

    def _extract_sections(self, content: str, 
                         boundaries: List[SectionBoundary]) -> List[SectionContent]:
        """Extract section content based on boundaries."""
        sections = []
        
        for boundary in boundaries:
            section_text = content[boundary.start_index:boundary.end_index].strip()
            
            section = SectionContent(
                section_type=boundary.section_type,
                content=section_text,
                boundary=boundary,
                quality_score=self.validate_section_quality(
                    section_text, boundary.section_type
                )
            )
            sections.append(section)
        
        return sections

    def _refine_section(self, section: SectionContent, 
                       context: Dict[str, Any]) -> SectionContent:
        """Refine individual section through multiple passes."""
        current_content = section.content
        total_improvement = 0.0
        
        for iteration in range(self.max_iterations):
            if section.quality_score >= 0.9:  # Already high quality
                break
            
            # Determine refinement pass type
            pass_type = self._determine_pass_type(section, iteration)
            
            # Apply refinement
            refined_content = self.refine_by_section(
                current_content, section.section_type, context
            )
            
            # Calculate improvement
            new_quality = self.validate_section_quality(
                refined_content, section.section_type
            )
            improvement = new_quality - section.quality_score
            
            if improvement > 0.05:  # Meaningful improvement
                current_content = refined_content
                section.quality_score = new_quality
                total_improvement += improvement
                section.refinement_count += 1
                
                logger.debug("Section %s iteration %d: quality %.3f (+%.3f)",
                           section.section_type.value, iteration + 1,
                           new_quality, improvement)
            else:
                break  # No significant improvement
        
        section.content = current_content
        return section

    def _reassemble_content(self, sections: List[SectionContent]) -> str:
        """Reassemble refined sections into complete content."""
        # Sort sections by logical order
        section_order = [
            SectionType.INTRODUCTION,
            SectionType.NEWS,
            SectionType.ANALYSIS,
            SectionType.TUTORIAL,
            SectionType.CONCLUSION
        ]
        
        ordered_sections = []
        for section_type in section_order:
            for section in sections:
                if section.section_type == section_type:
                    ordered_sections.append(section)
        
        # Add any remaining sections
        for section in sections:
            if section not in ordered_sections:
                ordered_sections.append(section)
        
        # Combine content with appropriate spacing
        combined_content = []
        for i, section in enumerate(ordered_sections):
            if i > 0:
                combined_content.append("\n\n")
            combined_content.append(section.content)
        
        return "".join(combined_content)

    def _validate_narrative_flow(self, content: str, context: Dict[str, Any]) -> str:
        """Validate and improve narrative flow between sections."""
        # Basic flow validation - check transitions
        sections = content.split('\n\n')
        improved_sections = []
        
        for i, section in enumerate(sections):
            if i > 0 and len(section.strip()) > 0:
                # Check if transition is needed
                prev_section = sections[i-1]
                if self._needs_transition(prev_section, section):
                    transition = self._generate_transition(prev_section, section)
                    improved_sections.append(transition)
            
            improved_sections.append(section)
        
        return '\n\n'.join(improved_sections)

    def _generate_refinement_prompt(self, content: str, section_type: SectionType,
                                  context: Dict[str, Any]) -> str:
        """Generate prompt for section refinement."""
        base_prompt = f"""
Refine the following {section_type.value} section for improved quality:

Original Content:
{content}

Refinement Guidelines:
- Improve clarity and readability
- Ensure content matches section purpose
- Maintain consistent tone and style
- Fix any factual or grammatical errors
- Enhance engagement and value for readers

Target Audience: {context.get('audience', 'General Tech Audience')}
Content Focus: {context.get('content_focus', 'General Tech News')}

Provide the refined version:
"""
        return base_prompt

    def _apply_basic_refinements(self, content: str, section_type: SectionType) -> str:
        """Apply basic text refinements."""
        # Remove extra whitespace
        refined = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Ensure proper sentence endings
        refined = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', refined)
        
        # Add section-specific improvements
        if section_type == SectionType.TUTORIAL:
            refined = self._improve_tutorial_structure(refined)
        elif section_type == SectionType.NEWS:
            refined = self._improve_news_structure(refined)
        
        return refined.strip()

    def _improve_tutorial_structure(self, content: str) -> str:
        """Improve tutorial section structure."""
        # Add step numbering if missing
        lines = content.split('\n')
        improved_lines = []
        step_count = 1
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                if any(word in line.lower() for word in ['first', 'next', 'then', 'finally']):
                    if not re.match(r'^\d+\.', line.strip()):
                        line = f"{step_count}. {line.strip()}"
                        step_count += 1
            improved_lines.append(line)
        
        return '\n'.join(improved_lines)

    def _improve_news_structure(self, content: str) -> str:
        """Improve news section structure."""
        # Ensure news items are clearly separated
        lines = content.split('\n')
        improved_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # Add bullet points for news items if missing
                if not line.strip().startswith('•') and not line.strip().startswith('-'):
                    if any(keyword in line.lower() for keyword in 
                          ['announced', 'released', 'launched', 'reported']):
                        line = f"• {line.strip()}"
            improved_lines.append(line)
        
        return '\n'.join(improved_lines)

    def _determine_pass_type(self, section: SectionContent, iteration: int) -> RefinementPass:
        """Determine the type of refinement pass to apply."""
        if iteration == 0:
            return RefinementPass.STRUCTURE
        elif iteration == 1:
            return RefinementPass.CONTENT
        else:
            return RefinementPass.STYLE

    def _validate_introduction_quality(self, content: str) -> float:
        """Validate introduction section quality."""
        score = 0.0
        
        # Check for hook
        if any(pattern in content.lower() for pattern in 
               ['?', 'imagine', 'consider', 'what if']):
            score += 0.2
        
        # Check for topic preview
        if any(word in content.lower() for word in 
               ['explore', 'discuss', 'cover', 'examine']):
            score += 0.2
        
        # Check length appropriateness
        if 100 <= len(content) <= 300:
            score += 0.2
        
        return score

    def _validate_analysis_quality(self, content: str) -> float:
        """Validate analysis section quality."""
        score = 0.0
        
        # Check for analytical language
        if any(word in content.lower() for word in 
               ['analysis', 'data', 'evidence', 'trend', 'pattern']):
            score += 0.3
        
        # Check for depth
        if len(content) > 300:
            score += 0.2
        
        return score

    def _validate_tutorial_quality(self, content: str) -> float:
        """Validate tutorial section quality."""
        score = 0.0
        
        # Check for step structure
        if re.search(r'\d+\.', content) or re.search(r'step \d+', content.lower()):
            score += 0.3
        
        # Check for action words
        if any(word in content.lower() for word in 
               ['create', 'install', 'configure', 'run', 'execute']):
            score += 0.2
        
        return score

    def _validate_news_quality(self, content: str) -> float:
        """Validate news section quality."""
        score = 0.0
        
        # Check for recent indicators
        if any(word in content.lower() for word in 
               ['recently', 'today', 'this week', 'announced', 'launched']):
            score += 0.3
        
        # Check for specific details
        if re.search(r'\d{4}|\$\d+|version \d+', content):
            score += 0.2
        
        return score

    def _validate_conclusion_quality(self, content: str) -> float:
        """Validate conclusion section quality."""
        score = 0.0
        
        # Check for summary language
        if any(word in content.lower() for word in 
               ['summary', 'conclusion', 'takeaway', 'key points']):
            score += 0.2
        
        # Check for call to action
        if any(word in content.lower() for word in 
               ['try', 'explore', 'learn more', 'contact', 'share']):
            score += 0.3
        
        return score

    def _needs_transition(self, prev_section: str, current_section: str) -> bool:
        """Check if transition is needed between sections."""
        # Simple heuristic - if sections end/start abruptly
        prev_ends_abruptly = not prev_section.strip().endswith(('.', '!', '?'))
        current_starts_abruptly = current_section.strip()[0].islower() if current_section.strip() else False
        
        return prev_ends_abruptly or current_starts_abruptly

    def _generate_transition(self, prev_section: str, current_section: str) -> str:
        """Generate transition between sections."""
        return "\n---\n"  # Simple transition marker


# Export main classes
__all__ = [
    'SectionAwareRefinementLoop',
    'SectionBoundaryDetector',
    'SectionBoundary',
    'SectionContent',
    'RefinementResult',
    'RefinementPass'
]