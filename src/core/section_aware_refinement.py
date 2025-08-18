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


class ToolAugmentedRefinementLoop(SectionAwareRefinementLoop):
    """
    Enhanced refinement with mandatory tool usage per iteration.
    
    Implements Phase 3 FR3.1 requirements for tool-augmented refinement
    with claim validation, information enrichment, and source verification.
    """
    
    def __init__(self, max_iterations: int = 3):
        """Initialize tool-augmented refinement loop."""
        super().__init__(max_iterations)
        
        # Import tool systems
        from core.claim_validator import ClaimExtractor, SourceValidator, CitationGenerator
        from core.information_enricher import InformationEnricher
        from core.tool_cache import get_tool_cache
        from storage import get_storage_provider
        
        # Initialize tool components
        self.claim_extractor = ClaimExtractor()
        self.source_validator = SourceValidator()
        self.citation_generator = CitationGenerator()
        self.info_enricher = InformationEnricher()
        self.tool_cache = get_tool_cache()
        self.vector_store = get_storage_provider()
        
        logger.info("Tool-augmented refinement loop initialized")
    
    def refine_with_tools(self, content: str, section_type: SectionType,
                         workflow_id: Optional[str] = None,
                         session_id: Optional[str] = None) -> str:
        """Multi-iteration refinement with tool consultation."""
        current_content = content
        iteration_results = []
        
        for iteration in range(self.max_iterations):
            logger.info(f"Tool-augmented refinement iteration {iteration + 1}/{self.max_iterations}")
            
            iteration_context = {}
            
            if iteration == 0:  # Structure and completeness
                iteration_context = self._analyze_structure_completeness(
                    current_content, section_type, workflow_id, session_id)
                
            elif iteration == 1:  # Accuracy and freshness
                iteration_context = self._analyze_accuracy_freshness(
                    current_content, section_type, workflow_id, session_id)
                
            elif iteration == 2:  # Authority and citations
                iteration_context = self._analyze_authority_citations(
                    current_content, section_type, workflow_id, session_id)
            
            # Apply refinement based on tool analysis
            refined_content = self._apply_tool_refinement(
                current_content, iteration_context, section_type)
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'original_content': current_content,
                'refined_content': refined_content,
                'tool_context': iteration_context,
                'improvement_score': self._calculate_improvement_score(
                    current_content, refined_content, iteration_context)
            }
            
            iteration_results.append(iteration_result)
            
            # Cache iteration results
            self.tool_cache.cache_analysis_results(
                f"tool_refinement_{section_type.value}_iter_{iteration + 1}",
                iteration_result,
                session_id=session_id,
                workflow_id=workflow_id
            )
            
            current_content = refined_content
        
        return current_content
    
    def _analyze_structure_completeness(self, content: str, section_type: SectionType,
                                      workflow_id: Optional[str] = None,
                                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze content structure and completeness using tools."""
        context = {}
        
        # Query vector database for relevant context
        try:
            section_query = f"{section_type.value} structure best practices examples"
            vector_results = self.vector_store.search(query=section_query, top_k=5)
            
            vector_context = []
            for result in vector_results:
                title = getattr(result.metadata, 'title', '') if result.metadata else ''
                snippet = (result.content or '')[:200]
                vector_context.append(f"- {title}: {snippet}")
            
            context['vector_context'] = "\n".join(vector_context)
            context['vector_results_count'] = len(vector_results)
            
            # Cache vector results
            self.tool_cache.cache_vector_query(
                section_query, vector_results, top_k=5,
                agent_name="ToolAugmentedRefinement",
                session_id=session_id,
                workflow_id=workflow_id
            )
            
        except Exception as e:
            logger.warning(f"Vector search failed in structure analysis: {e}")
            context['vector_context'] = ""
            context['vector_results_count'] = 0
        
        # Analyze content gaps
        context['gap_analysis'] = self._analyze_content_gaps(content, context, section_type)
        
        logger.debug(f"Structure analysis complete: {context['vector_results_count']} vector results")
        return context
    
    def _analyze_accuracy_freshness(self, content: str, section_type: SectionType,
                                   workflow_id: Optional[str] = None,
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze content accuracy and freshness using claim validation."""
        context = {}
        
        # Extract claims from content
        try:
            claims = self.claim_extractor.extract_claims(content)
            context['extracted_claims'] = len(claims)
            
            # Validate top claims
            validation_results = []
            for claim in claims[:5]:  # Limit to top 5 claims
                try:
                    # Check cache first
                    cached_validation = self.tool_cache.get_cached_claim_validation(claim.text)
                    
                    if cached_validation:
                        validation_result = cached_validation
                        logger.debug(f"Using cached validation for claim: {claim.text[:50]}...")
                    else:
                        # Perform validation
                        validation_result = self.source_validator.validate_claim(
                            claim, workflow_id=workflow_id, session_id=session_id)
                        
                        # Cache the result
                        self.tool_cache.cache_claim_validation(
                            claim.text, validation_result,
                            agent_name="ToolAugmentedRefinement",
                            session_id=session_id,
                            workflow_id=workflow_id
                        )
                    
                    validation_results.append(validation_result)
                    
                except Exception as e:
                    logger.warning(f"Claim validation failed: {e}")
            
            context['validation_results'] = validation_results
            context['verified_claims'] = [r for r in validation_results 
                                        if r.validation_status == "supported"]
            
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            context['extracted_claims'] = 0
            context['validation_results'] = []
            context['verified_claims'] = []
        
        # Query recent developments
        try:
            topic = f"{section_type.value} recent developments"
            recent_developments = self.info_enricher.query_recent_developments(
                topic, max_results=3, workflow_id=workflow_id, session_id=session_id)
            
            context['recent_developments'] = recent_developments
            
        except Exception as e:
            logger.warning(f"Recent developments query failed: {e}")
            context['recent_developments'] = []
        
        logger.debug(f"Accuracy analysis complete: {len(context['verified_claims'])} verified claims")
        return context
    
    def _analyze_authority_citations(self, content: str, section_type: SectionType,
                                   workflow_id: Optional[str] = None,
                                   session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze source authority and generate citations."""
        context = {}
        
        # Get validation results from previous iteration (if cached)
        try:
            cached_analysis = self.tool_cache.get_cached_analysis(
                f"tool_refinement_{section_type.value}_iter_2")
            
            if cached_analysis and 'validation_results' in cached_analysis.get('tool_context', {}):
                validation_results = cached_analysis['tool_context']['validation_results']
                
                # Rank sources by authority
                from core.source_ranker import SourceAuthorityRanker
                ranker = SourceAuthorityRanker()
                
                all_sources = []
                for result in validation_results:
                    if result.sources:
                        all_sources.extend(result.sources)
                
                if all_sources:
                    ranked_sources = ranker.rank_sources(all_sources)
                    context['ranked_sources'] = ranked_sources
                    
                    # Generate citations
                    try:
                        citations = self.citation_generator.generate_citations(
                            validation_results, format_style="apa")
                        context['generated_citations'] = citations
                        
                    except Exception as e:
                        logger.warning(f"Citation generation failed: {e}")
                        context['generated_citations'] = ""
                else:
                    context['ranked_sources'] = []
                    context['generated_citations'] = ""
            else:
                context['ranked_sources'] = []
                context['generated_citations'] = ""
                
        except Exception as e:
            logger.warning(f"Authority analysis failed: {e}")
            context['ranked_sources'] = []
            context['generated_citations'] = ""
        
        logger.debug(f"Authority analysis complete: {len(context['ranked_sources'])} ranked sources")
        return context
    
    def _analyze_content_gaps(self, content: str, vector_context: Dict[str, Any], 
                            section_type: SectionType) -> Dict[str, Any]:
        """Analyze content gaps using vector database context."""
        gaps = []
        
        # Compare content length with typical sections
        word_count = len(content.split())
        
        expected_lengths = {
            SectionType.INTRODUCTION: (100, 300),
            SectionType.NEWS: (200, 500),
            SectionType.ANALYSIS: (400, 800),
            SectionType.TUTORIAL: (300, 600),
            SectionType.CONCLUSION: (100, 250)
        }
        
        min_length, max_length = expected_lengths.get(section_type, (200, 400))
        
        if word_count < min_length:
            gaps.append(f"Content too short ({word_count} words, expected {min_length}-{max_length})")
        elif word_count > max_length:
            gaps.append(f"Content too long ({word_count} words, expected {min_length}-{max_length})")
        
        # Check for section-specific elements
        if section_type == SectionType.TUTORIAL:
            if not re.search(r'\d+\.|\bstep\b', content.lower()):
                gaps.append("Tutorial lacks numbered steps or clear structure")
        
        elif section_type == SectionType.ANALYSIS:
            analysis_keywords = ['data', 'evidence', 'analysis', 'trend', 'pattern', 'insight']
            if not any(keyword in content.lower() for keyword in analysis_keywords):
                gaps.append("Analysis section lacks analytical language")
        
        elif section_type == SectionType.NEWS:
            if not any(keyword in content.lower() for keyword in 
                      ['announced', 'released', 'launched', 'recent', 'new']):
                gaps.append("News section lacks recent development indicators")
        
        return {
            'identified_gaps': gaps,
            'word_count': word_count,
            'expected_range': f"{min_length}-{max_length} words",
            'completeness_score': max(0.0, min(1.0, word_count / max_length))
        }
    
    def _apply_tool_refinement(self, content: str, iteration_context: Dict[str, Any],
                             section_type: SectionType) -> str:
        """Apply refinement based on tool analysis."""
        refined_content = content
        
        # Apply vector context improvements
        if 'vector_context' in iteration_context and iteration_context['vector_context']:
            refined_content = self._integrate_vector_context(
                refined_content, iteration_context['vector_context'], section_type)
        
        # Apply gap analysis improvements
        if 'gap_analysis' in iteration_context:
            refined_content = self._address_content_gaps(
                refined_content, iteration_context['gap_analysis'], section_type)
        
        # Integrate verified claims
        if 'verified_claims' in iteration_context:
            refined_content = self._integrate_verified_claims(
                refined_content, iteration_context['verified_claims'])
        
        # Add recent developments
        if 'recent_developments' in iteration_context:
            refined_content = self._integrate_recent_developments(
                refined_content, iteration_context['recent_developments'], section_type)
        
        # Add citations
        if 'generated_citations' in iteration_context and iteration_context['generated_citations']:
            refined_content = self._integrate_citations(
                refined_content, iteration_context['generated_citations'])
        
        return refined_content
    
    def _integrate_vector_context(self, content: str, vector_context: str, 
                                section_type: SectionType) -> str:
        """Integrate vector database context into content."""
        # Add context as background information
        if vector_context and len(vector_context.strip()) > 50:
            context_note = f"\n\n*Based on best practices and examples:*\n{vector_context}\n"
            
            # Insert context appropriately based on section type
            if section_type == SectionType.TUTORIAL:
                # Add at the beginning as prerequisites
                return f"## Prerequisites and Context\n{context_note}\n{content}"
            else:
                # Add at the end as additional context
                return f"{content}{context_note}"
        
        return content
    
    def _address_content_gaps(self, content: str, gap_analysis: Dict[str, Any],
                            section_type: SectionType) -> str:
        """Address identified content gaps."""
        refined_content = content
        gaps = gap_analysis.get('identified_gaps', [])
        
        for gap in gaps:
            if "too short" in gap:
                # Expand content based on section type
                if section_type == SectionType.ANALYSIS:
                    refined_content += "\n\n*[Additional analysis and insights would strengthen this section.]*"
                elif section_type == SectionType.TUTORIAL:
                    refined_content += "\n\n*[Additional steps and examples would improve completeness.]*"
                else:
                    refined_content += "\n\n*[This section could benefit from additional detail and examples.]*"
            
            elif "lacks numbered steps" in gap:
                # Add step structure to tutorial
                refined_content = self._add_step_structure(refined_content)
            
            elif "lacks analytical language" in gap:
                # Add analytical framing
                refined_content = f"## Analysis\n\n{refined_content}\n\n*This analysis reveals important patterns and trends worth monitoring.*"
        
        return refined_content
    
    def _integrate_verified_claims(self, content: str, verified_claims) -> str:
        """Integrate verified claims into content."""
        if not verified_claims:
            return content
        
        # Add verification note for high-confidence claims
        high_confidence_claims = [c for c in verified_claims if c.confidence > 0.7]
        
        if high_confidence_claims:
            verification_note = f"\n\n*Note: Key claims in this section have been verified against {len(high_confidence_claims)} authoritative sources.*"
            return f"{content}{verification_note}"
        
        return content
    
    def _integrate_recent_developments(self, content: str, developments, section_type: SectionType) -> str:
        """Integrate recent developments into content."""
        if not developments or section_type not in [SectionType.NEWS, SectionType.ANALYSIS]:
            return content
        
        recent_section = "\n\n## Recent Developments\n"
        for dev in developments[:3]:  # Limit to top 3
            recent_section += f"- **{dev.title}** - {dev.summary[:100]}{'...' if len(dev.summary) > 100 else ''}\n"
        
        return f"{content}{recent_section}"
    
    def _integrate_citations(self, content: str, citations: str) -> str:
        """Integrate generated citations into content."""
        if citations and citations.strip():
            return f"{content}\n\n{citations}"
        return content
    
    def _add_step_structure(self, content: str) -> str:
        """Add step structure to tutorial content."""
        lines = content.split('\n')
        structured_lines = []
        step_count = 1
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # Look for action words that indicate steps
                if any(word in line.lower() for word in ['create', 'install', 'configure', 'run', 'setup']):
                    if not re.match(r'^\d+\.', line.strip()):
                        line = f"{step_count}. {line.strip()}"
                        step_count += 1
            structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def _calculate_improvement_score(self, original: str, refined: str, context: Dict[str, Any]) -> float:
        """Calculate improvement score for iteration."""
        score = 0.0
        
        # Length improvement
        original_words = len(original.split())
        refined_words = len(refined.split())
        
        if refined_words > original_words:
            score += 0.1
        
        # Tool integration score
        if context.get('vector_results_count', 0) > 0:
            score += 0.2
        
        if context.get('verified_claims'):
            score += 0.3
        
        if context.get('recent_developments'):
            score += 0.2
        
        if context.get('generated_citations'):
            score += 0.2
        
        return min(1.0, score)


# Export main classes
__all__ = [
    'SectionAwareRefinementLoop',
    'ToolAugmentedRefinementLoop',
    'SectionBoundaryDetector',
    'SectionBoundary',
    'SectionContent',
    'RefinementResult',
    'RefinementPass'
]