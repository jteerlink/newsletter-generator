"""
Section-Aware Expansion System

This module implements the SectionExpansionOrchestrator for managing section-specific
content enhancement strategies. It coordinates targeted expansion based on content type,
template requirements, and quality preservation goals.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import core.core as core
from core.content_expansion import ExpansionOpportunity, ExpansionPriority, ExpansionStrategy
from core.template_manager import NewsletterTemplate, NewsletterType

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """Types of newsletter sections with specific characteristics."""
    INTRODUCTION = "introduction"
    TECHNICAL_ANALYSIS = "technical_analysis"
    TUTORIAL = "tutorial"
    CODE_EXAMPLES = "code_examples"
    CASE_STUDIES = "case_studies"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    FUTURE_OUTLOOK = "future_outlook"
    CONCLUSION = "conclusion"
    GENERAL_CONTENT = "general_content"


@dataclass
class SectionExpansionTemplate:
    """Template for expanding specific section types."""
    section_type: SectionType
    target_word_range: Tuple[int, int]  # (min, max) words
    expansion_strategies: List[ExpansionStrategy]
    content_elements: List[str]
    quality_requirements: Dict[str, Any]
    technical_depth_level: float  # 0.0-1.0


@dataclass
class SectionExpansionPlan:
    """Detailed plan for expanding a specific section."""
    section_name: str
    section_type: SectionType
    current_content: str
    current_word_count: int
    target_word_count: int
    expansion_strategy: ExpansionStrategy
    content_elements: List[str]
    expansion_prompt: str
    validation_criteria: Dict[str, Any]
    estimated_execution_time: float


@dataclass
class SectionExpansionResult:
    """Result of section expansion operation."""
    section_name: str
    original_content: str
    expanded_content: str
    original_word_count: int
    final_word_count: int
    expansion_achieved: int
    quality_score: float
    execution_time: float
    strategy_used: ExpansionStrategy
    success: bool
    error_message: Optional[str] = None


class SectionExpansionOrchestrator:
    """
    Orchestrates section-specific content expansion with template awareness.
    
    Manages the expansion of individual sections based on their type, current content,
    and target requirements while maintaining quality and technical accuracy.
    """
    
    def __init__(self):
        """Initialize the section expansion orchestrator."""
        self.expansion_templates = self._initialize_expansion_templates()
        self.section_classifiers = self._initialize_section_classifiers()
        self.content_generators = self._initialize_content_generators()
        
        logger.info("SectionExpansionOrchestrator initialized")
    
    def _initialize_expansion_templates(self) -> Dict[SectionType, SectionExpansionTemplate]:
        """Initialize expansion templates for different section types."""
        templates = {
            SectionType.INTRODUCTION: SectionExpansionTemplate(
                section_type=SectionType.INTRODUCTION,
                target_word_range=(200, 400),
                expansion_strategies=[
                    ExpansionStrategy.ANALYSIS_ENHANCEMENT,
                    ExpansionStrategy.COMPARATIVE_ANALYSIS
                ],
                content_elements=[
                    "Problem statement and context",
                    "Industry relevance and impact",
                    "Key concepts overview",
                    "Article structure preview"
                ],
                quality_requirements={
                    "clarity_score": 0.8,
                    "engagement_score": 0.7,
                    "technical_accuracy": 0.9
                },
                technical_depth_level=0.4
            ),
            
            SectionType.TECHNICAL_ANALYSIS: SectionExpansionTemplate(
                section_type=SectionType.TECHNICAL_ANALYSIS,
                target_word_range=(600, 1200),
                expansion_strategies=[
                    ExpansionStrategy.TECHNICAL_DEEP_DIVE,
                    ExpansionStrategy.IMPLEMENTATION_DETAILS
                ],
                content_elements=[
                    "Architecture and design patterns",
                    "Technical specifications",
                    "Performance considerations",
                    "Security implications",
                    "Scalability factors"
                ],
                quality_requirements={
                    "technical_accuracy": 0.95,
                    "depth_score": 0.9,
                    "implementation_detail": 0.8
                },
                technical_depth_level=0.9
            ),
            
            SectionType.TUTORIAL: SectionExpansionTemplate(
                section_type=SectionType.TUTORIAL,
                target_word_range=(500, 1000),
                expansion_strategies=[
                    ExpansionStrategy.TUTORIAL_ENHANCEMENT,
                    ExpansionStrategy.IMPLEMENTATION_DETAILS
                ],
                content_elements=[
                    "Step-by-step instructions",
                    "Code examples and explanations",
                    "Common pitfalls and solutions",
                    "Best practices and recommendations",
                    "Testing and validation"
                ],
                quality_requirements={
                    "clarity_score": 0.9,
                    "completeness": 0.8,
                    "actionability": 0.9
                },
                technical_depth_level=0.7
            ),
            
            SectionType.CODE_EXAMPLES: SectionExpansionTemplate(
                section_type=SectionType.CODE_EXAMPLES,
                target_word_range=(300, 600),
                expansion_strategies=[
                    ExpansionStrategy.IMPLEMENTATION_DETAILS,
                    ExpansionStrategy.TUTORIAL_ENHANCEMENT
                ],
                content_elements=[
                    "Complete code implementations",
                    "Detailed explanations",
                    "Alternative approaches",
                    "Error handling examples",
                    "Performance optimizations"
                ],
                quality_requirements={
                    "code_quality": 0.9,
                    "explanation_clarity": 0.8,
                    "completeness": 0.85
                },
                technical_depth_level=0.8
            ),
            
            SectionType.ANALYSIS: SectionExpansionTemplate(
                section_type=SectionType.ANALYSIS,
                target_word_range=(400, 800),
                expansion_strategies=[
                    ExpansionStrategy.ANALYSIS_ENHANCEMENT,
                    ExpansionStrategy.COMPARATIVE_ANALYSIS
                ],
                content_elements=[
                    "Detailed evaluation criteria",
                    "Quantitative and qualitative analysis",
                    "Trade-offs and considerations",
                    "Industry implications",
                    "Expert insights"
                ],
                quality_requirements={
                    "analytical_depth": 0.8,
                    "objectivity": 0.9,
                    "evidence_support": 0.8
                },
                technical_depth_level=0.6
            ),
            
            SectionType.CONCLUSION: SectionExpansionTemplate(
                section_type=SectionType.CONCLUSION,
                target_word_range=(150, 300),
                expansion_strategies=[
                    ExpansionStrategy.ANALYSIS_ENHANCEMENT
                ],
                content_elements=[
                    "Key takeaways summary",
                    "Practical implications",
                    "Future considerations",
                    "Call to action"
                ],
                quality_requirements={
                    "synthesis_quality": 0.8,
                    "actionability": 0.7,
                    "memorability": 0.7
                },
                technical_depth_level=0.5
            )
        }
        
        return templates
    
    def _initialize_section_classifiers(self) -> Dict[str, SectionType]:
        """Initialize patterns for classifying section types."""
        return {
            # Introduction patterns
            r'(?i)\b(intro|introduction|overview|summary|background)\b': SectionType.INTRODUCTION,
            
            # Technical analysis patterns  
            r'(?i)\b(technical|architecture|implementation|design|system|algorithm)\b': SectionType.TECHNICAL_ANALYSIS,
            
            # Tutorial patterns
            r'(?i)\b(tutorial|guide|how[\s-]?to|step[\s-]?by[\s-]?step|walkthrough)\b': SectionType.TUTORIAL,
            
            # Code examples patterns
            r'(?i)\b(code|example|implementation|demonstration|sample)\b': SectionType.CODE_EXAMPLES,
            
            # Analysis patterns
            r'(?i)\b(analysis|evaluation|assessment|comparison|review)\b': SectionType.ANALYSIS,
            
            # Conclusion patterns
            r'(?i)\b(conclusion|summary|final|takeaway|wrap[\s-]?up)\b': SectionType.CONCLUSION,
            
            # Future outlook patterns
            r'(?i)\b(future|outlook|trend|roadmap|prediction)\b': SectionType.FUTURE_OUTLOOK
        }
    
    def _initialize_content_generators(self) -> Dict[ExpansionStrategy, Dict[str, Any]]:
        """Initialize content generation strategies."""
        return {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: {
                "focus_areas": [
                    "architectural patterns",
                    "performance optimization", 
                    "security considerations",
                    "scalability design",
                    "implementation details"
                ],
                "content_structure": [
                    "technical_overview",
                    "detailed_explanation", 
                    "implementation_guidance",
                    "best_practices"
                ],
                "quality_emphasis": "technical_accuracy"
            },
            
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: {
                "focus_areas": [
                    "step_by_step_guidance",
                    "practical_examples",
                    "common_pitfalls",
                    "troubleshooting",
                    "validation_methods"
                ],
                "content_structure": [
                    "prerequisites",
                    "step_by_step_instructions",
                    "examples_and_demonstrations",
                    "validation_and_testing"
                ],
                "quality_emphasis": "clarity_and_completeness"
            },
            
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: {
                "focus_areas": [
                    "comparative_analysis",
                    "industry_context",
                    "benefits_and_limitations",
                    "expert_insights",
                    "future_implications"
                ],
                "content_structure": [
                    "analytical_framework",
                    "detailed_evaluation",
                    "comparative_assessment",
                    "implications_and_insights"
                ],
                "quality_emphasis": "analytical_depth"
            },
            
            ExpansionStrategy.IMPLEMENTATION_DETAILS: {
                "focus_areas": [
                    "code_examples",
                    "configuration_details",
                    "error_handling",
                    "performance_considerations",
                    "production_deployment"
                ],
                "content_structure": [
                    "implementation_overview",
                    "detailed_code_examples",
                    "configuration_and_setup",
                    "deployment_considerations"
                ],
                "quality_emphasis": "practical_applicability"
            }
        }
    
    def expand_section(self, section_name: str, content: str, target_word_count: int,
                      template_type: str, metadata: Dict[str, Any]) -> SectionExpansionResult:
        """
        Expand a specific section based on its type and requirements.
        
        Args:
            section_name: Name of the section to expand
            content: Current section content
            target_word_count: Target word count for the section
            template_type: Newsletter template type
            metadata: Additional context and configuration
            
        Returns:
            SectionExpansionResult with expansion details
        """
        start_time = time.time()
        current_word_count = len(content.split())
        
        logger.info(f"Expanding section '{section_name}': {current_word_count} → {target_word_count} words")
        
        try:
            # Step 1: Classify section type
            section_type = self._classify_section_type(section_name, content)
            
            # Step 2: Create expansion plan
            expansion_plan = self._create_section_expansion_plan(
                section_name, section_type, content, current_word_count, 
                target_word_count, template_type, metadata
            )
            
            # Step 3: Execute expansion
            expanded_content = self._execute_section_expansion(expansion_plan, metadata)
            
            # Step 4: Validate expanded content
            quality_score = self._validate_section_quality(
                expanded_content, section_type, expansion_plan.validation_criteria
            )
            
            execution_time = time.time() - start_time
            final_word_count = len(expanded_content.split())
            
            return SectionExpansionResult(
                section_name=section_name,
                original_content=content,
                expanded_content=expanded_content,
                original_word_count=current_word_count,
                final_word_count=final_word_count,
                expansion_achieved=final_word_count - current_word_count,
                quality_score=quality_score,
                execution_time=execution_time,
                strategy_used=expansion_plan.expansion_strategy,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Section expansion failed for '{section_name}': {e}")
            
            return SectionExpansionResult(
                section_name=section_name,
                original_content=content,
                expanded_content=content,  # Return original on failure
                original_word_count=current_word_count,
                final_word_count=current_word_count,
                expansion_achieved=0,
                quality_score=0.0,
                execution_time=execution_time,
                strategy_used=ExpansionStrategy.ANALYSIS_ENHANCEMENT,
                success=False,
                error_message=str(e)
            )
    
    def _classify_section_type(self, section_name: str, content: str) -> SectionType:
        """Classify the type of section based on name and content."""
        # First, try to classify based on section name
        section_name_lower = section_name.lower().strip()
        
        for pattern, section_type in self.section_classifiers.items():
            if re.search(pattern, section_name_lower):
                logger.debug(f"Classified '{section_name}' as {section_type.value} based on name")
                return section_type
        
        # If name-based classification fails, analyze content
        content_lower = content.lower()
        
        # Look for content indicators
        content_indicators = {
            SectionType.CODE_EXAMPLES: ['```', 'code', 'import', 'def ', 'class ', 'function'],
            SectionType.TUTORIAL: ['step', 'first', 'next', 'then', 'how to', 'guide'],
            SectionType.TECHNICAL_ANALYSIS: ['architecture', 'implementation', 'algorithm', 'system'],
            SectionType.ANALYSIS: ['analysis', 'comparison', 'evaluation', 'assessment'],
            SectionType.CONCLUSION: ['conclusion', 'summary', 'in summary', 'takeaway']
        }
        
        best_match = SectionType.GENERAL_CONTENT
        best_score = 0
        
        for section_type, indicators in content_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > best_score:
                best_score = score
                best_match = section_type
        
        logger.debug(f"Classified '{section_name}' as {best_match.value} based on content")
        return best_match
    
    def _create_section_expansion_plan(self, section_name: str, section_type: SectionType,
                                     content: str, current_word_count: int, target_word_count: int,
                                     template_type: str, metadata: Dict[str, Any]) -> SectionExpansionPlan:
        """Create detailed expansion plan for a section."""
        # Get template for this section type
        template = self.expansion_templates.get(section_type, 
                                               self.expansion_templates[SectionType.GENERAL_CONTENT])
        
        # Determine expansion strategy
        expansion_strategy = self._select_expansion_strategy(
            section_type, current_word_count, target_word_count, template_type
        )
        
        # Build content elements list
        content_elements = self._select_content_elements(
            section_type, expansion_strategy, metadata
        )
        
        # Create expansion prompt
        expansion_prompt = self._build_section_expansion_prompt(
            section_name, section_type, content, target_word_count - current_word_count,
            expansion_strategy, content_elements, metadata
        )
        
        # Define validation criteria
        validation_criteria = template.quality_requirements.copy()
        validation_criteria.update({
            'word_count_target': target_word_count,
            'word_count_tolerance': 0.2,  # 20% tolerance
            'technical_depth_required': template.technical_depth_level
        })
        
        # Estimate execution time
        words_to_generate = target_word_count - current_word_count
        estimated_time = self._estimate_section_expansion_time(
            words_to_generate, expansion_strategy, template.technical_depth_level
        )
        
        return SectionExpansionPlan(
            section_name=section_name,
            section_type=section_type,
            current_content=content,
            current_word_count=current_word_count,
            target_word_count=target_word_count,
            expansion_strategy=expansion_strategy,
            content_elements=content_elements,
            expansion_prompt=expansion_prompt,
            validation_criteria=validation_criteria,
            estimated_execution_time=estimated_time
        )
    
    def _select_expansion_strategy(self, section_type: SectionType, current_words: int,
                                 target_words: int, template_type: str) -> ExpansionStrategy:
        """Select the best expansion strategy for a section."""
        template = self.expansion_templates.get(section_type)
        if not template:
            return ExpansionStrategy.ANALYSIS_ENHANCEMENT
        
        # Get available strategies for this section type
        available_strategies = template.expansion_strategies
        
        # Consider the expansion magnitude
        expansion_ratio = target_words / max(1, current_words)
        
        # Strategy selection logic
        if section_type == SectionType.TECHNICAL_ANALYSIS:
            if expansion_ratio > 2.0:
                return ExpansionStrategy.TECHNICAL_DEEP_DIVE
            else:
                return ExpansionStrategy.IMPLEMENTATION_DETAILS
                
        elif section_type == SectionType.TUTORIAL:
            return ExpansionStrategy.TUTORIAL_ENHANCEMENT
            
        elif section_type == SectionType.CODE_EXAMPLES:
            return ExpansionStrategy.IMPLEMENTATION_DETAILS
            
        elif section_type in [SectionType.ANALYSIS, SectionType.COMPARISON]:
            if expansion_ratio > 1.5:
                return ExpansionStrategy.COMPARATIVE_ANALYSIS
            else:
                return ExpansionStrategy.ANALYSIS_ENHANCEMENT
        
        # Default to first available strategy
        return available_strategies[0] if available_strategies else ExpansionStrategy.ANALYSIS_ENHANCEMENT
    
    def _select_content_elements(self, section_type: SectionType, 
                               expansion_strategy: ExpansionStrategy,
                               metadata: Dict[str, Any]) -> List[str]:
        """Select appropriate content elements for expansion."""
        template = self.expansion_templates.get(section_type)
        base_elements = template.content_elements if template else []
        
        # Get strategy-specific elements
        strategy_config = self.content_generators.get(expansion_strategy, {})
        strategy_elements = strategy_config.get("focus_areas", [])
        
        # Combine and prioritize elements
        combined_elements = list(base_elements) + strategy_elements
        
        # Remove duplicates while preserving order
        seen = set()
        unique_elements = []
        for element in combined_elements:
            if element not in seen:
                seen.add(element)
                unique_elements.append(element)
        
        # Limit to most relevant elements
        return unique_elements[:5]
    
    def _build_section_expansion_prompt(self, section_name: str, section_type: SectionType,
                                      current_content: str, words_needed: int,
                                      strategy: ExpansionStrategy, elements: List[str],
                                      metadata: Dict[str, Any]) -> str:
        """Build detailed prompt for section expansion."""
        # Get strategy configuration
        strategy_config = self.content_generators.get(strategy, {})
        quality_emphasis = strategy_config.get("quality_emphasis", "overall_quality")
        
        # Build base prompt
        base_prompt = f"""
        Expand the '{section_name}' section for a technical newsletter about {metadata.get('topic', 'the subject')}.
        
        Current content:
        {current_content}
        
        Section type: {section_type.value}
        Expansion strategy: {strategy.value}
        Target additional words: {words_needed}
        Quality emphasis: {quality_emphasis}
        """
        
        # Add strategy-specific instructions
        strategy_instructions = self._get_strategy_instructions(strategy, section_type)
        
        # Add content elements guidance
        elements_text = "Focus on these content elements:\n" + "\n".join(f"- {element}" for element in elements)
        
        # Add quality requirements
        template = self.expansion_templates.get(section_type)
        quality_requirements = ""
        if template:
            quality_reqs = template.quality_requirements
            quality_requirements = f"""
            Quality requirements:
            - Technical accuracy: {quality_reqs.get('technical_accuracy', 0.8):.0%}
            - Content depth: {quality_reqs.get('depth_score', 0.7):.0%}
            - Clarity: {quality_reqs.get('clarity_score', 0.8):.0%}
            """
        
        # Combine all parts
        full_prompt = f"""
        {base_prompt}
        
        {strategy_instructions}
        
        {elements_text}
        
        {quality_requirements}
        
        Guidelines:
        - Maintain technical accuracy and professional tone
        - Provide specific, actionable information
        - Include relevant examples where appropriate
        - Ensure seamless integration with existing content
        - Target AI/ML professionals as the audience
        - Use clear, well-structured writing
        
        Generate the expanded content now:
        """
        
        return full_prompt.strip()
    
    def _get_strategy_instructions(self, strategy: ExpansionStrategy, 
                                 section_type: SectionType) -> str:
        """Get specific instructions for expansion strategy."""
        instructions = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: """
                Strategy: Technical Deep-Dive
                - Provide detailed technical explanations and analysis
                - Include architectural considerations and design patterns
                - Explain implementation details and technical trade-offs
                - Add performance and security considerations
                - Use appropriate technical terminology and concepts
            """,
            
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: """
                Strategy: Tutorial Enhancement
                - Provide step-by-step instructions and guidance
                - Include clear, actionable examples and demonstrations
                - Explain common pitfalls and how to avoid them
                - Add troubleshooting and validation steps
                - Ensure content is practical and immediately applicable
            """,
            
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: """
                Strategy: Analysis Enhancement
                - Provide deeper analytical insights and evaluation
                - Include comparative analysis with alternatives
                - Explain benefits, limitations, and trade-offs
                - Add industry context and expert perspectives
                - Support conclusions with evidence and reasoning
            """,
            
            ExpansionStrategy.IMPLEMENTATION_DETAILS: """
                Strategy: Implementation Details
                - Provide specific implementation guidance and examples
                - Include code samples and configuration details
                - Explain setup, deployment, and operational considerations
                - Add error handling and edge case scenarios
                - Focus on practical, real-world application
            """,
            
            ExpansionStrategy.COMPARATIVE_ANALYSIS: """
                Strategy: Comparative Analysis
                - Compare with alternative approaches and solutions
                - Analyze strengths and weaknesses of different options
                - Provide decision-making frameworks and criteria
                - Include use case scenarios and recommendations
                - Support comparisons with quantitative data where possible
            """
        }
        
        return instructions.get(strategy, "Strategy: General Enhancement\n- Provide relevant, high-quality content expansion")
    
    def _estimate_section_expansion_time(self, words_needed: int, strategy: ExpansionStrategy,
                                       technical_depth: float) -> float:
        """Estimate time required for section expansion."""
        # Base time per word (seconds)
        base_time_per_word = 0.03
        
        # Strategy complexity multipliers
        strategy_multipliers = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: 1.5,
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: 1.3,
            ExpansionStrategy.IMPLEMENTATION_DETAILS: 1.2,
            ExpansionStrategy.COMPARATIVE_ANALYSIS: 1.1,
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: 1.0
        }
        
        strategy_multiplier = strategy_multipliers.get(strategy, 1.0)
        depth_multiplier = 1.0 + (technical_depth * 0.5)  # Up to 50% increase for high technical depth
        
        estimated_time = words_needed * base_time_per_word * strategy_multiplier * depth_multiplier
        
        # Add overhead for quality validation
        estimated_time *= 1.1  # 10% overhead
        
        return estimated_time
    
    def _execute_section_expansion(self, plan: SectionExpansionPlan,
                                 metadata: Dict[str, Any]) -> str:
        """Execute section expansion based on the plan."""
        try:
            # Generate expanded content using LLM
            expanded_content = core.query_llm(plan.expansion_prompt)
            
            # Basic validation
            if not expanded_content or len(expanded_content.strip()) < 50:
                raise ValueError("Generated content is too short or empty")
            
            # Integrate with original content
            integrated_content = self._integrate_section_content(
                plan.current_content, expanded_content, plan.section_type
            )
            
            return integrated_content
            
        except Exception as e:
            logger.error(f"Section expansion execution failed: {e}")
            # Fallback to original content with minimal enhancement
            return self._generate_fallback_content(plan)
    
    def _integrate_section_content(self, original_content: str, expanded_content: str,
                                 section_type: SectionType) -> str:
        """Integrate expanded content with original content."""
        if not original_content.strip():
            return expanded_content
        
        # For most sections, append the expanded content
        if section_type in [SectionType.CONCLUSION, SectionType.INTRODUCTION]:
            # For intro/conclusion, blend more carefully
            return self._blend_content(original_content, expanded_content)
        else:
            # For other sections, append with proper spacing
            return f"{original_content}\n\n{expanded_content}"
    
    def _blend_content(self, original: str, expanded: str) -> str:
        """Blend content for sections that need smooth integration."""
        # Simple blending - append with transition
        if original.strip() and expanded.strip():
            return f"{original}\n\n{expanded}"
        return expanded if expanded.strip() else original
    
    def _generate_fallback_content(self, plan: SectionExpansionPlan) -> str:
        """Generate fallback content when main expansion fails."""
        words_needed = plan.target_word_count - plan.current_word_count
        
        fallback_content = {
            SectionType.TECHNICAL_ANALYSIS: f"""
                Technical considerations for this implementation include several key factors that impact both development and deployment decisions.
                
                **Architecture and Design**: The system architecture requires careful planning to ensure scalability and maintainability. Key design patterns and architectural principles should guide implementation decisions.
                
                **Performance Optimization**: Performance characteristics depend on various factors including algorithm complexity, data structures, and system resources. Optimization strategies should focus on critical performance bottlenecks.
                
                **Security and Reliability**: Security considerations encompass authentication, authorization, data protection, and system hardening. Reliability requirements include error handling, fault tolerance, and recovery mechanisms.
            """,
            
            SectionType.TUTORIAL: f"""
                **Implementation Steps**:
                
                1. **Initial Setup**: Begin by preparing the development environment and installing necessary dependencies.
                
                2. **Core Implementation**: Implement the main functionality following established best practices and design patterns.
                
                3. **Testing and Validation**: Thoroughly test the implementation to ensure correctness and handle edge cases.
                
                4. **Deployment and Monitoring**: Deploy the solution with appropriate monitoring and logging for production use.
            """,
            
            SectionType.ANALYSIS: f"""
                **Detailed Analysis**:
                
                The analysis of this approach reveals several important considerations that influence adoption and implementation decisions. Key factors include technical feasibility, resource requirements, and long-term maintenance implications.
                
                **Comparative Assessment**: When evaluated against alternative solutions, this approach offers specific advantages in certain scenarios while presenting trade-offs that require careful consideration.
                
                **Industry Impact**: The broader implications of this technology extend beyond immediate implementation to influence development practices and architectural decisions.
            """
        }
        
        base_content = fallback_content.get(plan.section_type, 
            "Additional context and detailed information provide valuable insights for understanding the complete scope and implications of this topic.")
        
        # Adjust length based on words needed
        if words_needed > 200:
            base_content += "\n\nFurther exploration reveals additional considerations that impact both technical implementation and strategic decision-making processes."
        
        if plan.current_content.strip():
            return f"{plan.current_content}\n\n{base_content}"
        return base_content
    
    def _validate_section_quality(self, content: str, section_type: SectionType,
                                criteria: Dict[str, Any]) -> float:
        """Validate quality of expanded section content."""
        quality_scores = []
        
        # Word count validation
        word_count = len(content.split())
        target_word_count = criteria.get('word_count_target', 0)
        tolerance = criteria.get('word_count_tolerance', 0.2)
        
        if target_word_count > 0:
            word_count_ratio = word_count / target_word_count
            word_count_score = max(0.0, 1.0 - abs(1.0 - word_count_ratio) / tolerance)
            quality_scores.append(word_count_score)
        
        # Technical depth validation
        required_depth = criteria.get('technical_depth_required', 0.5)
        technical_terms = len(re.findall(
            r'\b(?:implementation|architecture|algorithm|framework|optimization|analysis|system|approach|method)\b',
            content, re.IGNORECASE
        ))
        technical_density = technical_terms / max(1, word_count / 100)
        technical_score = min(1.0, technical_density / required_depth) if required_depth > 0 else 1.0
        quality_scores.append(technical_score)
        
        # Content structure validation
        has_structure = bool(re.search(r'[\n\r].*?[\n\r]', content))  # Multiple lines
        has_details = len(content.split('.')) > 3  # Multiple sentences
        structure_score = 0.8 if has_structure and has_details else 0.5
        quality_scores.append(structure_score)
        
        # Calculate overall quality score
        overall_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        logger.debug(f"Section quality validation: {overall_score:.2f} (word_count: {word_count_score if 'word_count_score' in locals() else 'N/A'}, technical: {technical_score}, structure: {structure_score})")
        
        return overall_score
    
    def get_supported_section_types(self) -> List[SectionType]:
        """Get list of supported section types."""
        return list(self.expansion_templates.keys())
    
    def get_expansion_template(self, section_type: SectionType) -> Optional[SectionExpansionTemplate]:
        """Get expansion template for a specific section type."""
        return self.expansion_templates.get(section_type)


# Export main classes
__all__ = [
    'SectionExpansionOrchestrator',
    'SectionType',
    'SectionExpansionTemplate',
    'SectionExpansionPlan',
    'SectionExpansionResult'
]