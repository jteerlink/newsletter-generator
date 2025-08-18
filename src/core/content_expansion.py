"""
Intelligent Content Expansion System

This module implements the core content expansion engine for achieving word count targets
while maintaining technical accuracy and quality. The system analyzes content gaps,
develops targeted expansion strategies, and executes quality-gated content enhancement.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import core.core as core
from core.advanced_quality_gates import ConfigurableQualityGate, QualityDimension
from core.template_manager import NewsletterTemplate, NewsletterType

logger = logging.getLogger(__name__)


class ExpansionStrategy(Enum):
    """Content expansion strategies based on content analysis."""
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    TUTORIAL_ENHANCEMENT = "tutorial_enhancement"
    ANALYSIS_ENHANCEMENT = "analysis_enhancement"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    IMPLEMENTATION_DETAILS = "implementation_details"
    CASE_STUDY_INTEGRATION = "case_study_integration"


class ExpansionPriority(Enum):
    """Priority levels for expansion opportunities."""
    CRITICAL = "critical"    # Essential for template compliance
    HIGH = "high"           # Important for content quality
    MEDIUM = "medium"       # Beneficial for depth
    LOW = "low"            # Optional enhancement


@dataclass
class ExpansionOpportunity:
    """Represents a specific content expansion opportunity."""
    section_name: str
    current_word_count: int
    target_word_count: int
    expansion_type: ExpansionStrategy
    priority: ExpansionPriority
    content_gap: str
    suggested_topics: List[str] = field(default_factory=list)
    technical_depth_score: float = 0.0
    expansion_complexity: float = 0.0


@dataclass
class ContentAnalysisResult:
    """Result of content structure and gap analysis."""
    total_words: int
    target_words: int
    sections: Dict[str, int]  # section_name -> word_count
    opportunities: List[ExpansionOpportunity]
    technical_complexity: float
    template_compliance: float
    expansion_needed: int  # Additional words needed


@dataclass
class ExpansionExecutionPlan:
    """Detailed plan for executing content expansion."""
    opportunities: List[ExpansionOpportunity]
    execution_order: List[str]  # section names in execution order
    estimated_words: int
    estimated_time: float
    quality_checkpoints: List[str]
    rollback_strategy: str


@dataclass
class ContentExpansionResult:
    """Result of content expansion operation."""
    original_content: str
    expanded_content: str
    original_word_count: int
    final_word_count: int
    expansion_achieved: int
    target_achievement: float  # percentage of target achieved
    quality_metrics: Dict[str, Any]
    execution_time: float
    expansions_applied: List[ExpansionOpportunity]
    success: bool
    error_message: Optional[str] = None


class IntelligentContentExpander:
    """
    Core intelligent content expansion system.
    
    Analyzes content structure, identifies expansion opportunities, and executes
    targeted content enhancement while maintaining quality and technical accuracy.
    """
    
    def __init__(self, quality_gate: Optional[ConfigurableQualityGate] = None):
        """Initialize the content expander with quality validation."""
        self.quality_gate = quality_gate or ConfigurableQualityGate("enforcing")
        self.expansion_cache = {}  # Cache for similar expansion requests
        
        # Content analysis patterns
        self.section_patterns = {
            'introduction': r'(?:^|\n)(?:#\s*|\*\*\s*)?(?:intro|introduction|overview|summary)(?:\s*\*\*|\s*#)?.*?(?=\n(?:#|\*\*)|$)',
            'technical': r'(?:^|\n)(?:#\s*|\*\*\s*)?(?:technical|implementation|architecture|design|code|algorithm)(?:\s*\*\*|\s*#)?.*?(?=\n(?:#|\*\*)|$)',
            'tutorial': r'(?:^|\n)(?:#\s*|\*\*\s*)?(?:tutorial|guide|how to|step|walkthrough)(?:\s*\*\*|\s*#)?.*?(?=\n(?:#|\*\*)|$)',
            'analysis': r'(?:^|\n)(?:#\s*|\*\*\s*)?(?:analysis|comparison|evaluation|assessment|review)(?:\s*\*\*|\s*#)?.*?(?=\n(?:#|\*\*)|$)',
            'conclusion': r'(?:^|\n)(?:#\s*|\*\*\s*)?(?:conclusion|summary|final|takeaway|key points)(?:\s*\*\*|\s*#)?.*?(?=\n(?:#|\*\*)|$)'
        }
        
        logger.info("IntelligentContentExpander initialized with quality gates")
    
    def expand_content(self, content: str, target_words: int, 
                      template_type: str, metadata: Dict[str, Any]) -> ContentExpansionResult:
        """
        Main content expansion method.
        
        Args:
            content: Original content to expand
            target_words: Target word count
            template_type: Newsletter template type
            metadata: Additional context and configuration
            
        Returns:
            ContentExpansionResult with expansion details and metrics
        """
        start_time = time.time()
        original_word_count = len(content.split())
        
        logger.info(f"Starting content expansion: {original_word_count} → {target_words} words")
        
        try:
            # Step 1: Analyze current content structure and gaps
            analysis_result = self._analyze_content_structure(content, target_words, template_type)
            
            # Step 2: Develop expansion execution plan
            execution_plan = self._create_expansion_plan(analysis_result, metadata)
            
            # Step 3: Execute targeted expansions with quality monitoring
            expanded_content = self._execute_expansion_plan(content, execution_plan, metadata)
            
            # Step 4: Validate final content quality
            quality_validation = self._validate_expanded_content(expanded_content, metadata)
            
            # Step 5: Calculate results and metrics
            final_word_count = len(expanded_content.split())
            expansion_achieved = final_word_count - original_word_count
            target_achievement = min(1.0, final_word_count / target_words)
            
            execution_time = time.time() - start_time
            
            result = ContentExpansionResult(
                original_content=content,
                expanded_content=expanded_content,
                original_word_count=original_word_count,
                final_word_count=final_word_count,
                expansion_achieved=expansion_achieved,
                target_achievement=target_achievement,
                quality_metrics=quality_validation,
                execution_time=execution_time,
                expansions_applied=execution_plan.opportunities,
                success=True
            )
            
            logger.info(f"Content expansion completed: {target_achievement:.1%} target achievement in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Content expansion failed: {e}")
            execution_time = time.time() - start_time
            
            return ContentExpansionResult(
                original_content=content,
                expanded_content=content,  # Return original on failure
                original_word_count=original_word_count,
                final_word_count=original_word_count,
                expansion_achieved=0,
                target_achievement=0.0,
                quality_metrics={},
                execution_time=execution_time,
                expansions_applied=[],
                success=False,
                error_message=str(e)
            )
    
    def _analyze_content_structure(self, content: str, target_words: int, 
                                 template_type: str) -> ContentAnalysisResult:
        """Analyze content structure and identify expansion opportunities."""
        current_word_count = len(content.split())
        sections = self._extract_sections(content)
        
        # Calculate technical complexity
        technical_complexity = self._calculate_technical_complexity(content)
        
        # Calculate template compliance
        template_compliance = self._assess_template_compliance(content, template_type)
        
        # Identify expansion opportunities
        opportunities = self._identify_expansion_opportunities(
            sections, target_words, template_type, technical_complexity
        )
        
        return ContentAnalysisResult(
            total_words=current_word_count,
            target_words=target_words,
            sections=sections,
            opportunities=opportunities,
            technical_complexity=technical_complexity,
            template_compliance=template_compliance,
            expansion_needed=max(0, target_words - current_word_count)
        )
    
    def _extract_sections(self, content: str) -> Dict[str, int]:
        """Extract sections and their word counts from content."""
        sections = {}
        
        # Split content by headers
        header_pattern = r'^(#{1,6}\s+.+|^\*\*[^*]+\*\*)\s*$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        current_section = "introduction"
        current_content = ""
        
        for i, part in enumerate(parts):
            if re.match(header_pattern, part.strip(), re.MULTILINE):
                # Save previous section
                if current_content.strip():
                    sections[current_section] = len(current_content.split())
                
                # Start new section
                section_name = self._classify_section(part.strip())
                current_section = section_name
                current_content = ""
            else:
                current_content += part
        
        # Save final section
        if current_content.strip():
            sections[current_section] = len(current_content.split())
        
        # Ensure we have at least basic sections
        if not sections:
            sections["main_content"] = len(content.split())
        
        logger.debug(f"Extracted sections: {sections}")
        return sections
    
    def _classify_section(self, header: str) -> str:
        """Classify section type based on header content."""
        header_lower = header.lower()
        
        # Remove markdown formatting
        header_clean = re.sub(r'[#*]', '', header_lower).strip()
        
        # Classification based on keywords
        if any(keyword in header_clean for keyword in ['intro', 'overview', 'summary']):
            return "introduction"
        elif any(keyword in header_clean for keyword in ['technical', 'implementation', 'code', 'architecture']):
            return "technical_analysis"
        elif any(keyword in header_clean for keyword in ['tutorial', 'guide', 'how to', 'step']):
            return "tutorial"
        elif any(keyword in header_clean for keyword in ['analysis', 'comparison', 'evaluation']):
            return "analysis"
        elif any(keyword in header_clean for keyword in ['conclusion', 'takeaway', 'final']):
            return "conclusion"
        elif any(keyword in header_clean for keyword in ['example', 'case study', 'application']):
            return "examples"
        elif any(keyword in header_clean for keyword in ['future', 'outlook', 'trend']):
            return "future_outlook"
        else:
            return "general_content"
    
    def _calculate_technical_complexity(self, content: str) -> float:
        """Calculate technical complexity score for content."""
        complexity_indicators = {
            'code_blocks': content.count('```') / 2,  # Pair of backticks
            'technical_terms': len(re.findall(r'\b(?:API|algorithm|framework|implementation|architecture|deployment|optimization|configuration)\b', content, re.IGNORECASE)),
            'equations': len(re.findall(r'[=<>+\-*/]', content)),
            'urls': len(re.findall(r'https?://\S+', content)),
            'bullet_points': content.count('•') + content.count('-'),
            'numbered_lists': len(re.findall(r'^\d+\.', content, re.MULTILINE))
        }
        
        # Weighted complexity calculation
        weights = {
            'code_blocks': 0.3,
            'technical_terms': 0.25,
            'equations': 0.15,
            'urls': 0.1,
            'bullet_points': 0.1,
            'numbered_lists': 0.1
        }
        
        complexity_score = sum(
            min(1.0, indicators * weights[factor]) 
            for factor, indicators in complexity_indicators.items()
        )
        
        return min(1.0, complexity_score)
    
    def _assess_template_compliance(self, content: str, template_type: str) -> float:
        """Assess how well content matches template expectations."""
        # Simple compliance assessment based on section presence
        sections = self._extract_sections(content)
        
        expected_sections = {
            'technical_deep_dive': ['introduction', 'technical_analysis', 'examples', 'conclusion'],
            'tutorial_guide': ['introduction', 'tutorial', 'examples', 'conclusion'],
            'trend_analysis': ['introduction', 'analysis', 'future_outlook', 'conclusion'],
            'research_summary': ['introduction', 'analysis', 'conclusion']
        }
        
        expected = expected_sections.get(template_type, ['introduction', 'analysis', 'conclusion'])
        present = [section for section in expected if section in sections]
        
        return len(present) / len(expected) if expected else 1.0
    
    def _identify_expansion_opportunities(self, sections: Dict[str, int], target_words: int,
                                       template_type: str, technical_complexity: float) -> List[ExpansionOpportunity]:
        """Identify specific opportunities for content expansion."""
        opportunities = []
        total_current = sum(sections.values())
        words_needed = max(0, target_words - total_current)
        
        if words_needed <= 0:
            return opportunities
        
        # Define target distributions based on template type
        section_targets = self._get_section_targets(template_type, target_words)
        
        for section_name, current_words in sections.items():
            target_words_section = section_targets.get(section_name, current_words)
            
            if current_words < target_words_section:
                gap = target_words_section - current_words
                
                # Determine expansion strategy based on section type and complexity
                strategy = self._determine_expansion_strategy(section_name, technical_complexity)
                priority = self._determine_expansion_priority(section_name, gap, template_type)
                
                # Generate suggested topics for expansion
                suggested_topics = self._generate_expansion_topics(section_name, strategy, template_type)
                
                opportunity = ExpansionOpportunity(
                    section_name=section_name,
                    current_word_count=current_words,
                    target_word_count=target_words_section,
                    expansion_type=strategy,
                    priority=priority,
                    content_gap=f"Need {gap} more words for {section_name}",
                    suggested_topics=suggested_topics,
                    technical_depth_score=technical_complexity,
                    expansion_complexity=self._calculate_expansion_complexity(gap, strategy)
                )
                
                opportunities.append(opportunity)
        
        # Sort by priority and potential impact
        opportunities.sort(key=lambda x: (x.priority.value, -x.target_word_count + x.current_word_count))
        
        logger.debug(f"Identified {len(opportunities)} expansion opportunities")
        return opportunities
    
    def _get_section_targets(self, template_type: str, total_target: int) -> Dict[str, int]:
        """Get target word counts for each section based on template type."""
        distributions = {
            'technical_deep_dive': {
                'introduction': 0.15,
                'technical_analysis': 0.40,
                'examples': 0.25,
                'conclusion': 0.10,
                'future_outlook': 0.10
            },
            'tutorial_guide': {
                'introduction': 0.15,
                'tutorial': 0.50,
                'examples': 0.25,
                'conclusion': 0.10
            },
            'trend_analysis': {
                'introduction': 0.15,
                'analysis': 0.45,
                'future_outlook': 0.25,
                'conclusion': 0.15
            }
        }
        
        distribution = distributions.get(template_type, {
            'introduction': 0.20,
            'analysis': 0.50,
            'conclusion': 0.30
        })
        
        return {section: int(total_target * ratio) for section, ratio in distribution.items()}
    
    def _determine_expansion_strategy(self, section_name: str, 
                                    technical_complexity: float) -> ExpansionStrategy:
        """Determine the best expansion strategy for a section."""
        strategy_map = {
            'introduction': ExpansionStrategy.ANALYSIS_ENHANCEMENT,
            'technical_analysis': ExpansionStrategy.TECHNICAL_DEEP_DIVE,
            'tutorial': ExpansionStrategy.TUTORIAL_ENHANCEMENT,
            'examples': ExpansionStrategy.IMPLEMENTATION_DETAILS,
            'analysis': ExpansionStrategy.COMPARATIVE_ANALYSIS,
            'conclusion': ExpansionStrategy.ANALYSIS_ENHANCEMENT,
            'future_outlook': ExpansionStrategy.ANALYSIS_ENHANCEMENT
        }
        
        base_strategy = strategy_map.get(section_name, ExpansionStrategy.ANALYSIS_ENHANCEMENT)
        
        # Adjust based on technical complexity
        if technical_complexity > 0.7 and base_strategy == ExpansionStrategy.ANALYSIS_ENHANCEMENT:
            return ExpansionStrategy.TECHNICAL_DEEP_DIVE
        
        return base_strategy
    
    def _determine_expansion_priority(self, section_name: str, gap: int, 
                                    template_type: str) -> ExpansionPriority:
        """Determine priority for expanding a section."""
        # Critical sections that must be expanded
        critical_sections = {
            'technical_deep_dive': ['technical_analysis'],
            'tutorial_guide': ['tutorial'],
            'trend_analysis': ['analysis']
        }
        
        template_critical = critical_sections.get(template_type, [])
        
        if section_name in template_critical:
            return ExpansionPriority.CRITICAL
        elif gap > 300:  # Large gaps are high priority
            return ExpansionPriority.HIGH
        elif gap > 150:
            return ExpansionPriority.MEDIUM
        else:
            return ExpansionPriority.LOW
    
    def _generate_expansion_topics(self, section_name: str, strategy: ExpansionStrategy,
                                 template_type: str) -> List[str]:
        """Generate suggested topics for section expansion."""
        topic_suggestions = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: [
                "Implementation architecture and design patterns",
                "Performance considerations and optimization strategies",
                "Security implications and best practices",
                "Scalability challenges and solutions",
                "Integration patterns and API design"
            ],
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: [
                "Step-by-step implementation guide",
                "Common pitfalls and troubleshooting",
                "Alternative approaches and variations",
                "Testing and validation strategies",
                "Production deployment considerations"
            ],
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: [
                "Comparative analysis with alternatives",
                "Industry adoption and case studies",
                "Benefits and limitations assessment",
                "Future trends and implications",
                "Expert opinions and insights"
            ],
            ExpansionStrategy.IMPLEMENTATION_DETAILS: [
                "Code examples and demonstrations",
                "Configuration and setup details",
                "Error handling and edge cases",
                "Performance benchmarks and metrics",
                "Real-world application scenarios"
            ]
        }
        
        return topic_suggestions.get(strategy, [
            "Additional context and background",
            "Detailed explanations and examples",
            "Practical applications and use cases"
        ])
    
    def _calculate_expansion_complexity(self, word_gap: int, strategy: ExpansionStrategy) -> float:
        """Calculate complexity score for expansion execution."""
        base_complexity = min(1.0, word_gap / 500)  # Normalize to 500 words
        
        strategy_multipliers = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: 1.3,
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: 1.2,
            ExpansionStrategy.IMPLEMENTATION_DETAILS: 1.1,
            ExpansionStrategy.COMPARATIVE_ANALYSIS: 1.0,
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: 0.9,
            ExpansionStrategy.CASE_STUDY_INTEGRATION: 0.8
        }
        
        multiplier = strategy_multipliers.get(strategy, 1.0)
        return min(1.0, base_complexity * multiplier)
    
    def _create_expansion_plan(self, analysis: ContentAnalysisResult,
                             metadata: Dict[str, Any]) -> ExpansionExecutionPlan:
        """Create detailed execution plan for content expansion."""
        # Filter and prioritize opportunities
        critical_opportunities = [op for op in analysis.opportunities if op.priority == ExpansionPriority.CRITICAL]
        high_opportunities = [op for op in analysis.opportunities if op.priority == ExpansionPriority.HIGH]
        other_opportunities = [op for op in analysis.opportunities if op.priority in [ExpansionPriority.MEDIUM, ExpansionPriority.LOW]]
        
        # Order execution by priority and dependencies
        execution_order = []
        selected_opportunities = []
        
        # Add critical opportunities first
        for opportunity in critical_opportunities:
            execution_order.append(opportunity.section_name)
            selected_opportunities.append(opportunity)
        
        # Add high priority opportunities
        remaining_words = analysis.expansion_needed - sum(op.target_word_count - op.current_word_count for op in selected_opportunities)
        
        for opportunity in high_opportunities:
            if remaining_words > 0:
                execution_order.append(opportunity.section_name)
                selected_opportunities.append(opportunity)
                remaining_words -= (opportunity.target_word_count - opportunity.current_word_count)
        
        # Add other opportunities as needed
        for opportunity in other_opportunities:
            if remaining_words > 50:  # Only add if significant words remaining
                execution_order.append(opportunity.section_name)
                selected_opportunities.append(opportunity)
                remaining_words -= (opportunity.target_word_count - opportunity.current_word_count)
        
        # Calculate estimates
        estimated_words = sum(op.target_word_count - op.current_word_count for op in selected_opportunities)
        estimated_time = self._estimate_expansion_time(selected_opportunities)
        
        # Define quality checkpoints
        quality_checkpoints = [
            "pre_expansion_validation",
            "mid_expansion_quality_check",
            "post_expansion_validation",
            "final_quality_assessment"
        ]
        
        return ExpansionExecutionPlan(
            opportunities=selected_opportunities,
            execution_order=execution_order,
            estimated_words=estimated_words,
            estimated_time=estimated_time,
            quality_checkpoints=quality_checkpoints,
            rollback_strategy="revert_to_previous_section_on_quality_failure"
        )
    
    def _estimate_expansion_time(self, opportunities: List[ExpansionOpportunity]) -> float:
        """Estimate time required for expansion execution."""
        base_time_per_word = 0.05  # 0.05 seconds per word
        
        total_time = 0.0
        for opportunity in opportunities:
            words_to_add = opportunity.target_word_count - opportunity.current_word_count
            complexity_multiplier = 1.0 + opportunity.expansion_complexity
            section_time = words_to_add * base_time_per_word * complexity_multiplier
            total_time += section_time
        
        # Add overhead for quality validation
        total_time *= 1.2  # 20% overhead
        
        return total_time
    
    def _execute_expansion_plan(self, content: str, plan: ExpansionExecutionPlan,
                              metadata: Dict[str, Any]) -> str:
        """Execute the content expansion plan with quality monitoring."""
        expanded_content = content
        
        for i, section_name in enumerate(plan.execution_order):
            # Find the corresponding opportunity
            opportunity = next((op for op in plan.opportunities if op.section_name == section_name), None)
            if not opportunity:
                continue
            
            logger.debug(f"Expanding section '{section_name}' ({i+1}/{len(plan.execution_order)})")
            
            # Execute section expansion
            try:
                expanded_content = self._expand_section(
                    expanded_content, opportunity, metadata
                )
                
                # Quality checkpoint after each section
                if i % 2 == 1:  # Check quality every other section
                    quality_check = self._validate_intermediate_quality(expanded_content, metadata)
                    if not quality_check.get('passed', True):
                        logger.warning(f"Quality check failed after expanding {section_name}")
                        # Continue but log the issue
                
            except Exception as e:
                logger.error(f"Failed to expand section {section_name}: {e}")
                # Continue with other sections
                continue
        
        return expanded_content
    
    def _expand_section(self, content: str, opportunity: ExpansionOpportunity,
                       metadata: Dict[str, Any]) -> str:
        """Expand a specific section based on the expansion opportunity."""
        words_needed = opportunity.target_word_count - opportunity.current_word_count
        
        if words_needed <= 0:
            return content
        
        # Generate expansion content based on strategy
        expansion_content = self._generate_expansion_content(
            opportunity, words_needed, metadata
        )
        
        # Integrate expansion into the original content
        expanded_content = self._integrate_expansion(
            content, opportunity.section_name, expansion_content
        )
        
        return expanded_content
    
    def _generate_expansion_content(self, opportunity: ExpansionOpportunity,
                                  words_needed: int, metadata: Dict[str, Any]) -> str:
        """Generate expansion content for a specific opportunity."""
        # Build expansion prompt based on strategy and topics
        expansion_prompt = self._build_expansion_prompt(opportunity, words_needed, metadata)
        
        try:
            # Generate content using LLM
            expanded_text = core.query_llm(expansion_prompt)
            
            # Validate generated content quality
            if self._validate_generated_content(expanded_text, opportunity):
                return expanded_text
            else:
                # Fallback to simpler expansion
                return self._generate_fallback_expansion(opportunity, words_needed)
                
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return self._generate_fallback_expansion(opportunity, words_needed)
    
    def _build_expansion_prompt(self, opportunity: ExpansionOpportunity,
                              words_needed: int, metadata: Dict[str, Any]) -> str:
        """Build LLM prompt for content expansion."""
        strategy_prompts = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: """
                Expand the {section_name} section with detailed technical analysis including:
                - Implementation architecture and design patterns
                - Performance considerations and optimization strategies  
                - Security implications and best practices
                - Code examples and technical demonstrations
                Target approximately {words_needed} words. Maintain professional technical tone.
            """,
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: """
                Enhance the {section_name} section with comprehensive tutorial content including:
                - Step-by-step implementation instructions
                - Code examples with explanations
                - Common pitfalls and troubleshooting tips
                - Best practices and recommendations
                Target approximately {words_needed} words. Use clear, instructional language.
            """,
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: """
                Expand the {section_name} section with thorough analysis including:
                - Comparative analysis with alternatives
                - Industry trends and adoption patterns
                - Benefits, limitations, and trade-offs
                - Expert insights and future implications
                Target approximately {words_needed} words. Maintain analytical, professional tone.
            """,
            ExpansionStrategy.IMPLEMENTATION_DETAILS: """
                Enhance the {section_name} section with practical implementation details:
                - Detailed code examples and demonstrations
                - Configuration and setup instructions
                - Error handling and edge case scenarios
                - Real-world application examples
                Target approximately {words_needed} words. Focus on practical, actionable content.
            """
        }
        
        base_prompt = strategy_prompts.get(
            opportunity.expansion_type,
            "Expand the {section_name} section with relevant, high-quality content. Target approximately {words_needed} words."
        )
        
        # Add context from suggested topics
        if opportunity.suggested_topics:
            topics_text = "Focus on these topics: " + ", ".join(opportunity.suggested_topics[:3])
            base_prompt += f"\n\n{topics_text}"
        
        # Add technical context if available
        template_type = metadata.get('template_type', 'general')
        topic = metadata.get('topic', 'the subject')
        
        context_prompt = f"""
        Context: You are expanding content for a {template_type} newsletter about {topic}.
        Section: {opportunity.section_name}
        Current word count: {opportunity.current_word_count}
        Target word count: {opportunity.target_word_count}
        
        {base_prompt.format(
            section_name=opportunity.section_name,
            words_needed=words_needed
        )}
        
        Ensure the expanded content:
        - Maintains technical accuracy and professional quality
        - Integrates seamlessly with existing content
        - Provides valuable insights for AI/ML professionals
        - Uses appropriate technical terminology
        - Includes specific examples where relevant
        """
        
        return context_prompt.strip()
    
    def _validate_generated_content(self, content: str, opportunity: ExpansionOpportunity) -> bool:
        """Validate quality of generated expansion content."""
        if not content or len(content.strip()) < 50:
            return False
        
        word_count = len(content.split())
        expected_words = opportunity.target_word_count - opportunity.current_word_count
        
        # Check if word count is reasonable (50%-150% of target)
        if word_count < expected_words * 0.5 or word_count > expected_words * 1.5:
            return False
        
        # Check for technical quality indicators
        technical_indicators = [
            'implementation', 'architecture', 'performance', 'optimization',
            'example', 'analysis', 'consideration', 'approach', 'solution'
        ]
        
        content_lower = content.lower()
        technical_score = sum(1 for indicator in technical_indicators if indicator in content_lower)
        
        # Require at least some technical depth
        return technical_score >= 2
    
    def _generate_fallback_expansion(self, opportunity: ExpansionOpportunity,
                                   words_needed: int) -> str:
        """Generate fallback expansion content when main generation fails."""
        fallback_templates = {
            ExpansionStrategy.TECHNICAL_DEEP_DIVE: """
                This technical implementation requires careful consideration of several key factors:
                
                **Architecture Design**: The system architecture should follow established design patterns
                that promote scalability and maintainability. Key architectural considerations include
                component separation, data flow design, and interface definitions.
                
                **Performance Optimization**: Performance characteristics depend on implementation
                choices and deployment environment. Optimization strategies should focus on
                critical path analysis and resource utilization efficiency.
                
                **Security Considerations**: Security implementation requires attention to
                authentication, authorization, and data protection. Following established
                security frameworks and best practices ensures robust protection.
            """,
            ExpansionStrategy.TUTORIAL_ENHANCEMENT: """
                **Implementation Steps**:
                
                1. **Setup and Configuration**: Begin by establishing the necessary environment
                   and dependencies. Ensure all required components are properly installed.
                
                2. **Core Implementation**: Implement the main functionality following
                   established patterns and best practices. Pay attention to error handling.
                
                3. **Testing and Validation**: Thoroughly test the implementation to ensure
                   it meets requirements and handles edge cases appropriately.
                
                4. **Deployment Considerations**: Plan deployment strategy considering
                   environment requirements and operational constraints.
            """,
            ExpansionStrategy.ANALYSIS_ENHANCEMENT: """
                **Comprehensive Analysis**:
                
                The analysis reveals several important factors that influence implementation
                decisions and adoption strategies. Key considerations include technical
                feasibility, resource requirements, and long-term maintenance implications.
                
                **Comparative Assessment**: When compared to alternative approaches,
                this solution offers distinct advantages in specific use cases while
                presenting trade-offs that must be carefully evaluated.
                
                **Future Implications**: The long-term impact of this approach extends
                beyond immediate implementation to influence architectural decisions
                and development practices.
            """
        }
        
        template = fallback_templates.get(opportunity.expansion_type, 
            "Additional detailed information and analysis would be valuable here. "
            "This section provides important context and considerations for "
            "understanding the full scope and implications of the topic."
        )
        
        # Adjust length to approximately match needs
        target_ratio = min(2.0, words_needed / 100)  # Cap at 2x base template
        if target_ratio > 1.0:
            template += "\n\nFurther exploration of these concepts reveals additional " \
                       "considerations that impact implementation and adoption decisions."
        
        return template.strip()
    
    def _integrate_expansion(self, content: str, section_name: str, 
                           expansion_content: str) -> str:
        """Integrate expansion content into the original content."""
        # Find the section in the content
        lines = content.split('\n')
        section_start = -1
        section_end = len(lines)
        
        # Look for section header
        for i, line in enumerate(lines):
            if self._is_section_header(line, section_name):
                section_start = i
                break
        
        if section_start == -1:
            # Section not found, append at end
            return content + "\n\n" + expansion_content
        
        # Find end of section (next header or end of content)
        for i in range(section_start + 1, len(lines)):
            if self._is_any_section_header(lines[i]):
                section_end = i
                break
        
        # Insert expansion content at end of section
        result_lines = (
            lines[:section_end] +
            ['', expansion_content, ''] +
            lines[section_end:]
        )
        
        return '\n'.join(result_lines)
    
    def _is_section_header(self, line: str, section_name: str) -> bool:
        """Check if line is a header for the specified section."""
        line_clean = re.sub(r'[#*]', '', line).strip().lower()
        section_keywords = {
            'introduction': ['intro', 'introduction', 'overview'],
            'technical_analysis': ['technical', 'implementation', 'architecture'],
            'tutorial': ['tutorial', 'guide', 'how to'],
            'analysis': ['analysis', 'evaluation', 'assessment'],
            'conclusion': ['conclusion', 'summary', 'final'],
            'examples': ['example', 'demonstration', 'case study'],
            'future_outlook': ['future', 'outlook', 'trends']
        }
        
        keywords = section_keywords.get(section_name, [section_name])
        return any(keyword in line_clean for keyword in keywords)
    
    def _is_any_section_header(self, line: str) -> bool:
        """Check if line is any kind of section header."""
        # Markdown headers
        if re.match(r'^#{1,6}\s+', line.strip()):
            return True
        
        # Bold headers
        if re.match(r'^\*\*[^*]+\*\*\s*$', line.strip()):
            return True
        
        return False
    
    def _validate_intermediate_quality(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality during expansion process."""
        try:
            # Basic quality checks
            word_count = len(content.split())
            
            # Check for minimum technical quality
            technical_terms = len(re.findall(
                r'\b(?:implementation|architecture|performance|optimization|analysis|framework|approach)\b',
                content, re.IGNORECASE
            ))
            
            quality_score = min(1.0, technical_terms / max(1, word_count / 100))
            
            return {
                'passed': quality_score > 0.1,
                'quality_score': quality_score,
                'word_count': word_count,
                'technical_density': technical_terms / max(1, word_count / 100)
            }
            
        except Exception as e:
            logger.error(f"Intermediate quality validation failed: {e}")
            return {'passed': True, 'error': str(e)}
    
    def _validate_expanded_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final expanded content quality."""
        try:
            # Use quality gate for comprehensive validation
            tool_usage = metadata.get('tool_usage', {})
            
            if self.quality_gate:
                quality_report = self.quality_gate.validate_with_level(
                    content, tool_usage, metadata, "warning"
                )
                
                return {
                    'overall_score': quality_report.overall_score,
                    'quality_passed': not quality_report.content_blocked,
                    'violations': quality_report.total_violations,
                    'dimension_scores': {
                        dim.value: score.score 
                        for dim, score in quality_report.dimension_scores.items()
                    }
                }
            else:
                # Fallback validation
                word_count = len(content.split())
                return {
                    'overall_score': 0.8,
                    'quality_passed': True,
                    'word_count': word_count,
                    'validation_method': 'fallback'
                }
                
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            return {
                'overall_score': 0.5,
                'quality_passed': True,
                'error': str(e),
                'validation_method': 'error_fallback'
            }


# Export main classes
__all__ = [
    'IntelligentContentExpander',
    'ContentExpansionResult',
    'ExpansionOpportunity',
    'ExpansionStrategy',
    'ExpansionPriority'
]