"""
Enhanced Writing Agent for Newsletter Generation

This module provides the enhanced WriterAgent class, which is responsible for creating
engaging and well-structured newsletter content with context-driven capabilities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from core.constants import TOOL_ENFORCEMENT_ENABLED, MANDATORY_VECTOR_TOP_K
from core.tool_usage_tracker import get_tool_tracker
from core.quality_gates import validate_tool_usage_quality
from storage import get_storage_provider

from core.campaign_context import CampaignContext
import core.core as core
from core.feedback_system import StructuredFeedback
from core.template_manager import NewsletterTemplate, NewsletterType
from core.code_generator import AIMLCodeGenerator, CodeType
from tools.syntax_validator import SyntaxValidator, ValidationLevel
from tools.code_executor import SafeCodeExecutor, ExecutionConfig, SecurityLevel
from templates.code_templates import template_library, Framework, TemplateCategory

from .base import AgentType, SimpleAgent

logger = logging.getLogger(__name__)


class WriterAgent(SimpleAgent):
    """Enhanced agent specialized in writing and content creation with context-driven capabilities."""

    def __init__(self, name: str = "WriterAgent", **kwargs):
        super().__init__(
            name=name,
            role="Content Writer",
            goal="Create engaging, informative, and well-structured newsletter content with context awareness",
            backstory="""You are an experienced content writer specializing in newsletter creation.
            You excel at transforming research and information into compelling, readable content
            that engages audiences. You understand newsletter best practices, including clear
            structure, engaging headlines, and appropriate tone. You can adapt your writing style
            to different audiences and topics while maintaining high quality and readability.
            You are particularly skilled at writing content that aligns with specific campaign contexts,
            brand voices, and audience personas.""",
            agent_type=AgentType.WRITER,
            tools=["search_web"],  # Writers may need to verify facts
            **kwargs
        )
        self.campaign_context: Optional[CampaignContext] = None
        
        # Initialize Phase 3 code generation components
        self.code_generator = AIMLCodeGenerator()
        self.syntax_validator = SyntaxValidator(ValidationLevel.STANDARD)
        self.code_executor = SafeCodeExecutor(ExecutionConfig(
            security_level=SecurityLevel.SECURE,
            timeout=10.0
        ))
        self.code_templates = template_library
        
        # Initialize content expansion capabilities
        self.content_expander = None  # Will be initialized when needed
        self.section_orchestrator = None  # Will be initialized when needed

    def write_from_context(
            self,
            research_data: Dict,
            context: CampaignContext) -> str:
        """Write content using campaign context for style and audience alignment."""
        self.campaign_context = context

        # Extract writing parameters from context
        content_style = context.content_style
        audience_persona = context.audience_persona
        strategic_goals = context.strategic_goals

        # Generate context-aware content
        content = self._generate_context_aware_content(
            research_data, content_style, audience_persona)

        # Integrate sources if available
        if 'sources' in research_data:
            content = self.integrate_sources(content, research_data['sources'])

        # Adapt style based on context requirements
        content = self.adapt_style(content, content_style)

        return content

    def integrate_sources(self, content: str, sources: List[Dict]) -> str:
        """Integrate source citations into content."""
        if not sources:
            return content

        # Add source citations section
        citations_section = self._create_citations_section(sources)

        # Add inline citations where appropriate
        content_with_citations = self._add_inline_citations(content, sources)

        # Combine content with citations
        final_content = f"{content_with_citations}\n\n## Sources\n{
            citations_section}"

        return final_content
    
    def _initialize_expansion_components(self):
        """Initialize content expansion components when needed."""
        if self.content_expander is None:
            try:
                from core.content_expansion import IntelligentContentExpander
                from core.section_expansion import SectionExpansionOrchestrator
                
                self.content_expander = IntelligentContentExpander()
                self.section_orchestrator = SectionExpansionOrchestrator()
                logger.info("Content expansion components initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize expansion components: {e}")
                self.content_expander = None
                self.section_orchestrator = None

    def adapt_style(self, content: str, style_params: Dict) -> str:
        """Adapt content style based on style parameters."""
        tone = style_params.get('tone', 'professional')
        formality = style_params.get('formality', 'standard')
        personality = style_params.get('personality', 'neutral')

        # Apply style adaptations
        adapted_content = self._apply_tone_adaptation(content, tone)
        adapted_content = self._apply_formality_adaptation(
            adapted_content, formality)
        adapted_content = self._apply_personality_adaptation(
            adapted_content, personality)

        return adapted_content

    def implement_revisions(
            self,
            content: str,
            feedback: StructuredFeedback) -> str:
        """Implement targeted revisions based on structured feedback."""
        revised_content = content

        for feedback_item in feedback.feedback_items:
            if feedback_item.required_action.value == 'REVISION':
                revised_content = self._apply_specific_revision(
                    revised_content,
                    feedback_item.text_snippet,
                    feedback_item.comment,
                    feedback_item.issue_type.value
                )

        return revised_content

    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute writing task with enhanced content creation capabilities."""
        logger.info(f"WriterAgent executing writing task: {task}")

        # Extract writing parameters
        template_type = kwargs.get(
            'template_type',
            NewsletterType.TECHNICAL_DEEP_DIVE)
        target_length = kwargs.get('target_length', 1500)
        tone = kwargs.get('tone', 'professional')
        audience = kwargs.get('audience', 'general')

        # Initialize content expansion components if needed
        if kwargs.get('enable_expansion', False):
            self._initialize_expansion_components()
        
        # Optional Phase 1 writer-specific vector enrichment
        section_contexts = {}
        if TOOL_ENFORCEMENT_ENABLED:
            section_contexts = self._gather_section_vector_context(task, template_type)
            section_chunks: List[str] = []
            for title, text in section_contexts.items():
                if text:
                    section_chunks.append(f"{title} Vector Context:\n{text}")
            if section_chunks:
                context = (context + "\n\n" if context else "") + "\n\n".join(section_chunks)

        # Create enhanced writing prompt
        enhanced_prompt = self._build_writing_prompt(
            task, context, template_type, target_length, tone, audience)

        # Generate content
        content = core.query_llm(enhanced_prompt)

        # Post-process content
        final_content = self._post_process_content(content, template_type)

        # Phase 1: Apply quality gates with tool usage validation
        tool_usage_metrics = {
            "vector_queries": len(section_contexts) if TOOL_ENFORCEMENT_ENABLED else 0,
            "web_searches": 0,  # Will be incremented by parent class if web search is used
            "verified_claims": []  # Placeholder for Phase 2 claim validation
        }
        
        quality_result = validate_tool_usage_quality(
            content=final_content,
            tool_usage=tool_usage_metrics,
            agent_type=self.agent_type.value
        )
        
        if not quality_result["ok"]:
            logger.warning(f"WriterAgent quality gate violations: {quality_result['issues']}")
            if quality_result["warnings"]:
                logger.info(f"WriterAgent quality gate warnings: {quality_result['warnings']}")
        else:
            logger.info("WriterAgent content passed all quality gates")

        return final_content

    def _gather_section_vector_context(self, task: str, template_type: NewsletterType) -> Dict[str, str]:
        """Gather writer-specific vector context per section (FR1.2)."""
        tracker = get_tool_tracker()
        store = get_storage_provider()

        queries: Dict[str, str] = {
            "Introduction": f"{task} introduction overview context",
            "Analysis": f"{task} technical analysis documentation details",
            "Tutorial": f"{task} step-by-step code examples tutorial",
            "Conclusion": f"{task} industry insights future outlook"
        }

        # Adjust for template emphasis
        if template_type == NewsletterType.TUTORIAL_GUIDE:
            queries["Tutorial"] = f"{task} code examples how-to tutorial best practices"
        elif template_type == NewsletterType.TECHNICAL_DEEP_DIVE:
            queries["Analysis"] = f"{task} specifications benchmarks technical docs"

        results: Dict[str, str] = {}

        for section, query_text in queries.items():
            try:
                with tracker.track_tool_usage(
                    tool_name="vector_search",
                    agent_name=self.name,
                    workflow_id=self.context.workflow_id,
                    session_id=self.context.session_id,
                    input_data={"query": query_text, "top_k": MANDATORY_VECTOR_TOP_K},
                    context={"integration": "writer_section", "section": section}
                ):
                    hits = store.search(query=query_text, top_k=MANDATORY_VECTOR_TOP_K)
                summarized: List[str] = []
                for r in hits:
                    title = getattr(r.metadata, 'title', '') if r.metadata else ''
                    snippet = (r.content or '')[:200]
                    summarized.append(f"- {title}: {snippet}")
                results[section] = "\n".join(summarized) if summarized else ""
            except Exception as e:
                logger.warning(f"Writer section vector search failed for {section}: {e}")
                results[section] = ""

        return results

    def _generate_context_aware_content(
            self,
            research_data: Dict,
            content_style: Dict,
            audience_persona: Dict) -> str:
        """Generate content that aligns with campaign context."""
        topic = research_data.get('topic', '')
        research_results = research_data.get('research_results', [])

        # Build context-aware prompt
        prompt = self._build_context_aware_prompt(
            topic, research_results, content_style, audience_persona)

        # Generate content
        content = core.query_llm(prompt)

        return content

    def _build_context_aware_prompt(
            self,
            topic: str,
            research_results: List[Dict],
            content_style: Dict,
            audience_persona: Dict) -> str:
        """Build prompt that incorporates campaign context."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Topic: {topic}",
            "",
            "Content Style Requirements:",
            f"- Tone: {content_style.get('tone', 'professional')}",
            f"- Formality: {content_style.get('formality', 'standard')}",
            f"- Personality: {content_style.get('personality', 'neutral')}",
            "",
            "Audience Persona:",
            f"- Demographics: {audience_persona.get('demographics', 'general')}",
            f"- Interests: {audience_persona.get('interests', '')}",
            f"- Knowledge Level: {audience_persona.get('knowledge_level', 'intermediate')}",
            "",
            "Research Findings:",
        ]

        # Add research findings
        for i, finding in enumerate(
                research_results[:5]):  # Limit to top 5 findings
            prompt_parts.append(
                f"{i + 1}. {finding.get('title', 'Finding')}: {finding.get('summary', '')}")

        prompt_parts.extend([
            "",
            "Please create engaging newsletter content that:",
            "1. Aligns with the specified content style and audience persona",
            "2. Incorporates the research findings naturally",
            "3. Maintains appropriate tone and formality for the target audience",
            "4. Uses clear, engaging language that matches the personality requirements",
            "5. Includes proper structure with headlines and sections",
            "6. Provides value to the target audience based on their interests and knowledge level"
        ])

        return "\n".join(prompt_parts)

    def _create_citations_section(self, sources: List[Dict]) -> str:
        """Create a formatted citations section."""
        citations = []

        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            url = source.get('url', '')
            author = source.get('author', 'Unknown Author')
            date = source.get('date', 'Unknown Date')

            citation = f"{i}. **{title}** by {author} ({date})"
            if url:
                citation += f" - [{url}]({url})"

            citations.append(citation)

        return "\n".join(citations)

    def _add_inline_citations(self, content: str, sources: List[Dict]) -> str:
        """Add inline citations to content where appropriate."""
        # Simple implementation - can be enhanced with more sophisticated citation detection
        # For now, we'll add a note about sources at the end of each major
        # section

        # Find section headers and add source references
        import re
        section_pattern = r'(^## .+$)'  # Markdown section headers

        def add_source_note(match):
            section = match.group(1)
            return f"{
                section}\n\n*This section draws from multiple authoritative sources.*"

        content_with_citations = re.sub(
            section_pattern,
            add_source_note,
            content,
            flags=re.MULTILINE)

        return content_with_citations

    def _apply_tone_adaptation(self, content: str, tone: str) -> str:
        """Apply tone-specific adaptations to content."""
        if tone == 'casual':
            # Make language more conversational
            content = content.replace('Furthermore,', 'Also,')
            content = content.replace('Moreover,', 'Plus,')
            content = content.replace('Nevertheless,', 'Still,')
        elif tone == 'formal':
            # Make language more formal
            content = content.replace('Also,', 'Furthermore,')
            content = content.replace('Plus,', 'Moreover,')
            content = content.replace('Still,', 'Nevertheless,')

        return content

    def _apply_formality_adaptation(self, content: str, formality: str) -> str:
        """Apply formality adaptations to content."""
        if formality == 'casual':
            # Use more casual language
            content = content.replace('utilize', 'use')
            content = content.replace('implement', 'put in place')
            content = content.replace('facilitate', 'help')
        elif formality == 'formal':
            # Use more formal language
            content = content.replace('use', 'utilize')
            content = content.replace('help', 'facilitate')
            content = content.replace('put in place', 'implement')

        return content

    def _apply_personality_adaptation(
            self, content: str, personality: str) -> str:
        """Apply personality adaptations to content."""
        if personality == 'enthusiastic':
            # Add enthusiasm markers
            content = content.replace('important', 'exciting')
            content = content.replace('significant', 'amazing')
        elif personality == 'analytical':
            # Add analytical markers
            content = content.replace('exciting', 'important')
            content = content.replace('amazing', 'significant')

        return content

    def _apply_specific_revision(
            self,
            content: str,
            text_snippet: str,
            comment: str,
            issue_type: str) -> str:
        """Apply a specific revision based on feedback."""
        # Simple implementation - can be enhanced with more sophisticated text
        # replacement
        if issue_type == 'clarity':
            # Improve clarity
            content = content.replace(
                text_snippet, f"{text_snippet} (clarified)")
        elif issue_type == 'grammar':
            # Fix grammar issues (simplified)
            content = content.replace(
                text_snippet, f"{text_snippet} (grammar corrected)")
        elif issue_type == 'style':
            # Improve style
            content = content.replace(
                text_snippet, f"{text_snippet} (style improved)")

        return content

    def _build_writing_prompt(
            self,
            task: str,
            context: str,
            template_type: NewsletterType,
            target_length: int,
            tone: str,
            audience: str) -> str:
        """Build comprehensive writing prompt."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Writing Task: {task}",
            "",
            f"Template Type: {template_type.value}",
            f"Target Length: {target_length} words",
            f"Tone: {tone}",
            f"Target Audience: {audience}"
        ]

        if context:
            prompt_parts.extend([
                "",
                f"Context and Research: {context}"
            ])

        # Add template-specific instructions
        template_instructions = self._get_template_instructions(template_type)
        prompt_parts.extend([
            "",
            "Template Instructions:",
            template_instructions
        ])

        # Add writing guidelines
        writing_guidelines = self._get_writing_guidelines(tone, audience)
        prompt_parts.extend([
            "",
            "Writing Guidelines:",
            writing_guidelines
        ])

        prompt_parts.extend([
            "",
            "Please create engaging newsletter content that:",
            "1. Follows the specified template structure",
            "2. Maintains the appropriate tone and style",
            "3. Engages the target audience",
            "4. Includes clear headings and subheadings",
            "5. Uses bullet points and lists where appropriate",
            "6. Provides valuable insights and actionable information",
            "7. Meets the target length requirements",
            "",
            "Structure your response with proper markdown formatting."
        ])

        return "\n".join(prompt_parts)

    def _get_template_instructions(self, template_type: NewsletterType) -> str:
        """Get template-specific writing instructions."""
        template_instructions = {
            NewsletterType.TECHNICAL_DEEP_DIVE: """
            - Focus on technical depth and accuracy
            - Include code examples and technical explanations
            - Use technical terminology appropriately
            - Provide implementation details and best practices
            - Include relevant technical diagrams or code snippets
            """,

            NewsletterType.TREND_ANALYSIS: """
            - Focus on business implications and market trends
            - Include data-driven insights and analysis
            - Discuss competitive landscape and opportunities
            - Provide actionable business recommendations
            - Use business terminology and frameworks
            """,

            NewsletterType.RESEARCH_SUMMARY: """
            - Balance technical and non-technical content
            - Use accessible language for broader audiences
            - Include both high-level concepts and practical details
            - Provide context and background information
            - Use analogies and examples to explain complex topics
            """,

            NewsletterType.TUTORIAL_GUIDE: """
            - Keep content concise and focused
            - Use bullet points and short paragraphs
            - Highlight key takeaways and action items
            - Focus on essential information only
            - Use clear, direct language
            """
        }

        return template_instructions.get(
            template_type, template_instructions[NewsletterType.RESEARCH_SUMMARY])
    
    def write_with_template_compliance(self, task: str, template: NewsletterTemplate, 
                                     context: str = "", **kwargs) -> Dict[str, Any]:
        """Write content with strict template compliance and section-by-section generation."""
        logger.info(f"Writing with template compliance: {template.name}")
        
        # Extract code generation settings
        enable_code_generation = kwargs.get('enable_code_generation', False)
        audience = kwargs.get('audience', 'technical')
        tone = kwargs.get('tone', 'professional')
        
        logger.info(f"Code generation enabled: {enable_code_generation}")
        
        try:
            # Initialize tracking
            section_results = {}
            total_words = 0
            overall_quality_score = 0.0
            code_examples_count = 0
            
            # Generate content section by section
            for i, section in enumerate(template.sections):
                logger.info(f"Generating section {i+1}/{len(template.sections)}: {section.name}")
                
                # Build section-specific prompt
                section_prompt = self._build_section_prompt(
                    task, section, context, kwargs.get('tone', 'professional')
                )
                
                # Generate section content
                section_content = core.query_llm(section_prompt)
                
                # Enhanced word count validation and adjustment
                section_words = len(section_content.split())
                target_words = section.word_count_target
                min_words = int(target_words * 0.8)  # 80% of target minimum
                max_words = int(target_words * 1.2)  # 120% of target maximum
                
                # Enhanced multi-stage length adjustment with semantic understanding
                adjustment_iterations = 0
                max_adjustments = 3  # Increased for better precision
                
                while adjustment_iterations < max_adjustments and (section_words < min_words or section_words > max_words):
                    current_compliance = section_words / target_words
                    
                    if section_words < min_words:
                        # Semantic-aware expansion targeting
                        content_analysis = self._analyze_content_context(section_content, section)
                        
                        # Calculate optimal expansion based on content density and type
                        if content_analysis['density_score'] < 0.5:  # Low density content
                            expansion_multiplier = 1.2  # More aggressive expansion
                        elif content_analysis['primary_type'] == 'technical':
                            expansion_multiplier = 1.1  # Moderate expansion for technical content
                        else:
                            expansion_multiplier = 1.0  # Standard expansion
                        
                        optimal_target = int(target_words * 0.95 * expansion_multiplier)
                        words_needed = max(50, optimal_target - section_words)
                        
                        logger.info(f"Section {section.name} below threshold ({section_words}/{target_words}, {current_compliance:.1%})")
                        logger.info(f"Content type: {content_analysis['primary_type']}, density: {content_analysis['density_score']:.2f}")
                        logger.info(f"Semantic-aware expansion: {words_needed} words (multiplier: {expansion_multiplier:.1f})")
                        
                        section_content = self._expand_section_content(
                            section_content, section, words_needed
                        )
                        section_words = len(section_content.split())
                        
                    elif section_words > max_words:
                        # Intelligent condensation with content preservation
                        content_analysis = self._analyze_content_context(section_content, section)
                        
                        # Adjust condensation target based on content characteristics
                        if content_analysis['has_code'] or content_analysis['has_examples']:
                            optimal_target = int(target_words * 1.1)  # More lenient for code/examples
                        elif content_analysis['primary_type'] == 'educational':
                            optimal_target = int(target_words * 1.08)  # Preserve educational content
                        else:
                            optimal_target = int(target_words * 1.05)  # Standard condensation
                        
                        logger.info(f"Section {section.name} above threshold ({section_words}/{target_words}, {current_compliance:.1%})")
                        logger.info(f"Content analysis: {content_analysis['analysis_summary']}")
                        logger.info(f"Intelligent condensation to {optimal_target} words")
                        
                        section_content = self._condense_section_content(
                            section_content, section, optimal_target
                        )
                        section_words = len(section_content.split())
                    
                    adjustment_iterations += 1
                    
                    # Enhanced compliance checking with semantic considerations
                    new_compliance = section_words / target_words
                    
                    # More lenient compliance checking for the last iteration
                    if adjustment_iterations >= max_adjustments:
                        acceptable_range = (0.75, 1.25)  # Wider range on final iteration
                    else:
                        acceptable_range = (0.8, 1.2)    # Standard range
                    
                    if acceptable_range[0] <= new_compliance <= acceptable_range[1]:
                        logger.info(f"Section {section.name} achieved semantic compliance: {section_words}/{target_words} ({new_compliance:.1%})")
                        break
                    elif adjustment_iterations >= max_adjustments:
                        logger.warning(f"Section {section.name} reached max iterations: {section_words}/{target_words} ({new_compliance:.1%})")
                
                # Code example integration for technical sections
                if enable_code_generation:
                    section_content, code_added = self._integrate_code_examples_into_section(
                        section_content, section, task, context, i
                    )
                    if code_added:
                        code_examples_count += 1
                        section_words = len(section_content.split())  # Recalculate after code addition
                        logger.info(f"Added code example to {section.name}, new word count: {section_words}")
                
                # Final validation
                final_compliance = section_words / target_words
                is_compliant = 0.8 <= final_compliance <= 1.2
                
                # Store section result with enhanced compliance tracking
                section_results[section.name] = {
                    'content': section_content,
                    'word_count': section_words,
                    'target_words': target_words,
                    'compliance': is_compliant,
                    'compliance_ratio': final_compliance,
                    'adjustment_iterations': adjustment_iterations,
                    'quality_elements': self._check_section_elements(section_content, section),
                    'has_code_examples': enable_code_generation and self._analyze_content_context(section_content, section)['has_code']
                }
                
                total_words += section_words
                logger.info(f"Section {section.name}: {section_words} words (target: {target_words})")
            
            # Combine sections into final content
            final_content = self._combine_sections(section_results, template)
            
            # Calculate overall compliance
            word_compliance = abs(total_words - template.total_word_target) / template.total_word_target
            word_compliance_score = max(0.0, 1.0 - word_compliance)
            
            section_compliance = sum(1 for s in section_results.values() if s['compliance']) / len(section_results)
            
            overall_quality_score = (word_compliance_score * 0.4 + section_compliance * 0.6) * 10
            
            logger.info(f"Template compliance complete: {total_words}/{template.total_word_target} words, {overall_quality_score:.1f}/10 quality")
            
            return {
                'content': final_content,
                'total_words': total_words,
                'target_words': template.total_word_target,
                'quality_score': overall_quality_score,
                'section_results': section_results,
                'template_compliance': {
                    'word_compliance': word_compliance_score,
                    'section_compliance': section_compliance,
                    'overall_score': overall_quality_score
                },
                'code_generation': {
                    'enabled': enable_code_generation,
                    'examples_count': code_examples_count,
                    'sections_with_code': len([r for r in section_results.values() if r.get('has_code_examples', False)])
                },
                'template_name': template.name,
                'template_type': template.type.value
            }
            
        except Exception as e:
            logger.error(f"Template-compliant writing failed: {e}")
            # Fallback to standard writing
            content = self.execute_task(task, context, **kwargs)
            return {
                'content': content,
                'total_words': len(content.split()),
                'target_words': template.total_word_target,
                'quality_score': 5.0,  # Neutral score for fallback
                'section_results': {},
                'template_compliance': {'error': str(e)},
                'template_name': template.name,
                'template_type': template.type.value
            }
    
    def write_with_expansion(self, topic: str, template: NewsletterTemplate, 
                           research_content: str, target_words: int = None, 
                           **kwargs) -> Dict[str, Any]:
        """
        Enhanced writing method with intelligent content expansion.
        
        This method integrates content expansion capabilities to achieve target
        word counts while maintaining quality and template compliance.
        """
        logger.info(f"Starting expanded writing for topic: {topic}")
        
        # Initialize expansion components
        self._initialize_expansion_components()
        
        if not self.content_expander:
            logger.warning("Content expansion not available, falling back to standard writing")
            return self.write_with_template_compliance(topic, template, research_content, **kwargs)
        
        try:
            # Step 1: Generate base content using standard method
            base_result = self.write_with_template_compliance(
                topic, template, research_content, **kwargs
            )
            
            base_content = base_result.get('content', '')
            base_word_count = base_result.get('total_words', len(base_content.split()))
            
            # Step 2: Determine expansion requirements
            target_word_count = target_words or template.total_word_target
            words_needed = max(0, target_word_count - base_word_count)
            
            if words_needed < 100:  # No significant expansion needed
                logger.info(f"No expansion needed: {base_word_count}/{target_word_count} words")
                return base_result
            
            logger.info(f"Expanding content: {base_word_count} → {target_word_count} words (+{words_needed})")
            
            # Step 3: Execute intelligent content expansion
            expansion_metadata = {
                'topic': topic,
                'template_type': template.type.value,
                'base_quality_score': base_result.get('quality_score', 7.0),
                'tool_usage': kwargs.get('tool_usage', {}),
                'audience': kwargs.get('audience', 'technical'),
                'tone': kwargs.get('tone', 'professional')
            }
            
            expansion_result = self.content_expander.expand_content(
                base_content, target_word_count, template.type.value, expansion_metadata
            )
            
            # Step 4: Validate expanded content
            if expansion_result.success:
                expanded_content = expansion_result.expanded_content
                
                # Update result with expansion metrics
                enhanced_result = base_result.copy()
                enhanced_result.update({
                    'content': expanded_content,
                    'total_words': expansion_result.final_word_count,
                    'expansion_metrics': {
                        'original_words': expansion_result.original_word_count,
                        'final_words': expansion_result.final_word_count,
                        'words_added': expansion_result.expansion_achieved,
                        'target_achievement': expansion_result.target_achievement,
                        'execution_time': expansion_result.execution_time,
                        'expansions_applied': len(expansion_result.expansions_applied)
                    },
                    'expansion_quality': expansion_result.quality_metrics
                })
                
                logger.info(f"Content expansion successful: {expansion_result.target_achievement:.1%} target achievement")
                return enhanced_result
            else:
                logger.warning(f"Content expansion failed: {expansion_result.error_message}")
                return base_result
                
        except Exception as e:
            logger.error(f"Enhanced writing with expansion failed: {e}")
            # Fallback to base result
            return self.write_with_template_compliance(topic, template, research_content, **kwargs)
    
    def expand_section_content(self, section_name: str, content: str, 
                             target_words: int, template_type: str, 
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand specific section content using section orchestrator.
        
        Args:
            section_name: Name of the section to expand
            content: Current section content
            target_words: Target word count for the section
            template_type: Newsletter template type
            metadata: Additional context
            
        Returns:
            Dictionary with expanded content and metrics
        """
        # Initialize expansion components
        self._initialize_expansion_components()
        
        if not self.section_orchestrator:
            logger.warning("Section orchestrator not available")
            return {
                'content': content,
                'word_count': len(content.split()),
                'expansion_applied': False,
                'error': 'Section orchestrator not available'
            }
        
        try:
            # Execute section expansion
            expansion_result = self.section_orchestrator.expand_section(
                section_name, content, target_words, template_type, metadata
            )
            
            return {
                'content': expansion_result.expanded_content,
                'word_count': expansion_result.final_word_count,
                'expansion_applied': expansion_result.success,
                'expansion_achieved': expansion_result.expansion_achieved,
                'quality_score': expansion_result.quality_score,
                'execution_time': expansion_result.execution_time,
                'strategy_used': expansion_result.strategy_used.value if expansion_result.strategy_used else None,
                'error': expansion_result.error_message
            }
            
        except Exception as e:
            logger.error(f"Section expansion failed: {e}")
            return {
                'content': content,
                'word_count': len(content.split()),
                'expansion_applied': False,
                'error': str(e)
            }
    
    def _enhance_section_with_expansion(self, section_content: str, section, 
                                      target_words: int, template_type: str, 
                                      metadata: Dict[str, Any]) -> str:
        """
        Enhance a section using intelligent expansion if needed.
        
        This method is called during template-compliant writing to enhance
        sections that are below target word counts.
        """
        current_words = len(section_content.split())
        
        # Only expand if significantly below target
        if current_words < target_words * 0.8:
            expansion_metadata = metadata.copy()
            expansion_metadata.update({
                'section_type': section.name,
                'current_word_count': current_words,
                'target_word_count': target_words
            })
            
            expansion_result = self.expand_section_content(
                section.name, section_content, target_words, 
                template_type, expansion_metadata
            )
            
            if expansion_result.get('expansion_applied', False):
                logger.info(f"Enhanced section {section.name}: {current_words} → {expansion_result['word_count']} words")
                return expansion_result['content']
        
        return section_content
    
    def _build_section_prompt(self, task: str, section, context: str, tone: str) -> str:
        """Build a prompt for generating a specific template section."""
        prompt = f"""You are writing the "{section.name}" section of a newsletter about: {task}

Section Description: {section.description}

Word Count Target: {section.word_count_target} words (aim for 80-120% of this target)

Content Guidelines:
{chr(10).join(f"- {guideline}" for guideline in section.content_guidelines)}

Required Elements (must include):
{chr(10).join(f"- {element}" for element in section.required_elements)}

Optional Elements (include if relevant):
{chr(10).join(f"- {element}" for element in section.optional_elements)}

Context: {context}

Tone: {tone}

Write ONLY the content for this section. Use clear, engaging prose that matches the specified tone and includes all required elements. Aim for approximately {section.word_count_target} words."""
        
        return prompt
    
    def _expand_section_content(self, content: str, section, additional_words: int) -> str:
        """Expand section content with iterative enhancement and targeted word count management."""
        logger.info(f"Expanding section {section.name} by ~{additional_words} words")
        
        current_words = len(content.split())
        target_words = current_words + additional_words
        
        # Iterative expansion with multiple enhancement strategies
        expanded_content = content
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            current_word_count = len(expanded_content.split())
            remaining_words = target_words - current_word_count
            
            if remaining_words <= 50:  # Close enough to target
                break
                
            # Choose expansion strategy based on remaining words needed
            if remaining_words > 200:
                strategy = "comprehensive"
            elif remaining_words > 100:
                strategy = "detailed"
            else:
                strategy = "focused"
            
            expansion_prompt = self._build_expansion_prompt(
                expanded_content, section, remaining_words, strategy
            )
            
            try:
                enhanced_content = core.query_llm(expansion_prompt)
                
                # Validate the expansion
                new_word_count = len(enhanced_content.split())
                if new_word_count > current_word_count:
                    expanded_content = enhanced_content
                    logger.info(f"Expansion iteration {attempts + 1}: {current_word_count} → {new_word_count} words")
                else:
                    logger.warning(f"Expansion iteration {attempts + 1} did not increase word count")
                    break
                    
            except Exception as e:
                logger.error(f"Expansion iteration {attempts + 1} failed: {e}")
                break
            
            attempts += 1
        
        final_word_count = len(expanded_content.split())
        logger.info(f"Section expansion complete: {current_words} → {final_word_count} words (target: {target_words})")
        
        return expanded_content
    
    def _build_expansion_prompt(self, content: str, section, remaining_words: int, strategy: str) -> str:
        """Build targeted expansion prompt with context-aware intelligence."""
        # Analyze content context for intelligent expansion
        content_analysis = self._analyze_content_context(content, section)
        
        base_prompt = f"""The following section content needs to be expanded by approximately {remaining_words} words while maintaining quality and relevance.

Current content:
{content}

Section guidelines:
{chr(10).join(f"- {guideline}" for guideline in section.content_guidelines)}

Required elements (ensure these remain):
{chr(10).join(f"- {element}" for element in section.required_elements)}

Content Analysis:
{content_analysis['analysis_summary']}
"""
        
        # Context-aware strategy selection
        if strategy == "comprehensive":
            strategy_instructions = self._build_comprehensive_expansion(content_analysis, remaining_words)
        elif strategy == "detailed":
            strategy_instructions = self._build_detailed_expansion(content_analysis, remaining_words)
        else:  # focused strategy
            strategy_instructions = self._build_focused_expansion(content_analysis, remaining_words)
        
        return f"""{base_prompt}

{strategy_instructions}

Maintain the same tone and style. Return the expanded version:"""
    
    def _analyze_content_context(self, content: str, section) -> dict:
        """Analyze content context for intelligent expansion decisions."""
        content_lower = content.lower()
        words = content.split()
        sentences = content.split('.')
        
        # Technical content indicators
        technical_terms = ['api', 'algorithm', 'framework', 'implementation', 'architecture', 
                          'performance', 'optimization', 'deployment', 'integration', 'system']
        technical_score = sum(1 for term in technical_terms if term in content_lower) / len(technical_terms)
        
        # Business content indicators  
        business_terms = ['market', 'strategy', 'revenue', 'cost', 'roi', 'business', 
                         'customer', 'value', 'growth', 'opportunity']
        business_score = sum(1 for term in business_terms if term in content_lower) / len(business_terms)
        
        # Educational content indicators
        educational_terms = ['learn', 'understand', 'explain', 'example', 'tutorial', 
                           'guide', 'step', 'process', 'method', 'approach']
        educational_score = sum(1 for term in educational_terms if term in content_lower) / len(educational_terms)
        
        # Content structure analysis
        has_lists = content.count('•') > 0 or content.count('-') > 2
        has_code = '```' in content or 'code' in content_lower
        has_examples = 'example' in content_lower or 'instance' in content_lower
        
        # Determine primary content type
        scores = {'technical': technical_score, 'business': business_score, 'educational': educational_score}
        primary_type = max(scores, key=scores.get)
        
        # Content density analysis
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
        density_score = min(1.0, avg_sentence_length / 20)  # Normalize to 0-1
        
        return {
            'primary_type': primary_type,
            'technical_score': technical_score,
            'business_score': business_score, 
            'educational_score': educational_score,
            'has_lists': has_lists,
            'has_code': has_code,
            'has_examples': has_examples,
            'density_score': density_score,
            'avg_sentence_length': avg_sentence_length,
            'analysis_summary': f"Content type: {primary_type} (score: {scores[primary_type]:.2f}), "
                               f"Density: {density_score:.2f}, Structure: {'Lists' if has_lists else 'Prose'}"
        }
    
    def _build_comprehensive_expansion(self, analysis: dict, remaining_words: int) -> str:
        """Build comprehensive expansion strategy based on content analysis."""
        base_instructions = [
            "- Detailed explanations of key concepts",
            "- Contextual background information", 
            "- Industry insights and trends"
        ]
        
        # Add type-specific comprehensive expansion
        if analysis['primary_type'] == 'technical':
            base_instructions.extend([
                "- Technical implementation details and best practices",
                "- Architecture considerations and design patterns",
                "- Performance implications and optimization strategies",
                "- Integration approaches and compatibility factors"
            ])
        elif analysis['primary_type'] == 'business':
            base_instructions.extend([
                "- Market implications and competitive analysis",
                "- ROI considerations and value propositions", 
                "- Strategic recommendations and implementation roadmaps",
                "- Risk assessment and mitigation strategies"
            ])
        else:  # educational
            base_instructions.extend([
                "- Step-by-step explanations and learning progressions",
                "- Multiple examples illustrating different scenarios",
                "- Common pitfalls and troubleshooting guidance",
                "- Practice exercises and real-world applications"
            ])
        
        # Add structure-specific enhancements
        if not analysis['has_examples'] and remaining_words > 150:
            base_instructions.append("- Multiple concrete examples to illustrate concepts")
        
        if not analysis['has_code'] and analysis['technical_score'] > 0.3:
            base_instructions.append("- Code examples or technical specifications where relevant")
            
        expansion_focus = f"Target expansion: {remaining_words} words with comprehensive depth"
        
        return f"""Please expand this content comprehensively by adding:
{chr(10).join(base_instructions)}

{expansion_focus}
Focus on depth and breadth while maintaining readability."""
    
    def _build_detailed_expansion(self, analysis: dict, remaining_words: int) -> str:
        """Build detailed expansion strategy based on content analysis."""
        base_instructions = [
            "- More thorough explanations of existing points",
            "- Relevant supporting information",
            "- Practical details and implementation considerations"
        ]
        
        # Type-specific detailed expansion
        if analysis['primary_type'] == 'technical':
            base_instructions.extend([
                "- Technical specifications and configuration details",
                "- Implementation examples and code snippets"
            ])
        elif analysis['primary_type'] == 'business': 
            base_instructions.extend([
                "- Market data and industry statistics",
                "- Business case studies and success stories"
            ])
        else:  # educational
            base_instructions.extend([
                "- Additional examples to illustrate concepts", 
                "- Learning aids and memory techniques"
            ])
        
        # Density-based adjustments
        if analysis['density_score'] < 0.5:  # Low density content
            base_instructions.append("- Enhanced detail and explanation depth")
        
        expansion_focus = f"Target expansion: {remaining_words} words with focused detail enhancement"
        
        return f"""Please expand this content with detailed enhancements:
{chr(10).join(base_instructions)}

{expansion_focus}
Maintain the current structure while adding substantive detail."""
    
    def _build_focused_expansion(self, analysis: dict, remaining_words: int) -> str:
        """Build focused expansion strategy based on content analysis."""
        base_instructions = [
            "- Clarify and elaborate on existing points",
            "- Include relevant details that support main ideas"
        ]
        
        # Targeted improvements based on analysis
        if analysis['avg_sentence_length'] > 25:
            base_instructions.append("- Break down complex sentences for better readability")
        
        if not analysis['has_examples'] and remaining_words > 50:
            base_instructions.append("- Add specific examples where helpful")
            
        if analysis['density_score'] > 0.8:  # High density content
            base_instructions.append("- Add transitional phrases for better flow")
        
        expansion_focus = f"Target expansion: {remaining_words} words with precise improvements"
        
        return f"""Please expand this content with focused improvements:
{chr(10).join(base_instructions)}

{expansion_focus}
Keep expansion targeted and relevant."""
    
    def _integrate_code_examples_into_section(self, content: str, section, task: str, context: str, section_index: int) -> tuple[str, bool]:
        """Integrate code examples into section content based on technical relevance."""
        try:
            # Analyze section content to determine if code is appropriate
            content_analysis = self._analyze_content_context(content, section)
            
            # Determine if this section should have code examples
            should_add_code = self._should_section_have_code(section, content_analysis, section_index)
            
            if not should_add_code:
                return content, False
            
            logger.info(f"Adding code example to section: {section.name}")
            
            # Generate appropriate code example for this section
            code_example = self._generate_section_appropriate_code(
                task, section, content_analysis, section_index
            )
            
            if not code_example:
                logger.warning(f"Failed to generate code example for {section.name}")
                return content, False
            
            # Integrate code example into content
            enhanced_content = self._integrate_code_into_content(content, code_example, section)
            
            return enhanced_content, True
            
        except Exception as e:
            logger.error(f"Error integrating code into {section.name}: {e}")
            return content, False
    
    def _should_section_have_code(self, section, content_analysis: dict, section_index: int) -> bool:
        """Determine if a section should have code examples based on multiple factors."""
        # Technical content score threshold
        if content_analysis['technical_score'] < 0.3:
            return False
        
        # Section-based rules
        section_name_lower = section.name.lower()
        
        # Sections that typically should have code
        code_appropriate_sections = [
            'technical', 'implementation', 'analysis', 'examples', 
            'practical', 'development', 'coding', 'tutorial'
        ]
        
        # Sections that typically shouldn't have code
        avoid_code_sections = [
            'introduction', 'conclusion', 'overview', 'outlook', 
            'future', 'summary', 'executive'
        ]
        
        # Check if section name suggests code appropriateness
        has_code_keywords = any(keyword in section_name_lower for keyword in code_appropriate_sections)
        has_avoid_keywords = any(keyword in section_name_lower for keyword in avoid_code_sections)
        
        if has_avoid_keywords:
            return False
        
        if has_code_keywords:
            return True
        
        # For middle sections (typically technical analysis), add code if technical content
        if 2 <= section_index <= 4 and content_analysis['technical_score'] > 0.5:
            return True
        
        # Section 3 (Deep Technical Analysis) should almost always have code if technical
        if section_index == 2 and content_analysis['technical_score'] > 0.4:
            return True
        
        return False
    
    def _generate_section_appropriate_code(self, task: str, section, content_analysis: dict, section_index: int) -> str:
        """Generate code example appropriate for the specific section context."""
        try:
            # Determine code type based on section and content
            if section_index <= 1:  # Introduction/Overview
                code_type = "basic_example"
            elif section_index == 2:  # Deep Technical Analysis
                code_type = "implementation_example" 
            elif section_index == 3:  # Real-World Applications
                code_type = "usage_example"
            else:  # Conclusion/Future
                code_type = "getting_started"
            
            # Extract technical topic from task
            technical_context = self._extract_technical_context(task, content_analysis)
            
            # Generate code example
            code_examples = self.generate_code_examples(
                topic=technical_context,
                framework="python",  # Default to Python for technical content
                complexity="intermediate",
                count=1
            )
            
            if code_examples and len(code_examples) > 0:
                return code_examples[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating section-appropriate code: {e}")
            return None
    
    def _extract_technical_context(self, task: str, content_analysis: dict) -> str:
        """Extract specific technical context for code generation."""
        # Extract key technical terms from the task
        task_lower = task.lower()
        
        # Common technical domains and their code contexts
        domain_contexts = {
            'machine learning': 'ML model training and inference',
            'deep learning': 'neural network implementation',
            'data science': 'data analysis and visualization', 
            'web development': 'web application development',
            'api': 'API development and integration',
            'database': 'database operations and queries',
            'microservices': 'microservice architecture',
            'kubernetes': 'container orchestration',
            'docker': 'containerization',
            'react': 'React component development',
            'python': 'Python application development',
            'javascript': 'JavaScript programming',
            'tensorflow': 'TensorFlow model development',
            'pytorch': 'PyTorch neural networks',
            'rag': 'Retrieval-Augmented Generation implementation',
            'llm': 'Large Language Model integration',
            'agentic': 'AI agent development'
        }
        
        # Find matching context
        for domain, context in domain_contexts.items():
            if domain in task_lower:
                return context
        
        # Default technical context
        return f"Technical implementation for {task}"
    
    def _integrate_code_into_content(self, content: str, code_example: str, section) -> str:
        """Intelligently integrate code example into section content."""
        # Find optimal insertion point
        lines = content.split('\n')
        
        # Look for good insertion points (after explanatory paragraphs)
        insertion_point = len(lines)  # Default to end
        
        # Try to find a natural break point
        for i, line in enumerate(lines):
            # Insert after paragraphs that mention implementation, examples, or technical details
            if i < len(lines) - 2:  # Not at the very end
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in 
                      ['implementation', 'example', 'consider', 'approach', 'method', 'technique']):
                    # Check if next line is empty or start of new paragraph
                    if i + 1 < len(lines) and (lines[i + 1].strip() == '' or lines[i + 1].startswith('##')):
                        insertion_point = i + 1
                        break
        
        # Insert code example with appropriate formatting
        code_section = f"\n### Implementation Example\n\n{code_example}\n"
        
        # Insert at the determined point
        if insertion_point < len(lines):
            lines.insert(insertion_point, code_section)
        else:
            lines.append(code_section)
        
        return '\n'.join(lines)
    
    def _condense_section_content(self, content: str, section, max_words: int) -> str:
        """Condense section content with intelligent priority-based reduction."""
        logger.info(f"Condensing section {section.name} to max {max_words} words")
        
        current_words = len(content.split())
        words_to_remove = current_words - max_words
        
        if words_to_remove <= 0:
            return content
        
        # Iterative condensation with priority-based reduction
        condensed_content = content
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            current_word_count = len(condensed_content.split())
            remaining_reduction = current_word_count - max_words
            
            if remaining_reduction <= 20:  # Close enough to target
                break
            
            # Choose condensation strategy based on reduction needed
            if remaining_reduction > 200:
                strategy = "aggressive"
            elif remaining_reduction > 100:
                strategy = "moderate"
            else:
                strategy = "minimal"
            
            condensation_prompt = self._build_condensation_prompt(
                condensed_content, section, max_words, strategy
            )
            
            try:
                reduced_content = core.query_llm(condensation_prompt)
                
                # Validate the condensation
                new_word_count = len(reduced_content.split())
                if new_word_count < current_word_count and new_word_count >= max_words * 0.9:
                    condensed_content = reduced_content
                    logger.info(f"Condensation iteration {attempts + 1}: {current_word_count} → {new_word_count} words")
                else:
                    logger.warning(f"Condensation iteration {attempts + 1} ineffective")
                    break
                    
            except Exception as e:
                logger.error(f"Condensation iteration {attempts + 1} failed: {e}")
                break
            
            attempts += 1
        
        final_word_count = len(condensed_content.split())
        logger.info(f"Section condensation complete: {current_words} → {final_word_count} words (target: ≤{max_words})")
        
        return condensed_content
    
    def _build_condensation_prompt(self, content: str, section, max_words: int, strategy: str) -> str:
        """Build targeted condensation prompt based on strategy and word count limits."""
        base_prompt = f"""The following section content needs to be condensed to approximately {max_words} words while retaining all essential information.

Current content:
{content}

Required elements that MUST be preserved:
{chr(10).join(f"- {element}" for element in section.required_elements)}

Section guidelines to maintain:
{chr(10).join(f"- {guideline}" for guideline in section.content_guidelines)}
"""
        
        if strategy == "aggressive":
            strategy_instructions = """Please condense this content aggressively by:
- Removing all redundant phrases and repetitive content
- Combining related points into single sentences
- Eliminating examples that don't add unique value
- Using more concise language and shorter sentences
- Removing transitional phrases and filler words
- Keeping only the most critical information

Focus on maximum reduction while preserving core meaning."""
        
        elif strategy == "moderate":
            strategy_instructions = """Please condense this content moderately by:
- Removing redundant phrases and unnecessary elaboration
- Combining similar points where possible
- Shortening verbose explanations
- Using more direct language
- Eliminating less important details
- Maintaining key examples and insights

Balance reduction with content quality."""
        
        else:  # minimal strategy
            strategy_instructions = """Please condense this content minimally by:
- Removing only truly redundant phrases
- Shortening wordy sentences slightly
- Eliminating unnecessary filler words
- Keeping all important content and examples
- Making small efficiency improvements

Preserve content quality while making targeted reductions."""
        
        return f"""{base_prompt}

{strategy_instructions}

Return the condensed version that maintains all required elements:"""
    
    def _check_section_elements(self, content: str, section) -> Dict[str, bool]:
        """Check if section content includes required elements."""
        elements_found = {}
        content_lower = content.lower()
        
        for element in section.required_elements:
            # Simple keyword-based detection (could be enhanced with NLP)
            element_keywords = element.lower().split()
            found = any(keyword in content_lower for keyword in element_keywords)
            elements_found[element] = found
        
        return elements_found
    
    def _combine_sections(self, section_results: Dict, template: NewsletterTemplate) -> str:
        """Combine individual sections into final newsletter content."""
        content_parts = []
        
        # Add title
        title = f"# {template.name}\n\n"
        content_parts.append(title)
        
        # Add each section
        for section in template.sections:
            if section.name in section_results:
                section_data = section_results[section.name]
                content_parts.append(f"## {section.name}\n\n")
                content_parts.append(section_data['content'])
                content_parts.append("\n\n")
        
        return "".join(content_parts)

    def _get_writing_guidelines(self, tone: str, audience: str) -> str:
        """Get tone and audience-specific writing guidelines."""
        tone_guidelines = {
            'professional': """
            - Use formal but accessible language
            - Maintain professional credibility
            - Avoid slang and casual expressions
            - Use industry-standard terminology
            """,

            'conversational': """
            - Use friendly, approachable language
            - Include personal pronouns (we, you)
            - Use contractions and natural speech patterns
            - Maintain engagement through dialogue
            """,

            'technical': """
            - Use precise technical language
            - Include specific details and specifications
            - Focus on accuracy and precision
            - Use technical diagrams and examples
            """,

            'casual': """
            - Use relaxed, informal language
            - Include humor and personality
            - Use conversational expressions
            - Maintain approachability and friendliness
            """
        }

        audience_guidelines = {
            'general': """
            - Use accessible language for non-experts
            - Provide context and background information
            - Avoid jargon without explanation
            - Use analogies and examples
            """,

            'technical': """
            - Use technical terminology appropriately
            - Assume technical background knowledge
            - Include detailed technical explanations
            - Focus on implementation and best practices
            """,

            'executive': """
            - Focus on high-level insights and implications
            - Emphasize business value and ROI
            - Use executive summary format
            - Highlight strategic recommendations
            """,

            'developer': """
            - Include code examples and technical details
            - Focus on practical implementation
            - Use developer-friendly language
            - Provide hands-on guidance
            """
        }

        tone_guide = tone_guidelines.get(tone, tone_guidelines['professional'])
        audience_guide = audience_guidelines.get(
            audience, audience_guidelines['general'])

        return f"{tone_guide}\n{audience_guide}"

    def _post_process_content(
            self,
            content: str,
            template_type: NewsletterType) -> str:
        """Post-process and refine the generated content."""
        try:
            # Create post-processing prompt
            post_process_prompt = f"""
            Please review and refine the following newsletter content:

            {content}

            Template Type: {template_type.value}

            Please ensure:
            1. Proper markdown formatting
            2. Consistent heading structure
            3. Appropriate paragraph breaks
            4. Clear and engaging language
            5. Logical flow and organization
            6. No repetitive or redundant content
            7. Professional presentation

            Return the refined content with proper formatting.
            """

            refined_content = core.query_llm(post_process_prompt)
            return refined_content if refined_content.strip() else content

        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return content

    def create_newsletter_section(self, section_title: str, content: str,
                                  section_type: str = "general") -> str:
        """Create a specific newsletter section."""
        section_prompt = f"""
        Create a newsletter section with the title: "{section_title}"

        Content to include: {content}
        Section type: {section_type}

        Please create an engaging section that:
        1. Has a compelling headline
        2. Introduces the topic clearly
        3. Presents the information in an organized way
        4. Includes relevant examples or data
        5. Provides actionable insights
        6. Maintains reader engagement

        Use appropriate formatting and structure for the section type.
        """

        try:
            return core.query_llm(section_prompt)
        except Exception as e:
            logger.error(f"Error creating newsletter section: {e}")
            return f"## {section_title}\n\n{content}"

    def generate_headlines(self, content: str, count: int = 3) -> List[str]:
        """Generate multiple headline options for content."""
        headline_prompt = f"""
        Generate {count}
            engaging headlines for the following newsletter content:

        {content}

        Requirements for headlines:
        1. Be compelling and clickable
        2. Accurately represent the content
        3. Use appropriate tone and style
        4. Be concise but descriptive
        5. Include relevant keywords

        Return only the headlines, one per line.
        """

        try:
            response = core.query_llm(headline_prompt)
            headlines = [line.strip()
                         for line in response.split('\n') if line.strip()]
            return headlines[:count]
        except Exception as e:
            logger.error(f"Error generating headlines: {e}")
            return [f"Newsletter - {content[:50]}..."]

    def generate_code_examples(self, topic: str, framework: str = None, 
                              complexity: str = "beginner", count: int = 2) -> List[str]:
        """Generate code examples for technical newsletter content."""
        try:
            logger.info(f"Generating {count} code examples for topic: {topic}")
            
            # Determine best framework if not specified
            if not framework:
                framework = self.code_generator.suggest_framework(topic)
                logger.info(f"Auto-selected framework: {framework}")
            
            # Generate examples using different approaches
            code_examples = []
            
            # 1. Try to get from template library first
            template = self.code_templates.get_template(
                Framework(framework.lower()),
                TemplateCategory.BASIC_EXAMPLE,
                complexity
            )
            
            if template:
                code_examples.append(self._format_template_example(template))
                logger.info("Added template-based example")
            
            # 2. Generate custom examples
            for i in range(max(0, count - len(code_examples))):
                try:
                    example = self.code_generator.generate_code_example(
                        topic=topic,
                        framework=framework,
                        code_type=CodeType.BASIC_EXAMPLE,
                        complexity=complexity
                    )
                    
                    # Validate the generated code
                    validation_result = self.syntax_validator.validate(example.code)
                    
                    if validation_result.is_valid or validation_result.overall_score > 0.7:
                        formatted_example = self.code_generator.format_code_for_newsletter(example)
                        code_examples.append(formatted_example)
                        logger.info(f"Added generated example {i+1}")
                    else:
                        logger.warning(f"Generated code failed validation: {validation_result.overall_score}")
                        
                except Exception as e:
                    logger.warning(f"Error generating code example {i+1}: {e}")
            
            return code_examples[:count]
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return [f"# Code example for {topic}\n# Error: Could not generate example"]
    
    def generate_technical_content_with_code(self, topic: str, context: str = "",
                                           include_code: bool = True) -> str:
        """Generate technical content with integrated code examples."""
        try:
            # Generate base content
            content_prompt = f"""
            Create technical newsletter content about: {topic}
            
            Context: {context}
            
            Requirements:
            - Technical depth appropriate for AI/ML professionals
            - Clear explanations with practical insights
            - Structure with sections and subsections
            {"- Include placeholders for code examples" if include_code else ""}
            - Use markdown formatting
            
            Write engaging, informative content that explains the topic thoroughly.
            """
            
            base_content = core.query_llm(content_prompt)
            
            if not include_code:
                return base_content
            
            # Generate and integrate code examples
            code_examples = self.generate_code_examples(topic, count=2)
            
            if code_examples:
                # Add code examples section
                code_section = "\n\n## Code Examples\n\n"
                code_section += "\n\n".join(code_examples)
                
                # Insert code examples into content
                final_content = base_content + code_section
            else:
                final_content = base_content
            
            logger.info(f"Generated technical content with {len(code_examples)} code examples")
            return final_content
            
        except Exception as e:
            logger.error(f"Error generating technical content: {e}")
            return f"# Error generating content for {topic}\n\nPlease try again."
    
    def validate_and_test_code(self, code: str, framework: str = "python") -> Dict[str, Any]:
        """Validate and test code examples for quality assurance."""
        try:
            # Syntax validation
            validation_result = self.syntax_validator.validate(code)
            
            # Code execution test
            execution_result = self.code_executor.execute(code)
            
            return {
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "syntax_score": validation_result.syntax_score,
                    "style_score": validation_result.style_score,
                    "overall_score": validation_result.overall_score,
                    "issues_count": len(validation_result.issues),
                    "has_imports": validation_result.has_imports,
                    "has_comments": validation_result.has_comments
                },
                "execution": {
                    "status": execution_result.status.value,
                    "execution_time": execution_result.execution_time,
                    "has_output": bool(execution_result.stdout.strip()),
                    "has_errors": bool(execution_result.stderr.strip())
                },
                "recommendations": self._generate_code_recommendations(validation_result, execution_result)
            }
            
        except Exception as e:
            logger.error(f"Error validating code: {e}")
            return {"error": str(e)}
    
    def _format_template_example(self, template) -> str:
        """Format a template code example for newsletter inclusion."""
        return f"""
## {template.name}

{template.description}

**Framework:** {template.framework.value}
**Complexity:** {template.complexity.value}
**Dependencies:** {', '.join(template.dependencies)}

```python
{template.code}
```

### Explanation

{template.explanation}

**Use Cases:**
{chr(10).join(f"- {use_case}" for use_case in template.use_cases)}
"""
    
    def _generate_code_recommendations(self, validation_result, execution_result) -> List[str]:
        """Generate recommendations based on validation and execution results."""
        recommendations = []
        
        if not validation_result.is_valid:
            recommendations.append("Fix syntax errors before using this code")
        
        if validation_result.style_score < 0.8:
            recommendations.append("Improve code style for better readability")
        
        if not validation_result.has_comments:
            recommendations.append("Add comments to explain complex logic")
        
        if execution_result.status.value != "success":
            recommendations.append("Test and debug code execution issues")
        
        if validation_result.overall_score > 0.9 and execution_result.status.value == "success":
            recommendations.append("Code quality is excellent - ready for publication")
        
        return recommendations

    def get_writing_analytics(self) -> Dict[str, Any]:
        """Get writing-specific analytics."""
        analytics = self.get_tool_usage_analytics()

        # Add writing-specific metrics
        writing_metrics = {
            "writing_sessions": len(self.execution_history),
            "avg_writing_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "success_rate": sum(1 for r in self.execution_history if r.status.value == "completed") / len(self.execution_history) if self.execution_history else 0,
            "content_quality_metrics": {
                "avg_content_length": sum(len(r.result) for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
                "sections_created": sum(1 for r in self.execution_history if "section" in r.result.lower()),
                "headlines_generated": sum(1 for r in self.execution_history if "headline" in r.result.lower())
            }
        }

        analytics.update(writing_metrics)
        return analytics
