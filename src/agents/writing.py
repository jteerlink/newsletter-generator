"""
Enhanced Writing Agent for Newsletter Generation

This module provides the enhanced WriterAgent class, which is responsible for creating
engaging and well-structured newsletter content with context-driven capabilities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from src.core.constants import TOOL_ENFORCEMENT_ENABLED, MANDATORY_VECTOR_TOP_K
from src.core.tool_usage_tracker import get_tool_tracker
from src.core.quality_gates import validate_tool_usage_quality
from src.storage import get_storage_provider

from src.core.campaign_context import CampaignContext
import src.core.core as core
from src.core.feedback_system import StructuredFeedback
from src.core.template_manager import NewsletterTemplate, NewsletterType
from src.core.code_generator import AIMLCodeGenerator, CodeType
from src.tools.syntax_validator import SyntaxValidator, ValidationLevel
from src.tools.code_executor import SafeCodeExecutor, ExecutionConfig, SecurityLevel
from src.templates.code_templates import template_library, Framework, TemplateCategory

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
