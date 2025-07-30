"""
Writing Agent for Newsletter Generation

This module provides the WriterAgent class, which is responsible for creating
engaging and well-structured newsletter content.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from .base import SimpleAgent, AgentType
from src.core.core import query_llm
from src.core.template_manager import NewsletterTemplate, NewsletterType

logger = logging.getLogger(__name__)


class WriterAgent(SimpleAgent):
    """Agent specialized in writing and content creation."""
    
    def __init__(self, name: str = "WriterAgent", **kwargs):
        super().__init__(
            name=name,
            role="Content Writer",
            goal="Create engaging, informative, and well-structured newsletter content",
            backstory="""You are an experienced content writer specializing in newsletter creation. 
            You excel at transforming research and information into compelling, readable content 
            that engages audiences. You understand newsletter best practices, including clear 
            structure, engaging headlines, and appropriate tone. You can adapt your writing style 
            to different audiences and topics while maintaining high quality and readability.""",
            agent_type=AgentType.WRITER,
            tools=["search_web"],  # Writers may need to verify facts
            **kwargs
        )
    
    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute writing task with enhanced content creation capabilities."""
        logger.info(f"WriterAgent executing writing task: {task}")
        
        # Extract writing parameters
        template_type = kwargs.get('template_type', NewsletterType.TECHNICAL_DEEP_DIVE)
        target_length = kwargs.get('target_length', 1500)
        tone = kwargs.get('tone', 'professional')
        audience = kwargs.get('audience', 'general')
        
        # Create enhanced writing prompt
        enhanced_prompt = self._build_writing_prompt(task, context, template_type, target_length, tone, audience)
        
        # Generate content
        content = query_llm(enhanced_prompt)
        
        # Post-process content
        final_content = self._post_process_content(content, template_type)
        
        return final_content
    
    def _build_writing_prompt(self, task: str, context: str, template_type: NewsletterType, 
                             target_length: int, tone: str, audience: str) -> str:
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
        
        return template_instructions.get(template_type, template_instructions[NewsletterType.RESEARCH_SUMMARY])
    
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
        audience_guide = audience_guidelines.get(audience, audience_guidelines['general'])
        
        return f"{tone_guide}\n{audience_guide}"
    
    def _post_process_content(self, content: str, template_type: NewsletterType) -> str:
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
            
            refined_content = query_llm(post_process_prompt)
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
            return query_llm(section_prompt)
        except Exception as e:
            logger.error(f"Error creating newsletter section: {e}")
            return f"## {section_title}\n\n{content}"
    
    def generate_headlines(self, content: str, count: int = 3) -> List[str]:
        """Generate multiple headline options for content."""
        headline_prompt = f"""
        Generate {count} engaging headlines for the following newsletter content:
        
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
            response = query_llm(headline_prompt)
            headlines = [line.strip() for line in response.split('\n') if line.strip()]
            return headlines[:count]
        except Exception as e:
            logger.error(f"Error generating headlines: {e}")
            return [f"Newsletter - {content[:50]}..."]
    
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