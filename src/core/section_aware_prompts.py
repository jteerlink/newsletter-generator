"""
Section-Aware Prompt Engine for Newsletter Generation

This module implements the section-specific prompt templates and management system
as specified in Phase 1 FR1.1 of the Multi-Agent Enhancement PRD.

Features:
- Section-specific prompt templates for different newsletter content types
- Dynamic section detection and routing
- Context-aware prompt generation with audience and focus customization
- Backward compatibility with existing prompt system
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """Enumeration of newsletter section types."""
    INTRODUCTION = "introduction"
    ANALYSIS = "analysis"
    TUTORIAL = "tutorial"
    NEWS = "news"
    CONCLUSION = "conclusion"
    GENERAL = "general"  # Fallback for unspecified sections


@dataclass
class PromptContext:
    """Context information for prompt generation."""
    topic: str
    audience: str
    content_focus: str
    word_count: int
    special_requirements: List[str]
    section_type: SectionType
    tone: str = "professional"
    technical_level: str = "intermediate"
    include_examples: bool = False
    include_citations: bool = False


class SectionPromptTemplate(ABC):
    """Abstract base class for section-specific prompt templates."""

    @abstractmethod
    def generate_prompt(self, context: PromptContext) -> str:
        """Generate a section-specific prompt based on context."""
        pass

    @abstractmethod
    def get_section_guidelines(self) -> str:
        """Return section-specific writing guidelines."""
        pass

    def _format_requirements(self, requirements: List[str]) -> str:
        """Format special requirements for inclusion in prompts."""
        if not requirements:
            return ""
        
        formatted = "\n\nSpecial Requirements:\n"
        for req in requirements:
            formatted += f"- {req}\n"
        return formatted

    def _get_audience_guidance(self, audience: str, technical_level: str) -> str:
        """Generate audience-specific writing guidance."""
        guidance_map = {
            "AI/ML Engineers": "Use technical terminology appropriately, include implementation details",
            "Data Scientists": "Focus on analytical approaches, include statistical concepts",
            "Software Developers": "Emphasize practical implementation, code examples welcome",
            "Technical Leaders": "Balance technical depth with strategic implications",
            "Research Community": "Include methodological details, cite relevant papers",
            "Business Professionals": "Focus on business impact, minimize technical jargon",
            "General Tech Audience": "Explain technical concepts clearly, provide context"
        }
        
        base_guidance = guidance_map.get(audience, "Adjust technical level appropriately")
        return f"Audience: {audience} - {base_guidance} (Technical level: {technical_level})"


class IntroductionPromptTemplate(SectionPromptTemplate):
    """Prompt template for newsletter introduction sections."""

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate introduction-specific prompt."""
        base_prompt = f"""
Write an engaging introduction for a newsletter about {context.topic}.

{self._get_audience_guidance(context.audience, context.technical_level)}

Content Focus: {context.content_focus}

{self.get_section_guidelines()}

Target Length: {max(150, context.word_count // 8)} words

Requirements:
- Hook the reader's attention immediately
- Clearly state what the newsletter will cover
- Establish relevance to the target audience
- Set the tone for the entire newsletter
- Preview key insights or developments to be discussed{self._format_requirements(context.special_requirements)}
"""
        return base_prompt.strip()

    def get_section_guidelines(self) -> str:
        """Return introduction-specific guidelines."""
        return """
Introduction Guidelines:
- Start with a compelling hook (question, statistic, or current event)
- Briefly introduce the main topic and its significance
- Outline what readers will learn or gain
- Keep paragraphs short and scannable
- Use active voice and engaging language
"""


class AnalysisPromptTemplate(SectionPromptTemplate):
    """Prompt template for analysis sections."""

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate analysis-specific prompt."""
        base_prompt = f"""
Write a comprehensive analysis section for a newsletter about {context.topic}.

{self._get_audience_guidance(context.audience, context.technical_level)}

Content Focus: {context.content_focus}

{self.get_section_guidelines()}

Target Length: {max(400, context.word_count // 3)} words

Requirements:
- Provide deep insights and critical analysis
- Use data and evidence to support key points
- Examine multiple perspectives or approaches
- Identify trends, patterns, or implications
- Connect findings to broader industry context{self._format_requirements(context.special_requirements)}
"""
        return base_prompt.strip()

    def get_section_guidelines(self) -> str:
        """Return analysis-specific guidelines."""
        return """
Analysis Guidelines:
- Present information logically and systematically
- Support claims with credible sources and data
- Use clear subheadings to organize complex information
- Include relevant statistics, case studies, or examples
- Analyze implications and potential future developments
- Maintain objectivity while providing insights
"""


class TutorialPromptTemplate(SectionPromptTemplate):
    """Prompt template for tutorial/how-to sections."""

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate tutorial-specific prompt."""
        base_prompt = f"""
Write a practical tutorial section for a newsletter about {context.topic}.

{self._get_audience_guidance(context.audience, context.technical_level)}

Content Focus: {context.content_focus}

{self.get_section_guidelines()}

Target Length: {max(300, context.word_count // 4)} words

Requirements:
- Provide step-by-step, actionable guidance
- Include practical examples and use cases
- Anticipate common challenges and solutions
- Make complex concepts accessible
- Include implementation tips and best practices{self._format_requirements(context.special_requirements)}
"""
        return base_prompt.strip()

    def get_section_guidelines(self) -> str:
        """Return tutorial-specific guidelines."""
        return """
Tutorial Guidelines:
- Use numbered lists or clear step sequences
- Include code examples, screenshots, or diagrams where helpful
- Explain the 'why' behind each step, not just the 'how'
- Provide troubleshooting tips for common issues
- Include resources for further learning
- Test instructions for clarity and completeness
"""


class NewsPromptTemplate(SectionPromptTemplate):
    """Prompt template for news and updates sections."""

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate news-specific prompt."""
        base_prompt = f"""
Write a news and updates section for a newsletter about {context.topic}.

{self._get_audience_guidance(context.audience, context.technical_level)}

Content Focus: {context.content_focus}

{self.get_section_guidelines()}

Target Length: {max(250, context.word_count // 5)} words

Requirements:
- Cover the most recent and relevant developments
- Prioritize information by importance and relevance
- Provide context for why news items matter
- Include brief analysis of implications
- Keep information current and factual{self._format_requirements(context.special_requirements)}
"""
        return base_prompt.strip()

    def get_section_guidelines(self) -> str:
        """Return news-specific guidelines."""
        return """
News Guidelines:
- Lead with the most important or breaking news
- Use clear, concise language
- Include specific dates, companies, and figures
- Explain technical developments in accessible terms
- Connect news items to reader interests and concerns
- Verify information accuracy before inclusion
"""


class ConclusionPromptTemplate(SectionPromptTemplate):
    """Prompt template for conclusion sections."""

    def generate_prompt(self, context: PromptContext) -> str:
        """Generate conclusion-specific prompt."""
        base_prompt = f"""
Write a compelling conclusion for a newsletter about {context.topic}.

{self._get_audience_guidance(context.audience, context.technical_level)}

Content Focus: {context.content_focus}

{self.get_section_guidelines()}

Target Length: {max(100, context.word_count // 10)} words

Requirements:
- Summarize key insights and takeaways
- Reinforce the value provided to readers
- Include a clear call-to-action
- End on a forward-looking or inspiring note
- Encourage reader engagement{self._format_requirements(context.special_requirements)}
"""
        return base_prompt.strip()

    def get_section_guidelines(self) -> str:
        """Return conclusion-specific guidelines."""
        return """
Conclusion Guidelines:
- Recap the most important points covered
- Emphasize practical value for the reader
- Suggest next steps or actions readers can take
- Include relevant links or resources
- Invite feedback or discussion
- End with a memorable closing thought
"""


class SectionAwarePromptManager:
    """
    Enhanced prompt manager with section-specific templates.
    
    Implements FR1.1 requirements for section-aware prompt generation
    with backward compatibility for existing systems.
    """

    def __init__(self):
        """Initialize the section-aware prompt manager."""
        self.section_templates: Dict[SectionType, SectionPromptTemplate] = {
            SectionType.INTRODUCTION: IntroductionPromptTemplate(),
            SectionType.ANALYSIS: AnalysisPromptTemplate(),
            SectionType.TUTORIAL: TutorialPromptTemplate(),
            SectionType.NEWS: NewsPromptTemplate(),
            SectionType.CONCLUSION: ConclusionPromptTemplate(),
        }
        
        # Default template for backward compatibility
        self.default_template = AnalysisPromptTemplate()
        
        logger.info("Section-aware prompt manager initialized with %d templates", 
                   len(self.section_templates))

    def get_section_prompt(self, section_type: Union[str, SectionType], 
                          context: Dict[str, Any]) -> str:
        """
        Generate section-specific prompt with context.
        
        Args:
            section_type: The type of section to generate prompt for
            context: Dictionary containing prompt context information
            
        Returns:
            str: Generated section-specific prompt
        """
        try:
            # Convert string to SectionType if needed
            if isinstance(section_type, str):
                try:
                    section_type = SectionType(section_type.lower())
                except ValueError:
                    logger.warning("Unknown section type '%s', using general", section_type)
                    section_type = SectionType.GENERAL

            # Create PromptContext from dictionary
            prompt_context = self._create_prompt_context(context, section_type)
            
            # Get appropriate template
            template = self.section_templates.get(section_type, self.default_template)
            
            # Generate and return prompt
            prompt = template.generate_prompt(prompt_context)
            logger.debug("Generated prompt for section type %s (%d chars)", 
                        section_type.value, len(prompt))
            
            return prompt
            
        except Exception as e:
            logger.error("Error generating section prompt: %s", e)
            # Fallback to basic prompt for backward compatibility
            return self._generate_fallback_prompt(context)

    def detect_section_type(self, content: str, context: Dict[str, Any]) -> SectionType:
        """
        Detect the section type based on content and context.
        
        Args:
            content: The content to analyze
            context: Additional context for detection
            
        Returns:
            SectionType: Detected section type
        """
        content_lower = content.lower()
        
        # Introduction indicators
        if any(indicator in content_lower for indicator in 
               ['welcome', 'introduction', 'overview', 'begin', 'start']):
            return SectionType.INTRODUCTION
            
        # Tutorial indicators
        if any(indicator in content_lower for indicator in 
               ['step', 'how to', 'tutorial', 'guide', 'implement', 'setup']):
            return SectionType.TUTORIAL
            
        # News indicators  
        if any(indicator in content_lower for indicator in 
               ['news', 'update', 'announcement', 'release', 'launched']):
            return SectionType.NEWS
            
        # Conclusion indicators
        if any(indicator in content_lower for indicator in 
               ['conclusion', 'summary', 'finally', 'in summary', 'takeaway']):
            return SectionType.CONCLUSION
            
        # Default to analysis for substantial content
        return SectionType.ANALYSIS

    def get_available_sections(self) -> List[str]:
        """
        Get list of available section types.
        
        Returns:
            List[str]: Available section type names
        """
        return [section.value for section in SectionType if section != SectionType.GENERAL]

    def _create_prompt_context(self, context: Dict[str, Any], 
                             section_type: SectionType) -> PromptContext:
        """Create PromptContext from dictionary context."""
        return PromptContext(
            topic=context.get('topic', 'Technology Updates'),
            audience=context.get('audience', 'General Tech Audience'),
            content_focus=context.get('content_focus', 'General Tech News'),
            word_count=context.get('word_count', 3000),
            special_requirements=context.get('special_requirements', []),
            section_type=section_type,
            tone=context.get('tone', 'professional'),
            technical_level=context.get('technical_level', 'intermediate'),
            include_examples=context.get('include_examples', False),
            include_citations=context.get('include_citations', False)
        )

    def _generate_fallback_prompt(self, context: Dict[str, Any]) -> str:
        """Generate fallback prompt for backward compatibility."""
        topic = context.get('topic', 'technology updates')
        audience = context.get('audience', 'technology professionals')
        
        return f"""
Write content about {topic} for {audience}.

Make it informative, well-structured, and engaging for the target audience.
Include relevant examples and maintain a professional tone throughout.
"""


# Convenience functions for backward compatibility
def get_section_prompt(section_type: str, context: Dict[str, Any]) -> str:
    """
    Convenience function to get section-specific prompt.
    
    Args:
        section_type: Type of section ('introduction', 'analysis', etc.)
        context: Context dictionary for prompt generation
        
    Returns:
        str: Generated prompt
    """
    manager = SectionAwarePromptManager()
    return manager.get_section_prompt(section_type, context)


def detect_section_type(content: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to detect section type.
    
    Args:
        content: Content to analyze
        context: Optional context for detection
        
    Returns:
        str: Detected section type
    """
    manager = SectionAwarePromptManager()
    context = context or {}
    return manager.detect_section_type(content, context).value


# Export main classes and functions
__all__ = [
    'SectionType',
    'PromptContext', 
    'SectionAwarePromptManager',
    'SectionPromptTemplate',
    'IntroductionPromptTemplate',
    'AnalysisPromptTemplate',
    'TutorialPromptTemplate',
    'NewsPromptTemplate',
    'ConclusionPromptTemplate',
    'get_section_prompt',
    'detect_section_type'
]