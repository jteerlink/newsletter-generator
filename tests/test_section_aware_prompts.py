"""
Test Suite for Section-Aware Prompt Engine

Comprehensive tests for the section-aware prompt system including:
- Section type detection
- Prompt generation for all section types
- Context handling and validation
- Backward compatibility
"""

import pytest
from unittest.mock import patch, MagicMock

from src.core.section_aware_prompts import (
    SectionType,
    PromptContext,
    SectionAwarePromptManager,
    IntroductionPromptTemplate,
    AnalysisPromptTemplate,
    TutorialPromptTemplate,
    NewsPromptTemplate,
    ConclusionPromptTemplate,
    get_section_prompt,
    detect_section_type
)


class TestSectionType:
    """Test SectionType enum functionality."""
    
    def test_section_type_values(self):
        """Test that all expected section types exist."""
        expected_types = {
            'introduction', 'analysis', 'tutorial', 
            'news', 'conclusion', 'general'
        }
        actual_types = {section.value for section in SectionType}
        assert actual_types == expected_types
    
    def test_section_type_from_string(self):
        """Test converting strings to SectionType."""
        assert SectionType('introduction') == SectionType.INTRODUCTION
        assert SectionType('analysis') == SectionType.ANALYSIS
        
        with pytest.raises(ValueError):
            SectionType('invalid_type')


class TestPromptContext:
    """Test PromptContext dataclass."""
    
    def test_prompt_context_creation(self):
        """Test creating PromptContext with required fields."""
        context = PromptContext(
            topic="AI Research",
            audience="AI/ML Engineers",
            content_focus="Machine Learning",
            word_count=1000,
            special_requirements=["Include code examples"],
            section_type=SectionType.TUTORIAL
        )
        
        assert context.topic == "AI Research"
        assert context.audience == "AI/ML Engineers"
        assert context.section_type == SectionType.TUTORIAL
        assert context.tone == "professional"  # default value
        assert context.technical_level == "intermediate"  # default value
    
    def test_prompt_context_defaults(self):
        """Test default values in PromptContext."""
        context = PromptContext(
            topic="Test",
            audience="Test",
            content_focus="Test",
            word_count=100,
            special_requirements=[],
            section_type=SectionType.GENERAL
        )
        
        assert context.tone == "professional"
        assert context.technical_level == "intermediate"
        assert context.include_examples is False
        assert context.include_citations is False


class TestSectionPromptTemplates:
    """Test individual section prompt templates."""
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return PromptContext(
            topic="Artificial Intelligence",
            audience="Software Developers",
            content_focus="AI Development Tools",
            word_count=2000,
            special_requirements=["Include practical examples"],
            section_type=SectionType.ANALYSIS,
            tone="professional",
            technical_level="intermediate"
        )
    
    def test_introduction_template(self, sample_context):
        """Test introduction prompt template."""
        template = IntroductionPromptTemplate()
        sample_context.section_type = SectionType.INTRODUCTION
        
        prompt = template.generate_prompt(sample_context)
        
        assert "introduction" in prompt.lower()
        assert sample_context.topic in prompt
        assert sample_context.audience in prompt
        assert "hook the reader's attention" in prompt.lower()
        assert str(sample_context.word_count // 8) in prompt
    
    def test_analysis_template(self, sample_context):
        """Test analysis prompt template."""
        template = AnalysisPromptTemplate()
        sample_context.section_type = SectionType.ANALYSIS
        
        prompt = template.generate_prompt(sample_context)
        
        assert "analysis" in prompt.lower()
        assert "deep insights" in prompt.lower()
        assert "data and evidence" in prompt.lower()
        assert str(sample_context.word_count // 3) in prompt
    
    def test_tutorial_template(self, sample_context):
        """Test tutorial prompt template."""
        template = TutorialPromptTemplate()
        sample_context.section_type = SectionType.TUTORIAL
        
        prompt = template.generate_prompt(sample_context)
        
        assert "tutorial" in prompt.lower()
        assert "step-by-step" in prompt.lower()
        assert "actionable guidance" in prompt.lower()
        assert str(sample_context.word_count // 4) in prompt
    
    def test_news_template(self, sample_context):
        """Test news prompt template."""
        template = NewsPromptTemplate()
        sample_context.section_type = SectionType.NEWS
        
        prompt = template.generate_prompt(sample_context)
        
        assert "news" in prompt.lower()
        assert "recent" in prompt.lower()
        assert "developments" in prompt.lower()
        assert str(sample_context.word_count // 5) in prompt
    
    def test_conclusion_template(self, sample_context):
        """Test conclusion prompt template."""
        template = ConclusionPromptTemplate()
        sample_context.section_type = SectionType.CONCLUSION
        
        prompt = template.generate_prompt(sample_context)
        
        assert "conclusion" in prompt.lower()
        assert "summarize" in prompt.lower()
        assert "call-to-action" in prompt.lower()
        assert str(sample_context.word_count // 10) in prompt
    
    def test_special_requirements_formatting(self, sample_context):
        """Test special requirements are properly formatted."""
        template = AnalysisPromptTemplate()
        sample_context.special_requirements = ["Include charts", "Add citations"]
        
        prompt = template.generate_prompt(sample_context)
        
        assert "Special Requirements:" in prompt
        assert "Include charts" in prompt
        assert "Add citations" in prompt
    
    def test_audience_guidance(self, sample_context):
        """Test audience-specific guidance generation."""
        template = AnalysisPromptTemplate()
        
        # Test known audience
        sample_context.audience = "AI/ML Engineers"
        prompt = template.generate_prompt(sample_context)
        assert "technical terminology" in prompt.lower()
        
        # Test unknown audience
        sample_context.audience = "Unknown Audience"
        prompt = template.generate_prompt(sample_context)
        assert "technical level appropriately" in prompt.lower()


class TestSectionAwarePromptManager:
    """Test SectionAwarePromptManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create prompt manager instance."""
        return SectionAwarePromptManager()
    
    @pytest.fixture
    def sample_context_dict(self):
        """Sample context dictionary."""
        return {
            'topic': 'Machine Learning',
            'audience': 'Data Scientists',
            'content_focus': 'Neural Networks',
            'word_count': 1500,
            'special_requirements': ['Include examples'],
            'tone': 'professional',
            'technical_level': 'advanced'
        }
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.section_templates) == 5
        assert SectionType.INTRODUCTION in manager.section_templates
        assert SectionType.ANALYSIS in manager.section_templates
        assert manager.default_template is not None
    
    def test_get_section_prompt_with_enum(self, manager, sample_context_dict):
        """Test getting prompt with SectionType enum."""
        prompt = manager.get_section_prompt(SectionType.INTRODUCTION, sample_context_dict)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "introduction" in prompt.lower()
        assert sample_context_dict['topic'] in prompt
    
    def test_get_section_prompt_with_string(self, manager, sample_context_dict):
        """Test getting prompt with string section type."""
        prompt = manager.get_section_prompt('tutorial', sample_context_dict)
        
        assert isinstance(prompt, str)
        assert "tutorial" in prompt.lower()
        assert "step-by-step" in prompt.lower()
    
    def test_get_section_prompt_invalid_string(self, manager, sample_context_dict):
        """Test getting prompt with invalid string section type."""
        with patch('src.core.section_aware_prompts.logger') as mock_logger:
            prompt = manager.get_section_prompt('invalid_type', sample_context_dict)
            
            assert isinstance(prompt, str)
            mock_logger.warning.assert_called_once()
    
    @patch('src.core.section_aware_prompts.logger')
    def test_error_handling(self, mock_logger, manager):
        """Test error handling in prompt generation."""
        # Create invalid context that will cause an error
        invalid_context = {'invalid_key': 'invalid_value'}
        
        prompt = manager.get_section_prompt(SectionType.ANALYSIS, invalid_context)
        
        # Should return fallback prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        mock_logger.error.assert_called_once()
    
    def test_detect_section_type(self, manager):
        """Test section type detection."""
        test_cases = [
            ("Welcome to our newsletter introduction", SectionType.INTRODUCTION),
            ("Let's start with an overview", SectionType.INTRODUCTION),
            ("Step 1: Install the package", SectionType.TUTORIAL),
            ("How to implement the algorithm", SectionType.TUTORIAL),
            ("Breaking news: Company announced", SectionType.NEWS),
            ("Latest updates from the conference", SectionType.NEWS),
            ("In conclusion, we have seen", SectionType.CONCLUSION),
            ("Summary of key takeaways", SectionType.CONCLUSION),
            ("This is a general analysis", SectionType.ANALYSIS),
        ]
        
        for content, expected_type in test_cases:
            detected_type = manager.detect_section_type(content, {})
            assert detected_type == expected_type, f"Failed for content: {content}"
    
    def test_get_available_sections(self, manager):
        """Test getting available section types."""
        sections = manager.get_available_sections()
        
        expected_sections = ['introduction', 'analysis', 'tutorial', 'news', 'conclusion']
        assert set(sections) == set(expected_sections)
        assert 'general' not in sections  # Should exclude GENERAL
    
    def test_create_prompt_context(self, manager, sample_context_dict):
        """Test creating PromptContext from dictionary."""
        context = manager._create_prompt_context(sample_context_dict, SectionType.ANALYSIS)
        
        assert isinstance(context, PromptContext)
        assert context.topic == sample_context_dict['topic']
        assert context.audience == sample_context_dict['audience']
        assert context.section_type == SectionType.ANALYSIS
    
    def test_fallback_prompt(self, manager):
        """Test fallback prompt generation."""
        context = {'topic': 'Test Topic', 'audience': 'Test Audience'}
        fallback = manager._generate_fallback_prompt(context)
        
        assert isinstance(fallback, str)
        assert 'Test Topic' in fallback
        assert 'Test Audience' in fallback


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility."""
    
    def test_get_section_prompt_function(self):
        """Test get_section_prompt convenience function."""
        context = {
            'topic': 'Test Topic',
            'audience': 'Test Audience',
            'content_focus': 'Test Focus',
            'word_count': 1000,
            'special_requirements': []
        }
        
        prompt = get_section_prompt('introduction', context)
        
        assert isinstance(prompt, str)
        assert 'introduction' in prompt.lower()
        assert 'Test Topic' in prompt
    
    def test_detect_section_type_function(self):
        """Test detect_section_type convenience function."""
        content = "Welcome to our tutorial on machine learning"
        
        section_type = detect_section_type(content)
        
        assert section_type == 'tutorial'
    
    def test_detect_section_type_with_context(self):
        """Test detect_section_type with context parameter."""
        content = "Let's analyze the data"
        context = {'additional_info': 'test'}
        
        section_type = detect_section_type(content, context)
        
        assert section_type == 'analysis'


class TestIntegration:
    """Integration tests for the complete prompt system."""
    
    def test_end_to_end_prompt_generation(self):
        """Test complete prompt generation workflow."""
        manager = SectionAwarePromptManager()
        
        context = {
            'topic': 'Quantum Computing',
            'audience': 'Research Community', 
            'content_focus': 'Quantum Algorithms',
            'word_count': 3000,
            'special_requirements': ['Include mathematical formulas', 'Add research citations'],
            'tone': 'academic',
            'technical_level': 'expert'
        }
        
        # Test all section types
        for section_type in SectionType:
            if section_type != SectionType.GENERAL:
                prompt = manager.get_section_prompt(section_type, context)
                
                assert isinstance(prompt, str)
                assert len(prompt) > 100
                assert context['topic'] in prompt
                assert context['audience'] in prompt
                assert 'Include mathematical formulas' in prompt
                assert 'Add research citations' in prompt
    
    def test_section_detection_and_prompt_generation(self):
        """Test section detection followed by prompt generation."""
        manager = SectionAwarePromptManager()
        
        test_content = "Step 1: Download the software package"
        context = {
            'topic': 'Software Installation',
            'audience': 'Software Developers',
            'content_focus': 'Development Tools',
            'word_count': 1000,
            'special_requirements': []
        }
        
        # Detect section type
        detected_type = manager.detect_section_type(test_content, context)
        assert detected_type == SectionType.TUTORIAL
        
        # Generate prompt for detected type
        prompt = manager.get_section_prompt(detected_type, context)
        assert 'tutorial' in prompt.lower()
        assert 'step-by-step' in prompt.lower()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_context(self):
        """Test handling of empty context."""
        manager = SectionAwarePromptManager()
        
        prompt = manager.get_section_prompt(SectionType.ANALYSIS, {})
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should use default values
        assert 'Technology Updates' in prompt
        assert 'General Tech Audience' in prompt
    
    def test_missing_context_fields(self):
        """Test handling of partial context."""
        manager = SectionAwarePromptManager()
        
        partial_context = {
            'topic': 'Test Topic'
            # Missing other required fields
        }
        
        prompt = manager.get_section_prompt(SectionType.NEWS, partial_context)
        
        assert isinstance(prompt, str)
        assert 'Test Topic' in prompt
        # Should use defaults for missing fields
        assert 'General Tech Audience' in prompt
    
    def test_very_large_word_count(self):
        """Test handling of very large word counts."""
        manager = SectionAwarePromptManager()
        
        context = {
            'topic': 'Large Document',
            'audience': 'Readers',
            'content_focus': 'Comprehensive Guide',
            'word_count': 50000,  # Very large
            'special_requirements': []
        }
        
        prompt = manager.get_section_prompt(SectionType.INTRODUCTION, context)
        
        assert isinstance(prompt, str)
        # Should handle large word count gracefully
        expected_intro_length = max(150, 50000 // 8)
        assert str(expected_intro_length) in prompt
    
    def test_zero_word_count(self):
        """Test handling of zero word count."""
        manager = SectionAwarePromptManager()
        
        context = {
            'topic': 'Short Content',
            'audience': 'Readers', 
            'content_focus': 'Brief Update',
            'word_count': 0,
            'special_requirements': []
        }
        
        prompt = manager.get_section_prompt(SectionType.CONCLUSION, context)
        
        assert isinstance(prompt, str)
        # Should use minimum word count
        expected_min_length = max(100, 0 // 10)  # Should be 100
        assert str(expected_min_length) in prompt


if __name__ == '__main__':
    pytest.main([__file__])