"""
Test Suite for Section-Aware Refinement System

Comprehensive tests for the multi-pass section processing system including:
- Section boundary detection
- Section content extraction
- Multi-pass refinement logic
- Quality validation
- Content reassembly
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.section_aware_prompts import SectionType
from src.core.section_aware_refinement import (
    RefinementPass,
    RefinementResult,
    SectionAwareRefinementLoop,
    SectionBoundary,
    SectionBoundaryDetector,
    SectionContent,
)


class TestRefinementPass:
    """Test RefinementPass enum."""
    
    def test_refinement_pass_values(self):
        """Test that all expected refinement passes exist."""
        expected_passes = {
            'structure', 'content', 'style', 'technical', 'final'
        }
        actual_passes = {pass_type.value for pass_type in RefinementPass}
        assert actual_passes == expected_passes


class TestSectionBoundary:
    """Test SectionBoundary dataclass."""
    
    def test_section_boundary_creation(self):
        """Test creating SectionBoundary instance."""
        boundary = SectionBoundary(
            section_type=SectionType.INTRODUCTION,
            start_index=0,
            end_index=100,
            title="Introduction",
            confidence=0.9
        )
        
        assert boundary.section_type == SectionType.INTRODUCTION
        assert boundary.start_index == 0
        assert boundary.end_index == 100
        assert boundary.title == "Introduction"
        assert boundary.confidence == 0.9
    
    def test_section_boundary_defaults(self):
        """Test default values in SectionBoundary."""
        boundary = SectionBoundary(
            section_type=SectionType.ANALYSIS,
            start_index=50,
            end_index=150
        )
        
        assert boundary.title == ""
        assert boundary.confidence == 0.0


class TestSectionContent:
    """Test SectionContent dataclass."""
    
    def test_section_content_creation(self):
        """Test creating SectionContent instance."""
        boundary = SectionBoundary(
            section_type=SectionType.TUTORIAL,
            start_index=0,
            end_index=200
        )
        
        content = SectionContent(
            section_type=SectionType.TUTORIAL,
            content="This is tutorial content",
            boundary=boundary,
            quality_score=0.8,
            refinement_count=2
        )
        
        assert content.section_type == SectionType.TUTORIAL
        assert content.content == "This is tutorial content"
        assert content.boundary == boundary
        assert content.quality_score == 0.8
        assert content.refinement_count == 2
    
    def test_section_content_defaults(self):
        """Test default values in SectionContent."""
        boundary = SectionBoundary(
            section_type=SectionType.NEWS,
            start_index=0,
            end_index=100
        )
        
        content = SectionContent(
            section_type=SectionType.NEWS,
            content="News content",
            boundary=boundary
        )
        
        assert content.quality_score == 0.0
        assert content.refinement_count == 0
        assert content.issues == []
        assert content.improvements == []


class TestSectionBoundaryDetector:
    """Test section boundary detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create boundary detector instance."""
        return SectionBoundaryDetector()
    
    def test_detect_explicit_boundaries(self, detector):
        """Test detection of explicit section headers."""
        content = """# Introduction
Welcome to our newsletter.

## Latest News
Here are the latest updates.

### Technical Analysis
Let's dive into the details.

## Conclusion
Thank you for reading.
"""
        
        boundaries = detector.detect_boundaries(content)
        
        assert len(boundaries) == 4
        assert boundaries[0].section_type == SectionType.INTRODUCTION
        assert boundaries[1].section_type == SectionType.NEWS
        assert boundaries[2].section_type == SectionType.ANALYSIS
        assert boundaries[3].section_type == SectionType.CONCLUSION
    
    def test_detect_mixed_patterns(self, detector):
        """Test detection with mixed header patterns."""
        content = """# Overview
This is the introduction.

News: Latest updates
Company announced new features.

Tutorial: How to get started
Step 1: Download the software.

Summary:
Key takeaways from this newsletter.
"""
        
        boundaries = detector.detect_boundaries(content)
        
        assert len(boundaries) == 4
        section_types = [b.section_type for b in boundaries]
        assert SectionType.INTRODUCTION in section_types
        assert SectionType.NEWS in section_types
        assert SectionType.TUTORIAL in section_types
        assert SectionType.CONCLUSION in section_types
    
    def test_no_explicit_boundaries(self, detector):
        """Test handling content with no explicit boundaries."""
        content = """This is just regular content without any section headers.
It should be treated as a single analysis section."""
        
        boundaries = detector.detect_boundaries(content)
        
        assert len(boundaries) == 1
        assert boundaries[0].section_type == SectionType.ANALYSIS
        assert boundaries[0].start_index == 0
        assert boundaries[0].end_index == len(content)
        assert boundaries[0].confidence == 0.5
    
    def test_boundary_positions(self, detector):
        """Test correct boundary position calculation."""
        content = """First paragraph.

# Section Header
Second paragraph.

Third paragraph."""
        
        boundaries = detector.detect_boundaries(content)
        
        # Should detect the header
        assert len(boundaries) >= 1
        
        # Check that positions are calculated correctly
        for boundary in boundaries:
            assert 0 <= boundary.start_index < len(content)
            assert boundary.start_index < boundary.end_index <= len(content)
    
    def test_confidence_scores(self, detector):
        """Test confidence scoring for different header types."""
        content = """# Markdown Header
This should have high confidence.

Introduction:
This should have medium confidence.
"""
        
        boundaries = detector.detect_boundaries(content)
        
        # Markdown headers should have higher confidence
        markdown_boundaries = [b for b in boundaries if b.title.startswith('#')]
        colon_boundaries = [b for b in boundaries if b.title.endswith(':')]
        
        if markdown_boundaries and colon_boundaries:
            assert markdown_boundaries[0].confidence > colon_boundaries[0].confidence


class TestSectionAwareRefinementLoop:
    """Test the main refinement loop functionality."""
    
    @pytest.fixture
    def refinement_loop(self):
        """Create refinement loop instance."""
        return SectionAwareRefinementLoop(max_iterations=3)
    
    @pytest.fixture
    def sample_content(self):
        """Sample newsletter content for testing."""
        return """# Introduction
Welcome to our AI newsletter.

# Latest News
- OpenAI released GPT-4.5
- Google announced new Gemini features

# Technical Analysis
The recent developments in language models show significant improvements.

# Conclusion
Stay tuned for more updates next week.
"""
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for refinement."""
        return {
            'topic': 'AI Newsletter',
            'audience': 'AI Researchers',
            'content_focus': 'Language Models',
            'word_count': 1000
        }
    
    def test_initialization(self, refinement_loop):
        """Test refinement loop initialization."""
        assert refinement_loop.max_iterations == 3
        assert isinstance(refinement_loop.boundary_detector, SectionBoundaryDetector)
        assert len(refinement_loop.quality_thresholds) == 5
    
    def test_refine_newsletter_workflow(self, refinement_loop, sample_content, sample_context):
        """Test complete newsletter refinement workflow."""
        with patch.object(refinement_loop, '_refine_section') as mock_refine:
            # Mock the section refinement to return improved content
            def side_effect(section, context):
                section.content = f"Refined: {section.content}"
                section.quality_score = 0.9
                return section
            
            mock_refine.side_effect = side_effect
            
            result = refinement_loop.refine_newsletter(sample_content, sample_context)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Refined:" in result
            assert mock_refine.call_count > 0
    
    def test_extract_sections(self, refinement_loop, sample_content):
        """Test section extraction from content."""
        boundaries = refinement_loop.boundary_detector.detect_boundaries(sample_content)
        sections = refinement_loop._extract_sections(sample_content, boundaries)
        
        assert len(sections) > 0
        for section in sections:
            assert isinstance(section, SectionContent)
            assert len(section.content) > 0
            assert 0.0 <= section.quality_score <= 1.0
    
    def test_section_quality_validation(self, refinement_loop):
        """Test section quality validation methods."""
        test_cases = [
            (SectionType.INTRODUCTION, "Welcome to our newsletter overview", 0.0),
            (SectionType.TUTORIAL, "Step 1: Install the package", 0.0),
            (SectionType.NEWS, "Company announced new features today", 0.0),
            (SectionType.ANALYSIS, "The data shows interesting trends", 0.0),
            (SectionType.CONCLUSION, "In summary, these are key takeaways", 0.0)
        ]
        
        for section_type, content, min_score in test_cases:
            score = refinement_loop.validate_section_quality(content, section_type)
            assert min_score <= score <= 1.0
    
    def test_refine_section_iterations(self, refinement_loop):
        """Test section refinement with multiple iterations."""
        boundary = SectionBoundary(
            section_type=SectionType.ANALYSIS,
            start_index=0,
            end_index=100
        )
        
        section = SectionContent(
            section_type=SectionType.ANALYSIS,
            content="Basic analysis content that needs improvement.",
            boundary=boundary,
            quality_score=0.5
        )
        
        context = {'topic': 'Test', 'audience': 'Test', 'content_focus': 'Test'}
        
        with patch.object(refinement_loop, 'refine_by_section') as mock_refine, \
             patch.object(refinement_loop, 'validate_section_quality') as mock_validate:
            
            # Mock progressive improvement
            mock_refine.side_effect = lambda content, section_type, context: f"Improved: {content}"
            mock_validate.side_effect = [0.6, 0.7, 0.8, 0.9]  # Progressive improvement
            
            result = refinement_loop._refine_section(section, context)
            
            assert result.refinement_count > 0
            assert "Improved:" in result.content
            assert result.quality_score > 0.5
    
    def test_reassemble_content(self, refinement_loop):
        """Test content reassembly with proper ordering."""
        sections = [
            SectionContent(
                section_type=SectionType.CONCLUSION,
                content="Conclusion content",
                boundary=SectionBoundary(SectionType.CONCLUSION, 300, 400)
            ),
            SectionContent(
                section_type=SectionType.INTRODUCTION,
                content="Introduction content",
                boundary=SectionBoundary(SectionType.INTRODUCTION, 0, 100)
            ),
            SectionContent(
                section_type=SectionType.ANALYSIS,
                content="Analysis content",
                boundary=SectionBoundary(SectionType.ANALYSIS, 100, 300)
            )
        ]
        
        result = refinement_loop._reassemble_content(sections)
        
        assert "Introduction content" in result
        assert "Analysis content" in result
        assert "Conclusion content" in result
        
        # Check ordering - introduction should come before conclusion
        intro_pos = result.find("Introduction content")
        conclusion_pos = result.find("Conclusion content")
        assert intro_pos < conclusion_pos
    
    def test_validate_narrative_flow(self, refinement_loop, sample_context):
        """Test narrative flow validation."""
        content = """Introduction section ends abruptly
        
        next section starts without transition
        
        Another section follows."""
        
        result = refinement_loop._validate_narrative_flow(content, sample_context)
        
        assert isinstance(result, str)
        assert len(result) >= len(content)  # May add transitions
    
    def test_error_handling(self, refinement_loop, sample_context):
        """Test error handling in refinement process."""
        # Test with invalid content that might cause errors
        invalid_content = ""
        
        result = refinement_loop.refine_newsletter(invalid_content, sample_context)
        
        # Should not crash and return some result
        assert isinstance(result, str)
    
    def test_determine_pass_type(self, refinement_loop):
        """Test refinement pass type determination."""
        boundary = SectionBoundary(SectionType.ANALYSIS, 0, 100)
        section = SectionContent(SectionType.ANALYSIS, "content", boundary)
        
        # Test pass type progression
        assert refinement_loop._determine_pass_type(section, 0) == RefinementPass.STRUCTURE
        assert refinement_loop._determine_pass_type(section, 1) == RefinementPass.CONTENT
        assert refinement_loop._determine_pass_type(section, 2) == RefinementPass.STYLE
    
    def test_basic_refinements(self, refinement_loop):
        """Test basic text refinements."""
        test_cases = [
            ("Text with   extra   spaces", "Text with extra spaces"),
            ("Sentence.Next sentence", "Sentence. Next sentence"),
            ("Text\n\n\n\nwith extra newlines", "Text\n\nwith extra newlines")
        ]
        
        for input_text, expected_pattern in test_cases:
            result = refinement_loop._apply_basic_refinements(input_text, SectionType.ANALYSIS)
            
            # Check that basic cleanup was applied
            assert "  " not in result  # No double spaces
            assert "\n\n\n" not in result  # No triple newlines
    
    def test_tutorial_structure_improvement(self, refinement_loop):
        """Test tutorial-specific structure improvements."""
        tutorial_content = """First, you need to install the software.
Next, configure the settings.
Then, run the application.
Finally, test the installation."""
        
        result = refinement_loop._improve_tutorial_structure(tutorial_content)
        
        # Should add step numbering
        assert "1." in result
        assert "2." in result or any(char.isdigit() for char in result)
    
    def test_news_structure_improvement(self, refinement_loop):
        """Test news-specific structure improvements."""
        news_content = """Company announced new product launch.
        Startup released innovative feature.
        Research team reported breakthrough."""
        
        result = refinement_loop._improve_news_structure(news_content)
        
        # Should add bullet points for news items
        assert "•" in result or "-" in result


class TestQualityValidation:
    """Test section-specific quality validation methods."""
    
    @pytest.fixture
    def refinement_loop(self):
        """Create refinement loop for testing."""
        return SectionAwareRefinementLoop()
    
    def test_introduction_quality_validation(self, refinement_loop):
        """Test introduction quality validation."""
        good_intro = "What if we could revolutionize AI? This newsletter will explore cutting-edge developments."
        poor_intro = "This is a newsletter."
        
        good_score = refinement_loop._validate_introduction_quality(good_intro)
        poor_score = refinement_loop._validate_introduction_quality(poor_intro)
        
        assert good_score > poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0
    
    def test_analysis_quality_validation(self, refinement_loop):
        """Test analysis quality validation."""
        good_analysis = """This comprehensive analysis reveals significant trends in the data. 
        Research evidence shows clear patterns emerging across multiple dimensions."""
        poor_analysis = "Some things happened."
        
        good_score = refinement_loop._validate_analysis_quality(good_analysis)
        poor_score = refinement_loop._validate_analysis_quality(poor_analysis)
        
        assert good_score > poor_score
    
    def test_tutorial_quality_validation(self, refinement_loop):
        """Test tutorial quality validation."""
        good_tutorial = """Step 1: Create a new project folder.
        Step 2: Install the required dependencies.
        Step 3: Configure the settings file."""
        poor_tutorial = "Do some stuff to set it up."
        
        good_score = refinement_loop._validate_tutorial_quality(good_tutorial)
        poor_score = refinement_loop._validate_tutorial_quality(poor_tutorial)
        
        assert good_score > poor_score
    
    def test_news_quality_validation(self, refinement_loop):
        """Test news quality validation."""
        good_news = "Today, TechCorp announced the release of version 2.5 with $50M funding."
        poor_news = "Something was announced."
        
        good_score = refinement_loop._validate_news_quality(good_news)
        poor_score = refinement_loop._validate_news_quality(poor_news)
        
        assert good_score > poor_score
    
    def test_conclusion_quality_validation(self, refinement_loop):
        """Test conclusion quality validation."""
        good_conclusion = """In summary, these key takeaways show the importance of AI advancement. 
        Try implementing these strategies in your next project."""
        poor_conclusion = "That's all."
        
        good_score = refinement_loop._validate_conclusion_quality(good_conclusion)
        poor_score = refinement_loop._validate_conclusion_quality(poor_conclusion)
        
        assert good_score > poor_score


class TestTransitionLogic:
    """Test transition detection and generation."""
    
    @pytest.fixture
    def refinement_loop(self):
        """Create refinement loop for testing."""
        return SectionAwareRefinementLoop()
    
    def test_needs_transition_detection(self, refinement_loop):
        """Test transition need detection."""
        # Should need transition (abrupt ending)
        abrupt_prev = "This section ends without proper punctuation"
        normal_current = "This section starts normally."
        
        needs_transition = refinement_loop._needs_transition(abrupt_prev, normal_current)
        assert needs_transition
        
        # Should not need transition (proper ending)
        proper_prev = "This section ends properly."
        proper_current = "This section starts normally."
        
        no_transition = refinement_loop._needs_transition(proper_prev, proper_current)
        assert not no_transition
    
    def test_transition_generation(self, refinement_loop):
        """Test transition generation."""
        prev_section = "Previous section content."
        current_section = "Current section content."
        
        transition = refinement_loop._generate_transition(prev_section, current_section)
        
        assert isinstance(transition, str)
        assert len(transition) > 0


class TestIntegration:
    """Integration tests for the complete refinement system."""
    
    def test_full_refinement_pipeline(self):
        """Test complete refinement pipeline."""
        refinement_loop = SectionAwareRefinementLoop(max_iterations=2)
        
        content = """# Introduction
        Basic intro.
        
        # Analysis  
        Some analysis here.
        
        # Conclusion
        Basic conclusion."""
        
        context = {
            'topic': 'AI Development',
            'audience': 'Developers',
            'content_focus': 'Machine Learning',
            'word_count': 500
        }
        
        result = refinement_loop.refine_newsletter(content, context)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction" in result or "intro" in result.lower()
        assert "conclusion" in result.lower()
    
    def test_boundary_detection_integration(self):
        """Test integration between boundary detection and refinement."""
        refinement_loop = SectionAwareRefinementLoop()
        
        content_with_boundaries = """## Welcome
        Welcome to our newsletter.
        
        ### Latest Updates
        Here are the news items.
        
        ## Summary
        Thank you for reading."""
        
        boundaries = refinement_loop.boundary_detector.detect_boundaries(content_with_boundaries)
        sections = refinement_loop._extract_sections(content_with_boundaries, boundaries)
        
        assert len(boundaries) > 0
        assert len(sections) == len(boundaries)
        
        for section in sections:
            assert section.section_type != SectionType.GENERAL
            assert len(section.content.strip()) > 0


if __name__ == '__main__':
    pytest.main([__file__])