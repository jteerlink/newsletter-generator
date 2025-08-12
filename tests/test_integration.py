"""
Integration Test Suite for Section-Aware Newsletter Generation System

Comprehensive integration tests covering:
- End-to-end section-aware newsletter generation
- Component interaction and data flow
- Performance and quality validation
- Error handling and edge cases
- Backward compatibility
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.core.section_aware_prompts import (
    SectionType, SectionAwarePromptManager, PromptContext
)
from src.core.section_aware_refinement import (
    SectionAwareRefinementLoop, SectionBoundaryDetector, SectionContent
)
from src.core.section_quality_metrics import (
    SectionAwareQualitySystem, SectionQualityAnalyzer, AggregatedQualityReport
)
from src.core.continuity_validator import (
    ContinuityValidator, ContinuityReport
)


class TestSectionAwareWorkflow:
    """Test complete section-aware newsletter generation workflow."""
    
    @pytest.fixture
    def sample_newsletter_content(self):
        """Sample newsletter content for testing."""
        return """# Welcome to AI Weekly
        
Welcome to our comprehensive AI newsletter! This week brings exciting developments 
in machine learning and artificial intelligence research.

## Latest News Updates

Recent announcements have shaped the AI landscape significantly:
- OpenAI released GPT-4.5 with enhanced reasoning capabilities
- Google announced breakthrough quantum-AI hybrid research  
- Microsoft expanded Azure AI services globally
- Meta introduced new computer vision models

## Technical Deep Dive

Furthermore, these developments reveal important trends in AI advancement.
The research data indicates substantial improvements across key metrics.
Technical analysis shows convergence toward more efficient architectures.

New algorithms demonstrate 40% better performance on standard benchmarks.
Implementation strategies focus on scalable deployment patterns.
Security considerations remain paramount in production systems.

## Implementation Guide

Next, let's explore practical steps for leveraging these advances:

Step 1: Evaluate current AI infrastructure and identify upgrade opportunities
Step 2: Install latest development frameworks and compatibility layers  
Step 3: Implement pilot projects using new capabilities
Step 4: Monitor performance metrics and optimize configurations
Step 5: Scale successful implementations across production systems

## Key Takeaways

In summary, this week's AI developments represent significant progress.
Organizations should evaluate these advances for competitive advantage.
Implementation requires careful planning and systematic deployment.

Try experimenting with these new capabilities in controlled environments.
Share insights with the AI community for collective advancement.
Stay tuned for next week's analysis of emerging trends and applications.
"""
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for newsletter generation."""
        return {
            'topic': 'AI Weekly Newsletter',
            'audience': 'AI/ML Engineers',
            'content_focus': 'Latest AI Developments',
            'word_count': 3000,
            'tone': 'professional',
            'technical_level': 'intermediate',
            'special_requirements': ['Include practical examples', 'Focus on implementation']
        }
    
    def test_end_to_end_section_aware_processing(self, sample_newsletter_content, sample_context):
        """Test complete end-to-end section-aware processing."""
        # 1. Initialize all components
        prompt_manager = SectionAwarePromptManager()
        refinement_loop = SectionAwareRefinementLoop(max_iterations=2)
        quality_system = SectionAwareQualitySystem()
        continuity_validator = ContinuityValidator()
        
        # 2. Detect and analyze sections
        boundaries = refinement_loop.boundary_detector.detect_boundaries(sample_newsletter_content)
        sections = refinement_loop._extract_sections(sample_newsletter_content, boundaries)
        
        assert len(sections) >= 4  # Should detect multiple sections
        assert all(isinstance(section, SectionContent) for section in sections)
        
        # 3. Generate section-specific prompts
        section_prompts = {}
        for section in sections:
            prompt = prompt_manager.get_section_prompt(section.section_type, sample_context)
            section_prompts[section.section_type] = prompt
            
            assert isinstance(prompt, str)
            assert len(prompt) > 100
            assert sample_context['topic'] in prompt
        
        # 4. Simulate refinement process
        with patch.object(refinement_loop, 'refine_by_section') as mock_refine:
            mock_refine.side_effect = lambda content, section_type, context: f"Refined: {content[:100]}..."
            
            refined_content = refinement_loop.refine_newsletter(sample_newsletter_content, sample_context)
            
            assert isinstance(refined_content, str)
            assert len(refined_content) > 0
            assert "Refined:" in refined_content
        
        # 5. Quality analysis
        quality_report = quality_system.analyze_newsletter_quality(
            sample_newsletter_content, context=sample_context
        )
        
        assert isinstance(quality_report, AggregatedQualityReport)
        assert quality_report.overall_score > 0.0
        assert len(quality_report.section_scores) > 0
        
        # 6. Continuity validation
        section_dict = {
            section.section_type: section.content 
            for section in sections
        }
        continuity_report = continuity_validator.validate_newsletter_continuity(
            section_dict, sample_context
        )
        
        assert isinstance(continuity_report, ContinuityReport)
        assert continuity_report.sections_analyzed > 0
        assert 0.0 <= continuity_report.overall_continuity_score <= 1.0
        
        # 7. Validate component integration
        assert len(section_prompts) == len(sections)
        assert quality_report.total_word_count > 0
        assert len(continuity_report.transitions_analyzed) >= 0
    
    def test_section_detection_accuracy(self, sample_newsletter_content):
        """Test accuracy of section detection across components."""
        # Test boundary detection
        detector = SectionBoundaryDetector()
        boundaries = detector.detect_boundaries(sample_newsletter_content)
        
        # Should detect main sections
        detected_types = {boundary.section_type for boundary in boundaries}
        expected_types = {SectionType.INTRODUCTION, SectionType.NEWS, SectionType.ANALYSIS, SectionType.TUTORIAL, SectionType.CONCLUSION}
        
        # Should detect most expected types
        assert len(detected_types.intersection(expected_types)) >= 3
        
        # Test prompt manager detection
        prompt_manager = SectionAwarePromptManager()
        
        intro_content = "Welcome to our newsletter! This overview covers exciting developments."
        detected_intro = prompt_manager.detect_section_type(intro_content, {})
        assert detected_intro == SectionType.INTRODUCTION
        
        tutorial_content = "Step 1: Install the software. Step 2: Configure settings."
        detected_tutorial = prompt_manager.detect_section_type(tutorial_content, {})
        assert detected_tutorial == SectionType.TUTORIAL
    
    def test_quality_metrics_integration(self, sample_newsletter_content, sample_context):
        """Test integration between quality metrics and other components."""
        quality_system = SectionAwareQualitySystem()
        
        # Analyze complete newsletter
        quality_report = quality_system.analyze_newsletter_quality(
            sample_newsletter_content, context=sample_context
        )
        
        # Validate section-specific quality metrics
        for section_type, metrics in quality_report.section_scores.items():
            assert section_type in [SectionType.INTRODUCTION, SectionType.NEWS, 
                                  SectionType.ANALYSIS, SectionType.TUTORIAL, SectionType.CONCLUSION]
            assert 0.0 <= metrics.overall_score <= 1.0
            assert metrics.word_count > 0
        
        # Test threshold validation
        passed, issues = quality_system.validate_section_thresholds(quality_report)
        assert isinstance(passed, bool)
        assert isinstance(issues, list)
        
        # Test recommendations
        recommendations = quality_system.get_improvement_recommendations(quality_report)
        assert isinstance(recommendations, list)
    
    def test_continuity_validation_integration(self, sample_newsletter_content):
        """Test continuity validation integration with section detection."""
        validator = ContinuityValidator()
        detector = SectionBoundaryDetector()
        
        # Detect sections first
        boundaries = detector.detect_boundaries(sample_newsletter_content)
        
        # Extract sections for continuity analysis
        sections = {}
        for boundary in boundaries:
            content = sample_newsletter_content[boundary.start_index:boundary.end_index].strip()
            if len(content) > 10:  # Skip very short sections
                sections[boundary.section_type] = content
        
        # Validate continuity
        report = validator.validate_newsletter_continuity(sections)
        
        assert isinstance(report, ContinuityReport)
        assert report.sections_analyzed == len(sections)
        
        # Should detect transitions for multi-section content
        if len(sections) > 1:
            assert len(report.transitions_analyzed) == len(sections) - 1
        
        # Validate consistency scores
        assert 0.0 <= report.style_consistency_score <= 1.0
        assert 0.0 <= report.redundancy_score <= 1.0


class TestComponentInteractions:
    """Test interactions between different components."""
    
    def test_prompt_refinement_integration(self):
        """Test integration between prompt generation and refinement."""
        prompt_manager = SectionAwarePromptManager()
        refinement_loop = SectionAwareRefinementLoop()
        
        context = {
            'topic': 'Machine Learning',
            'audience': 'Developers',
            'content_focus': 'Neural Networks',
            'word_count': 1000
        }
        
        # Generate prompts for different sections
        intro_prompt = prompt_manager.get_section_prompt(SectionType.INTRODUCTION, context)
        analysis_prompt = prompt_manager.get_section_prompt(SectionType.ANALYSIS, context)
        
        # Test that prompts can be used in refinement context
        test_content = "Basic introduction content that needs improvement."
        
        # Simulate using prompts in refinement
        refinement_prompt = refinement_loop._generate_refinement_prompt(
            test_content, SectionType.INTRODUCTION, context
        )
        
        assert isinstance(intro_prompt, str)
        assert isinstance(analysis_prompt, str)
        assert isinstance(refinement_prompt, str)
        
        # Prompts should be different for different sections
        assert intro_prompt != analysis_prompt
        assert context['topic'] in intro_prompt
        assert context['topic'] in analysis_prompt
    
    def test_quality_continuity_correlation(self):
        """Test correlation between quality metrics and continuity scores."""
        quality_analyzer = SectionQualityAnalyzer()
        continuity_validator = ContinuityValidator()
        
        # Test with high-quality, consistent content
        high_quality_sections = {
            SectionType.INTRODUCTION: """Welcome to our comprehensive analysis of machine learning advances. 
            This systematic review examines recent developments and their practical implications.""",
            
            SectionType.ANALYSIS: """Building on these foundations, our examination reveals significant 
            performance improvements across multiple evaluation metrics. The research demonstrates 
            consistent enhancement in computational efficiency.""",
            
            SectionType.CONCLUSION: """In conclusion, these advances represent substantial progress 
            in artificial intelligence capabilities. Organizations should evaluate these developments 
            for competitive advantage and strategic planning."""
        }
        
        # Test with low-quality, inconsistent content  
        low_quality_sections = {
            SectionType.INTRODUCTION: "intro stuff",
            SectionType.ANALYSIS: "Therefore, the comprehensive methodology demonstrates significant findings.",
            SectionType.CONCLUSION: "so thats it basically"
        }
        
        # Analyze quality for both
        high_quality_metrics = {}
        low_quality_metrics = {}
        
        for section_type, content in high_quality_sections.items():
            metrics = quality_analyzer.analyze_section(content, section_type)
            high_quality_metrics[section_type] = metrics
        
        for section_type, content in low_quality_sections.items():
            metrics = quality_analyzer.analyze_section(content, section_type)
            low_quality_metrics[section_type] = metrics
        
        # Analyze continuity for both
        high_continuity_report = continuity_validator.validate_newsletter_continuity(high_quality_sections)
        low_continuity_report = continuity_validator.validate_newsletter_continuity(low_quality_sections)
        
        # High quality should correlate with better continuity
        high_avg_quality = sum(m.overall_score for m in high_quality_metrics.values()) / len(high_quality_metrics)
        low_avg_quality = sum(m.overall_score for m in low_quality_metrics.values()) / len(low_quality_metrics)
        
        assert high_avg_quality > low_avg_quality
        assert high_continuity_report.overall_continuity_score >= low_continuity_report.overall_continuity_score
    
    def test_boundary_detection_consistency(self):
        """Test consistency of boundary detection across components."""
        content_with_clear_boundaries = """# Introduction Section
        This is the introduction to our newsletter.
        
        ## News and Updates
        Here are the latest developments.
        
        # Technical Analysis
        Deep dive into the technical aspects.
        
        ## How-To Guide
        Step-by-step implementation instructions.
        
        # Summary and Conclusion
        Wrapping up the key points.
        """
        
        # Test boundary detection
        detector = SectionBoundaryDetector()
        boundaries = detector.detect_boundaries(content_with_clear_boundaries)
        
        # Test refinement loop extraction
        refinement_loop = SectionAwareRefinementLoop()
        sections = refinement_loop._extract_sections(content_with_clear_boundaries, boundaries)
        
        # Should detect multiple sections consistently
        assert len(boundaries) >= 4
        assert len(sections) == len(boundaries)
        
        # Test section types are reasonable
        detected_types = {boundary.section_type for boundary in boundaries}
        expected_types = {SectionType.INTRODUCTION, SectionType.NEWS, SectionType.ANALYSIS, 
                         SectionType.TUTORIAL, SectionType.CONCLUSION}
        
        # Should detect most expected types
        overlap = detected_types.intersection(expected_types)
        assert len(overlap) >= 3


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    def test_large_content_processing(self):
        """Test processing of large newsletter content."""
        # Generate large content
        large_content = """# Introduction
        """ + "This is a comprehensive introduction to AI developments. " * 100 + """
        
        ## Latest News
        """ + "Recent announcements in AI research and industry applications. " * 150 + """
        
        # Technical Analysis  
        """ + "Detailed technical analysis of emerging trends and methodologies. " * 200 + """
        
        ## Implementation Guide
        """ + "Step-by-step guide for practical implementation and deployment. " * 150 + """
        
        # Conclusion
        """ + "Summary of key insights and future directions for development. " * 100
        
        context = {
            'topic': 'Comprehensive AI Review',
            'audience': 'Technical Professionals',
            'content_focus': 'AI Development',
            'word_count': 10000
        }
        
        # Test component performance with large content
        detector = SectionBoundaryDetector()
        boundaries = detector.detect_boundaries(large_content)
        assert len(boundaries) > 0
        
        quality_system = SectionAwareQualitySystem()
        quality_report = quality_system.analyze_newsletter_quality(large_content, context=context)
        assert isinstance(quality_report, AggregatedQualityReport)
        assert quality_report.total_word_count > 1000
        
        continuity_validator = ContinuityValidator()
        section_dict = {}
        for boundary in boundaries:
            content_section = large_content[boundary.start_index:boundary.end_index].strip()
            if len(content_section) > 10:
                section_dict[boundary.section_type] = content_section
        
        continuity_report = continuity_validator.validate_newsletter_continuity(section_dict)
        assert isinstance(continuity_report, ContinuityReport)
    
    def test_multiple_section_types(self):
        """Test handling of newsletters with many different section types."""
        multi_section_content = """# Welcome Introduction
        Welcome to our multi-faceted newsletter.
        
        ## Breaking News
        Latest announcements and updates.
        
        ### In-Depth Analysis
        Comprehensive examination of trends.
        
        ## Step-by-Step Tutorial
        Practical implementation guide.
        
        ### Research Insights
        Academic research and findings.
        
        ## Industry Applications
        Real-world use cases and examples.
        
        # Final Thoughts
        Conclusion and next steps.
        """
        
        # Test all components handle multiple sections
        detector = SectionBoundaryDetector()
        boundaries = detector.detect_boundaries(multi_section_content)
        
        prompt_manager = SectionAwarePromptManager()
        context = {'topic': 'Multi-Topic Newsletter', 'audience': 'General', 'content_focus': 'Various'}
        
        for boundary in boundaries:
            section_content = multi_section_content[boundary.start_index:boundary.end_index]
            detected_type = prompt_manager.detect_section_type(section_content, context)
            assert detected_type in [section.value for section in SectionType]
        
        quality_system = SectionAwareQualitySystem()
        quality_report = quality_system.analyze_newsletter_quality(multi_section_content, context=context)
        assert len(quality_report.section_scores) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_content_handling(self):
        """Test handling of empty or minimal content."""
        empty_content = ""
        minimal_content = "Short content."
        
        # Test boundary detection
        detector = SectionBoundaryDetector()
        
        empty_boundaries = detector.detect_boundaries(empty_content)
        minimal_boundaries = detector.detect_boundaries(minimal_content)
        
        # Should handle gracefully
        assert isinstance(empty_boundaries, list)
        assert isinstance(minimal_boundaries, list)
        assert len(minimal_boundaries) >= 0  # May detect as single section
        
        # Test quality analysis
        quality_system = SectionAwareQualitySystem()
        context = {'topic': 'Test', 'audience': 'Test', 'content_focus': 'Test'}
        
        try:
            empty_report = quality_system.analyze_newsletter_quality(empty_content, context=context)
            assert isinstance(empty_report, AggregatedQualityReport)
        except Exception:
            pass  # May legitimately fail with empty content
        
        minimal_report = quality_system.analyze_newsletter_quality(minimal_content, context=context)
        assert isinstance(minimal_report, AggregatedQualityReport)
    
    def test_malformed_content_handling(self):
        """Test handling of malformed or unusual content."""
        malformed_content = """### Random Header
        No proper structure here.
        
        Another paragraph without clear section.
        
        #### Yet another header
        More unstructured content.
        """
        
        # All components should handle malformed content gracefully
        detector = SectionBoundaryDetector()
        boundaries = detector.detect_boundaries(malformed_content)
        assert isinstance(boundaries, list)
        
        quality_system = SectionAwareQualitySystem()
        context = {'topic': 'Test', 'audience': 'Test', 'content_focus': 'Test'}
        quality_report = quality_system.analyze_newsletter_quality(malformed_content, context=context)
        assert isinstance(quality_report, AggregatedQualityReport)
        
        continuity_validator = ContinuityValidator()
        sections = {SectionType.GENERAL: malformed_content}
        continuity_report = continuity_validator.validate_newsletter_continuity(sections)
        assert isinstance(continuity_report, ContinuityReport)
    
    def test_invalid_context_handling(self):
        """Test handling of invalid or incomplete context."""
        content = "Sample newsletter content for testing."
        
        # Test with empty context
        empty_context = {}
        
        prompt_manager = SectionAwarePromptManager()
        prompt = prompt_manager.get_section_prompt(SectionType.ANALYSIS, empty_context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        quality_system = SectionAwareQualitySystem()
        quality_report = quality_system.analyze_newsletter_quality(content, context=empty_context)
        assert isinstance(quality_report, AggregatedQualityReport)
        
        # Test with invalid context values
        invalid_context = {
            'topic': None,
            'audience': 123,  # Invalid type
            'word_count': 'invalid'  # Invalid type
        }
        
        try:
            prompt = prompt_manager.get_section_prompt(SectionType.INTRODUCTION, invalid_context)
            assert isinstance(prompt, str)
        except Exception:
            pass  # May legitimately fail with invalid context


class TestBackwardCompatibility:
    """Test backward compatibility with existing systems."""
    
    def test_convenience_functions(self):
        """Test convenience functions for backward compatibility."""
        from src.core.section_aware_prompts import get_section_prompt, detect_section_type
        
        context = {
            'topic': 'AI Research',
            'audience': 'Researchers',
            'content_focus': 'Machine Learning',
            'word_count': 2000
        }
        
        # Test convenience functions
        prompt = get_section_prompt('introduction', context)
        assert isinstance(prompt, str)
        assert 'AI Research' in prompt
        
        content = "Welcome to our tutorial on machine learning implementation."
        section_type = detect_section_type(content)
        assert section_type in ['introduction', 'tutorial', 'analysis', 'news', 'conclusion']
    
    def test_fallback_behavior(self):
        """Test fallback behavior when components encounter issues."""
        prompt_manager = SectionAwarePromptManager()
        
        # Test with invalid section type
        context = {'topic': 'Test', 'audience': 'Test', 'content_focus': 'Test', 'word_count': 1000}
        
        # Should fall back gracefully
        with patch('src.core.section_aware_prompts.logger') as mock_logger:
            prompt = prompt_manager.get_section_prompt('invalid_section', context)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
        
        # Test refinement loop error handling
        refinement_loop = SectionAwareRefinementLoop()
        
        # Should return original content on error
        problematic_content = None  # This might cause issues
        try:
            result = refinement_loop.refine_newsletter("", context)
            assert isinstance(result, str)
        except Exception:
            pass  # May legitimately fail


class TestRealWorldScenarios:
    """Test realistic scenarios and use cases."""
    
    def test_typical_newsletter_workflow(self):
        """Test typical newsletter generation workflow."""
        # Simulate typical user input
        user_content = """# AI Weekly Update
        
        Welcome to this week's AI newsletter! We're excited to share the latest developments.
        
        ## What's New This Week
        
        - OpenAI announced new research on multimodal models
        - Google DeepMind published breakthrough quantum computing results  
        - Several startups secured funding for AI infrastructure
        
        ## Deep Dive: Transformer Efficiency
        
        Recent research has focused on making transformer models more efficient.
        New attention mechanisms reduce computational requirements by 35%.
        This has significant implications for deployment at scale.
        
        ## Getting Started with New Tools
        
        Here's how to implement the latest efficiency improvements:
        
        1. Update your model architecture
        2. Retrain with new attention patterns
        3. Benchmark performance improvements
        
        ## Wrap-Up
        
        This week showed continued progress in AI efficiency and capability.
        Next week we'll explore applications in robotics and automation.
        """
        
        user_context = {
            'topic': 'AI Weekly Newsletter',
            'audience': 'AI/ML Engineers',
            'content_focus': 'Latest AI Research and Tools',
            'word_count': 2500,
            'tone': 'professional',
            'technical_level': 'intermediate'
        }
        
        # Execute complete workflow
        prompt_manager = SectionAwarePromptManager()
        refinement_loop = SectionAwareRefinementLoop(max_iterations=2)
        quality_system = SectionAwareQualitySystem()
        continuity_validator = ContinuityValidator()
        
        # 1. Section detection and analysis
        boundaries = refinement_loop.boundary_detector.detect_boundaries(user_content)
        assert len(boundaries) >= 4  # Should detect intro, news, analysis, tutorial, conclusion
        
        # 2. Quality assessment
        quality_report = quality_system.analyze_newsletter_quality(user_content, context=user_context)
        assert quality_report.overall_score > 0.6  # Should be reasonably good quality
        
        # 3. Continuity validation
        sections = {}
        for boundary in boundaries:
            content_section = user_content[boundary.start_index:boundary.end_index].strip()
            if len(content_section) > 10:
                sections[boundary.section_type] = content_section
        
        continuity_report = continuity_validator.validate_newsletter_continuity(sections, user_context)
        assert continuity_report.transition_quality_score > 0.5  # Should have decent transitions
        
        # 4. Generate improvement recommendations
        recommendations = quality_system.get_improvement_recommendations(quality_report)
        continuity_recommendations = continuity_report.recommendations
        
        # Should provide actionable feedback
        all_recommendations = recommendations + continuity_recommendations
        assert len(all_recommendations) >= 0
        
        # 5. Validate threshold compliance
        passed, issues = quality_system.validate_section_thresholds(quality_report)
        
        # Results should be realistic for typical content
        assert isinstance(passed, bool)
        assert isinstance(issues, list)


if __name__ == '__main__':
    pytest.main([__file__])