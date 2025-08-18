"""
Comprehensive Test Suite for Content Expansion System

Tests for Phase 1: Intelligent Content Expansion including unit tests,
integration tests, and quality validation.
"""

import os
import sys
import unittest
from typing import Any, Dict
from unittest.mock import Mock

# Import test utilities
from test_utils import TestAssertions, TestConstants, TestDataFactory

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.content_expansion import (
    ContentExpansionResult,
    ExpansionPriority,
    ExpansionStrategy,
    IntelligentContentExpander,
)
from core.section_expansion import SectionExpansionOrchestrator, SectionExpansionResult, SectionType
from core.template_manager import NewsletterTemplate


class TestIntelligentContentExpander(unittest.TestCase):
    """Test suite for IntelligentContentExpander."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expander = IntelligentContentExpander()
        
        # Use test utilities for consistent test data
        self.sample_content = TestDataFactory.create_sample_newsletter_content()
        self.sample_metadata = TestDataFactory.create_test_metadata()
    
    def test_initialization(self):
        """Test proper initialization of content expander."""
        self.assertIsInstance(self.expander.expansion_strategies, dict)
        self.assertIsInstance(self.expander.content_templates, dict)
        self.assertIsInstance(self.expander.quality_thresholds, dict)
        
        # Check that all required strategies are present
        required_strategies = [
            ExpansionStrategy.TECHNICAL_DEEP_DIVE,
            ExpansionStrategy.TUTORIAL_ENHANCEMENT,
            ExpansionStrategy.ANALYSIS_ENHANCEMENT
        ]
        for strategy in required_strategies:
            self.assertIn(strategy, self.expander.expansion_strategies)
    
    def test_expand_content_basic(self):
        """Test basic content expansion functionality."""
        target_words = 1000
        template_type = "technical_newsletter"
        
        result = self.expander.expand_content(
            content=self.sample_content,
            target_words=target_words,
            template_type=template_type,
            metadata=self.sample_metadata
        )
        
        # Verify result structure using test utilities
        TestAssertions.assert_expansion_result_valid(result, self)
        self.assertEqual(result.original_content, self.sample_content)
        TestAssertions.assert_word_count_improvement(self.sample_content, result.expanded_content, self)
    
    def test_expansion_target_achievement(self):
        """Test that expansion achieves target word counts."""
        target_words = 800
        
        result = self.expander.expand_content(
            content=self.sample_content,
            target_words=target_words,
            template_type="newsletter",
            metadata=self.sample_metadata
        )
        
        expanded_word_count = len(result.expanded_content.split())
        original_word_count = len(self.sample_content.split())
        
        # Should significantly increase word count
        self.assertGreater(expanded_word_count, original_word_count)
        
        # Should achieve reasonable percentage of target
        achievement_ratio = expanded_word_count / target_words
        self.assertGreater(achievement_ratio, 0.7)  # At least 70% of target
    
    def test_content_structure_analysis(self):
        """Test content structure analysis functionality."""
        analysis = self.expander._analyze_content_structure(self.sample_content)
        
        # Verify analysis structure
        required_keys = [
            'sections', 'section_analysis', 'content_gaps',
            'expansion_opportunities', 'quality_indicators'
        ]
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Verify sections are identified
        self.assertGreater(len(analysis['sections']), 0)
        
        # Verify expansion opportunities are found
        self.assertIsInstance(analysis['expansion_opportunities'], list)
    
    def test_expansion_strategy_selection(self):
        """Test expansion strategy selection logic."""
        # Test technical content
        technical_metadata = {'newsletter_type': 'technical', 'complexity_level': 'advanced'}
        strategy = self.expander._select_expansion_strategy(
            self.sample_content, technical_metadata
        )
        self.assertEqual(strategy, ExpansionStrategy.TECHNICAL_DEEP_DIVE)
        
        # Test tutorial content
        tutorial_metadata = {'newsletter_type': 'tutorial', 'target_audience': 'beginners'}
        strategy = self.expander._select_expansion_strategy(
            self.sample_content, tutorial_metadata
        )
        self.assertEqual(strategy, ExpansionStrategy.TUTORIAL_ENHANCEMENT)
    
    def test_quality_preservation(self):
        """Test that expansion preserves content quality."""
        result = self.expander.expand_content(
            content=self.sample_content,
            target_words=600,
            template_type="newsletter",
            metadata=self.sample_metadata
        )
        
        # Quality score should be reasonable
        self.assertGreater(result.quality_score, 0.7)
        
        # Technical accuracy should be preserved
        self.assertGreater(result.technical_accuracy_score, 0.8)
        
        # Content coherence should be maintained
        self.assertGreater(result.content_coherence_score, 0.8)
    
    def test_expansion_with_insufficient_content(self):
        """Test expansion behavior with very short content."""
        short_content = "AI is important."
        
        result = self.expander.expand_content(
            content=short_content,
            target_words=500,
            template_type="newsletter",
            metadata=self.sample_metadata
        )
        
        # Should still succeed but with appropriate handling
        self.assertTrue(result.success)
        self.assertGreater(len(result.expanded_content), len(short_content))
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms."""
        # Test with invalid input
        with patch.object(self.expander, '_generate_section_expansion') as mock_generate:
            mock_generate.side_effect = Exception("Simulated failure")
            
            result = self.expander.expand_content(
                content=self.sample_content,
                target_words=500,
                template_type="newsletter",
                metadata=self.sample_metadata
            )
            
            # Should handle error gracefully
            self.assertFalse(result.success)
            self.assertGreater(len(result.warnings), 0)


class TestSectionExpansionOrchestrator(unittest.TestCase):
    """Test suite for SectionExpansionOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = SectionExpansionOrchestrator()
        
        self.sample_section_content = """
Machine learning algorithms are becoming more sophisticated. 
They can now process vast amounts of data efficiently.
"""
        
        self.sample_metadata = {
            'technical_depth': 'intermediate',
            'target_audience': 'developers'
        }
    
    def test_initialization(self):
        """Test proper initialization of section orchestrator."""
        self.assertIsInstance(self.orchestrator.section_templates, dict)
        self.assertIsInstance(self.orchestrator.expansion_patterns, dict)
        
        # Check that all section types have templates
        for section_type in SectionType:
            self.assertIn(section_type, self.orchestrator.section_templates)
    
    def test_expand_section_basic(self):
        """Test basic section expansion functionality."""
        result = self.orchestrator.expand_section(
            section_name="Technical Overview",
            content=self.sample_section_content,
            target_word_count=200,
            template_type="technical_newsletter",
            metadata=self.sample_metadata
        )
        
        # Verify result structure
        self.assertIsInstance(result, SectionExpansionResult)
        self.assertEqual(result.original_content, self.sample_section_content)
        self.assertGreater(len(result.expanded_content), len(self.sample_section_content))
        self.assertTrue(result.success)
    
    def test_section_type_classification(self):
        """Test section type classification logic."""
        # Test introduction section
        intro_content = "Welcome to our newsletter about artificial intelligence..."
        section_type = self.orchestrator._classify_section_type("Introduction", intro_content)
        self.assertEqual(section_type, SectionType.INTRODUCTION)
        
        # Test technical content section
        tech_content = "The algorithm uses gradient descent optimization..."
        section_type = self.orchestrator._classify_section_type("Algorithm Details", tech_content)
        self.assertEqual(section_type, SectionType.TECHNICAL_CONTENT)
        
        # Test tutorial section
        tutorial_content = "Step 1: Install the required dependencies..."
        section_type = self.orchestrator._classify_section_type("Getting Started", tutorial_content)
        self.assertEqual(section_type, SectionType.TUTORIAL_STEP)
    
    def test_expansion_plan_creation(self):
        """Test expansion plan creation for different section types."""
        plan = self.orchestrator._create_section_expansion_plan(
            SectionType.TECHNICAL_CONTENT,
            self.sample_section_content,
            200,
            self.sample_metadata
        )
        
        # Verify plan structure
        self.assertIn('expansion_elements', plan)
        self.assertIn('target_additions', plan)
        self.assertIn('quality_requirements', plan)
        
        # Verify expansion elements are appropriate for technical content
        elements = plan['expansion_elements']
        self.assertIsInstance(elements, list)
        self.assertGreater(len(elements), 0)
    
    def test_word_count_targeting(self):
        """Test that section expansion achieves target word counts."""
        target_count = 150
        
        result = self.orchestrator.expand_section(
            section_name="Main Content",
            content=self.sample_section_content,
            target_word_count=target_count,
            template_type="newsletter",
            metadata=self.sample_metadata
        )
        
        expanded_word_count = len(result.expanded_content.split())
        original_word_count = len(self.sample_section_content.split())
        
        # Should increase word count significantly
        self.assertGreater(expanded_word_count, original_word_count)
        
        # Should achieve reasonable percentage of target
        achievement_ratio = expanded_word_count / target_count
        self.assertGreater(achievement_ratio, 0.6)  # At least 60% of target


class TestContentExpansionIntegration(unittest.TestCase):
    """Integration tests for content expansion system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.expander = IntelligentContentExpander()
        self.orchestrator = SectionExpansionOrchestrator()
        
        # Sample newsletter template
        self.template = Mock()
        self.template.total_word_target = 2000
        self.template.sections = {
            'introduction': {'target_words': 300},
            'main_content': {'target_words': 1200},
            'conclusion': {'target_words': 500}
        }
        
        # Sample newsletter content
        self.newsletter_content = """
# Weekly AI Newsletter

## Introduction
This week we explore the latest developments in artificial intelligence.

## Main Content
### Machine Learning Advances
Recent breakthroughs in deep learning have opened new possibilities.

### Industry Applications
Companies are implementing AI solutions across various sectors.

## Conclusion
The AI revolution continues to accelerate across industries.
"""
    
    def test_end_to_end_expansion(self):
        """Test complete end-to-end content expansion workflow."""
        metadata = {
            'newsletter_type': 'technical',
            'target_audience': 'professionals',
            'complexity_level': 'intermediate'
        }
        
        # Perform expansion
        result = self.expander.expand_content(
            content=self.newsletter_content,
            target_words=self.template.total_word_target,
            template_type="technical_newsletter",
            metadata=metadata
        )
        
        # Verify successful expansion
        self.assertTrue(result.success)
        self.assertGreater(len(result.expanded_content), len(self.newsletter_content))
        
        # Verify quality metrics
        self.assertGreater(result.quality_score, 0.7)
        self.assertGreater(result.technical_accuracy_score, 0.8)
        
        # Verify expansion details
        self.assertIsInstance(result.expansion_details, dict)
        self.assertIn('sections_expanded', result.expansion_details)
    
    def test_template_compliance(self):
        """Test that expansion complies with template requirements."""
        # Mock template with specific requirements
        template_metadata = {
            'template_type': 'technical_newsletter',
            'required_sections': ['introduction', 'technical_content', 'conclusion'],
            'style_requirements': ['technical_accuracy', 'clear_examples']
        }
        
        result = self.expander.expand_content(
            content=self.newsletter_content,
            target_words=1500,
            template_type="technical_newsletter",
            metadata=template_metadata
        )
        
        # Should maintain template compliance
        self.assertTrue(result.template_compliance_score > 0.8)
        
        # Should preserve required sections
        expanded_content = result.expanded_content
        self.assertIn("# Weekly AI Newsletter", expanded_content)
        self.assertIn("## Introduction", expanded_content)
        self.assertIn("## Main Content", expanded_content)
        self.assertIn("## Conclusion", expanded_content)
    
    def test_quality_gate_integration(self):
        """Test integration with quality gate system."""
        # Test that expansion results include quality metrics needed for gates
        result = self.expander.expand_content(
            content=self.newsletter_content,
            target_words=1000,
            template_type="newsletter",
            metadata={'quality_level': 'high'}
        )
        
        # Verify all required quality metrics are present
        required_metrics = [
            'quality_score',
            'technical_accuracy_score', 
            'content_coherence_score',
            'template_compliance_score'
        ]
        
        for metric in required_metrics:
            self.assertTrue(hasattr(result, metric))
            self.assertIsInstance(getattr(result, metric), (int, float))
            self.assertGreaterEqual(getattr(result, metric), 0.0)
            self.assertLessEqual(getattr(result, metric), 1.0)
    
    def test_performance_benchmarks(self):
        """Test that expansion meets performance requirements."""
        import time
        
        start_time = time.time()
        
        result = self.expander.expand_content(
            content=self.newsletter_content,
            target_words=1500,
            template_type="newsletter",
            metadata={}
        )
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 30.0)  # 30 seconds max
        
        # Should achieve reasonable expansion ratio
        original_words = len(self.newsletter_content.split())
        expanded_words = len(result.expanded_content.split())
        expansion_ratio = expanded_words / original_words
        
        self.assertGreater(expansion_ratio, 1.5)  # At least 50% increase


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)