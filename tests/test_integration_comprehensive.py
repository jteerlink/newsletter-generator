"""
Comprehensive Integration Test Suite

End-to-end testing for the complete newsletter enhancement system including
Phase 1 (content expansion) and Phase 2 (mobile optimization) integration.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.writing import WriterAgent
from core.advanced_quality_gates import ConfigurableQualityGate, QualityDimension
from core.content_expansion import IntelligentContentExpander
from core.mobile_optimizer import MobileContentOptimizer, MobileOptimizationLevel
from core.readability_analyzer import MobileReadabilityAnalyzer
from core.section_expansion import SectionExpansionOrchestrator
from core.template_manager import NewsletterTemplate
from core.typography_manager import DeviceType, ResponsiveTypographyManager
from core.workflow_orchestrator import WorkflowOrchestrator


class TestComprehensiveNewsletterEnhancement(unittest.TestCase):
    """Integration test suite for complete newsletter enhancement workflow."""
    
    def setUp(self):
        """Set up comprehensive test environment."""
        # Initialize all components
        self.content_expander = IntelligentContentExpander()
        self.section_orchestrator = SectionExpansionOrchestrator()
        self.mobile_optimizer = MobileContentOptimizer()
        self.readability_analyzer = MobileReadabilityAnalyzer()
        self.typography_manager = ResponsiveTypographyManager()
        self.quality_gates = ConfigurableQualityGate()
        
        # Mock workflow orchestrator for testing
        self.workflow_orchestrator = Mock(spec=WorkflowOrchestrator)
        
        # Sample newsletter content for testing
        self.base_newsletter_content = """
# Weekly AI Newsletter

## Introduction
Artificial intelligence continues to evolve rapidly.

## Main Content
### Recent Developments
New breakthroughs in machine learning have emerged.

### Industry Impact
Companies are adopting AI at an unprecedented pace.

## Conclusion
The future of AI remains bright and full of possibilities.
"""
        
        # Newsletter template configuration
        self.newsletter_template = Mock(spec=NewsletterTemplate)
        self.newsletter_template.total_word_target = 2000
        self.newsletter_template.sections = {
            'introduction': {'target_words': 400},
            'main_content': {'target_words': 1200},
            'conclusion': {'target_words': 400}
        }
        self.newsletter_template.mobile_optimization_required = True
        self.newsletter_template.target_readability_score = 0.85
        
        # Test metadata
        self.test_metadata = {
            'newsletter_type': 'technical',
            'target_audience': 'professionals',
            'complexity_level': 'intermediate',
            'quality_level': 'high'
        }
    
    def test_complete_enhancement_workflow(self):
        """Test the complete enhancement workflow from start to finish."""
        # Phase 1: Content Expansion
        print("\n=== Testing Phase 1: Content Expansion ===")
        
        expansion_result = self.content_expander.expand_content(
            content=self.base_newsletter_content,
            target_words=self.newsletter_template.total_word_target,
            template_type="technical_newsletter",
            metadata=self.test_metadata
        )
        
        # Verify expansion success
        self.assertTrue(expansion_result.success)
        self.assertGreater(len(expansion_result.expanded_content), len(self.base_newsletter_content))
        self.assertGreater(expansion_result.quality_score, 0.7)
        
        print(f"✓ Content expanded from {len(self.base_newsletter_content.split())} to {len(expansion_result.expanded_content.split())} words")
        print(f"✓ Expansion quality score: {expansion_result.quality_score:.2f}")
        
        # Phase 2: Mobile Optimization
        print("\n=== Testing Phase 2: Mobile Optimization ===")
        
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.ENHANCED,
            target_metrics={
                'readability_score': 0.85,
                'mobile_score': 0.9
            }
        )
        
        # Verify mobile optimization success
        self.assertTrue(mobile_result.success)
        self.assertGreater(mobile_result.mobile_readability_score, 0.8)
        self.assertGreater(mobile_result.mobile_compatibility_score, 0.8)
        
        print(f"✓ Mobile readability score: {mobile_result.mobile_readability_score:.2f}")
        print(f"✓ Mobile compatibility score: {mobile_result.mobile_compatibility_score:.2f}")
        
        # Phase 3: Quality Gate Validation
        print("\n=== Testing Phase 3: Quality Validation ===")
        
        validation_result = self.quality_gates.validate_with_level(
            content=mobile_result.optimized_content,
            level=QualityLevel.HIGH,
            context={
                'expansion_result': expansion_result,
                'mobile_result': mobile_result,
                'template': self.newsletter_template,
                'metadata': self.test_metadata
            }
        )
        
        # Verify quality gate success
        self.assertTrue(validation_result.passed)
        self.assertGreater(validation_result.overall_score, 0.8)
        
        print(f"✓ Quality validation passed: {validation_result.overall_score:.2f}")
        print(f"✓ Quality dimensions validated: {len(validation_result.dimension_scores)}")
        
        # Final verification
        final_content = mobile_result.optimized_content
        final_word_count = len(final_content.split())
        word_target_achievement = final_word_count / self.newsletter_template.total_word_target
        
        print(f"\n=== Final Results ===")
        print(f"✓ Final word count: {final_word_count}")
        print(f"✓ Target achievement: {word_target_achievement:.1%}")
        print(f"✓ Overall enhancement successful")
        
        # Assert final success criteria
        self.assertGreater(word_target_achievement, 0.8)  # At least 80% of target
        self.assertGreater(final_word_count, len(self.base_newsletter_content.split()) * 2)  # At least doubled
    
    def test_quality_preservation_throughout_workflow(self):
        """Test that quality is preserved throughout the enhancement workflow."""
        # Track quality at each stage
        quality_scores = {}
        
        # Stage 1: Original content baseline
        baseline_readability = self.readability_analyzer.analyze_mobile_readability(
            self.base_newsletter_content
        )
        quality_scores['baseline'] = baseline_readability.overall_mobile_score
        
        # Stage 2: After content expansion
        expansion_result = self.content_expander.expand_content(
            content=self.base_newsletter_content,
            target_words=1500,
            template_type="newsletter",
            metadata=self.test_metadata
        )
        
        expanded_readability = self.readability_analyzer.analyze_mobile_readability(
            expansion_result.expanded_content
        )
        quality_scores['after_expansion'] = expanded_readability.overall_mobile_score
        
        # Stage 3: After mobile optimization
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        final_readability = self.readability_analyzer.analyze_mobile_readability(
            mobile_result.optimized_content
        )
        quality_scores['final'] = final_readability.overall_mobile_score
        
        # Verify quality improvements or maintenance
        print(f"\nQuality progression:")
        print(f"Baseline: {quality_scores['baseline']:.3f}")
        print(f"After expansion: {quality_scores['after_expansion']:.3f}")
        print(f"Final: {quality_scores['final']:.3f}")
        
        # Final score should be better than or close to baseline
        self.assertGreaterEqual(quality_scores['final'], quality_scores['baseline'] - 0.1)
        
        # Mobile optimization should improve mobile readability
        self.assertGreater(quality_scores['final'], 0.7)
    
    def test_template_compliance_integration(self):
        """Test that enhanced content maintains template compliance."""
        # Create detailed template requirements
        template_requirements = {
            'required_sections': ['introduction', 'main_content', 'conclusion'],
            'min_word_count': 1500,
            'max_word_count': 2500,
            'readability_target': 0.8,
            'mobile_score_target': 0.85
        }
        
        # Run complete enhancement
        expansion_result = self.content_expander.expand_content(
            content=self.base_newsletter_content,
            target_words=template_requirements['min_word_count'],
            template_type="technical_newsletter",
            metadata={**self.test_metadata, 'template_requirements': template_requirements}
        )
        
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.ENHANCED,
            target_metrics={
                'readability_score': template_requirements['readability_target'],
                'mobile_score': template_requirements['mobile_score_target']
            }
        )
        
        final_content = mobile_result.optimized_content
        
        # Verify template compliance
        # Check required sections are present
        for section in template_requirements['required_sections']:
            if section == 'introduction':
                self.assertIn('## Introduction', final_content)
            elif section == 'main_content':
                self.assertIn('## Main Content', final_content)
            elif section == 'conclusion':
                self.assertIn('## Conclusion', final_content)
        
        # Check word count compliance
        final_word_count = len(final_content.split())
        self.assertGreaterEqual(final_word_count, template_requirements['min_word_count'])
        self.assertLessEqual(final_word_count, template_requirements['max_word_count'])
        
        # Check quality targets
        self.assertGreater(mobile_result.mobile_readability_score, template_requirements['readability_target'] - 0.05)
        self.assertGreater(expansion_result.template_compliance_score, 0.8)
    
    def test_performance_benchmarks(self):
        """Test that the complete workflow meets performance requirements."""
        # Test with larger content for performance stress testing
        large_content = self.base_newsletter_content * 3  # Triple the content
        
        start_time = time.time()
        
        # Run complete workflow
        expansion_result = self.content_expander.expand_content(
            content=large_content,
            target_words=3000,
            template_type="newsletter",
            metadata=self.test_metadata
        )
        
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.STANDARD  # Use standard for performance
        )
        
        total_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(total_time, 60.0)  # Should complete within 60 seconds
        self.assertTrue(expansion_result.success)
        self.assertTrue(mobile_result.success)
        
        print(f"✓ Complete workflow completed in {total_time:.2f} seconds")
        print(f"✓ Performance benchmark met")
    
    def test_error_recovery_and_resilience(self):
        """Test error recovery and system resilience."""
        # Test with problematic content
        problematic_content = """
# Incomplete Newsletter

## Introduction
This newsletter has issues.

## Main Content
Missing proper structure and
```python
# Unclosed code block
def broken_function():
    return "missing quote
"""
        
        # Content expansion should handle errors gracefully
        expansion_result = self.content_expander.expand_content(
            content=problematic_content,
            target_words=1000,
            template_type="newsletter",
            metadata=self.test_metadata
        )
        
        # Mobile optimization should also handle errors
        if expansion_result.success:
            mobile_result = self.mobile_optimizer.optimize_for_mobile(
                content=expansion_result.expanded_content,
                optimization_level=MobileOptimizationLevel.BASIC
            )
        else:
            # Try mobile optimization on original content
            mobile_result = self.mobile_optimizer.optimize_for_mobile(
                content=problematic_content,
                optimization_level=MobileOptimizationLevel.BASIC
            )
        
        # At least one process should succeed or provide fallback
        self.assertTrue(expansion_result.success or mobile_result.success)
        
        # Should provide warnings about issues
        if not expansion_result.success:
            self.assertGreater(len(expansion_result.warnings), 0)
        if not mobile_result.success:
            self.assertGreater(len(mobile_result.warnings), 0)
    
    def test_different_optimization_levels_integration(self):
        """Test integration with different optimization levels."""
        optimization_levels = [
            MobileOptimizationLevel.BASIC,
            MobileOptimizationLevel.STANDARD,
            MobileOptimizationLevel.ENHANCED,
            MobileOptimizationLevel.PREMIUM
        ]
        
        results = {}
        
        for level in optimization_levels:
            # Run expansion once
            expansion_result = self.content_expander.expand_content(
                content=self.base_newsletter_content,
                target_words=1200,
                template_type="newsletter",
                metadata=self.test_metadata
            )
            
            # Test different mobile optimization levels
            mobile_result = self.mobile_optimizer.optimize_for_mobile(
                content=expansion_result.expanded_content,
                optimization_level=level
            )
            
            results[level] = {
                'mobile_score': mobile_result.mobile_readability_score,
                'compatibility_score': mobile_result.mobile_compatibility_score,
                'processing_time': mobile_result.processing_time,
                'optimizations_count': len(mobile_result.optimizations_applied)
            }
        
        # Verify that higher levels generally produce better results
        basic_score = results[MobileOptimizationLevel.BASIC]['mobile_score']
        premium_score = results[MobileOptimizationLevel.PREMIUM]['mobile_score']
        
        # Premium should be at least as good as basic
        self.assertGreaterEqual(premium_score, basic_score - 0.05)  # Allow small variance
        
        # Premium should have more optimizations applied
        basic_optimizations = results[MobileOptimizationLevel.BASIC]['optimizations_count']
        premium_optimizations = results[MobileOptimizationLevel.PREMIUM]['optimizations_count']
        
        self.assertGreaterEqual(premium_optimizations, basic_optimizations)
    
    def test_real_world_newsletter_scenario(self):
        """Test with a realistic newsletter scenario."""
        # Real-world newsletter content
        real_newsletter = """
# AI Weekly: Breakthrough in Natural Language Processing

## Introduction
This week brings exciting developments in the world of artificial intelligence, particularly in natural language processing and machine learning applications.

## Featured Story: GPT-4 Applications
Recent implementations of GPT-4 have shown remarkable capabilities in various domains.

## Industry News
Several major tech companies announced new AI initiatives this week.

## Technical Deep Dive
Understanding transformer architectures and their impact on modern NLP.

## Tools and Resources
New AI development tools that can help developers build better applications.

## Conclusion
The rapid pace of AI development continues to create new opportunities and challenges.
"""
        
        # Real-world metadata
        real_metadata = {
            'newsletter_type': 'weekly_digest',
            'target_audience': 'ai_professionals',
            'complexity_level': 'intermediate',
            'publication_date': '2024-01-15',
            'author': 'AI Research Team'
        }
        
        # Real-world template requirements
        real_template = Mock()
        real_template.total_word_target = 2500
        real_template.mobile_optimization_required = True
        real_template.target_readability_score = 0.85
        
        print("\n=== Real-World Newsletter Enhancement Test ===")
        
        # Phase 1: Expansion
        expansion_result = self.content_expander.expand_content(
            content=real_newsletter,
            target_words=real_template.total_word_target,
            template_type="weekly_digest",
            metadata=real_metadata
        )
        
        self.assertTrue(expansion_result.success)
        expanded_words = len(expansion_result.expanded_content.split())
        original_words = len(real_newsletter.split())
        
        print(f"✓ Expanded from {original_words} to {expanded_words} words")
        
        # Phase 2: Mobile optimization
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.ENHANCED,
            target_metrics={'readability_score': 0.85, 'mobile_score': 0.9}
        )
        
        self.assertTrue(mobile_result.success)
        print(f"✓ Mobile readability: {mobile_result.mobile_readability_score:.2f}")
        
        # Phase 3: Final validation
        final_readability = self.readability_analyzer.analyze_mobile_readability(
            mobile_result.optimized_content
        )
        
        self.assertGreater(final_readability.overall_mobile_score, 0.7)
        print(f"✓ Final mobile score: {final_readability.overall_mobile_score:.2f}")
        
        # Verify content structure preservation
        final_content = mobile_result.optimized_content
        self.assertIn('# AI Weekly:', final_content)
        self.assertIn('## Introduction', final_content)
        self.assertIn('## Conclusion', final_content)
        
        # Verify practical word count achievement
        word_achievement = expanded_words / real_template.total_word_target
        self.assertGreater(word_achievement, 0.7)  # At least 70% of target
        
        print(f"✓ Target achievement: {word_achievement:.1%}")
        print("✓ Real-world scenario test completed successfully")


class TestQualityGateIntegration(unittest.TestCase):
    """Test quality gate integration with enhancement systems."""
    
    def setUp(self):
        """Set up quality gate integration tests."""
        self.quality_gates = ConfigurableQualityGate()
        self.content_expander = IntelligentContentExpander()
        self.mobile_optimizer = MobileContentOptimizer()
        
    def test_quality_gate_validation_workflow(self):
        """Test complete quality gate validation workflow."""
        sample_content = """
# Test Newsletter

## Introduction
AI technology is advancing rapidly.

## Main Content
Machine learning algorithms are becoming more sophisticated and capable.

## Conclusion
The future holds many possibilities for AI applications.
"""
        
        # Step 1: Content expansion
        expansion_result = self.content_expander.expand_content(
            content=sample_content,
            target_words=1000,
            template_type="newsletter",
            metadata={'quality_level': 'high'}
        )
        
        # Step 2: Mobile optimization  
        mobile_result = self.mobile_optimizer.optimize_for_mobile(
            content=expansion_result.expanded_content,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        # Step 3: Quality gate validation
        validation_context = {
            'expansion_result': expansion_result,
            'mobile_result': mobile_result,
            'original_content': sample_content
        }
        
        validation_result = self.quality_gates.validate_with_level(
            content=mobile_result.optimized_content,
            level=QualityLevel.HIGH,
            context=validation_context
        )
        
        # Verify quality gate integration
        self.assertTrue(validation_result.passed)
        self.assertGreater(validation_result.overall_score, 0.7)
        
        # Check that specific quality dimensions are evaluated
        expected_dimensions = [
            'content_expansion_quality',
            'mobile_readability', 
            'expansion_target_achievement'
        ]
        
        for dimension in expected_dimensions:
            self.assertIn(dimension, validation_result.dimension_scores)


if __name__ == '__main__':
    # Configure test runner for detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestComprehensiveNewsletterEnhancement))
    suite.addTest(unittest.makeSuite(TestQualityGateIntegration))
    
    print("=" * 80)
    print("COMPREHENSIVE NEWSLETTER ENHANCEMENT INTEGRATION TESTS")
    print("=" * 80)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"\n✗ {len(result.failures + result.errors)} TESTS FAILED")
    
    print("=" * 80)