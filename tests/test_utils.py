"""
Test Utilities and Common Test Infrastructure

Shared utilities, fixtures, and helper functions for the test suite.
"""

import os
import sys
from typing import Any, Dict, Optional
from unittest.mock import Mock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.content_expansion import ContentExpansionResult
from core.mobile_optimizer import MobileOptimizationResult
from core.readability_analyzer import (
    MobileFriendlinessMetrics,
    MobileReadabilityReport,
    ReadabilityMetrics,
)


class TestDataFactory:
    """Factory for creating test data and mock objects."""
    
    @staticmethod
    def create_sample_newsletter_content() -> str:
        """Create sample newsletter content for testing."""
        return """
# Weekly AI Newsletter

## Introduction
Artificial intelligence continues to evolve rapidly across multiple domains.

## Main Content
### Recent Developments
New breakthroughs in machine learning have emerged this week.

### Industry Applications
Companies are adopting AI at an unprecedented pace.

## Technical Deep Dive
The latest advances in neural network architectures show promising results.

## Conclusion
The future of AI remains bright and full of possibilities.
"""
    
    @staticmethod
    def create_complex_newsletter_content() -> str:
        """Create complex newsletter content for advanced testing."""
        return """
# AI Newsletter: Machine Learning Breakthroughs

## Introduction
Artificial intelligence continues to revolutionize industries across the globe. This comprehensive overview explores the latest developments in machine learning, natural language processing, and computer vision technologies that are shaping our digital future.

## Technical Deep Dive
### Deep Learning Advances
Recent breakthroughs in deep neural networks have achieved unprecedented accuracy in image recognition tasks. Researchers at leading universities have developed new architectures that combine the efficiency of convolutional networks with the power of transformer models.

Key developments include:
- Improved vision transformers with 95% accuracy on ImageNet
- New attention mechanisms that reduce computational complexity
- Novel training techniques for better generalization

```python
import torch
import torch.nn as nn

class ImprovedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        
    def forward(self, src, tgt):
        return self.transformer(src, tgt)
```

### Performance Optimization
Modern AI systems require careful optimization to achieve real-world performance. Techniques include model pruning, quantization, and knowledge distillation.

## Conclusion
The rapid advancement of AI technology continues to open new possibilities across industries.
"""
    
    @staticmethod
    def create_mock_expansion_result(
        success: bool = True,
        quality_score: float = 0.85,
        original_word_count: int = 150,
        expanded_word_count: int = 800,
        target_word_count: int = 1000
    ) -> Mock:
        """Create mock ContentExpansionResult."""
        mock_result = Mock(spec=ContentExpansionResult)
        mock_result.success = success
        mock_result.quality_score = quality_score
        mock_result.technical_accuracy_score = 0.90
        mock_result.content_coherence_score = 0.88
        mock_result.template_compliance_score = 0.92
        mock_result.original_word_count = original_word_count
        mock_result.expanded_word_count = expanded_word_count
        mock_result.target_word_count = target_word_count
        mock_result.expanded_content = TestDataFactory.create_complex_newsletter_content()
        mock_result.warnings = []
        mock_result.processing_time = 5.2
        mock_result.expansion_details = {
            'sections_expanded': 3,
            'strategies_used': ['technical_deep_dive', 'tutorial_enhancement']
        }
        return mock_result
    
    @staticmethod
    def create_mock_mobile_result(
        success: bool = True,
        mobile_readability_score: float = 0.87,
        mobile_compatibility_score: float = 0.89
    ) -> Mock:
        """Create mock MobileOptimizationResult."""
        mock_result = Mock(spec=MobileOptimizationResult)
        mock_result.success = success
        mock_result.mobile_readability_score = mobile_readability_score
        mock_result.mobile_compatibility_score = mobile_compatibility_score
        mock_result.typography_score = 0.85
        mock_result.navigation_score = 0.88
        mock_result.optimized_content = TestDataFactory.create_complex_newsletter_content()
        mock_result.optimizations_applied = [
            'paragraph_optimization',
            'heading_improvement', 
            'code_block_formatting'
        ]
        mock_result.processing_time = 3.1
        mock_result.warnings = []
        return mock_result
    
    @staticmethod
    def create_mock_readability_report(
        overall_mobile_score: float = 0.86
    ) -> Mock:
        """Create mock MobileReadabilityReport."""
        mock_report = Mock(spec=MobileReadabilityReport)
        mock_report.overall_mobile_score = overall_mobile_score
        
        # Mock readability metrics
        mock_readability = Mock(spec=ReadabilityMetrics)
        mock_readability.flesch_reading_ease = 75.0
        mock_readability.flesch_kincaid_grade = 8.5
        mock_readability.avg_sentence_length = 15.2
        mock_readability.complex_words_ratio = 0.12
        mock_report.readability_metrics = mock_readability
        
        # Mock mobile metrics
        mock_mobile = Mock(spec=MobileFriendlinessMetrics)
        mock_mobile.paragraph_length_score = 0.85
        mock_mobile.navigation_clarity_score = 0.88
        mock_mobile.touch_target_accessibility = 0.90
        mock_mobile.visual_hierarchy_score = 0.82
        mock_report.mobile_metrics = mock_mobile
        
        mock_report.recommendations = [
            "Improve paragraph structure for mobile reading",
            "Add more section headings for navigation"
        ]
        mock_report.improvement_priorities = [
            "Paragraph length optimization",
            "Navigation clarity improvement"
        ]
        
        return mock_report
    
    @staticmethod
    def create_test_metadata(
        newsletter_type: str = 'technical',
        target_audience: str = 'professionals',
        complexity_level: str = 'intermediate'
    ) -> Dict[str, Any]:
        """Create test metadata dictionary."""
        return {
            'newsletter_type': newsletter_type,
            'target_audience': target_audience,
            'complexity_level': complexity_level,
            'quality_level': 'high',
            'template_requirements': {
                'min_word_count': 1500,
                'target_readability': 0.8,
                'mobile_required': True
            }
        }


class TestAssertions:
    """Custom assertion helpers for newsletter enhancement tests."""
    
    @staticmethod
    def assert_expansion_result_valid(result, test_case):
        """Assert that expansion result is valid."""
        test_case.assertIsNotNone(result)
        test_case.assertTrue(hasattr(result, 'success'))
        test_case.assertTrue(hasattr(result, 'quality_score'))
        test_case.assertTrue(hasattr(result, 'expanded_content'))
        
        if result.success:
            test_case.assertGreater(result.quality_score, 0.0)
            test_case.assertLessEqual(result.quality_score, 1.0)
            test_case.assertIsNotNone(result.expanded_content)
    
    @staticmethod
    def assert_mobile_result_valid(result, test_case):
        """Assert that mobile optimization result is valid."""
        test_case.assertIsNotNone(result)
        test_case.assertTrue(hasattr(result, 'success'))
        test_case.assertTrue(hasattr(result, 'mobile_readability_score'))
        test_case.assertTrue(hasattr(result, 'mobile_compatibility_score'))
        
        if result.success:
            test_case.assertGreaterEqual(result.mobile_readability_score, 0.0)
            test_case.assertLessEqual(result.mobile_readability_score, 1.0)
            test_case.assertGreaterEqual(result.mobile_compatibility_score, 0.0)
            test_case.assertLessEqual(result.mobile_compatibility_score, 1.0)
    
    @staticmethod
    def assert_word_count_improvement(original_content: str, enhanced_content: str, test_case, min_ratio: float = 1.5):
        """Assert that word count has improved significantly."""
        original_count = len(original_content.split())
        enhanced_count = len(enhanced_content.split())
        
        test_case.assertGreater(enhanced_count, original_count)
        
        improvement_ratio = enhanced_count / original_count
        test_case.assertGreater(improvement_ratio, min_ratio, 
                               f"Expected at least {min_ratio}x improvement, got {improvement_ratio:.2f}x")
    
    @staticmethod
    def assert_content_structure_preserved(original_content: str, enhanced_content: str, test_case):
        """Assert that content structure is preserved after enhancement."""
        # Check that main headings are preserved
        original_headings = [line for line in original_content.split('\n') if line.startswith('#')]
        enhanced_headings = [line for line in enhanced_content.split('\n') if line.startswith('#')]
        
        for original_heading in original_headings:
            test_case.assertIn(original_heading, enhanced_content, 
                             f"Original heading '{original_heading}' not found in enhanced content")


class TestConstants:
    """Constants used across test suites."""
    
    # Performance thresholds
    MAX_PROCESSING_TIME_EXPANSION = 30.0  # seconds
    MAX_PROCESSING_TIME_MOBILE = 20.0     # seconds
    MAX_PROCESSING_TIME_COMPLETE = 60.0   # seconds
    
    # Quality thresholds
    MIN_QUALITY_SCORE = 0.7
    MIN_MOBILE_SCORE = 0.8
    MIN_EXPANSION_ACHIEVEMENT = 0.7  # 70% of target word count
    
    # Word count targets
    DEFAULT_TARGET_WORDS = 1500
    LARGE_TARGET_WORDS = 2500
    SMALL_TARGET_WORDS = 800
    
    # Test content categories
    CONTENT_TYPES = [
        'technical_newsletter',
        'weekly_digest', 
        'industry_analysis',
        'tutorial_guide'
    ]
    
    COMPLEXITY_LEVELS = ['basic', 'intermediate', 'advanced']
    TARGET_AUDIENCES = ['general', 'professionals', 'developers', 'researchers']


# Export main classes for easy importing
__all__ = [
    'TestDataFactory',
    'TestAssertions', 
    'TestConstants'
]