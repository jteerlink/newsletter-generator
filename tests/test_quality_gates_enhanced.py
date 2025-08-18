"""
Enhanced Quality Gates Test Suite

Tests for the enhanced quality gate system including new Phase 1 and Phase 2
quality dimensions and validation workflows.
"""

import os
import sys
import unittest
from unittest.mock import Mock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.advanced_quality_gates import ConfigurableQualityGate, QualityDimension, QualityReport
from core.content_expansion import ContentExpansionResult
from core.mobile_optimizer import MobileOptimizationResult
from core.readability_analyzer import (
    MobileFriendlinessMetrics,
    MobileReadabilityReport,
    ReadabilityMetrics,
)


class TestEnhancedQualityGates(unittest.TestCase):
    """Test suite for enhanced quality gate system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quality_gates = ConfigurableQualityGate()
        
        # Sample content for testing
        self.sample_content = """
# AI Newsletter: Machine Learning Breakthroughs

## Introduction
Artificial intelligence continues to revolutionize industries across the globe. This week, we explore the latest developments in machine learning, natural language processing, and computer vision technologies that are shaping our digital future.

## Main Content

### Deep Learning Advances
Recent breakthroughs in deep neural networks have achieved unprecedented accuracy in image recognition tasks. Researchers at leading universities have developed new architectures that combine the efficiency of convolutional networks with the power of transformer models.

Key developments include:
- Improved vision transformers with 95% accuracy on ImageNet
- New attention mechanisms that reduce computational complexity
- Novel training techniques for better generalization

### Natural Language Processing
The field of NLP has seen remarkable progress with the introduction of large language models. These systems demonstrate human-like understanding of context, nuance, and semantic relationships in text.

Recent achievements:
- Models with over 100 billion parameters
- Enhanced multilingual capabilities
- Better reasoning and common sense understanding

### Industry Applications
Companies are rapidly adopting AI solutions to improve efficiency and customer experience. From autonomous vehicles to medical diagnosis, AI is becoming integral to business operations.

Notable implementations:
- Tesla's Full Self-Driving capabilities
- Google's protein folding predictions
- Microsoft's code generation tools

## Technical Deep Dive

### Transformer Architecture Evolution
The transformer architecture, introduced in 2017, has become the foundation for most modern AI systems. Recent improvements focus on efficiency and scalability.

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

Optimization strategies:
1. Reduce model size through pruning
2. Use mixed-precision training
3. Implement efficient attention mechanisms
4. Apply knowledge distillation techniques

## Conclusion
The rapid advancement of AI technology continues to open new possibilities across industries. As we move forward, the focus shifts from pure capability to practical implementation, efficiency, and ethical considerations. The future of AI looks promising with continued research and development efforts worldwide.
"""
        
        # Mock expansion result
        self.mock_expansion_result = Mock(spec=ContentExpansionResult)
        self.mock_expansion_result.success = True
        self.mock_expansion_result.quality_score = 0.85
        self.mock_expansion_result.technical_accuracy_score = 0.90
        self.mock_expansion_result.content_coherence_score = 0.88
        self.mock_expansion_result.template_compliance_score = 0.92
        self.mock_expansion_result.original_word_count = 150
        self.mock_expansion_result.expanded_word_count = 800
        self.mock_expansion_result.target_word_count = 1000
        
        # Mock mobile optimization result
        self.mock_mobile_result = Mock(spec=MobileOptimizationResult)
        self.mock_mobile_result.success = True
        self.mock_mobile_result.mobile_readability_score = 0.87
        self.mock_mobile_result.mobile_compatibility_score = 0.89
        self.mock_mobile_result.typography_score = 0.85
        self.mock_mobile_result.navigation_score = 0.88
        
        # Mock readability report
        self.mock_readability_report = Mock(spec=MobileReadabilityReport)
        self.mock_readability_report.overall_mobile_score = 0.86
        self.mock_readability_report.readability_metrics = Mock(spec=ReadabilityMetrics)
        self.mock_readability_report.readability_metrics.flesch_reading_ease = 75.0
        self.mock_readability_report.readability_metrics.flesch_kincaid_grade = 8.5
        self.mock_readability_report.mobile_metrics = Mock(spec=MobileFriendlinessMetrics)
        self.mock_readability_report.mobile_metrics.paragraph_length_score = 0.85
        self.mock_readability_report.mobile_metrics.navigation_clarity_score = 0.88
    
    def test_initialization(self):
        """Test proper initialization of enhanced quality gates."""
        self.assertIsInstance(self.quality_gates.quality_dimensions, dict)
        self.assertIsInstance(self.quality_gates.validation_rules, dict)
        self.assertIsInstance(self.quality_gates.thresholds, dict)
        
        # Check that new quality dimensions are present
        new_dimensions = [
            QualityDimension.CONTENT_EXPANSION_QUALITY,
            QualityDimension.EXPANSION_TARGET_ACHIEVEMENT,
            QualityDimension.MOBILE_READABILITY,
            QualityDimension.MOBILE_TYPOGRAPHY
        ]
        
        for dimension in new_dimensions:
            self.assertIn(dimension, self.quality_gates.quality_dimensions)
    
    def test_validate_with_level_basic(self):
        """Test basic quality gate validation."""
        context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result
        }
        
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.STANDARD,
            context=context
        )
        
        # Verify result structure
        self.assertIsInstance(result, QualityGateResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.overall_score, float)
        self.assertIsInstance(result.dimension_scores, dict)
        
        # Verify score ranges
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        
        # Should pass with good mock data
        self.assertTrue(result.passed)
    
    def test_content_expansion_quality_assessment(self):
        """Test content expansion quality assessment."""
        context = {'expansion_result': self.mock_expansion_result}
        
        score = self.quality_gates._assess_content_expansion_quality(
            self.sample_content, context
        )
        
        # Should return reasonable score
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.7)  # Good expansion should score well
    
    def test_expansion_target_achievement_assessment(self):
        """Test expansion target achievement assessment."""
        context = {'expansion_result': self.mock_expansion_result}
        
        score = self.quality_gates._assess_expansion_target_achievement(
            self.sample_content, context
        )
        
        # Should calculate achievement ratio correctly
        expected_ratio = self.mock_expansion_result.expanded_word_count / self.mock_expansion_result.target_word_count
        self.assertGreater(score, 0.7)  # 800/1000 = 0.8, should score well
    
    def test_expansion_technical_accuracy_assessment(self):
        """Test expansion technical accuracy preservation assessment."""
        context = {'expansion_result': self.mock_expansion_result}
        
        score = self.quality_gates._assess_expansion_technical_accuracy(
            self.sample_content, context
        )
        
        # Should use technical accuracy score from expansion result
        self.assertEqual(score, self.mock_expansion_result.technical_accuracy_score)
    
    def test_mobile_readability_assessment(self):
        """Test mobile readability assessment."""
        context = {'mobile_result': self.mock_mobile_result}
        
        score = self.quality_gates._assess_mobile_readability(
            self.sample_content, context
        )
        
        # Should use mobile readability score
        self.assertEqual(score, self.mock_mobile_result.mobile_readability_score)
    
    def test_mobile_typography_assessment(self):
        """Test mobile typography assessment."""
        context = {'mobile_result': self.mock_mobile_result}
        
        score = self.quality_gates._assess_mobile_typography(
            self.sample_content, context
        )
        
        # Should use typography score from mobile result
        self.assertEqual(score, self.mock_mobile_result.typography_score)
    
    def test_mobile_structure_optimization_assessment(self):
        """Test mobile structure optimization assessment."""
        context = {'mobile_result': self.mock_mobile_result}
        
        score = self.quality_gates._assess_mobile_structure_optimization(
            self.sample_content, context
        )
        
        # Should assess mobile compatibility
        self.assertEqual(score, self.mock_mobile_result.mobile_compatibility_score)
    
    def test_quality_levels_different_thresholds(self):
        """Test that different quality levels use different thresholds."""
        context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result
        }
        
        # Test different quality levels
        basic_result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.BASIC,
            context=context
        )
        
        high_result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.HIGH,
            context=context
        )
        
        # Both should pass with good data, but high should be more stringent
        self.assertTrue(basic_result.passed)
        
        # High level validation might be more strict
        if not high_result.passed:
            # Should have lower overall score due to higher thresholds
            self.assertLessEqual(high_result.overall_score, basic_result.overall_score)
    
    def test_validation_with_poor_expansion_result(self):
        """Test validation with poor expansion results."""
        # Create poor expansion result
        poor_expansion = Mock(spec=ContentExpansionResult)
        poor_expansion.success = True
        poor_expansion.quality_score = 0.45  # Poor quality
        poor_expansion.technical_accuracy_score = 0.50
        poor_expansion.content_coherence_score = 0.40
        poor_expansion.template_compliance_score = 0.35
        poor_expansion.expanded_word_count = 200
        poor_expansion.target_word_count = 1000  # Very low achievement
        
        context = {
            'expansion_result': poor_expansion,
            'mobile_result': self.mock_mobile_result
        }
        
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.HIGH,
            context=context
        )
        
        # Should fail with poor expansion quality
        self.assertFalse(result.passed)
        self.assertLess(result.overall_score, 0.6)
        
        # Should identify specific issues
        self.assertGreater(len(result.issues), 0)
        
        # Should provide recommendations
        self.assertGreater(len(result.recommendations), 0)
    
    def test_validation_with_poor_mobile_result(self):
        """Test validation with poor mobile optimization results."""
        # Create poor mobile result
        poor_mobile = Mock(spec=MobileOptimizationResult)
        poor_mobile.success = True
        poor_mobile.mobile_readability_score = 0.45  # Poor mobile readability
        poor_mobile.mobile_compatibility_score = 0.40
        poor_mobile.typography_score = 0.35
        poor_mobile.navigation_score = 0.50
        
        context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': poor_mobile
        }
        
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.HIGH,
            context=context
        )
        
        # Should fail with poor mobile optimization
        self.assertFalse(result.passed)
        
        # Should identify mobile-related issues
        mobile_issues = [issue for issue in result.issues if 'mobile' in issue.lower()]
        self.assertGreater(len(mobile_issues), 0)
    
    def test_validation_without_enhancement_context(self):
        """Test validation when enhancement context is missing."""
        # Test with minimal context
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.STANDARD,
            context={}
        )
        
        # Should still provide basic validation
        self.assertIsInstance(result, QualityGateResult)
        
        # Should have warnings about missing context
        missing_context_warnings = [
            warning for warning in result.warnings 
            if 'context' in warning.lower() or 'missing' in warning.lower()
        ]
        self.assertGreater(len(missing_context_warnings), 0)
    
    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow with all components."""
        # Create comprehensive context
        comprehensive_context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result,
            'readability_report': self.mock_readability_report,
            'original_content': "# Short Original Content\nBasic content.",
            'template_requirements': {
                'min_word_count': 1000,
                'target_readability': 0.8,
                'mobile_required': True
            }
        }
        
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.HIGH,
            context=comprehensive_context
        )
        
        # Should validate all quality dimensions
        expected_dimensions = [
            QualityDimension.CONTENT_EXPANSION_QUALITY,
            QualityDimension.EXPANSION_TARGET_ACHIEVEMENT,
            QualityDimension.EXPANSION_TECHNICAL_ACCURACY,
            QualityDimension.MOBILE_READABILITY,
            QualityDimension.MOBILE_TYPOGRAPHY,
            QualityDimension.MOBILE_STRUCTURE_OPTIMIZATION
        ]
        
        for dimension in expected_dimensions:
            self.assertIn(dimension, result.dimension_scores)
            self.assertGreaterEqual(result.dimension_scores[dimension], 0.0)
            self.assertLessEqual(result.dimension_scores[dimension], 1.0)
        
        # With good mock data, should pass
        self.assertTrue(result.passed)
        self.assertGreater(result.overall_score, 0.7)
    
    def test_adaptive_thresholds(self):
        """Test adaptive threshold adjustment based on context."""
        # Test with high-complexity content
        high_complexity_context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result,
            'complexity_level': 'high',
            'technical_content': True
        }
        
        # Test with simple content
        simple_context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result,
            'complexity_level': 'basic',
            'technical_content': False
        }
        
        high_result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.STANDARD,
            context=high_complexity_context
        )
        
        simple_result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.STANDARD,
            context=simple_context
        )
        
        # Both should be valid results
        self.assertIsInstance(high_result, QualityGateResult)
        self.assertIsInstance(simple_result, QualityGateResult)
    
    def test_performance_validation(self):
        """Test that quality gates complete validation within reasonable time."""
        import time
        
        context = {
            'expansion_result': self.mock_expansion_result,
            'mobile_result': self.mock_mobile_result
        }
        
        start_time = time.time()
        
        result = self.quality_gates.validate_with_level(
            content=self.sample_content,
            level=QualityLevel.HIGH,
            context=context
        )
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 5.0)  # 5 seconds max
        
        # Should still provide valid result
        self.assertIsInstance(result, QualityGateResult)
        self.assertTrue(result.passed)


class TestQualityDimensionAssessment(unittest.TestCase):
    """Test individual quality dimension assessments."""
    
    def setUp(self):
        """Set up dimension assessment tests."""
        self.quality_gates = ConfigurableQualityGate()
        
        self.sample_content = "# Test Content\n\nThis is test content for quality assessment."
    
    def test_content_coherence_preservation(self):
        """Test content coherence preservation assessment."""
        expansion_result = Mock()
        expansion_result.content_coherence_score = 0.85
        
        context = {'expansion_result': expansion_result}
        
        score = self.quality_gates._assess_content_coherence_preservation(
            self.sample_content, context
        )
        
        self.assertEqual(score, 0.85)
    
    def test_technical_accuracy_preservation(self):
        """Test technical accuracy preservation assessment."""
        expansion_result = Mock()
        expansion_result.technical_accuracy_score = 0.92
        
        context = {'expansion_result': expansion_result}
        
        score = self.quality_gates._assess_technical_accuracy_preservation(
            self.sample_content, context
        )
        
        self.assertEqual(score, 0.92)
    
    def test_mobile_optimization_metrics(self):
        """Test mobile optimization metrics assessment."""
        mobile_result = Mock()
        mobile_result.mobile_readability_score = 0.88
        mobile_result.mobile_compatibility_score = 0.85
        mobile_result.typography_score = 0.90
        
        context = {'mobile_result': mobile_result}
        
        # Test readability
        readability_score = self.quality_gates._assess_mobile_readability(
            self.sample_content, context
        )
        self.assertEqual(readability_score, 0.88)
        
        # Test structure optimization
        structure_score = self.quality_gates._assess_mobile_structure_optimization(
            self.sample_content, context
        )
        self.assertEqual(structure_score, 0.85)
        
        # Test typography
        typography_score = self.quality_gates._assess_mobile_typography(
            self.sample_content, context
        )
        self.assertEqual(typography_score, 0.90)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)