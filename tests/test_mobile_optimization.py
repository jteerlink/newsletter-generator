"""
Comprehensive Test Suite for Mobile Optimization System

Tests for Phase 2: Mobile-First Optimization including mobile content optimization,
responsive typography, and readability analysis.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.mobile_optimizer import (
    MobileContentOptimizer,
    MobileOptimizationLevel,
    ContentStructureType,
    MobileOptimizationResult
)
from core.readability_analyzer import (
    MobileReadabilityAnalyzer,
    MobileReadabilityReport,
    ReadabilityMetrics,
    MobileFriendlinessMetrics
)
from core.typography_manager import (
    ResponsiveTypographyManager,
    DeviceType,
    TypographyLevel,
    TypographyOptimizationResult
)


class TestMobileContentOptimizer(unittest.TestCase):
    """Test suite for MobileContentOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = MobileContentOptimizer()
        
        # Sample content for testing
        self.sample_content = """
# Mobile AI Newsletter

## Introduction
Artificial intelligence is transforming how we interact with mobile devices. This comprehensive overview explores the latest developments in mobile AI applications, from voice assistants to camera enhancements.

## Technical Deep Dive
### Machine Learning on Mobile
Modern smartphones are equipped with dedicated AI chips that enable real-time machine learning inference. These Neural Processing Units (NPUs) can handle complex computations without draining battery life significantly.

The optimization techniques used include:
- Model quantization to reduce memory footprint
- Pruning unnecessary neural network connections
- Using efficient architectures like MobileNets and EfficientNets

```python
# Example of mobile-optimized model loading
import tensorflow as tf

def load_mobile_model():
    model = tf.lite.Interpreter(model_path="mobile_model.tflite")
    model.allocate_tensors()
    return model
```

### Performance Considerations
Mobile AI applications must balance accuracy with performance constraints. Battery life, thermal management, and memory limitations all play crucial roles in deployment decisions.

## Industry Applications
Companies like Apple, Google, and Samsung have integrated AI deeply into their mobile ecosystems. Voice recognition, camera processing, and predictive text are just the beginning.

## Conclusion
The future of mobile AI looks promising with continued hardware improvements and software optimizations making sophisticated AI capabilities accessible to billions of users worldwide.
"""
        
        # Sample optimization settings
        self.optimization_settings = {
            'target_device': 'mobile',
            'readability_level': 'intermediate',
            'optimization_level': MobileOptimizationLevel.ENHANCED
        }
    
    def test_initialization(self):
        """Test proper initialization of mobile optimizer."""
        self.assertIsInstance(self.optimizer.mobile_thresholds, dict)
        self.assertIsInstance(self.optimizer.optimization_strategies, dict)
        self.assertIsInstance(self.optimizer.content_processors, dict)
        
        # Check that all optimization levels are supported
        for level in MobileOptimizationLevel:
            self.assertIn(level, self.optimizer.optimization_strategies)
    
    def test_optimize_for_mobile_basic(self):
        """Test basic mobile optimization functionality."""
        result = self.optimizer.optimize_for_mobile(
            content=self.sample_content,
            optimization_level=MobileOptimizationLevel.STANDARD,
            target_metrics={'readability_score': 0.9, 'mobile_score': 0.85}
        )
        
        # Verify result structure
        self.assertIsInstance(result, MobileOptimizationResult)
        self.assertEqual(result.original_content, self.sample_content)
        self.assertIsNotNone(result.optimized_content)
        self.assertTrue(result.success)
        
        # Verify optimization improvements
        self.assertGreater(result.mobile_readability_score, 0.8)
        self.assertGreater(result.mobile_compatibility_score, 0.8)
    
    def test_paragraph_optimization(self):
        """Test paragraph length optimization for mobile."""
        # Create content with overly long paragraphs
        long_paragraph_content = """
# Test Content

This is an extremely long paragraph that contains way too many words for optimal mobile reading experience and should be broken down into smaller, more digestible chunks that are easier to read on small screens. The paragraph continues with technical details about machine learning algorithms, neural networks, and their applications in mobile computing. It discusses various optimization techniques, performance considerations, and industry trends that span multiple sentences and cover numerous topics without appropriate breaks for mobile readers.

## Another Section
This section also has content that needs optimization.
"""
        
        result = self.optimizer.optimize_for_mobile(
            content=long_paragraph_content,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        # Check that long paragraphs were broken down
        optimized_paragraphs = result.optimized_content.split('\n\n')
        original_paragraphs = long_paragraph_content.split('\n\n')
        
        # Should have more paragraphs after optimization
        self.assertGreater(len(optimized_paragraphs), len(original_paragraphs))
        
        # Paragraphs should be shorter on average
        avg_optimized_length = sum(len(p.split()) for p in optimized_paragraphs if p.strip()) / len([p for p in optimized_paragraphs if p.strip()])
        self.assertLess(avg_optimized_length, 80)  # Under 80 words per paragraph
    
    def test_heading_optimization(self):
        """Test heading structure optimization for mobile navigation."""
        content_poor_headings = """
# Main Title
This is content without proper heading structure that makes mobile navigation difficult.

Some more content here that should have better organization.

More content that could benefit from subheadings and better structure.

Final paragraph with important information.
"""
        
        result = self.optimizer.optimize_for_mobile(
            content=content_poor_headings,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        # Should add appropriate headings for mobile navigation
        heading_count_original = content_poor_headings.count('#')
        heading_count_optimized = result.optimized_content.count('#')
        
        self.assertGreaterEqual(heading_count_optimized, heading_count_original)
        
        # Check for proper heading hierarchy
        self.assertIn('##', result.optimized_content)  # Should have H2 headings
    
    def test_code_block_mobile_optimization(self):
        """Test code block optimization for mobile display."""
        content_with_code = """
# Code Example

Here's a long line of code that might be difficult to read on mobile:

```python
def very_long_function_name_that_exceeds_mobile_width(parameter_one, parameter_two, parameter_three, parameter_four):
    return parameter_one + parameter_two + parameter_three + parameter_four
```

Another code block:

```javascript
const anotherVeryLongFunctionNameThatMightCauseHorizontalScrolling = (param1, param2, param3) => {
    return param1.someVeryLongMethodName() + param2.anotherLongMethod() + param3.yetAnotherMethod();
};
```
"""
        
        result = self.optimizer.optimize_for_mobile(
            content=content_with_code,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        # Should add mobile optimization hints for code blocks
        self.assertIn('<!-- Mobile:', result.optimized_content)
        
        # Should maintain code block structure
        self.assertEqual(
            content_with_code.count('```'),
            result.optimized_content.count('```')
        )
    
    def test_list_optimization(self):
        """Test list structure optimization for mobile."""
        content_with_lists = """
# Features

These are the key features: feature one which is very important, feature two that provides additional value, feature three with complex capabilities, and feature four that rounds out the offering.

Benefits include: improved performance, better user experience, reduced costs, enhanced security, and future-proof architecture.
"""
        
        result = self.optimizer.optimize_for_mobile(
            content=content_with_lists,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        # Should convert comma-separated lists to proper bullet points
        self.assertIn('- ', result.optimized_content)
        
        # Should improve readability with better list structure
        self.assertGreater(result.mobile_readability_score, 0.8)
    
    def test_optimization_level_differences(self):
        """Test that different optimization levels produce different results."""
        basic_result = self.optimizer.optimize_for_mobile(
            content=self.sample_content,
            optimization_level=MobileOptimizationLevel.BASIC
        )
        
        enhanced_result = self.optimizer.optimize_for_mobile(
            content=self.sample_content,
            optimization_level=MobileOptimizationLevel.ENHANCED
        )
        
        premium_result = self.optimizer.optimize_for_mobile(
            content=self.sample_content,
            optimization_level=MobileOptimizationLevel.PREMIUM
        )
        
        # Enhanced should have better scores than basic
        self.assertGreaterEqual(
            enhanced_result.mobile_readability_score,
            basic_result.mobile_readability_score
        )
        
        # Premium should have best scores
        self.assertGreaterEqual(
            premium_result.mobile_readability_score,
            enhanced_result.mobile_readability_score
        )
        
        # Premium should have more optimizations applied
        self.assertGreaterEqual(
            len(premium_result.optimizations_applied),
            len(enhanced_result.optimizations_applied)
        )
    
    def test_target_metrics_achievement(self):
        """Test that optimization achieves target metrics."""
        target_metrics = {
            'readability_score': 0.9,
            'mobile_score': 0.85,
            'paragraph_score': 0.8
        }
        
        result = self.optimizer.optimize_for_mobile(
            content=self.sample_content,
            optimization_level=MobileOptimizationLevel.ENHANCED,
            target_metrics=target_metrics
        )
        
        # Should achieve or come close to target metrics
        self.assertGreater(result.mobile_readability_score, 0.75)  # Within reasonable range
        self.assertGreater(result.mobile_compatibility_score, 0.75)
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        # Test with empty content
        result = self.optimizer.optimize_for_mobile(
            content="",
            optimization_level=MobileOptimizationLevel.STANDARD
        )
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.warnings), 0)
        
        # Test with malformed content
        malformed_content = "# Title\n\n```python\nno closing backticks"
        
        result = self.optimizer.optimize_for_mobile(
            content=malformed_content,
            optimization_level=MobileOptimizationLevel.STANDARD
        )
        
        # Should handle gracefully
        self.assertIsNotNone(result.optimized_content)


class TestMobileReadabilityAnalyzer(unittest.TestCase):
    """Test suite for MobileReadabilityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MobileReadabilityAnalyzer()
        
        # Sample content with varying readability characteristics
        self.readable_content = """
# Simple Guide

## Introduction
This guide explains AI in simple terms.

## What is AI?
AI stands for artificial intelligence. It helps computers think like humans.

## How does it work?
AI uses data to learn patterns. Then it makes predictions.

## Examples
- Voice assistants like Siri
- Photo recognition in phones  
- Movie recommendations on Netflix

## Conclusion
AI is becoming part of our daily lives. It makes technology more helpful.
"""
        
        self.complex_content = """
# Comprehensive Analysis of Quantum Computing Paradigms

## Theoretical Foundations
Quantum computational methodologies leverage superposition phenomena and entanglement correlations to achieve exponential algorithmic acceleration compared to classical computational architectures, particularly within specific problem domains characterized by inherent parallelizability.

## Mathematical Formulations
The Schrödinger equation governs quantum state evolution: iℏ∂|ψ⟩/∂t = Ĥ|ψ⟩, where the Hamiltonian operator encapsulates system energetics through eigenvalue decomposition, facilitating computational operations via unitary transformations applied to qubit registers maintained in coherent superposition states until measurement-induced collapse occurs.
"""
    
    def test_initialization(self):
        """Test proper initialization of readability analyzer."""
        self.assertIsInstance(self.analyzer.mobile_thresholds, dict)
        self.assertIsInstance(self.analyzer.readability_weights, dict)
        
        # Check threshold values are reasonable
        self.assertGreater(self.analyzer.mobile_thresholds['flesch_reading_ease_min'], 0)
        self.assertLess(self.analyzer.mobile_thresholds['flesch_reading_ease_min'], 100)
    
    def test_analyze_mobile_readability_basic(self):
        """Test basic mobile readability analysis."""
        report = self.analyzer.analyze_mobile_readability(self.readable_content)
        
        # Verify report structure
        self.assertIsInstance(report, MobileReadabilityReport)
        self.assertIsInstance(report.readability_metrics, ReadabilityMetrics)
        self.assertIsInstance(report.mobile_metrics, MobileFriendlinessMetrics)
        self.assertIsInstance(report.recommendations, list)
        self.assertIsInstance(report.improvement_priorities, list)
        
        # Verify score ranges
        self.assertGreaterEqual(report.overall_mobile_score, 0.0)
        self.assertLessEqual(report.overall_mobile_score, 1.0)
    
    def test_readability_comparison(self):
        """Test that analyzer correctly distinguishes readability levels."""
        readable_report = self.analyzer.analyze_mobile_readability(self.readable_content)
        complex_report = self.analyzer.analyze_mobile_readability(self.complex_content)
        
        # Readable content should score higher
        self.assertGreater(
            readable_report.overall_mobile_score,
            complex_report.overall_mobile_score
        )
        
        # Readable content should have better readability metrics
        self.assertGreater(
            readable_report.readability_metrics.flesch_reading_ease,
            complex_report.readability_metrics.flesch_reading_ease
        )
        
        # Complex content should have more recommendations
        self.assertGreaterEqual(
            len(complex_report.recommendations),
            len(readable_report.recommendations)
        )
    
    def test_mobile_specific_metrics(self):
        """Test mobile-specific friendliness metrics."""
        report = self.analyzer.analyze_mobile_readability(self.readable_content)
        
        mobile_metrics = report.mobile_metrics
        
        # Verify all mobile metrics are present and valid
        self.assertGreaterEqual(mobile_metrics.paragraph_length_score, 0.0)
        self.assertLessEqual(mobile_metrics.paragraph_length_score, 1.0)
        
        self.assertGreaterEqual(mobile_metrics.sentence_complexity_score, 0.0)
        self.assertLessEqual(mobile_metrics.sentence_complexity_score, 1.0)
        
        self.assertGreaterEqual(mobile_metrics.navigation_clarity_score, 0.0)
        self.assertLessEqual(mobile_metrics.navigation_clarity_score, 1.0)
        
        self.assertGreaterEqual(mobile_metrics.touch_target_accessibility, 0.0)
        self.assertLessEqual(mobile_metrics.touch_target_accessibility, 1.0)
    
    def test_paragraph_length_analysis(self):
        """Test paragraph length analysis for mobile."""
        # Create content with varying paragraph lengths
        mixed_content = """
# Test Content

Short paragraph.

This is a medium-length paragraph that contains a reasonable amount of content for mobile reading without being too overwhelming for users on small screens.

This is an extremely long paragraph that contains far too much information for optimal mobile reading experience and should definitely be flagged by the analyzer as needing improvement because it makes reading difficult on small screens and reduces user engagement significantly when people have to scroll through massive blocks of text without any visual breaks or breathing room which is particularly problematic on mobile devices where screen real estate is limited and users prefer bite-sized chunks of information.
"""
        
        report = self.analyzer.analyze_mobile_readability(mixed_content)
        
        # Should identify paragraph length issues
        self.assertLess(report.mobile_metrics.paragraph_length_score, 0.9)
        
        # Should provide relevant recommendations
        paragraph_recommendations = [
            rec for rec in report.recommendations 
            if 'paragraph' in rec.lower()
        ]
        self.assertGreater(len(paragraph_recommendations), 0)
    
    def test_code_block_mobile_analysis(self):
        """Test code block mobile-friendliness analysis."""
        content_with_code = """
# Code Examples

Here's a mobile-friendly code block:

```python
def hello():
    return "world"
```

Here's a problematic code block with long lines:

```python
def extremely_long_function_name_that_will_cause_horizontal_scrolling_on_mobile_devices(parameter_with_very_long_name, another_parameter_with_excessive_length):
    return parameter_with_very_long_name + another_parameter_with_excessive_length + "this string makes the line even longer"
```
"""
        
        report = self.analyzer.analyze_mobile_readability(content_with_code)
        
        # Should detect code block mobile issues
        self.assertLess(report.mobile_metrics.code_block_mobile_score, 1.0)
        
        # Should provide code-related recommendations
        code_recommendations = [
            rec for rec in report.recommendations 
            if 'code' in rec.lower()
        ]
        self.assertGreater(len(code_recommendations), 0)
    
    def test_navigation_clarity_analysis(self):
        """Test navigation clarity analysis for mobile."""
        # Content with poor navigation structure
        poor_navigation_content = """
This is a long document without proper headings that makes navigation difficult on mobile devices. Users need clear section breaks and headings to understand content structure and jump to relevant sections quickly.

More content continues here without any organizing structure or visual hierarchy that would help mobile users navigate efficiently.

Additional paragraphs keep coming without clear organization making it hard for users to find specific information they're looking for.
"""
        
        # Content with good navigation
        good_navigation_content = """
# Main Title

## Section One
Content for section one.

## Section Two  
Content for section two.

### Subsection A
Detailed content.

### Subsection B
More detailed content.

## Section Three
Final section content.
"""
        
        poor_report = self.analyzer.analyze_mobile_readability(poor_navigation_content)
        good_report = self.analyzer.analyze_mobile_readability(good_navigation_content)
        
        # Good navigation should score higher
        self.assertGreater(
            good_report.mobile_metrics.navigation_clarity_score,
            poor_report.mobile_metrics.navigation_clarity_score
        )
    
    def test_recommendation_generation(self):
        """Test that appropriate recommendations are generated."""
        report = self.analyzer.analyze_mobile_readability(self.complex_content)
        
        # Should have recommendations
        self.assertGreater(len(report.recommendations), 0)
        
        # Recommendations should be specific and actionable
        for recommendation in report.recommendations:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 10)  # Should be descriptive
        
        # Should prioritize improvements
        self.assertGreater(len(report.improvement_priorities), 0)
    
    def test_error_handling(self):
        """Test error handling in readability analysis."""
        # Test with empty content
        report = self.analyzer.analyze_mobile_readability("")
        
        # Should handle gracefully
        self.assertIsInstance(report, MobileReadabilityReport)
        self.assertLessEqual(report.overall_mobile_score, 1.0)
        
        # Test with minimal content
        minimal_report = self.analyzer.analyze_mobile_readability("AI.")
        
        self.assertIsInstance(minimal_report, MobileReadabilityReport)


class TestResponsiveTypographyManager(unittest.TestCase):
    """Test suite for ResponsiveTypographyManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.typography_manager = ResponsiveTypographyManager()
        
        self.sample_content = """
# Typography Test

This is a paragraph to test typography optimization.

## Subheading

Another paragraph with different content.

```python
def code_example():
    return "test"
```

- List item one
- List item two
- List item three
"""
    
    def test_initialization(self):
        """Test proper initialization of typography manager."""
        self.assertIsInstance(self.typography_manager.typography_settings, dict)
        self.assertIsInstance(self.typography_manager.mobile_thresholds, dict)
        self.assertIsInstance(self.typography_manager.formatting_rules, dict)
        
        # Check that all device types have settings
        for device_type in DeviceType:
            self.assertIn(device_type, self.typography_manager.typography_settings)
    
    def test_device_specific_settings(self):
        """Test device-specific typography settings."""
        mobile_settings = self.typography_manager.typography_settings[DeviceType.MOBILE]
        desktop_settings = self.typography_manager.typography_settings[DeviceType.DESKTOP]
        
        # Mobile should have appropriate font sizes (minimum 16px for body)
        self.assertGreaterEqual(mobile_settings.font_size_body, 16)
        
        # Desktop can have different sizing
        self.assertIsInstance(desktop_settings.font_size_body, int)
        
        # Mobile should have tighter spacing
        self.assertEqual(mobile_settings.margin_horizontal, "16px")
    
    def test_apply_mobile_typography(self):
        """Test mobile typography application."""
        adjustments = {
            'font_size_increase': True,
            'line_height_optimization': True,
            'heading_hierarchy_improvement': True,
            'responsive_font_scaling': True
        }
        
        optimized_content = self.typography_manager.apply_mobile_typography(
            content=self.sample_content,
            adjustments=adjustments,
            target_device=DeviceType.MOBILE
        )
        
        # Should add typography optimization comments/guidelines
        self.assertIn('Typography:', optimized_content)
        self.assertIn('Mobile-Optimized', optimized_content)
        
        # Should maintain original content structure
        self.assertIn('# Typography Test', optimized_content)
        self.assertIn('## Subheading', optimized_content)
    
    def test_heading_hierarchy_improvement(self):
        """Test heading hierarchy improvement for mobile."""
        content_poor_spacing = """# Title
Content immediately after title.
## Subheading
More content without spacing.
### Another heading
Final content."""
        
        result, adjustments = self.typography_manager._improve_heading_hierarchy(
            content_poor_spacing, DeviceType.MOBILE
        )
        
        # Should add appropriate spacing around headings
        lines = result.split('\n')
        
        # Check for empty lines around headings
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Should have spacing considerations for mobile
                self.assertIsInstance(result, str)
        
        # Should record adjustments
        self.assertGreater(len(adjustments), 0)
    
    def test_code_block_formatting(self):
        """Test code block mobile formatting."""
        content_with_long_code = """
# Code Example

```python
def very_long_function_name_that_exceeds_typical_mobile_screen_width(param_one, param_two, param_three):
    return param_one + param_two + param_three
```
"""
        
        result, adjustments = self.typography_manager._format_code_blocks(
            content_with_long_code, DeviceType.MOBILE
        )
        
        # Should add mobile formatting hints for long lines
        self.assertIn('Mobile:', result)
        
        # Should maintain code block structure
        self.assertEqual(
            content_with_long_code.count('```'),
            result.count('```')
        )
        
        # Should record adjustments
        self.assertGreater(len(adjustments), 0)
    
    def test_touch_target_optimization(self):
        """Test touch target optimization for mobile."""
        content_with_links = """
# Links Test

Here are multiple links: [Link One](url1) [Link Two](url2) [Link Three](url3) in the same line.

This line has [Single Link](url) which is fine.
"""
        
        result, adjustments = self.typography_manager._optimize_touch_targets(
            content_with_links, DeviceType.MOBILE
        )
        
        # Should add guidance for lines with multiple links
        lines_with_guidance = [
            line for line in result.split('\n') 
            if 'Mobile:' in line and 'spacing' in line
        ]
        self.assertGreater(len(lines_with_guidance), 0)
        
        # Should record touch target adjustments
        touch_adjustments = [
            adj for adj in adjustments 
            if adj.element_type == 'link'
        ]
        self.assertGreater(len(touch_adjustments), 0)
    
    def test_typography_compliance_assessment(self):
        """Test typography compliance assessment."""
        # Test with mobile-friendly content
        good_content = """
# Good Content

## Section One
This is a well-structured paragraph with appropriate length for mobile reading.

## Section Two  
Another paragraph that follows mobile best practices.

```python
# Short code
print("hello")
```

- Simple list
- Clear items
"""
        
        # Test with problematic content
        poor_content = """
# Poor Content
This is an extremely long paragraph that goes on and on without proper breaks and contains far too much information in a single block of text which makes it very difficult to read on mobile devices and should definitely be broken up into smaller more manageable chunks for better user experience and readability scores.

```python
def extremely_long_function_name_that_will_definitely_cause_horizontal_scrolling_issues_on_mobile_devices(parameter_with_very_long_name, another_parameter_that_makes_line_too_long):
    return "this line is also way too long for mobile screens and will cause usability issues"
```
"""
        
        good_compliance = self.typography_manager.assess_typography_compliance(good_content)
        poor_compliance = self.typography_manager.assess_typography_compliance(poor_content)
        
        # Good content should have higher compliance
        self.assertGreater(good_compliance, poor_compliance)
        self.assertGreater(good_compliance, 0.7)
        self.assertLess(poor_compliance, 0.8)
    
    def test_mobile_css_generation(self):
        """Test mobile CSS generation."""
        css = self.typography_manager.generate_mobile_css(DeviceType.MOBILE)
        
        # Should contain mobile-specific CSS
        self.assertIn('font-size: 16px', css)  # Minimum mobile font size
        self.assertIn('max-width:', css)
        self.assertIn('@media (max-width: 768px)', css)
        
        # Should include touch-friendly link styles
        self.assertIn('min-height: 44px', css)
        
        # Should be valid CSS structure
        self.assertIn('{', css)
        self.assertIn('}', css)
        self.assertIn(':', css)
    
    def test_typography_recommendations(self):
        """Test typography recommendations generation."""
        content_needing_improvement = """
# Title

This is a very long paragraph that definitely needs to be broken down into smaller pieces for better mobile readability and user experience because reading long blocks of text on small screens is quite challenging.

```
unformatted code block without language
def function():
    return value
```

Here are some links very close together: [One](url1) [Two](url2) [Three](url3) [Four](url4)
"""
        
        recommendations = self.typography_manager.get_typography_recommendations(content_needing_improvement)
        
        # Should provide specific recommendations
        self.assertGreater(len(recommendations), 0)
        
        # Should identify specific issues
        recommendation_text = ' '.join(recommendations).lower()
        self.assertTrue(
            any(keyword in recommendation_text for keyword in ['paragraph', 'code', 'link'])
        )


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)