"""
Tests for Technical Quality Validator

Tests the unified technical quality validation functionality.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.quality.base import QualityMetrics, QualityReport, QualityStatus
from src.quality.technical_validator import TechnicalQualityValidator


class TestTechnicalQualityValidator:
    """Test cases for TechnicalQualityValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a TechnicalQualityValidator instance."""
        return TechnicalQualityValidator()
    
    @pytest.fixture
    def sample_content(self):
        """Sample newsletter content for testing."""
        return {
            'content': '''
            # AI/ML Newsletter
            
            ## Neural Network Developments
            
            Recent advances in neural networks have shown remarkable improvements in performance.
            
            ```python
            import torch
            import torch.nn as nn
            
            class Transformer(nn.Module):
                def __init__(self, d_model, nhead):
                    super().__init__()
                    self.attention = nn.MultiheadAttention(d_model, nhead)
            ```
            
            This demonstrates the power of modern AI frameworks.
            ''',
            'subject': 'AI/ML Newsletter - Latest Developments in Neural Networks'
        }
    
    def test_validator_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.name == "TechnicalQualityValidator"
        assert len(validator.technical_terms_db) > 0
        assert len(validator.code_patterns) > 0
        assert len(validator.mobile_thresholds) > 0
        
        # Check technical terms database structure
        expected_categories = ['ai_ml_terms', 'programming_terms', 'cloud_terms', 'data_terms']
        for category in expected_categories:
            assert category in validator.technical_terms_db
            assert len(validator.technical_terms_db[category]) > 0
    
    def test_extract_text_content(self, validator):
        """Test text content extraction."""
        # Test string content
        content = "This is a test content"
        result = validator._extract_text_content(content)
        assert result == "This is a test content"
        
        # Test dictionary content
        content_dict = {
            'content': 'Main content here',
            'subject': 'Test subject',
            'preview': 'Test preview'
        }
        result = validator._extract_text_content(content_dict)
        assert result == 'Main content here'
    
    def test_load_technical_terms_database(self, validator):
        """Test technical terms database loading."""
        terms_db = validator._load_technical_terms_database()
        
        expected_categories = ['ai_ml_terms', 'programming_terms', 'cloud_terms', 'data_terms']
        for category in expected_categories:
            assert category in terms_db
            assert isinstance(terms_db[category], list)
            assert len(terms_db[category]) > 0
        
        # Check for specific terms
        assert 'neural network' in terms_db['ai_ml_terms']
        assert 'API' in terms_db['programming_terms']
        assert 'AWS' in terms_db['cloud_terms']
        assert 'data science' in terms_db['data_terms']
    
    def test_extract_technical_claims(self, validator):
        """Test technical claims extraction."""
        content = """
        Neural networks are transforming AI applications.
        Machine learning algorithms show remarkable performance.
        Transformer models have revolutionized NLP.
        This is a regular sentence without technical terms.
        """
        
        claims = validator._extract_technical_claims(content)
        
        assert isinstance(claims, list)
        assert len(claims) >= 3  # Should extract sentences with technical terms
        
        # Check that extracted claims contain technical terms
        technical_terms = ['neural network', 'machine learning', 'transformer']
        for term in technical_terms:
            assert any(term.lower() in claim.lower() for claim in claims)
    
    def test_validate_single_claim(self, validator):
        """Test single technical claim validation."""
        # Valid technical claim
        valid_claim = "Neural networks use backpropagation for training"
        result = validator._validate_single_claim(valid_claim)
        
        assert 'claim' in result
        assert 'is_accurate' in result
        assert 'validation_score' in result
        assert 'issues' in result
        assert 'technical_terms_found' in result
        
        assert result['claim'] == valid_claim
        assert isinstance(result['is_accurate'], bool)
        assert 0 <= result['validation_score'] <= 1
        assert isinstance(result['issues'], list)
        assert result['technical_terms_found'] > 0
        
        # Invalid technical claim
        invalid_claim = "AI can think like humans and is infallible"
        result = validator._validate_single_claim(invalid_claim)
        
        assert result['is_accurate'] is False
        assert len(result['issues']) > 0
    
    def test_calculate_accuracy_score(self, validator):
        """Test accuracy score calculation."""
        validation_results = [
            {'validation_score': 0.8},
            {'validation_score': 0.9},
            {'validation_score': 0.7}
        ]
        
        score = validator._calculate_accuracy_score(validation_results)
        expected_score = (0.8 + 0.9 + 0.7) / 3
        assert score == pytest.approx(expected_score)
        
        # Test with empty results
        score = validator._calculate_accuracy_score([])
        assert score == 0.0
    
    def test_validate_technical_accuracy(self, validator):
        """Test technical accuracy validation."""
        content = """
        Neural networks use gradient descent for optimization.
        Transformer models employ attention mechanisms.
        Machine learning algorithms require training data.
        """
        
        result = validator._validate_technical_accuracy(content)
        
        assert 'accuracy_score' in result
        assert 'technical_claims_count' in result
        assert 'validated_claims' in result
        assert 'processing_time_seconds' in result
        assert 'passes_threshold' in result
        assert 'issues_found' in result
        assert 'validation_timestamp' in result
        
        assert isinstance(result['accuracy_score'], float)
        assert 0 <= result['accuracy_score'] <= 1
        assert result['technical_claims_count'] > 0
        assert len(result['validated_claims']) > 0
        assert isinstance(result['processing_time_seconds'], float)
        assert isinstance(result['passes_threshold'], bool)
    
    def test_validate_code_examples(self, validator):
        """Test code examples validation."""
        content = """Here's a Python example:
```python
import numpy as np

def calculate_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)
```

And a JavaScript example:
```javascript
function calculateAccuracy(predictions, targets) {
    return predictions.filter((p, i) => p === targets[i]).length / targets.length;
}
```"""
        
        result = validator._validate_code_examples(content)
        
        assert 'code_score' in result
        assert 'code_blocks_count' in result
        assert 'validation_results' in result
        assert 'languages_found' in result
        assert 'passes_threshold' in result
        
        assert isinstance(result['code_score'], float)
        assert 0 <= result['code_score'] <= 1
        assert result['code_blocks_count'] > 0
        assert len(result['validation_results']) > 0
        assert len(result['languages_found']) > 0
        assert isinstance(result['passes_threshold'], bool)
    
    def test_validate_code_block(self, validator):
        """Test individual code block validation."""
        # Valid Python code
        python_code = "import numpy as np\ndef test():\n    return np.array([1, 2, 3])"
        result = validator._validate_code_block(python_code, 'python')
        
        assert 'code' in result
        assert 'language' in result
        assert 'validation_score' in result
        assert 'issues' in result
        assert 'is_valid' in result
        
        assert result['language'] == 'python'
        assert result['validation_score'] > 0.6  # Should be valid
        assert result['is_valid'] is True
        
        # Invalid Python code
        invalid_python = "import numpy as np\ndef test():\n    return np.array([1, 2, 3"  # Missing closing bracket
        result = validator._validate_code_block(invalid_python, 'python')
        
        assert result['validation_score'] < 0.5  # Should have lower score
        assert result['is_valid'] is False
        assert len(result['issues']) > 0
    
    def test_calculate_code_score(self, validator):
        """Test code score calculation."""
        validation_results = [
            {'validation_score': 0.8},
            {'validation_score': 0.9},
            {'validation_score': 0.7}
        ]
        
        score = validator._calculate_code_score(validation_results)
        expected_score = (0.8 + 0.9 + 0.7) / 3
        assert score == pytest.approx(expected_score)
        
        # Test with empty results
        score = validator._calculate_code_score([])
        assert score == 0.0
    
    def test_validate_mobile_readability(self, validator):
        """Test mobile readability validation."""
        content = {
            'subject': 'AI/ML Newsletter',
            'preview': 'Latest developments in machine learning and artificial intelligence.',
            'content': '''
            # AI/ML Newsletter
            
            ## Introduction
            
            Machine learning is transforming industries.
            
            ## Key Points
            
            - Neural networks are powerful
            - Transformers are revolutionary
            - Attention mechanisms work well
            
            ## Conclusion
            
            The future looks bright for AI.
            '''
        }
        
        result = validator._validate_mobile_readability(content)
        
        assert 'mobile_score' in result
        assert 'subject_validation' in result
        assert 'preview_validation' in result
        assert 'paragraph_validation' in result
        assert 'headline_validation' in result
        assert 'structure_validation' in result
        assert 'whitespace_validation' in result
        assert 'passes_threshold' in result
        
        assert isinstance(result['mobile_score'], float)
        assert 0 <= result['mobile_score'] <= 1
        assert isinstance(result['passes_threshold'], bool)
    
    def test_validate_subject_line(self, validator):
        """Test subject line validation."""
        # Optimal subject line
        subject = "AI/ML Newsletter"
        result = validator._validate_subject_line(subject)
        
        assert 'length' in result
        assert 'is_optimal' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert result['length'] == len(subject)
        assert result['is_optimal'] is True
        assert result['score'] == 1.0
        
        # Too long subject line
        long_subject = "This is a very long subject line that exceeds the recommended length for mobile devices"
        result = validator._validate_subject_line(long_subject)
        
        assert result['is_optimal'] is False
        assert result['score'] < 1.0
    
    def test_validate_preview_text(self, validator):
        """Test preview text validation."""
        # Optimal preview text
        preview = "Latest developments in machine learning and artificial intelligence."
        result = validator._validate_preview_text(preview)
        
        assert 'length' in result
        assert 'is_optimal' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert result['length'] == len(preview)
        assert result['is_optimal'] is True
        assert result['score'] == 1.0
        
        # Too long preview text
        long_preview = "This is a very long preview text that exceeds the recommended length for mobile devices and should be shortened to improve readability and user experience"
        result = validator._validate_preview_text(long_preview)
        
        assert result['is_optimal'] is False
        assert result['score'] < 1.0
    
    def test_validate_paragraph_lengths(self, validator):
        """Test paragraph length validation."""
        content = """
        This is a short paragraph.
        
        This is a very long paragraph that contains many words and should be broken down into smaller, more manageable chunks for better mobile readability and user experience.
        
        This is another short paragraph.
        """
        
        result = validator._validate_paragraph_lengths(content)
        
        assert 'total_paragraphs' in result
        assert 'optimal_paragraphs' in result
        assert 'long_paragraphs' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert result['total_paragraphs'] > 0
        assert result['optimal_paragraphs'] >= 0
        assert len(result['long_paragraphs']) >= 0
        assert 0 <= result['score'] <= 1
    
    def test_validate_headline_readability(self, validator):
        """Test headline readability validation."""
        content = """# Short headline

## This is a very long headline that exceeds the recommended length for mobile devices

### Another short headline"""
        
        result = validator._validate_headline_readability(content)
        
        assert 'total_headlines' in result
        assert 'optimal_headlines' in result
        assert 'long_headlines' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert result['total_headlines'] > 0
    
    def test_validate_content_structure(self, validator):
        """Test content structure validation."""
        content = """
        # Main heading
        
        This is a paragraph.
        
        - List item 1
        - List item 2
        
        ```python
        print("Code block")
        ```
        
        [Link text](https://example.com)
        """
        
        result = validator._validate_content_structure(content)
        
        assert 'structure_elements' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert isinstance(result['structure_elements'], dict)
        assert 0 <= result['score'] <= 1
        assert isinstance(result['recommendation'], str)
    
    def test_validate_white_space_usage(self, validator):
        """Test white space usage validation."""
        # Good white space usage
        content = """
        This content has good white space usage.
        
        It includes proper spacing between paragraphs.
        
        And uses whitespace effectively.
        """
        
        result = validator._validate_white_space_usage(content)
        
        assert 'whitespace_ratio' in result
        assert 'is_optimal' in result
        assert 'score' in result
        assert 'recommendation' in result
        
        assert isinstance(result['whitespace_ratio'], float)
        assert 0 <= result['whitespace_ratio'] <= 1
        assert isinstance(result['is_optimal'], bool)
        assert 0 <= result['score'] <= 1
    
    def test_calculate_mobile_score(self, validator):
        """Test mobile score calculation."""
        validation_results = {
            'subject': {'score': 0.8},
            'preview': {'score': 0.9},
            'paragraphs': {'score': 0.7},
            'headlines': {'score': 0.8},
            'structure': {'score': 0.9},
            'whitespace': {'score': 0.8}
        }
        
        score = validator._calculate_mobile_score(validation_results)
        expected_score = (0.8 + 0.9 + 0.7 + 0.8 + 0.9 + 0.8) / 6
        assert score == pytest.approx(expected_score)
    
    def test_calculate_technical_metrics(self, validator):
        """Test technical metrics calculation."""
        technical_accuracy = {'accuracy_score': 0.8}
        code_validation = {'code_score': 0.7}
        mobile_readability = {
            'mobile_score': 0.8,
            'structure_validation': {'score': 0.9}
        }
        
        result = validator._calculate_technical_metrics(
            technical_accuracy, code_validation, mobile_readability
        )
        
        assert isinstance(result, QualityMetrics)
        assert result.technical_accuracy_score > 0
        assert result.content_quality_score > 0
        assert result.readability_score > 0
        assert result.code_quality_score > 0
        assert result.mobile_readability_score > 0
        assert result.structure_score > 0
    
    def test_generate_technical_issues(self, validator):
        """Test technical issue generation."""
        technical_accuracy = {'accuracy_score': 0.6}
        code_validation = {'code_score': 0.5}
        mobile_readability = {'mobile_score': 0.5}
        
        issues, warnings, recommendations, blocking_issues = validator._generate_technical_issues(
            technical_accuracy, code_validation, mobile_readability
        )
        
        assert isinstance(issues, list)
        assert isinstance(warnings, list)
        assert isinstance(recommendations, list)
        assert isinstance(blocking_issues, list)
        
        # Should have issues due to low scores
        assert len(issues) > 0
    
    def test_determine_technical_status(self, validator):
        """Test technical status determination."""
        # Test passed status
        issues = []
        warnings = []
        blocking_issues = []
        metrics = QualityMetrics(
            overall_score=8.0,
            technical_accuracy_score=8.0,
            content_quality_score=8.0,
            readability_score=8.0,
            engagement_score=8.0,
            structure_score=8.0,
            code_quality_score=8.0,
            mobile_readability_score=8.0,
            source_credibility_score=8.0,
            content_balance_score=8.0,
            performance_score=8.0
        )
        
        status = validator._determine_technical_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.PASSED
        
        # Test failed status
        blocking_issues = ["Critical technical issue"]
        status = validator._determine_technical_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.FAILED
        
        # Test needs review status (technical accuracy < 7.0 but >= 5.0)
        blocking_issues = []
        metrics.technical_accuracy_score = 6.0
        status = validator._determine_technical_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.NEEDS_REVIEW
    
    def test_identify_technical_strengths(self, validator):
        """Test technical strengths identification."""
        technical_accuracy = {'accuracy_score': 0.8}
        code_validation = {'code_score': 0.7}
        mobile_readability = {'mobile_score': 0.9}
        
        strengths = validator._identify_technical_strengths(
            technical_accuracy, code_validation, mobile_readability
        )
        
        assert isinstance(strengths, list)
        assert len(strengths) > 0
    
    def test_validate_comprehensive(self, validator, sample_content):
        """Test comprehensive technical validation."""
        report = validator.validate(sample_content)
        
        assert isinstance(report, QualityReport)
        assert report.status in QualityStatus
        assert isinstance(report.metrics, QualityMetrics)
        assert isinstance(report.issues, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.blocking_issues, list)
        assert isinstance(report.strengths, list)
        assert isinstance(report.detailed_analysis, dict)
        
        # Check that detailed analysis contains expected keys
        expected_keys = [
            'technical_accuracy', 'code_validation', 'mobile_readability',
            'content_length', 'processing_time'
        ]
        for key in expected_keys:
            assert key in report.detailed_analysis
    
    def test_get_metrics(self, validator, sample_content):
        """Test metrics extraction."""
        metrics = validator.get_metrics(sample_content)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_score > 0
        assert metrics.technical_accuracy_score > 0
        assert metrics.code_quality_score > 0
    
    def test_validate_batch(self, validator):
        """Test batch validation functionality."""
        contents = [
            "Neural networks are powerful AI tools.",
            "Machine learning algorithms require data.",
            "Transformer models use attention mechanisms."
        ]
        
        reports = validator.validate_batch(contents)
        
        assert isinstance(reports, list)
        assert len(reports) == 3
        
        for report in reports:
            assert isinstance(report, QualityReport)
            assert report.status in QualityStatus
    
    def test_error_handling(self, validator):
        """Test error handling in validation."""
        # Test with empty content
        report = validator.validate("")
        assert isinstance(report, QualityReport)
        assert report.status in QualityStatus
        
        # Test with None content
        report = validator.validate(None)
        assert isinstance(report, QualityReport)
        assert report.status in QualityStatus
        
        # Test with invalid content type
        report = validator.validate(123)
        assert isinstance(report, QualityReport)
        assert report.status in QualityStatus


if __name__ == "__main__":
    pytest.main([__file__])