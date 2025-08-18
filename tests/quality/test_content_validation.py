"""
Tests for Content Quality Validator

Tests the unified content quality validation functionality.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.quality.base import QualityMetrics, QualityReport, QualityStatus
from src.quality.content_validator import ContentQualityValidator


class TestContentQualityValidator:
    """Test cases for ContentQualityValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a ContentQualityValidator instance."""
        return ContentQualityValidator()
    
    @pytest.fixture
    def sample_content(self):
        """Sample newsletter content for testing."""
        return """
        # AI/ML Newsletter
        
        Machine learning algorithms are transforming the way we approach data analysis. 
        Neural networks have shown remarkable performance in various applications.
        
        ## Key Developments
        
        Recent studies show that 85% of companies are adopting AI technologies. 
        According to research, machine learning models can improve efficiency by 40%.
        
        "The future of AI depends on responsible development," says Dr. Smith from MIT.
        
        ## Code Example
        
        ```python
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        ```
        
        This demonstrates the power of modern AI frameworks.
        """
    
    def test_validator_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator.name == "ContentQualityValidator"
        assert len(validator.suspicious_expert_patterns) > 0
        assert len(validator.generic_expert_quotes) > 0
        assert len(validator.factual_claim_patterns) > 0
    
    def test_extract_text_content_string(self, validator):
        """Test text extraction from string content."""
        content = "This is a test content"
        result = validator._extract_text_content(content)
        assert result == "This is a test content"
    
    def test_extract_text_content_dict(self, validator):
        """Test text extraction from dictionary content."""
        content = {
            'content': 'Main content here',
            'subject': 'Test subject',
            'preview': 'Test preview'
        }
        result = validator._extract_text_content(content)
        assert result == 'Main content here'
    
    def test_extract_text_content_dict_fallback(self, validator):
        """Test text extraction fallback for dictionary."""
        content = {
            'body': 'Body content here',
            'other_field': 'Other data'
        }
        result = validator._extract_text_content(content)
        assert result == 'Body content here'
    
    def test_detect_repetition(self, validator):
        """Test repetition detection functionality."""
        content = """
        This is a test sentence. This is a test sentence. 
        Another sentence here. This is a test sentence.
        """
        
        result = validator._detect_repetition(content)
        
        assert 'repetition_score' in result
        assert 'repetitive_sentences' in result
        assert 'sentence_groups' in result
        assert 'concept_repetition' in result
        assert 'total_sentences' in result
        assert 'repetitive_count' in result
        
        # Should detect repetition in this content
        assert result['repetition_score'] > 0
    
    def test_analyze_expert_quotes(self, validator):
        """Test expert quote analysis."""
        content = '''
        "The future of AI depends on responsible development," says Dr. Smith from MIT.
        "Machine learning is critical for success," according to Professor Johnson.
        "This is a generic quote about AI."
        '''
        
        result = validator._analyze_expert_quotes(content)
        
        assert 'total_quotes' in result
        assert 'suspicious_quotes' in result
        assert 'suspicion_score' in result
        assert 'average_suspicion' in result
        
        assert result['total_quotes'] == 3
        # The suspicious quotes detection depends on the patterns, so we just check it's a list
        assert isinstance(result['suspicious_quotes'], list)
    
    def test_analyze_factual_claims(self, validator):
        """Test factual claim analysis."""
        content = """
        Studies show that 85% of companies are adopting AI.
        Research indicates that 60% of users prefer mobile apps.
        According to recent survey, 90% of developers use Python.
        """
        
        result = validator._analyze_factual_claims(content)
        
        assert 'total_claims' in result
        assert 'claims' in result
        assert 'verification_needed' in result
        assert 'claim_density' in result
        
        assert result['total_claims'] > 0
        assert result['verification_needed'] is True
    
    def test_assess_content_quality(self, validator):
        """Test content quality assessment."""
        content = """
        # AI/ML Newsletter
        
        Machine learning algorithms are transforming data analysis.
        Neural networks show remarkable performance.
        
        ```python
        import tensorflow as tf
        model = tf.keras.Sequential()
        ```
        """
        
        result = validator._assess_content_quality(content)
        
        expected_metrics = [
            'technical_accuracy', 'information_density', 'readability',
            'engagement', 'structure', 'ai_ml_relevance', 'code_quality',
            'citation_quality', 'practical_value', 'innovation_factor'
        ]
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], float)
            assert 0 <= result[metric] <= 10
    
    def test_calculate_content_metrics(self, validator):
        """Test content metrics calculation."""
        quality_metrics = {
            'technical_accuracy': 8.0,
            'information_density': 7.0,
            'readability': 6.0,
            'engagement': 5.0,
            'structure': 7.0,
            'ai_ml_relevance': 8.0,
            'code_quality': 6.0,
            'citation_quality': 5.0,
            'practical_value': 6.0,
            'innovation_factor': 7.0
        }
        
        repetition_analysis = {'repetition_score': 0.1}
        expert_analysis = {'suspicion_score': 0.2}
        factual_analysis = {'claim_density': 0.05}
        
        result = validator._calculate_content_metrics(
            quality_metrics, repetition_analysis, expert_analysis, factual_analysis
        )
        
        assert isinstance(result, QualityMetrics)
        assert result.technical_accuracy_score > 0
        assert result.content_quality_score > 0
        assert result.readability_score > 0
        assert result.overall_score > 0
    
    def test_generate_content_issues(self, validator):
        """Test content issue generation."""
        repetition_analysis = {'repetition_score': 0.4}
        expert_analysis = {'suspicious_quotes': [{'quote': 'test', 'suspicion_score': 0.8}]}
        factual_analysis = {'total_claims': 6}
        quality_metrics = {'technical_accuracy': 4.0, 'readability': 3.0}
        
        issues, warnings, recommendations, blocking_issues = validator._generate_content_issues(
            repetition_analysis, expert_analysis, factual_analysis, quality_metrics
        )
        
        assert isinstance(issues, list)
        assert isinstance(warnings, list)
        assert isinstance(recommendations, list)
        assert isinstance(blocking_issues, list)
        
        # Should have issues due to low scores
        assert len(issues) > 0
    
    def test_determine_content_status(self, validator):
        """Test content status determination."""
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
        
        status = validator._determine_content_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.PASSED
        
        # Test failed status
        blocking_issues = ["Critical issue"]
        status = validator._determine_content_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.FAILED
        
        # Test needs review status
        blocking_issues = []
        metrics.overall_score = 6.0
        status = validator._determine_content_status(issues, warnings, blocking_issues, metrics)
        assert status == QualityStatus.NEEDS_REVIEW
    
    def test_validate_comprehensive(self, validator, sample_content):
        """Test comprehensive content validation."""
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
            'repetition_analysis', 'expert_analysis', 'factual_analysis',
            'quality_metrics', 'content_length', 'word_count'
        ]
        for key in expected_keys:
            assert key in report.detailed_analysis
    
    def test_get_metrics(self, validator, sample_content):
        """Test metrics extraction."""
        metrics = validator.get_metrics(sample_content)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.overall_score > 0
        assert metrics.technical_accuracy_score > 0
        assert metrics.content_quality_score > 0
    
    def test_validate_batch(self, validator):
        """Test batch validation functionality."""
        contents = [
            "This is the first test content with AI and machine learning.",
            "This is the second test content with neural networks.",
            "This is the third test content with algorithms."
        ]
        
        reports = validator.validate_batch(contents)
        
        assert isinstance(reports, list)
        assert len(reports) == 3
        
        for report in reports:
            assert isinstance(report, QualityReport)
            assert report.status in QualityStatus
    
    def test_calculate_similarity(self, validator):
        """Test similarity calculation."""
        text1 = "This is a test sentence"
        text2 = "This is a test sentence"
        text3 = "This is a different sentence"
        
        # Identical sentences should have high similarity
        similarity1 = validator._calculate_similarity(text1, text2)
        assert similarity1 > 0.9
        
        # Different sentences should have lower similarity
        similarity2 = validator._calculate_similarity(text1, text3)
        assert similarity2 < similarity1
    
    def test_detect_concept_repetition(self, validator):
        """Test concept repetition detection."""
        content = "Machine learning is important. Machine learning is powerful. Machine learning is the future. Machine learning is critical. AI is key."
        
        result = validator._detect_concept_repetition(content)
        
        assert isinstance(result, list)
        # Should detect repetition of "machine" (appears 4 times and is longer than 4 characters)
        assert len(result) > 0
        assert any('machine' in concept['concept'] for concept in result)
    
    def test_calculate_quote_suspicion(self, validator):
        """Test quote suspicion calculation."""
        # Generic quote
        generic_quote = "The future of AI depends on responsible development"
        suspicion1 = validator._calculate_quote_suspicion(generic_quote)
        assert suspicion1 > 0
        
        # Specific quote
        specific_quote = "Our research shows that transformer models outperform previous architectures"
        suspicion2 = validator._calculate_quote_suspicion(specific_quote)
        assert suspicion2 < suspicion1
    
    def test_get_suspicion_reasons(self, validator):
        """Test suspicion reason generation."""
        quote = "The future of AI depends on responsible development"
        reasons = validator._get_suspicion_reasons(quote)
        
        assert isinstance(reasons, list)
        assert len(reasons) > 0
    
    def test_quality_assessment_methods(self, validator):
        """Test individual quality assessment methods."""
        content = "AI and machine learning are transforming technology. Neural networks show promise."
        
        # Test technical accuracy
        accuracy = validator._assess_technical_accuracy(content)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 10
        
        # Test information density
        density = validator._assess_information_density(content)
        assert isinstance(density, float)
        assert 0 <= density <= 10
        
        # Test readability
        readability = validator._assess_readability(content)
        assert isinstance(readability, float)
        assert 0 <= readability <= 10
        
        # Test AI/ML relevance
        relevance = validator._assess_ai_ml_relevance(content)
        assert isinstance(relevance, float)
        assert 0 <= relevance <= 10
        
        # Test code quality
        code_content = "```python\nimport tensorflow as tf\n```"
        code_quality = validator._assess_code_quality(code_content)
        assert isinstance(code_quality, float)
        assert 0 <= code_quality <= 10
    
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