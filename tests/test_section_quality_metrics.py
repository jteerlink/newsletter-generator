"""
Test Suite for Section-Level Quality Metrics System

Comprehensive tests for the granular quality assessment including:
- Quality dimension analysis
- Section-specific quality metrics
- Aggregated quality reporting
- Quality threshold validation
- Improvement recommendations
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.section_aware_prompts import SectionType
from src.core.section_quality_metrics import (
    AggregatedQualityReport,
    QualityDimension,
    QualityMetric,
    SectionAwareQualitySystem,
    SectionQualityAnalyzer,
    SectionQualityMetrics,
)


class TestQualityDimension:
    """Test QualityDimension enum."""
    
    def test_quality_dimension_values(self):
        """Test that all expected quality dimensions exist."""
        expected_dimensions = {
            'clarity', 'relevance', 'completeness', 'accuracy',
            'engagement', 'structure', 'consistency', 'readability'
        }
        actual_dimensions = {dim.value for dim in QualityDimension}
        assert actual_dimensions == expected_dimensions


class TestQualityMetric:
    """Test QualityMetric dataclass."""
    
    def test_quality_metric_creation(self):
        """Test creating QualityMetric instance."""
        metric = QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=0.85,
            weight=1.5,
            details="Good clarity score",
            issues=["Minor issue"],
            suggestions=["Improvement suggestion"]
        )
        
        assert metric.dimension == QualityDimension.CLARITY
        assert metric.score == 0.85
        assert metric.weight == 1.5
        assert metric.details == "Good clarity score"
        assert len(metric.issues) == 1
        assert len(metric.suggestions) == 1
    
    def test_quality_metric_defaults(self):
        """Test default values in QualityMetric."""
        metric = QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=0.7
        )
        
        assert metric.weight == 1.0
        assert metric.details == ""
        assert metric.issues == []
        assert metric.suggestions == []


class TestSectionQualityMetrics:
    """Test SectionQualityMetrics dataclass and methods."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample quality metrics."""
        metrics = {
            QualityDimension.CLARITY: QualityMetric(
                dimension=QualityDimension.CLARITY,
                score=0.8,
                weight=1.5
            ),
            QualityDimension.RELEVANCE: QualityMetric(
                dimension=QualityDimension.RELEVANCE,
                score=0.9,
                weight=2.0
            ),
            QualityDimension.COMPLETENESS: QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=0.7,
                weight=1.0
            )
        }
        return metrics
    
    def test_section_quality_metrics_creation(self, sample_metrics):
        """Test creating SectionQualityMetrics instance."""
        section_metrics = SectionQualityMetrics(
            section_type=SectionType.ANALYSIS,
            section_content="Sample analysis content",
            overall_score=0.8,
            metrics=sample_metrics,
            word_count=150,
            sentence_count=8,
            paragraph_count=3
        )
        
        assert section_metrics.section_type == SectionType.ANALYSIS
        assert section_metrics.overall_score == 0.8
        assert len(section_metrics.metrics) == 3
        assert section_metrics.word_count == 150
    
    def test_get_weighted_score(self, sample_metrics):
        """Test weighted score calculation."""
        section_metrics = SectionQualityMetrics(
            section_type=SectionType.TUTORIAL,
            section_content="Tutorial content",
            overall_score=0.0,  # Will be calculated
            metrics=sample_metrics
        )
        
        weighted_score = section_metrics.get_weighted_score()
        
        # Calculate expected: (0.8*1.5 + 0.9*2.0 + 0.7*1.0) / (1.5+2.0+1.0) = 3.7/4.5
        expected_score = (0.8 * 1.5 + 0.9 * 2.0 + 0.7 * 1.0) / (1.5 + 2.0 + 1.0)
        assert abs(weighted_score - expected_score) < 0.01
    
    def test_get_weighted_score_empty_metrics(self):
        """Test weighted score with empty metrics."""
        section_metrics = SectionQualityMetrics(
            section_type=SectionType.NEWS,
            section_content="News content",
            overall_score=0.75,
            metrics={}
        )
        
        weighted_score = section_metrics.get_weighted_score()
        assert weighted_score == 0.75  # Should return overall_score
    
    def test_get_issues(self, sample_metrics):
        """Test getting all issues from metrics."""
        # Add issues to metrics
        sample_metrics[QualityDimension.CLARITY].issues = ["Clarity issue 1", "Clarity issue 2"]
        sample_metrics[QualityDimension.RELEVANCE].issues = ["Relevance issue"]
        
        section_metrics = SectionQualityMetrics(
            section_type=SectionType.INTRODUCTION,
            section_content="Intro content",
            overall_score=0.8,
            metrics=sample_metrics
        )
        
        all_issues = section_metrics.get_issues()
        assert len(all_issues) == 3
        assert "Clarity issue 1" in all_issues
        assert "Relevance issue" in all_issues
    
    def test_get_suggestions(self, sample_metrics):
        """Test getting all suggestions from metrics."""
        # Add suggestions to metrics
        sample_metrics[QualityDimension.CLARITY].suggestions = ["Improve clarity"]
        sample_metrics[QualityDimension.COMPLETENESS].suggestions = ["Add more details", "Include examples"]
        
        section_metrics = SectionQualityMetrics(
            section_type=SectionType.CONCLUSION,
            section_content="Conclusion content",
            overall_score=0.7,
            metrics=sample_metrics
        )
        
        all_suggestions = section_metrics.get_suggestions()
        assert len(all_suggestions) == 3
        assert "Improve clarity" in all_suggestions
        assert "Add more details" in all_suggestions


class TestAggregatedQualityReport:
    """Test AggregatedQualityReport dataclass and methods."""
    
    @pytest.fixture
    def sample_section_scores(self):
        """Create sample section scores."""
        return {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Intro",
                overall_score=0.85,
                word_count=100
            ),
            SectionType.ANALYSIS: SectionQualityMetrics(
                section_type=SectionType.ANALYSIS,
                section_content="Analysis",
                overall_score=0.90,
                word_count=400
            ),
            SectionType.CONCLUSION: SectionQualityMetrics(
                section_type=SectionType.CONCLUSION,
                section_content="Conclusion",
                overall_score=0.80,
                word_count=150
            )
        }
    
    def test_aggregated_quality_report_creation(self, sample_section_scores):
        """Test creating AggregatedQualityReport."""
        report = AggregatedQualityReport(
            overall_score=0.85,
            section_scores=sample_section_scores,
            total_word_count=650,
            average_readability=0.75,
            consistency_score=0.80,
            flow_score=0.85
        )
        
        assert report.overall_score == 0.85
        assert len(report.section_scores) == 3
        assert report.total_word_count == 650
    
    def test_get_section_breakdown(self, sample_section_scores):
        """Test section breakdown generation."""
        report = AggregatedQualityReport(
            overall_score=0.85,
            section_scores=sample_section_scores,
            total_word_count=650,
            average_readability=0.75,
            consistency_score=0.80,
            flow_score=0.85
        )
        
        breakdown = report.get_section_breakdown()
        
        assert len(breakdown) == 3
        assert breakdown['introduction'] == 0.85
        assert breakdown['analysis'] == 0.90
        assert breakdown['conclusion'] == 0.80
    
    def test_get_quality_summary(self, sample_section_scores):
        """Test quality summary generation."""
        report = AggregatedQualityReport(
            overall_score=0.85,
            section_scores=sample_section_scores,
            total_word_count=650,
            average_readability=0.75,
            consistency_score=0.80,
            flow_score=0.85
        )
        
        summary = report.get_quality_summary()
        
        expected_keys = {
            'overall_score', 'total_sections', 'word_count',
            'readability', 'consistency', 'flow', 'section_breakdown'
        }
        assert set(summary.keys()) == expected_keys
        assert summary['overall_score'] == 0.85
        assert summary['total_sections'] == 3
        assert summary['word_count'] == 650


class TestSectionQualityAnalyzer:
    """Test SectionQualityAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create quality analyzer instance."""
        return SectionQualityAnalyzer()
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for analysis."""
        return {
            'topic': 'Machine Learning',
            'audience': 'AI/ML Engineers',
            'content_focus': 'Neural Networks'
        }
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.section_weights) == 5  # 5 section types
        assert SectionType.INTRODUCTION in analyzer.section_weights
        assert len(analyzer.default_weights) == 8  # 8 quality dimensions
    
    def test_analyze_section_basic(self, analyzer, sample_context):
        """Test basic section analysis."""
        content = """This is a comprehensive introduction to machine learning.
        We will explore various algorithms and their applications.
        By the end, you'll understand the fundamentals."""
        
        metrics = analyzer.analyze_section(content, SectionType.INTRODUCTION, sample_context)
        
        assert isinstance(metrics, SectionQualityMetrics)
        assert metrics.section_type == SectionType.INTRODUCTION
        assert 0.0 <= metrics.overall_score <= 1.0
        assert metrics.word_count > 0
        assert len(metrics.metrics) == 8  # All quality dimensions
    
    def test_section_specific_weights(self, analyzer):
        """Test section-specific quality weights."""
        intro_weights = analyzer.section_weights[SectionType.INTRODUCTION]
        analysis_weights = analyzer.section_weights[SectionType.ANALYSIS]
        
        # Introduction should weight engagement higher than analysis
        assert intro_weights[QualityDimension.ENGAGEMENT] > analysis_weights.get(QualityDimension.ENGAGEMENT, 1.0)
        
        # Analysis should weight accuracy higher than introduction
        assert analysis_weights[QualityDimension.ACCURACY] > intro_weights.get(QualityDimension.ACCURACY, 1.0)
    
    def test_clarity_assessment(self, analyzer):
        """Test clarity dimension assessment."""
        # Good clarity - short sentences, clear language
        clear_content = "This is clear. Each sentence is short. Technical terms are explained."
        
        # Poor clarity - long sentences, complex structure
        unclear_content = """This is a very long and complex sentence that contains multiple clauses and subclauses which make it difficult to understand and follow the main point being communicated."""
        
        clear_metric = analyzer._assess_clarity(clear_content, SectionType.ANALYSIS)
        unclear_metric = analyzer._assess_clarity(unclear_content, SectionType.ANALYSIS)
        
        assert clear_metric.score > unclear_metric.score
        assert clear_metric.dimension == QualityDimension.CLARITY
    
    def test_relevance_assessment(self, analyzer):
        """Test relevance dimension assessment."""
        context = {
            'topic': 'machine learning algorithms',
            'audience': 'AI/ML Engineers'
        }
        
        relevant_content = "Machine learning algorithms like neural networks provide powerful tools for AI engineers."
        irrelevant_content = "Cooking recipes are important for kitchen management."
        
        relevant_metric = analyzer._assess_relevance(relevant_content, SectionType.ANALYSIS, context)
        irrelevant_metric = analyzer._assess_relevance(irrelevant_content, SectionType.ANALYSIS, context)
        
        assert relevant_metric.score > irrelevant_metric.score
    
    def test_completeness_assessment(self, analyzer):
        """Test completeness dimension assessment."""
        # Test different section types
        test_cases = [
            (SectionType.INTRODUCTION, "Welcome! This overview covers key topics. We'll explore main concepts.", 0.0),
            (SectionType.TUTORIAL, "Step 1: Install package. Step 2: Configure settings. Follow these instructions carefully.", 0.0),
            (SectionType.NEWS, "Company announced new features. Updates include performance improvements. Details are available.", 0.0)
        ]
        
        for section_type, content, min_score in test_cases:
            metric = analyzer._assess_completeness(content, section_type)
            assert metric.score >= min_score
            assert metric.dimension == QualityDimension.COMPLETENESS
    
    def test_accuracy_assessment(self, analyzer):
        """Test accuracy dimension assessment."""
        # Content with supported claims
        supported_content = "According to recent research, studies show that AI performance has improved significantly."
        
        # Content with unsupported claims
        unsupported_content = "Research indicates that AI is perfect. Studies show definitive results."
        
        supported_metric = analyzer._assess_accuracy(supported_content, SectionType.ANALYSIS)
        unsupported_metric = analyzer._assess_accuracy(unsupported_content, SectionType.ANALYSIS)
        
        assert supported_metric.score >= unsupported_metric.score
    
    def test_engagement_assessment(self, analyzer):
        """Test engagement dimension assessment."""
        engaging_content = """Are you ready to discover machine learning? 
        For example, neural networks can recognize patterns. 
        You can implement these techniques in your projects."""
        
        boring_content = "Machine learning is a field. It has algorithms. They process data."
        
        engaging_metric = analyzer._assess_engagement_dimension(engaging_content, SectionType.INTRODUCTION)
        boring_metric = analyzer._assess_engagement_dimension(boring_content, SectionType.INTRODUCTION)
        
        assert engaging_metric.score > boring_metric.score
    
    def test_structure_assessment(self, analyzer):
        """Test structure dimension assessment."""
        well_structured = """# Main Topic
        
        This is the introduction paragraph.
        
        ## Subtopic
        
        - Point one
        - Point two
        - Point three
        
        Conclusion paragraph."""
        
        poorly_structured = "This is just one long paragraph without any structure or organization at all."
        
        good_metric = analyzer._assess_structure(well_structured, SectionType.TUTORIAL)
        poor_metric = analyzer._assess_structure(poorly_structured, SectionType.TUTORIAL)
        
        assert good_metric.score > poor_metric.score
    
    def test_readability_calculation(self, analyzer):
        """Test readability score calculation."""
        # Simple, readable text
        simple_text = "This is easy. Read this text. It is clear."
        
        # Complex, difficult text  
        complex_text = "Utilizing sophisticated methodologies and implementing comprehensive analytical frameworks."
        
        simple_score = analyzer._calculate_readability(simple_text)
        complex_score = analyzer._calculate_readability(complex_text)
        
        assert simple_score > complex_score
        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
    
    def test_syllable_counting(self, analyzer):
        """Test syllable counting functionality."""
        test_words = [
            ("cat", 1),
            ("hello", 2),
            ("computer", 3),
            ("information", 4),
            ("communication", 5)
        ]
        
        for word, expected_syllables in test_words:
            actual_syllables = analyzer._count_syllables(word)
            # Allow some tolerance for heuristic-based counting
            assert abs(actual_syllables - expected_syllables) <= 1


class TestSectionAwareQualitySystem:
    """Test SectionAwareQualitySystem functionality."""
    
    @pytest.fixture
    def quality_system(self):
        """Create quality system instance."""
        return SectionAwareQualitySystem()
    
    @pytest.fixture
    def sample_newsletter_content(self):
        """Sample newsletter content with sections."""
        return """# Introduction
Welcome to our AI newsletter. This week we explore machine learning advances.

# Latest News
- OpenAI released new model
- Google announced research breakthrough  
- Microsoft expanded AI services

# Technical Analysis
The recent developments show significant progress in transformer architectures.
New attention mechanisms improve efficiency by 30%.
These advances enable better natural language understanding.

# Conclusion
In summary, AI continues rapid advancement. Stay tuned for next week's updates.
Try implementing these new techniques in your projects.
"""
    
    def test_system_initialization(self, quality_system):
        """Test quality system initialization."""
        assert isinstance(quality_system.analyzer, SectionQualityAnalyzer)
        assert len(quality_system.quality_thresholds) == 5
        assert quality_system.quality_thresholds[SectionType.INTRODUCTION] == 0.8
    
    def test_analyze_newsletter_quality(self, quality_system, sample_newsletter_content):
        """Test complete newsletter quality analysis."""
        context = {
            'topic': 'AI Newsletter',
            'audience': 'AI Researchers',
            'content_focus': 'Machine Learning'
        }
        
        report = quality_system.analyze_newsletter_quality(sample_newsletter_content, context=context)
        
        assert isinstance(report, AggregatedQualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.section_scores) > 0
        assert report.total_word_count > 0
        assert 0.0 <= report.average_readability <= 1.0
        assert 0.0 <= report.consistency_score <= 1.0
        assert 0.0 <= report.flow_score <= 1.0
    
    def test_validate_section_thresholds(self, quality_system):
        """Test section threshold validation."""
        # Create mock quality report
        section_scores = {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Good intro",
                overall_score=0.85  # Above threshold
            ),
            SectionType.ANALYSIS: SectionQualityMetrics(
                section_type=SectionType.ANALYSIS,
                section_content="Poor analysis",
                overall_score=0.70  # Below threshold (0.85)
            )
        }
        
        report = AggregatedQualityReport(
            overall_score=0.75,
            section_scores=section_scores,
            total_word_count=300,
            average_readability=0.7,
            consistency_score=0.8,
            flow_score=0.75
        )
        
        all_passed, issues = quality_system.validate_section_thresholds(report)
        
        assert not all_passed  # Should fail due to analysis section
        assert len(issues) == 1
        assert "analysis" in issues[0].lower()
    
    def test_get_improvement_recommendations(self, quality_system):
        """Test improvement recommendation generation."""
        # Create quality report with various issues
        metrics_with_issues = {
            QualityDimension.CLARITY: QualityMetric(
                dimension=QualityDimension.CLARITY,
                score=0.6,
                suggestions=["Improve sentence clarity", "Simplify language"]
            )
        }
        
        section_scores = {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Intro",
                overall_score=0.70,  # Below threshold
                metrics=metrics_with_issues
            )
        }
        
        report = AggregatedQualityReport(
            overall_score=0.70,  # Below 0.8 threshold
            section_scores=section_scores,
            total_word_count=200,
            average_readability=0.6,
            consistency_score=0.7,  # Below 0.8 threshold
            flow_score=0.7  # Below 0.8 threshold
        )
        
        recommendations = quality_system.get_improvement_recommendations(report)
        
        assert len(recommendations) > 0
        # Should include global recommendations
        assert any("overall quality" in rec.lower() for rec in recommendations)
        assert any("consistency" in rec.lower() for rec in recommendations)
        assert any("flow" in rec.lower() for rec in recommendations)
        # Should include section-specific recommendations
        assert any("introduction" in rec.lower() for rec in recommendations)
    
    def test_cross_section_consistency_calculation(self, quality_system):
        """Test cross-section consistency calculation."""
        # Consistent section scores
        consistent_scores = {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Intro",
                overall_score=0.80
            ),
            SectionType.ANALYSIS: SectionQualityMetrics(
                section_type=SectionType.ANALYSIS,
                section_content="Analysis",
                overall_score=0.82
            ),
            SectionType.CONCLUSION: SectionQualityMetrics(
                section_type=SectionType.CONCLUSION,
                section_content="Conclusion",
                overall_score=0.81
            )
        }
        
        # Inconsistent section scores
        inconsistent_scores = {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Intro",
                overall_score=0.90
            ),
            SectionType.ANALYSIS: SectionQualityMetrics(
                section_type=SectionType.ANALYSIS,
                section_content="Analysis",
                overall_score=0.50
            )
        }
        
        consistent_score = quality_system._calculate_cross_section_consistency(consistent_scores)
        inconsistent_score = quality_system._calculate_cross_section_consistency(inconsistent_scores)
        
        assert consistent_score > inconsistent_score
        assert 0.0 <= consistent_score <= 1.0
        assert 0.0 <= inconsistent_score <= 1.0
    
    def test_narrative_flow_calculation(self, quality_system):
        """Test narrative flow score calculation."""
        section_scores = {
            SectionType.INTRODUCTION: SectionQualityMetrics(
                section_type=SectionType.INTRODUCTION,
                section_content="Intro",
                overall_score=0.8
            )
        }
        
        # Content with good transitions
        good_flow_content = """Introduction section.

However, the analysis shows different results.

Furthermore, the conclusion reinforces this point."""
        
        # Content without transitions
        poor_flow_content = """Introduction section.

Analysis section starts abruptly.

Conclusion appears suddenly."""
        
        good_flow_score = quality_system._calculate_narrative_flow_score(section_scores, good_flow_content)
        poor_flow_score = quality_system._calculate_narrative_flow_score(section_scores, poor_flow_content)
        
        assert good_flow_score >= poor_flow_score
        assert 0.0 <= good_flow_score <= 1.0
        assert 0.0 <= poor_flow_score <= 1.0


class TestIntegration:
    """Integration tests for the complete quality metrics system."""
    
    def test_end_to_end_quality_analysis(self):
        """Test complete quality analysis workflow."""
        quality_system = SectionAwareQualitySystem()
        
        newsletter_content = """# Introduction
Welcome to our comprehensive AI newsletter! What exciting developments await us this week?

# Latest News
- OpenAI announced GPT-5 with revolutionary capabilities
- Google released new research on transformer efficiency
- Microsoft expanded Azure AI services globally

# Technical Analysis
The recent breakthroughs in large language models demonstrate significant progress.
Research data indicates 40% improvement in reasoning capabilities.
These developments have important implications for AI applications.

# How-To Guide
Step 1: Install the latest AI development tools
Step 2: Configure your development environment
Step 3: Run your first AI model

# Conclusion
In summary, we've covered major AI advances this week.
Try implementing these new techniques in your next project.
Stay tuned for more exciting updates next week!
"""
        
        context = {
            'topic': 'Artificial Intelligence Weekly',
            'audience': 'AI/ML Engineers',
            'content_focus': 'Latest AI Developments',
            'word_count': 3000
        }
        
        # Analyze complete newsletter
        report = quality_system.analyze_newsletter_quality(newsletter_content, context=context)
        
        # Validate results
        assert isinstance(report, AggregatedQualityReport)
        assert report.overall_score > 0.0
        assert len(report.section_scores) >= 4  # Should detect multiple sections
        assert report.total_word_count > 100
        
        # Validate thresholds
        all_passed, issues = quality_system.validate_section_thresholds(report)
        assert isinstance(all_passed, bool)
        assert isinstance(issues, list)
        
        # Get recommendations
        recommendations = quality_system.get_improvement_recommendations(report)
        assert isinstance(recommendations, list)
        
        # Check quality summary
        summary = report.get_quality_summary()
        assert 'overall_score' in summary
        assert 'section_breakdown' in summary
    
    def test_quality_threshold_customization(self):
        """Test custom quality thresholds."""
        custom_thresholds = {
            SectionType.INTRODUCTION: 0.9,  # Very high threshold
            SectionType.ANALYSIS: 0.95,     # Very high threshold
            SectionType.CONCLUSION: 0.5     # Low threshold
        }
        
        quality_system = SectionAwareQualitySystem(quality_thresholds=custom_thresholds)
        
        assert quality_system.quality_thresholds[SectionType.INTRODUCTION] == 0.9
        assert quality_system.quality_thresholds[SectionType.ANALYSIS] == 0.95
        assert quality_system.quality_thresholds[SectionType.CONCLUSION] == 0.5
    
    def test_boundary_detection_integration(self):
        """Test integration with section boundary detection."""
        quality_system = SectionAwareQualitySystem()
        
        # Content without explicit boundaries 
        implicit_content = """This newsletter covers AI developments.
        Recent research shows promising results.
        Implementation requires careful planning.
        Thank you for reading our update."""
        
        context = {'topic': 'AI', 'audience': 'Engineers', 'content_focus': 'Development'}
        
        # Should handle content without explicit boundaries
        report = quality_system.analyze_newsletter_quality(implicit_content, context=context)
        
        assert isinstance(report, AggregatedQualityReport)
        assert len(report.section_scores) > 0
        assert report.overall_score >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])