"""
Test Suite for Enhanced Continuity Validation System

Comprehensive tests for the cross-section coherence validation including:
- Transition analysis between sections
- Style consistency checking
- Content redundancy detection
- Continuity issue identification
- Comprehensive continuity reporting
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.continuity_validator import (
    ContinuityIssue,
    ContinuityIssueType,
    ContinuityReport,
    ContinuityValidator,
    RedundancyDetector,
    StyleAnalysis,
    StyleConsistencyAnalyzer,
    TransitionAnalysis,
    TransitionAnalyzer,
)
from src.core.section_aware_prompts import SectionType


class TestContinuityIssueType:
    """Test ContinuityIssueType enum."""
    
    def test_continuity_issue_types(self):
        """Test that all expected issue types exist."""
        expected_types = {
            'abrupt_transition', 'style_inconsistency', 'content_redundancy',
            'logical_gap', 'tone_shift', 'reference_mismatch', 'structural_break'
        }
        actual_types = {issue.value for issue in ContinuityIssueType}
        assert actual_types == expected_types


class TestContinuityIssue:
    """Test ContinuityIssue dataclass."""
    
    def test_continuity_issue_creation(self):
        """Test creating ContinuityIssue instance."""
        issue = ContinuityIssue(
            issue_type=ContinuityIssueType.ABRUPT_TRANSITION,
            severity=0.7,
            location="Between introduction and analysis",
            description="Sections lack connecting language",
            suggestion="Add transitional phrases",
            section_1=SectionType.INTRODUCTION,
            section_2=SectionType.ANALYSIS,
            content_snippet="...end of intro\n\nStart of analysis..."
        )
        
        assert issue.issue_type == ContinuityIssueType.ABRUPT_TRANSITION
        assert issue.severity == 0.7
        assert issue.location == "Between introduction and analysis"
        assert issue.section_1 == SectionType.INTRODUCTION
        assert issue.section_2 == SectionType.ANALYSIS
    
    def test_continuity_issue_defaults(self):
        """Test default values in ContinuityIssue."""
        issue = ContinuityIssue(
            issue_type=ContinuityIssueType.STYLE_INCONSISTENCY,
            severity=0.5,
            location="Section 1",
            description="Style issue",
            suggestion="Fix style"
        )
        
        assert issue.section_1 is None
        assert issue.section_2 is None
        assert issue.content_snippet == ""


class TestTransitionAnalysis:
    """Test TransitionAnalysis dataclass."""
    
    def test_transition_analysis_creation(self):
        """Test creating TransitionAnalysis instance."""
        analysis = TransitionAnalysis(
            from_section=SectionType.INTRODUCTION,
            to_section=SectionType.NEWS,
            transition_quality=0.75,
            has_explicit_transition=True,
            transition_text="...intro ends. Meanwhile, news section begins...",
            issues=[],
            suggestions=["Good transition detected"]
        )
        
        assert analysis.from_section == SectionType.INTRODUCTION
        assert analysis.to_section == SectionType.NEWS
        assert analysis.transition_quality == 0.75
        assert analysis.has_explicit_transition is True
    
    def test_transition_analysis_defaults(self):
        """Test default values in TransitionAnalysis."""
        analysis = TransitionAnalysis(
            from_section=SectionType.ANALYSIS,
            to_section=SectionType.CONCLUSION,
            transition_quality=0.5,
            has_explicit_transition=False,
            transition_text="Basic transition"
        )
        
        assert analysis.issues == []
        assert analysis.suggestions == []


class TestTransitionAnalyzer:
    """Test TransitionAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create transition analyzer instance."""
        return TransitionAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.transition_indicators) == 6  # 6 categories
        assert 'continuation' in analyzer.transition_indicators
        assert 'contrast' in analyzer.transition_indicators
        assert len(analyzer.expected_transitions) > 0
    
    def test_analyze_transition_with_explicit_indicator(self, analyzer):
        """Test transition analysis with explicit transition words."""
        from_content = "This concludes our introduction to machine learning."
        to_content = "However, the implementation challenges are significant."
        
        analysis = analyzer.analyze_transition(
            from_content, SectionType.INTRODUCTION,
            to_content, SectionType.ANALYSIS
        )
        
        assert isinstance(analysis, TransitionAnalysis)
        assert analysis.from_section == SectionType.INTRODUCTION
        assert analysis.to_section == SectionType.ANALYSIS
        assert analysis.has_explicit_transition is True
        assert analysis.transition_quality > 0.5  # Should be good with "However"
    
    def test_analyze_transition_without_indicator(self, analyzer):
        """Test transition analysis without explicit transition words."""
        from_content = "Machine learning is important."
        to_content = "Neural networks process data efficiently."
        
        analysis = analyzer.analyze_transition(
            from_content, SectionType.INTRODUCTION,
            to_content, SectionType.TUTORIAL
        )
        
        assert analysis.has_explicit_transition is False
        assert analysis.transition_quality < 0.5  # Should be poor without transitions
        assert len(analysis.issues) > 0  # Should identify issues
    
    def test_has_explicit_transition(self, analyzer):
        """Test explicit transition detection."""
        # Text with transition words
        with_transition = "Previous section ended. Furthermore, we need to consider..."
        assert analyzer._has_explicit_transition(with_transition) is True
        
        # Text without transition words
        without_transition = "Previous section ended. We need to consider..."
        assert analyzer._has_explicit_transition(without_transition) is False
    
    def test_calculate_transition_quality(self, analyzer):
        """Test transition quality calculation."""
        # Good transition with expected phrases
        from_end = "This analysis reveals important insights."
        to_start = "These developments indicate future trends."
        
        quality_good = analyzer._calculate_transition_quality(
            from_end, to_start, SectionType.ANALYSIS, SectionType.NEWS
        )
        
        # Poor transition without connections
        from_end_poor = "Random content here."
        to_start_poor = "Completely different topic."
        
        quality_poor = analyzer._calculate_transition_quality(
            from_end_poor, to_start_poor, SectionType.INTRODUCTION, SectionType.CONCLUSION
        )
        
        assert 0.0 <= quality_good <= 1.0
        assert 0.0 <= quality_poor <= 1.0
        assert quality_good >= quality_poor
    
    def test_identify_transition_issues(self, analyzer):
        """Test transition issue identification."""
        # Abrupt transition case
        from_end = "Introduction content ends"  # No punctuation
        to_start = "analysis begins immediately"  # Lowercase start
        
        issues = analyzer._identify_transition_issues(
            from_end, to_start, SectionType.INTRODUCTION, SectionType.ANALYSIS
        )
        
        assert len(issues) > 0
        assert any(issue.issue_type == ContinuityIssueType.ABRUPT_TRANSITION for issue in issues)
    
    def test_generate_transition_suggestions(self, analyzer):
        """Test transition suggestion generation."""
        issues = [
            ContinuityIssue(
                issue_type=ContinuityIssueType.LOGICAL_GAP,
                severity=0.5,
                location="test",
                description="test",
                suggestion="test"
            )
        ]
        
        suggestions = analyzer._generate_transition_suggestions(
            SectionType.INTRODUCTION, SectionType.ANALYSIS, False, issues
        )
        
        assert len(suggestions) > 0
        assert any("transition" in suggestion.lower() for suggestion in suggestions)
    
    def test_logical_gap_detection(self, analyzer):
        """Test logical gap detection."""
        # Case with unresolved question
        from_end = "What are the implications of this research?"
        to_start = "Neural networks are complex systems."
        
        has_gap = analyzer._has_logical_gap(from_end, to_start)
        assert has_gap is True
        
        # Case without logical gap
        from_end_good = "This research shows important results."
        to_start_good = "These findings indicate future directions."
        
        has_gap_good = analyzer._has_logical_gap(from_end_good, to_start_good)
        assert has_gap_good is False


class TestStyleConsistencyAnalyzer:
    """Test StyleConsistencyAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create style consistency analyzer."""
        return StyleConsistencyAnalyzer()
    
    @pytest.fixture
    def sample_sections(self):
        """Sample sections with different styles."""
        return {
            SectionType.INTRODUCTION: "Welcome! This is exciting stuff. You'll love it.",
            SectionType.ANALYSIS: "Therefore, the data indicates significant correlations. Furthermore, the methodology demonstrates validity.",
            SectionType.CONCLUSION: "So basically, that's the main takeaway. Pretty cool results overall."
        }
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.formal_indicators) > 0
        assert len(analyzer.informal_indicators) > 0
        assert len(analyzer.technical_indicators) > 0
    
    def test_analyze_style_consistency(self, analyzer, sample_sections):
        """Test complete style consistency analysis."""
        analysis = analyzer.analyze_style_consistency(sample_sections)
        
        assert isinstance(analysis, StyleAnalysis)
        assert 0.0 <= analysis.overall_consistency <= 1.0
        assert 0.0 <= analysis.tone_consistency <= 1.0
        assert 0.0 <= analysis.vocabulary_consistency <= 1.0
        assert 0.0 <= analysis.structure_consistency <= 1.0
    
    def test_single_section_consistency(self, analyzer):
        """Test consistency analysis with single section."""
        single_section = {SectionType.INTRODUCTION: "Single section content"}
        
        analysis = analyzer.analyze_style_consistency(single_section)
        
        # Should return perfect consistency for single section
        assert analysis.overall_consistency == 1.0
        assert analysis.tone_consistency == 1.0
    
    def test_create_style_profile(self, analyzer):
        """Test style profile creation."""
        content = """Therefore, this technical analysis demonstrates significant findings. 
        Furthermore, the implementation requires careful consideration.
        You should explore these approaches."""
        
        profile = analyzer._create_style_profile(content)
        
        assert 'formality_score' in profile
        assert 'technical_density' in profile
        assert 'avg_sentence_length' in profile
        assert 'personal_pronoun_ratio' in profile
        assert 'word_count' in profile
        assert profile['word_count'] > 0
    
    def test_tone_consistency_calculation(self, analyzer):
        """Test tone consistency calculation."""
        # Consistent formal tone
        consistent_profiles = {
            SectionType.INTRODUCTION: {
                'formality_score': 0.1,
                'personal_pronoun_ratio': 0.02
            },
            SectionType.ANALYSIS: {
                'formality_score': 0.12,
                'personal_pronoun_ratio': 0.01
            }
        }
        
        # Inconsistent tone (formal vs informal)
        inconsistent_profiles = {
            SectionType.INTRODUCTION: {
                'formality_score': 0.2,   # Formal
                'personal_pronoun_ratio': 0.01
            },
            SectionType.ANALYSIS: {
                'formality_score': -0.1,  # Informal
                'personal_pronoun_ratio': 0.1
            }
        }
        
        consistent_score = analyzer._calculate_tone_consistency(consistent_profiles)
        inconsistent_score = analyzer._calculate_tone_consistency(inconsistent_profiles)
        
        assert consistent_score > inconsistent_score
        assert 0.0 <= consistent_score <= 1.0
        assert 0.0 <= inconsistent_score <= 1.0
    
    def test_identify_style_issues(self, analyzer):
        """Test style issue identification."""
        inconsistent_profiles = {
            SectionType.INTRODUCTION: {
                'formality_score': 0.1,  # More formal
            },
            SectionType.CONCLUSION: {
                'formality_score': -0.1,  # More informal
            }
        }
        
        sections = {
            SectionType.INTRODUCTION: "Formal content",
            SectionType.CONCLUSION: "Informal content"
        }
        
        issues = analyzer._identify_style_issues(inconsistent_profiles, sections)
        
        # Should detect tone shifts
        tone_issues = [issue for issue in issues if issue.issue_type == ContinuityIssueType.TONE_SHIFT]
        assert len(tone_issues) > 0
    
    def test_variance_calculation(self, analyzer):
        """Test variance calculation utility."""
        # No variance
        identical_values = [0.5, 0.5, 0.5]
        assert analyzer._calculate_variance(identical_values) == 0.0
        
        # Some variance
        varied_values = [0.1, 0.5, 0.9]
        variance = analyzer._calculate_variance(varied_values)
        assert variance > 0.0
        
        # Single value
        single_value = [0.5]
        assert analyzer._calculate_variance(single_value) == 0.0


class TestRedundancyDetector:
    """Test RedundancyDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create redundancy detector."""
        return RedundancyDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.similarity_threshold == 0.7
    
    def test_detect_redundancy_high_similarity(self, detector):
        """Test redundancy detection with high similarity."""
        redundant_sections = {
            SectionType.INTRODUCTION: "Machine learning algorithms process data efficiently using neural networks",
            SectionType.ANALYSIS: "Machine learning algorithms process data efficiently using neural networks"
        }
        
        issues = detector.detect_redundancy(redundant_sections)
        
        assert len(issues) > 0
        assert issues[0].issue_type == ContinuityIssueType.CONTENT_REDUNDANCY
        assert issues[0].severity > 0.7
    
    def test_detect_redundancy_low_similarity(self, detector):
        """Test redundancy detection with low similarity."""
        different_sections = {
            SectionType.INTRODUCTION: "Welcome to our newsletter about artificial intelligence",
            SectionType.CONCLUSION: "Thank you for reading about database optimization"
        }
        
        issues = detector.detect_redundancy(different_sections)
        
        # Should not detect redundancy
        assert len(issues) == 0
    
    def test_calculate_content_similarity(self, detector):
        """Test content similarity calculation."""
        # Identical content
        identical_1 = "machine learning algorithms"
        identical_2 = "machine learning algorithms"
        similarity_identical = detector._calculate_content_similarity(identical_1, identical_2)
        assert similarity_identical == 1.0
        
        # Completely different content
        different_1 = "machine learning algorithms"
        different_2 = "database optimization techniques"
        similarity_different = detector._calculate_content_similarity(different_1, different_2)
        assert similarity_different < 0.5
        
        # Partially similar content
        similar_1 = "machine learning algorithms for data processing"
        similar_2 = "neural network algorithms for data analysis"
        similarity_partial = detector._calculate_content_similarity(similar_1, similar_2)
        assert 0.0 < similarity_partial < 1.0
    
    def test_empty_content_similarity(self, detector):
        """Test similarity calculation with empty content."""
        # Both empty
        assert detector._calculate_content_similarity("", "") == 1.0
        
        # One empty
        assert detector._calculate_content_similarity("content", "") == 0.0
        assert detector._calculate_content_similarity("", "content") == 0.0


class TestContinuityValidator:
    """Test ContinuityValidator main functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create continuity validator."""
        return ContinuityValidator()
    
    @pytest.fixture
    def sample_sections(self):
        """Sample sections for testing."""
        return {
            SectionType.INTRODUCTION: """Welcome to our AI newsletter! 
            This week we explore machine learning advances and their implications.""",
            
            SectionType.NEWS: """However, recent developments show exciting progress.
            OpenAI announced new model capabilities.
            Google released research on efficiency improvements.""",
            
            SectionType.ANALYSIS: """Furthermore, these developments indicate significant trends.
            The data shows 40% improvement in processing speed.
            Technical analysis reveals architectural innovations.""",
            
            SectionType.CONCLUSION: """In summary, AI continues rapid advancement.
            Try implementing these techniques in your projects.
            Stay tuned for next week's updates."""
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator.transition_analyzer, TransitionAnalyzer)
        assert isinstance(validator.style_analyzer, StyleConsistencyAnalyzer)
        assert isinstance(validator.redundancy_detector, RedundancyDetector)
    
    def test_validate_newsletter_continuity(self, validator, sample_sections):
        """Test complete newsletter continuity validation."""
        context = {
            'topic': 'AI Newsletter',
            'audience': 'Tech Professionals'
        }
        
        report = validator.validate_newsletter_continuity(sample_sections, context)
        
        assert isinstance(report, ContinuityReport)
        assert 0.0 <= report.overall_continuity_score <= 1.0
        assert 0.0 <= report.narrative_flow_score <= 1.0
        assert 0.0 <= report.style_consistency_score <= 1.0
        assert 0.0 <= report.transition_quality_score <= 1.0
        assert 0.0 <= report.redundancy_score <= 1.0
        assert report.sections_analyzed == len(sample_sections)
        assert len(report.transitions_analyzed) == len(sample_sections) - 1
        assert isinstance(report.style_analysis, StyleAnalysis)
        assert isinstance(report.issues, list)
        assert isinstance(report.recommendations, list)
    
    def test_analyze_all_transitions(self, validator, sample_sections):
        """Test transition analysis for all section pairs."""
        transitions = validator._analyze_all_transitions(sample_sections)
        
        assert len(transitions) == len(sample_sections) - 1  # n-1 transitions for n sections
        
        for transition in transitions:
            assert isinstance(transition, TransitionAnalysis)
            assert transition.from_section in sample_sections
            assert transition.to_section in sample_sections
    
    def test_calculate_narrative_flow_score(self, validator):
        """Test narrative flow score calculation."""
        # Good transitions
        good_transitions = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.ANALYSIS,
                transition_quality=0.8,
                has_explicit_transition=True,
                transition_text="good transition"
            ),
            TransitionAnalysis(
                from_section=SectionType.ANALYSIS,
                to_section=SectionType.CONCLUSION,
                transition_quality=0.9,
                has_explicit_transition=True,
                transition_text="excellent transition"
            )
        ]
        
        # Poor transitions
        poor_transitions = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.CONCLUSION,
                transition_quality=0.2,
                has_explicit_transition=False,
                transition_text="poor transition"
            )
        ]
        
        good_score = validator._calculate_narrative_flow_score(good_transitions)
        poor_score = validator._calculate_narrative_flow_score(poor_transitions)
        
        assert good_score > poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0
    
    def test_calculate_transition_quality_score(self, validator):
        """Test transition quality score calculation."""
        transitions_with_explicit = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.ANALYSIS,
                transition_quality=0.8,
                has_explicit_transition=True,
                transition_text="with transition"
            ),
            TransitionAnalysis(
                from_section=SectionType.ANALYSIS,
                to_section=SectionType.CONCLUSION,
                transition_quality=0.6,
                has_explicit_transition=True,
                transition_text="with transition"
            )
        ]
        
        transitions_without_explicit = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.CONCLUSION,
                transition_quality=0.5,
                has_explicit_transition=False,
                transition_text="no transition"
            )
        ]
        
        score_with = validator._calculate_transition_quality_score(transitions_with_explicit)
        score_without = validator._calculate_transition_quality_score(transitions_without_explicit)
        
        assert score_with > score_without
        assert score_with == 1.0  # All have explicit transitions
        assert score_without == 0.0  # None have explicit transitions
    
    def test_generate_recommendations(self, validator):
        """Test recommendation generation."""
        # Create scenarios with various issues
        poor_transitions = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.ANALYSIS,
                transition_quality=0.3,  # Poor quality
                has_explicit_transition=False,
                transition_text="poor"
            )
        ]
        
        poor_style_analysis = StyleAnalysis(
            overall_consistency=0.6,  # Below 0.7 threshold
            tone_consistency=0.5,
            vocabulary_consistency=0.6,
            structure_consistency=0.7
        )
        
        redundancy_issues = [
            ContinuityIssue(
                issue_type=ContinuityIssueType.CONTENT_REDUNDANCY,
                severity=0.9,  # High redundancy
                location="test",
                description="test",
                suggestion="test"
            )
        ]
        
        recommendations = validator._generate_recommendations(
            poor_transitions, poor_style_analysis, redundancy_issues
        )
        
        assert len(recommendations) > 0
        # Should include transition recommendations
        assert any("transition" in rec.lower() for rec in recommendations)
        # Should include style recommendations
        assert any("tone" in rec.lower() for rec in recommendations)
        # Should include redundancy recommendations
        assert any("redundant" in rec.lower() for rec in recommendations)
    
    def test_empty_sections_handling(self, validator):
        """Test handling of empty sections."""
        empty_sections = {}
        
        report = validator.validate_newsletter_continuity(empty_sections)
        
        assert isinstance(report, ContinuityReport)
        assert report.sections_analyzed == 0
        assert len(report.transitions_analyzed) == 0
        assert report.overall_continuity_score >= 0.0
    
    def test_single_section_handling(self, validator):
        """Test handling of single section."""
        single_section = {
            SectionType.ANALYSIS: "This is a single analysis section."
        }
        
        report = validator.validate_newsletter_continuity(single_section)
        
        assert report.sections_analyzed == 1
        assert len(report.transitions_analyzed) == 0
        # Should have high scores for single section (no conflicts)
        assert report.style_consistency_score == 1.0
        assert report.redundancy_score == 1.0


class TestContinuityReport:
    """Test ContinuityReport dataclass functionality."""
    
    def test_continuity_report_creation(self):
        """Test creating comprehensive continuity report."""
        transitions = [
            TransitionAnalysis(
                from_section=SectionType.INTRODUCTION,
                to_section=SectionType.ANALYSIS,
                transition_quality=0.8,
                has_explicit_transition=True,
                transition_text="good transition"
            )
        ]
        
        style_analysis = StyleAnalysis(
            overall_consistency=0.8,
            tone_consistency=0.7,
            vocabulary_consistency=0.8,
            structure_consistency=0.9
        )
        
        issues = [
            ContinuityIssue(
                issue_type=ContinuityIssueType.ABRUPT_TRANSITION,
                severity=0.5,
                location="test",
                description="test issue",
                suggestion="test suggestion"
            )
        ]
        
        report = ContinuityReport(
            overall_continuity_score=0.85,
            narrative_flow_score=0.8,
            style_consistency_score=0.8,
            transition_quality_score=0.9,
            redundancy_score=0.95,
            sections_analyzed=3,
            transitions_analyzed=transitions,
            style_analysis=style_analysis,
            issues=issues,
            recommendations=["Improve transitions", "Enhance consistency"]
        )
        
        assert report.overall_continuity_score == 0.85
        assert len(report.transitions_analyzed) == 1
        assert len(report.issues) == 1
        assert len(report.recommendations) == 2


class TestIntegration:
    """Integration tests for the complete continuity validation system."""
    
    def test_end_to_end_continuity_validation(self):
        """Test complete continuity validation workflow."""
        validator = ContinuityValidator()
        
        newsletter_sections = {
            SectionType.INTRODUCTION: """Welcome to our comprehensive AI newsletter! 
            What exciting developments have emerged this week in artificial intelligence?
            This overview will explore cutting-edge research and practical applications.""",
            
            SectionType.NEWS: """Meanwhile, several major announcements have shaped the landscape.
            OpenAI released GPT-4.5 with enhanced reasoning capabilities.
            Google announced breakthrough research in quantum-classical hybrid models.
            Microsoft expanded Azure AI services to include new vision capabilities.""",
            
            SectionType.ANALYSIS: """Furthermore, these developments reveal significant trends.
            The data indicates 45% improvement in model efficiency across benchmarks.
            Technical analysis shows convergence toward multimodal architectures.
            Research methodologies demonstrate reproducible experimental frameworks.""",
            
            SectionType.TUTORIAL: """Next, let's explore practical implementation steps.
            Step 1: Install the latest AI development frameworks.
            Step 2: Configure your development environment for optimal performance.
            Step 3: Implement basic model training pipelines.""",
            
            SectionType.CONCLUSION: """In summary, these advances represent substantial progress.
            Try implementing these new techniques in your next project.
            Share your experiences with the community for collective learning.
            Stay tuned for next week's comprehensive update on emerging trends."""
        }
        
        context = {
            'topic': 'AI Weekly Newsletter',
            'audience': 'AI Researchers and Developers',
            'content_focus': 'Latest AI Developments'
        }
        
        # Perform complete validation
        report = validator.validate_newsletter_continuity(newsletter_sections, context)
        
        # Validate report structure
        assert isinstance(report, ContinuityReport)
        assert report.sections_analyzed == 5
        assert len(report.transitions_analyzed) == 4  # 4 transitions for 5 sections
        
        # Validate score ranges
        assert 0.0 <= report.overall_continuity_score <= 1.0
        assert 0.0 <= report.narrative_flow_score <= 1.0
        assert 0.0 <= report.style_consistency_score <= 1.0
        assert 0.0 <= report.transition_quality_score <= 1.0
        assert 0.0 <= report.redundancy_score <= 1.0
        
        # Should detect good transitions (content has "Meanwhile", "Furthermore", "Next")
        assert report.transition_quality_score > 0.5
        
        # Should have recommendations
        assert len(report.recommendations) >= 0
        
        # Validate individual transitions
        for transition in report.transitions_analyzed:
            assert isinstance(transition, TransitionAnalysis)
            assert transition.from_section in newsletter_sections
            assert transition.to_section in newsletter_sections
    
    def test_problematic_content_detection(self):
        """Test detection of various continuity problems."""
        validator = ContinuityValidator()
        
        problematic_sections = {
            SectionType.INTRODUCTION: """This newsletter covers AI topics""",  # Abrupt ending
            
            SectionType.ANALYSIS: """machine learning algorithms process data efficiently 
            using advanced neural network architectures""",  # Lowercase start, no transition
            
            SectionType.NEWS: """Therefore, the comprehensive analysis demonstrates significant 
            findings. Furthermore, the implementation requires careful consideration.""",  # Formal tone
            
            SectionType.CONCLUSION: """So basically, that's it. Pretty cool stuff overall."""  # Informal tone
        }
        
        report = validator.validate_newsletter_continuity(problematic_sections)
        
        # Should detect multiple issues
        assert len(report.issues) > 0
        
        # Should detect style inconsistency (formal vs informal)
        style_issues = [issue for issue in report.issues 
                      if issue.issue_type == ContinuityIssueType.TONE_SHIFT]
        assert len(style_issues) > 0
        
        # Should have recommendations for improvement
        assert len(report.recommendations) > 0
        
        # Should have lower consistency scores
        assert report.style_consistency_score < 0.8
    
    def test_high_quality_content_validation(self):
        """Test validation of high-quality, well-structured content."""
        validator = ContinuityValidator()
        
        high_quality_sections = {
            SectionType.INTRODUCTION: """Welcome to our weekly AI research digest. 
            This comprehensive overview examines recent breakthroughs in machine learning 
            and their implications for practical applications.""",
            
            SectionType.ANALYSIS: """Building on these foundations, our analysis reveals 
            significant performance improvements across multiple benchmarks. 
            The research demonstrates consistent 30% efficiency gains in processing.""",
            
            SectionType.CONCLUSION: """In conclusion, these developments represent substantial 
            progress in artificial intelligence capabilities. 
            We recommend exploring these techniques for enhanced performance."""
        }
        
        report = validator.validate_newsletter_continuity(high_quality_sections)
        
        # Should have high overall scores
        assert report.overall_continuity_score > 0.7
        assert report.style_consistency_score > 0.7
        
        # Should have fewer issues
        assert len(report.issues) <= 2
        
        # Should detect good transitions
        assert report.transition_quality_score > 0.5


if __name__ == '__main__':
    pytest.main([__file__])