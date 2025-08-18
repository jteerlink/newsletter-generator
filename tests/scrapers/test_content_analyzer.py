# tests/scrapers/test_content_analyzer.py
from datetime import datetime, timezone

import pytest

from src.scrapers.content_analyzer import ContentAnalyzer


class TestContentAnalyzer:
    @pytest.fixture
    def analyzer(self):
        """Create a ContentAnalyzer instance for testing."""
        return ContentAnalyzer()

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "url": "https://arxiv.org/abs/2023.12345",
            "source": "arXiv",
            "title": "Advances in Machine Learning",
            "description": "This paper presents new developments in artificial intelligence and deep learning.",
            "published": datetime.now(timezone.utc).isoformat(),
            "author": "John Doe",
            "category": "Research",
            "tags": ["AI", "ML", "research"],
        }

    def test_analyze_content_basic(self, analyzer, sample_metadata):
        """Test basic content analysis functionality."""
        content = "This is a comprehensive study of machine learning algorithms and their applications in artificial intelligence."
        analysis = analyzer.analyze_content(content, sample_metadata)

        # Check basic analysis results
        assert "quality_score" in analysis
        assert "category" in analysis
        assert "tags" in analysis
        assert "word_count" in analysis
        assert "content_hash" in analysis
        assert isinstance(analysis["quality_score"], float)
        assert 0.0 <= analysis["quality_score"] <= 1.0
        assert analysis["word_count"] > 0
        assert len(analysis["content_hash"]) > 0

    def test_categorization(self, analyzer):
        """Test content categorization."""
        # AI/ML content
        ai_content = "Deep learning and neural networks are transforming artificial intelligence."
        ai_metadata = {"url": "https://example.com", "source": "Test"}
        ai_analysis = analyzer.analyze_content(ai_content, ai_metadata)
        assert ai_analysis["category"] == "AI/ML"

        # Research content
        research_content = (
            "This research paper presents findings from our scientific study."
        )
        research_analysis = analyzer.analyze_content(research_content, ai_metadata)
        assert research_analysis["category"] == "Research"

        # Technology content
        tech_content = "New software technology platform for system integration."
        tech_analysis = analyzer.analyze_content(tech_content, ai_metadata)
        assert tech_analysis["category"] == "Technology"

    def test_quality_scoring(self, analyzer, sample_metadata):
        """Test quality scoring functionality."""
        # High-quality content
        high_quality_content = """
        This comprehensive research paper presents groundbreaking findings in machine learning.
        The study involved extensive experimentation with deep neural networks and provides
        detailed analysis of results. Our methodology follows established scientific principles
        and includes thorough validation procedures.
        """
        high_analysis = analyzer.analyze_content(high_quality_content, sample_metadata)

        # Low-quality content
        low_quality_content = "Short content."
        low_analysis = analyzer.analyze_content(low_quality_content, sample_metadata)

        # The low-quality score should be less than the high-quality score
        assert low_analysis["quality_score"] < high_analysis["quality_score"]

    def test_source_reliability(self, analyzer):
        """Test source reliability assessment."""
        # High-reliability source
        arxiv_metadata = {"url": "https://arxiv.org/abs/2023.12345", "source": "arXiv"}
        arxiv_analysis = analyzer.analyze_content("Test content", arxiv_metadata)
        assert arxiv_analysis["source_reliability"] > 0.9

        # Medium-reliability source
        medium_metadata = {"url": "https://medium.com/article", "source": "Medium"}
        medium_analysis = analyzer.analyze_content("Test content", medium_metadata)
        assert 0.5 < medium_analysis["source_reliability"] < 0.8

        # Unknown source
        unknown_metadata = {"url": "https://unknown-site.com", "source": "Unknown"}
        unknown_analysis = analyzer.analyze_content("Test content", unknown_metadata)
        assert unknown_analysis["source_reliability"] == 0.5

    def test_freshness_scoring(self, analyzer):
        """Test freshness scoring."""
        # Recent content
        recent_metadata = {
            "url": "https://example.com",
            "source": "Test",
            "published": datetime.now(timezone.utc).isoformat(),
        }
        recent_analysis = analyzer.analyze_content("Test content", recent_metadata)
        assert recent_analysis["freshness_score"] > 0.9

        # Old content
        old_metadata = {
            "url": "https://example.com",
            "source": "Test",
            "published": "2020-01-01T00:00:00Z",
        }
        old_analysis = analyzer.analyze_content("Test content", old_metadata)
        assert old_analysis["freshness_score"] < 0.5

    def test_completeness_scoring(self, analyzer):
        """Test completeness scoring."""
        # Complete content
        complete_metadata = {
            "url": "https://example.com",
            "source": "Test",
            "title": "Complete Article",
            "description": "Detailed description",
            "author": "Author Name",
        }
        complete_content = "This is a comprehensive article with substantial content and detailed information."
        complete_analysis = analyzer.analyze_content(
            complete_content, complete_metadata
        )
        assert complete_analysis["completeness_score"] > 0.7

        # Incomplete content
        incomplete_metadata = {"url": "https://example.com", "source": "Test"}
        incomplete_content = "Short."
        incomplete_analysis = analyzer.analyze_content(
            incomplete_content, incomplete_metadata
        )
        assert incomplete_analysis["completeness_score"] < 0.5

    def test_duplicate_detection(self, analyzer):
        """Test duplicate detection functionality."""
        content = "This is unique content for testing duplicate detection."
        content_hash = analyzer._generate_content_hash(content)

        # Test with empty existing hashes
        assert not analyzer.detect_duplicates(content_hash, [])

        # Test with existing hash
        existing_hashes = [content_hash, "other_hash"]
        assert analyzer.detect_duplicates(content_hash, existing_hashes)

        # Test with different hash
        different_hash = "different_hash"
        assert not analyzer.detect_duplicates(different_hash, existing_hashes)

    def test_tag_extraction(self, analyzer, sample_metadata):
        """Test tag extraction functionality."""
        content = "This article discusses machine learning, artificial intelligence, and deep learning algorithms."
        analysis = analyzer.analyze_content(content, sample_metadata)

        # Should extract relevant tags
        tags = analysis["tags"]
        assert len(tags) > 0
        assert any(
            "machine learning" in tag.lower()
            or "artificial intelligence" in tag.lower()
            for tag in tags
        )

    def test_readability_calculation(self, analyzer):
        """Test readability calculation."""
        # Simple, readable content
        simple_content = "This is simple content. It has short sentences. Easy to read."
        simple_readability = analyzer._calculate_readability(simple_content)
        assert simple_readability > 0.6

        # Complex content
        complex_content = "This extraordinarily complex content contains numerous polysyllabic words and intricate sentence structures that challenge comprehension."
        complex_readability = analyzer._calculate_readability(complex_content)
        assert complex_readability < simple_readability

    def test_word_count(self, analyzer):
        """Test word counting functionality."""
        content = "This is a test content with multiple words."
        word_count = analyzer._count_words(content)
        assert word_count == 8

        # Empty content
        empty_count = analyzer._count_words("")
        assert empty_count == 0

        # Content with special characters
        special_content = "Word1, word2! word3? word4."
        special_count = analyzer._count_words(special_content)
        assert special_count == 4

    def test_metadata_enrichment(self, analyzer, sample_metadata):
        """Test metadata enrichment functionality."""
        content = "Test content for metadata enrichment."
        analysis = analyzer.analyze_content(content, sample_metadata)
        enriched_metadata = analysis["enriched_metadata"]

        # Check that original metadata is preserved
        assert enriched_metadata["url"] == sample_metadata["url"]
        assert enriched_metadata["title"] == sample_metadata["title"]

        # Check that analysis results are added
        assert "quality_score" in enriched_metadata
        assert "category" in enriched_metadata
        assert "tags" in enriched_metadata
        assert "analyzed_at" in enriched_metadata

    def test_filter_by_quality(self, analyzer):
        """Test quality filtering functionality."""
        articles = [
            {"quality_score": 0.8, "title": "High Quality"},
            {"quality_score": 0.5, "title": "Medium Quality"},
            {"quality_score": 0.2, "title": "Low Quality"},
        ]

        # Filter by minimum quality
        high_quality = analyzer.filter_by_quality(articles, min_quality=0.7)
        assert len(high_quality) == 1
        assert high_quality[0]["title"] == "High Quality"

        # Filter by lower threshold
        medium_plus = analyzer.filter_by_quality(articles, min_quality=0.4)
        assert len(medium_plus) == 2

    def test_sort_by_relevance(self, analyzer):
        """Test relevance sorting functionality."""
        articles = [
            {
                "quality_score": 0.8,
                "freshness_score": 0.9,
                "title": "High Quality, Fresh",
            },
            {
                "quality_score": 0.9,
                "freshness_score": 0.5,
                "title": "High Quality, Old",
            },
            {
                "quality_score": 0.5,
                "freshness_score": 0.8,
                "title": "Medium Quality, Fresh",
            },
        ]

        sorted_articles = analyzer.sort_by_relevance(articles)

        # Should be sorted by relevance (quality * 0.7 + freshness * 0.3)
        assert len(sorted_articles) == 3
        # First should be high quality, fresh
        assert sorted_articles[0]["title"] == "High Quality, Fresh"
