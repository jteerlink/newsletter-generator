"""
Tests for Phase 3 Tool Integration

This module tests the grammar linter and enhanced search tools
implemented in Phase 3 of the enhanced agent architecture.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.grammar_linter import GrammarAndStyleLinter, LinterResult, GrammarIssue, StyleIssue
from tools.enhanced_search import EnhancedSearchTool, SearchResult, SearchQuery

class TestGrammarAndStyleLinter:
    """Test cases for the GrammarAndStyleLinter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.linter = GrammarAndStyleLinter()
    
    def test_initialization(self):
        """Test linter initialization."""
        assert self.linter is not None
        assert hasattr(self.linter, 'grammar_rules')
        assert hasattr(self.linter, 'style_rules')
        assert hasattr(self.linter, 'severity_mapping')
    
    def test_check_content_basic(self):
        """Test basic content checking."""
        content = "This is a test sentence. It has some basic content."
        result = self.linter.check_content(content)
        
        assert isinstance(result, LinterResult)
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'grammar_score')
        assert hasattr(result, 'style_score')
        assert hasattr(result, 'total_issues')
        assert hasattr(result, 'suggestions')
        assert hasattr(result, 'summary')
    
    def test_check_content_with_issues(self):
        """Test content checking with known issues."""
        content = "This sentence has  double  spaces.  It also has repetitive words words."
        result = self.linter.check_content(content)
        
        assert result.total_issues > 0
        assert result.overall_score < 1.0
    
    def test_check_content_empty(self):
        """Test content checking with empty content."""
        result = self.linter.check_content("")
        
        assert result.overall_score == 0.0
        assert result.total_issues == 0
    
    def test_get_quick_feedback(self):
        """Test quick feedback functionality."""
        content = "This is a test sentence with some issues."
        feedback = self.linter.get_quick_feedback(content)
        
        assert isinstance(feedback, dict)
        assert 'score' in feedback
        assert 'grammar_score' in feedback
        assert 'style_score' in feedback
        assert 'total_issues' in feedback
        assert 'suggestions' in feedback
        assert 'needs_revision' in feedback
    
    def test_style_rules_detection(self):
        """Test detection of style issues."""
        content = "The project was optimized to leverage synergies and think outside the box."
        result = self.linter.check_content(content)
        
        # Should detect jargon and clichés
        style_issues = [issue for issue in result.style_issues if issue.category == 'style']
        assert len(style_issues) > 0
    
    def test_grammar_rules_detection(self):
        """Test detection of grammar issues."""
        content = "This sentence has  double  spaces  and its missing punctuation"
        result = self.linter.check_content(content)
        
        # Should detect double spaces and missing punctuation
        grammar_issues = [issue for issue in result.grammar_issues if issue.category == 'grammar']
        assert len(grammar_issues) > 0
    
    def test_score_calculation(self):
        """Test score calculation logic."""
        # Perfect content should have high scores
        perfect_content = "This is a well-written sentence. It has proper grammar and style."
        result = self.linter.check_content(perfect_content)
        
        assert result.overall_score > 0.8
        assert result.grammar_score > 0.8
        assert result.style_score > 0.8
    
    def test_error_handling(self):
        """Test error handling in content checking."""
        with patch.object(self.linter, '_check_grammar', side_effect=Exception("Test error")):
            result = self.linter.check_content("Test content")
            
            assert result.overall_score == 0.0
            assert "Error during analysis" in result.suggestions[0]

class TestEnhancedSearchTool:
    """Test cases for the EnhancedSearchTool class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_tool = EnhancedSearchTool()
    
    def test_initialization(self):
        """Test search tool initialization."""
        assert self.search_tool is not None
        assert hasattr(self.search_tool, 'search_providers')
        assert hasattr(self.search_tool, 'cache')
        assert 'duckduckgo' in self.search_tool.search_providers
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.search_tool._generate_cache_key("test query", "context", 10)
        key2 = self.search_tool._generate_cache_key("test query", "context", 10)
        key3 = self.search_tool._generate_cache_key("different query", "context", 10)
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
    
    def test_expand_query(self):
        """Test query expansion functionality."""
        query = "technology news"
        context = "latest developments in AI and machine learning"
        expanded = self.search_tool._expand_query(query, context)
        
        assert "technology" in expanded
        assert len(expanded) > len(query)  # Should be expanded
    
    def test_extract_key_terms(self):
        """Test key term extraction."""
        text = "The latest developments in artificial intelligence and machine learning"
        terms = self.search_tool._extract_key_terms(text)
        
        assert len(terms) > 0
        assert "developments" in terms
        assert "artificial" in terms
        assert "intelligence" in terms
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        result = {
            'title': 'Technology News Latest Updates',
            'snippet': 'Latest technology news and updates',
            'url': 'https://example.com'
        }
        query = "technology news"
        
        score = self.search_tool._calculate_confidence_score(result, query)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have good confidence for matching terms
    
    def test_calculate_relevance_score(self):
        """Test relevance score calculation."""
        result = {
            'title': 'Technology News',
            'snippet': 'Latest technology updates',
            'url': 'https://example.com'
        }
        query = "technology news"
        context = "latest developments"
        
        score = self.search_tool._calculate_relevance_score(result, query, context)
        
        assert 0.0 <= score <= 1.0
    
    def test_calculate_authority_score(self):
        """Test authority score calculation."""
        # Test authoritative domain
        result = {'url': 'https://reuters.com/article'}
        score1 = self.search_tool._calculate_authority_score(result)
        
        # Test non-authoritative domain
        result2 = {'url': 'https://random-blog.com/article'}
        score2 = self.search_tool._calculate_authority_score(result2)
        
        assert score1 > score2  # Authoritative domain should score higher
    
    def test_rank_results(self):
        """Test result ranking."""
        results = [
            SearchResult(
                title="Result 1",
                url="https://example1.com",
                snippet="First result",
                source="test",
                confidence_score=0.8,
                relevance_score=0.7,
                freshness_score=0.6,
                authority_score=0.5,
                overall_score=0.65,
                metadata={},
                timestamp=datetime.now()
            ),
            SearchResult(
                title="Result 2",
                url="https://example2.com",
                snippet="Second result",
                source="test",
                confidence_score=0.9,
                relevance_score=0.8,
                freshness_score=0.7,
                authority_score=0.6,
                overall_score=0.75,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        ranked = self.search_tool._rank_results(results)
        
        assert ranked[0].overall_score > ranked[1].overall_score
    
    def test_deduplicate_results(self):
        """Test result deduplication."""
        results = [
            SearchResult(
                title="Result 1",
                url="https://example.com",
                snippet="First result",
                source="test",
                confidence_score=0.8,
                relevance_score=0.7,
                freshness_score=0.6,
                authority_score=0.5,
                overall_score=0.65,
                metadata={},
                timestamp=datetime.now()
            ),
            SearchResult(
                title="Result 2",
                url="https://example.com",  # Same URL
                snippet="Second result",
                source="test",
                confidence_score=0.9,
                relevance_score=0.8,
                freshness_score=0.7,
                authority_score=0.6,
                overall_score=0.75,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        deduplicated = self.search_tool._deduplicate_results(results)
        
        assert len(deduplicated) == 1  # Should remove duplicate
    
    @patch('requests.get')
    def test_search_duckduckgo(self, mock_get):
        """Test DuckDuckGo search functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'AbstractText': 'Test abstract text',
            'AbstractURL': 'https://example.com',
            'AbstractSource': 'Test Source',
            'RelatedTopics': [
                {'Text': 'Related topic 1', 'FirstURL': 'https://example1.com'},
                {'Text': 'Related topic 2', 'FirstURL': 'https://example2.com'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = self.search_tool._search_duckduckgo("test query", 5)
        
        assert len(results) > 0
        assert all('source' in result for result in results)
        assert all(result['source'] == 'duckduckgo' for result in results)
    
    def test_search_with_filters(self):
        """Test search with filters."""
        # Mock search results
        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://reuters.com/article",
                snippet="Test result",
                source="test",
                confidence_score=0.8,
                relevance_score=0.7,
                freshness_score=0.6,
                authority_score=0.5,
                overall_score=0.65,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        with patch.object(self.search_tool, 'search_with_confidence', return_value=mock_results):
            filters = {'allowed_domains': ['reuters.com']}
            filtered_results = self.search_tool.search_with_filters("test query", filters)
            
            assert len(filtered_results) == 1
    
    def test_get_search_analytics(self):
        """Test search analytics functionality."""
        # Mock search results
        mock_results = [
            SearchResult(
                title="Result 1",
                url="https://example.com",
                snippet="Test result",
                source="test",
                confidence_score=0.8,
                relevance_score=0.7,
                freshness_score=0.6,
                authority_score=0.5,
                overall_score=0.65,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        with patch.object(self.search_tool, 'search_with_confidence', return_value=mock_results):
            analytics = self.search_tool.get_search_analytics("test query")
            
            assert 'query' in analytics
            assert 'total_results' in analytics
            assert 'average_scores' in analytics
            assert 'source_distribution' in analytics
            assert 'top_results' in analytics

class TestIntegration:
    """Integration tests for Phase 3 tools."""
    
    def test_linter_with_search_integration(self):
        """Test integration between linter and search tools."""
        linter = GrammarAndStyleLinter()
        search_tool = EnhancedSearchTool()
        
        # Create content that would benefit from search verification
        content = "The latest technology news shows that AI is becoming very important."
        
        # Check content with linter
        linter_result = linter.check_content(content)
        
        # If there are issues that need verification, search for more information
        if linter_result.total_issues > 0:
            # This would be used in a real scenario to verify claims
            assert linter_result.overall_score < 1.0
    
    def test_tool_availability(self):
        """Test that all Phase 3 tools are available."""
        from tools import GrammarAndStyleLinter, EnhancedSearchTool
        
        # Test linter availability
        linter = GrammarAndStyleLinter()
        assert linter is not None
        
        # Test search tool availability
        search_tool = EnhancedSearchTool()
        assert search_tool is not None

if __name__ == "__main__":
    pytest.main([__file__]) 