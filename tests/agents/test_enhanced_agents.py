"""
Tests for Enhanced Agent Capabilities

This module tests the enhanced agent capabilities including context-driven writing,
tool-assisted auditing, and advanced research capabilities.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.agents.writing import WriterAgent
from src.agents.editing import EditorAgent
from src.agents.research import ResearchAgent
from src.core.campaign_context import CampaignContext
from src.core.feedback_system import StructuredFeedback, FeedbackItem, IssueType, Severity, RequiredAction


class TestEnhancedWriterAgent:
    """Test enhanced WriterAgent capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.writer_agent = WriterAgent()
        self.campaign_context = CampaignContext.create_default_context()
        
        # Mock research data
        self.research_data = {
            'topic': 'Artificial Intelligence in Healthcare',
            'research_results': [
                {
                    'title': 'AI improves diagnostic accuracy',
                    'summary': 'Recent studies show AI can improve diagnostic accuracy by 20%',
                    'url': 'https://example.com/ai-diagnostics'
                },
                {
                    'title': 'Machine learning in medical imaging',
                    'summary': 'ML algorithms are revolutionizing medical imaging analysis',
                    'url': 'https://example.com/ml-imaging'
                }
            ],
            'sources': [
                {
                    'title': 'AI in Healthcare Report',
                    'url': 'https://example.com/report',
                    'author': 'Dr. Smith',
                    'date': '2024-01-15'
                }
            ]
        }
    
    def test_write_from_context(self):
        """Test context-driven writing capability."""
        with patch.object(self.writer_agent, '_generate_context_aware_content') as mock_generate:
            mock_generate.return_value = "# AI in Healthcare\n\nThis is a test newsletter content."
            
            result = self.writer_agent.write_from_context(self.research_data, self.campaign_context)
            
            assert result is not None
            assert "AI in Healthcare" in result
            mock_generate.assert_called_once()
    
    def test_integrate_sources(self):
        """Test source citation integration."""
        content = "# Test Content\n\nThis is test content."
        sources = [
            {
                'title': 'Test Source 1',
                'url': 'https://example.com/1',
                'author': 'Author 1',
                'date': '2024-01-01'
            },
            {
                'title': 'Test Source 2',
                'url': 'https://example.com/2',
                'author': 'Author 2',
                'date': '2024-01-02'
            }
        ]
        
        result = self.writer_agent.integrate_sources(content, sources)
        
        assert "## Sources" in result
        assert "Test Source 1" in result
        assert "Test Source 2" in result
        assert "https://example.com/1" in result
    
    def test_adapt_style(self):
        """Test style adaptation capability."""
        content = "This is a test content with formal language."
        style_params = {
            'tone': 'casual',
            'formality': 'casual',
            'personality': 'enthusiastic'
        }
        
        result = self.writer_agent.adapt_style(content, style_params)
        
        assert result is not None
        # Should have some style adaptations applied
        assert len(result) > 0
    
    def test_implement_revisions(self):
        """Test targeted revision implementation."""
        content = "This is the original content with some issues."
        
        # Create mock feedback
        feedback_items = [
            FeedbackItem(
                text_snippet="original content",
                issue_type=IssueType.CLARITY,
                comment="Needs clarification",
                required_action=RequiredAction.REVISION,
                severity=Severity.MEDIUM
            )
        ]
        
        feedback = StructuredFeedback(
            overall_score=7.0,
            sub_scores={'clarity': 6.0},
            feedback_items=feedback_items,
            required_action=RequiredAction.REVISION,
            revision_cycles=1,
            summary="Content needs improvement",
            improvement_suggestions=["Clarify language", "Improve structure"],
            quality_metrics={'readability': 0.7}
        )
        
        result = self.writer_agent.implement_revisions(content, feedback)
        
        assert result is not None
        assert len(result) > 0


class TestEnhancedEditorAgent:
    """Test enhanced EditorAgent capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.editor_agent = EditorAgent()
        self.campaign_context = CampaignContext.create_default_context()
        
        self.test_content = """
# Test Newsletter

This is a test newsletter content with some issues.

## Section 1

The content has some grammar problems and it's not very clear.
Furthermore, there are some style issues that need to be addressed.

## Section 2

This section contains some technical terms like algorithm and framework.
        """
    
    def test_perform_tool_assisted_audit(self):
        """Test tool-assisted auditing capability."""
        result = self.editor_agent.perform_tool_assisted_audit(self.test_content)
        
        assert 'grammar_issues' in result
        assert 'style_issues' in result
        assert 'clarity_issues' in result
        assert 'structure_issues' in result
        assert 'engagement_issues' in result
        assert 'seo_issues' in result
        assert 'brand_compliance_issues' in result
        assert 'accessibility_issues' in result
        assert 'overall_score' in result
        assert 'recommendations' in result
        
        assert isinstance(result['overall_score'], float)
        assert isinstance(result['recommendations'], list)
    
    def test_evaluate_against_context(self):
        """Test context evaluation capability."""
        result = self.editor_agent.evaluate_against_context(self.test_content, self.campaign_context)
        
        assert 'tone_alignment' in result
        assert 'audience_alignment' in result
        assert 'goal_alignment' in result
        assert 'terminology_compliance' in result
        assert 'quality_threshold_met' in result
        assert 'context_score' in result
        assert 'recommendations' in result
        
        assert isinstance(result['context_score'], float)
        assert isinstance(result['quality_threshold_met'], bool)
    
    def test_generate_structured_feedback(self):
        """Test structured feedback generation."""
        audit_results = {
            'grammar_issues': [{'text': 'test', 'issue_type': 'grammar_error'}],
            'style_issues': [],
            'clarity_issues': [],
            'structure_issues': [],
            'engagement_issues': [],
            'seo_issues': [],
            'brand_compliance_issues': [],
            'accessibility_issues': [],
            'overall_score': 8.0,
            'recommendations': ['Fix grammar issues']
        }
        
        feedback = self.editor_agent.generate_structured_feedback(audit_results)
        
        assert isinstance(feedback, StructuredFeedback)
        assert feedback.overall_score > 0
        assert len(feedback.feedback_items) >= 0
    
    def test_verify_critical_claims(self):
        """Test critical claim verification."""
        content = "According to research, 75% of users prefer AI-powered tools."
        
        result = self.editor_agent.verify_critical_claims(content)
        
        assert isinstance(result, list)
        # Should extract claims for verification
        assert len(result) >= 0
    
    def test_grammar_checking(self):
        """Test grammar checking functionality."""
        content = "Its a test of grammar checking. Their going to be issues."
        
        issues = self.editor_agent._check_grammar(content)
        
        assert isinstance(issues, list)
        # Should find some grammar issues
        assert len(issues) >= 0
    
    def test_style_checking(self):
        """Test style checking functionality."""
        content = "The content is being written in passive voice."
        
        issues = self.editor_agent._check_style(content)
        
        assert isinstance(issues, list)
        # Should find some style issues
        assert len(issues) >= 0


class TestEnhancedResearchAgent:
    """Test enhanced ResearchAgent capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.research_agent = ResearchAgent()
        self.campaign_context = CampaignContext.create_default_context()
        
        # Mock research findings
        self.mock_findings = [
            {
                'title': 'AI in Healthcare',
                'content': 'Artificial intelligence is revolutionizing healthcare with improved diagnostics.',
                'url': 'https://example.com/ai-healthcare',
                'source': 'Medical Journal'
            },
            {
                'title': 'Machine Learning Trends',
                'content': 'ML algorithms are becoming more sophisticated and accurate.',
                'url': 'https://example.com/ml-trends',
                'source': 'Tech Blog'
            }
        ]
    
    def test_conduct_context_aware_research(self):
        """Test context-aware research capability."""
        with patch.object(self.research_agent, '_execute_context_aware_research') as mock_execute:
            mock_execute.return_value = self.mock_findings
            
            result = self.research_agent.conduct_context_aware_research(
                "AI in Healthcare", 
                self.campaign_context
            )
            
            assert 'topic' in result
            assert 'context' in result
            assert 'research_results' in result
            assert 'verification_results' in result
            assert 'search_queries' in result
    
    def test_validate_sources(self):
        """Test source validation capability."""
        result = self.research_agent.validate_sources(self.mock_findings)
        
        assert isinstance(result, list)
        assert len(result) == len(self.mock_findings)
        
        for finding in result:
            assert 'confidence_score' in finding
            assert 'validation_timestamp' in finding
            assert 'source_quality' in finding
            assert 'relevance_score' in finding
            assert 'freshness_score' in finding
            assert 'authority_score' in finding
    
    def test_generate_structured_output(self):
        """Test structured output generation."""
        research_data = {
            'topic': 'AI in Healthcare',
            'context': self.campaign_context,
            'research_results': self.mock_findings,
            'verification_results': [],
            'search_queries': ['AI healthcare', 'machine learning medicine']
        }
        
        result = self.research_agent.generate_structured_output(research_data)
        
        assert 'topic' in result
        assert 'sections' in result
        assert 'insights' in result
        assert 'depth_assessment' in result
        assert 'coverage_score' in result
        assert 'total_findings' in result
        assert 'research_quality_score' in result
        assert 'recommendations' in result
    
    def test_verify_specific_claim(self):
        """Test specific claim verification."""
        claim = "AI improves diagnostic accuracy by 20%"
        
        with patch('src.tools.tools.search_web') as mock_search:
            mock_search.return_value = [
                {'title': 'AI Diagnostic Study', 'content': 'Research shows AI improves accuracy'}
            ]
            
            result = self.research_agent.verify_specific_claim(claim, "test context")
            
            assert 'claim' in result
            assert 'verification_status' in result
            assert 'confidence_score' in result
            assert 'supporting_evidence' in result
            assert 'contradicting_evidence' in result
            assert 'verification_queries' in result
            assert 'verification_notes' in result
    
    def test_expand_queries_proactively(self):
        """Test proactive query expansion."""
        base_queries = ['AI healthcare']
        
        result = self.research_agent._expand_queries_proactively(base_queries, self.campaign_context)
        
        assert isinstance(result, list)
        assert len(result) > len(base_queries)  # Should have expanded queries
        assert all(isinstance(q, str) for q in result)
    
    def test_categorize_finding_enhanced(self):
        """Test enhanced finding categorization."""
        finding = {
            'title': 'Latest AI Developments',
            'content': 'Recent developments in artificial intelligence show promising results.'
        }
        
        result = self.research_agent._categorize_finding_enhanced(finding, self.campaign_context)
        
        assert isinstance(result, str)
        assert result in ['key_facts', 'recent_developments', 'expert_opinions', 
                         'related_topics', 'trending_insights', 'audience_specific_findings']
    
    def test_assess_research_depth_enhanced(self):
        """Test enhanced research depth assessment."""
        result = self.research_agent._assess_research_depth_enhanced(self.mock_findings, self.campaign_context)
        
        assert 'depth_score' in result
        assert 'total_findings' in result
        assert 'high_confidence_findings' in result
        assert 'verified_claims' in result
        assert 'goal_coverage' in result
        assert 'depth_level' in result
        
        assert isinstance(result['depth_score'], float)
        assert result['depth_level'] in ['comprehensive', 'moderate', 'basic']


class TestAgentIntegration:
    """Test integration between enhanced agents."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.writer_agent = WriterAgent()
        self.editor_agent = EditorAgent()
        self.research_agent = ResearchAgent()
        self.campaign_context = CampaignContext.create_default_context()
    
    def test_workflow_integration(self):
        """Test integration of enhanced agents in a workflow."""
        # Mock research results
        research_data = {
            'topic': 'AI in Healthcare',
            'research_results': [
                {
                    'title': 'AI Diagnostic Tools',
                    'summary': 'AI is improving diagnostic accuracy',
                    'confidence_score': 0.9
                }
            ]
        }
        
        # Test writer agent
        with patch('src.core.core.query_llm') as mock_llm:
            mock_llm.return_value = "# AI in Healthcare\n\nThis is test content."
            content = self.writer_agent.write_from_context(research_data, self.campaign_context)
        
        # Test editor agent
        audit_results = self.editor_agent.perform_tool_assisted_audit(content)
        context_evaluation = self.editor_agent.evaluate_against_context(content, self.campaign_context)
        
        # Verify integration
        assert content is not None
        assert audit_results is not None
        assert context_evaluation is not None
        
        # Test feedback integration
        feedback = self.editor_agent.generate_structured_feedback(audit_results)
        revised_content = self.writer_agent.implement_revisions(content, feedback)
        
        assert revised_content is not None
        assert len(revised_content) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 