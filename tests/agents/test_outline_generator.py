"""
Tests for Enhanced Newsletter Outline Generator

Tests audience-specific outline generation, content analysis, section creation,
and outline validation functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.agents.planning.outline_generator import OutlineGenerator, OutlineSection, NewsletterOutline


class TestOutlineGenerator:
    """Test cases for OutlineGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OutlineGenerator()
        
        # Sample content items for testing
        self.sample_content_items = [
            Mock(
                themes=['machine_learning', 'deep_learning'],
                entities={'companies': ['OpenAI', 'Google'], 'models': ['GPT-4', 'BERT']},
                sentiment={'label': 'positive', 'score': 0.8},
                content='OpenAI released GPT-4 with breakthrough performance.'
            ),
            Mock(
                themes=['neural_networks', 'algorithms'],
                entities={'companies': ['NVIDIA'], 'models': ['Transformer']},
                sentiment={'label': 'neutral', 'score': 0.5},
                content='NVIDIA unveiled new GPU architecture for AI training.'
            ),
            Mock(
                themes=['business', 'market'],
                entities={'companies': ['Microsoft', 'Meta'], 'models': []},
                sentiment={'label': 'positive', 'score': 0.7},
                content='AI market shows strong growth with major investments.'
            )
        ]
    
    def test_outline_generator_initialization(self):
        """Test OutlineGenerator initialization."""
        assert hasattr(self.generator, 'content_analyzer')
        assert 'business' in self.generator.audience_templates
        assert 'technical' in self.generator.audience_templates
        assert 'mixed' in self.generator.audience_templates
        assert 'executive_summary' in self.generator.section_type_mapping
    
    def test_audience_templates(self):
        """Test audience-specific templates."""
        # Business template
        business_template = self.generator.audience_templates['business']
        assert len(business_template['sections']) == 5
        assert business_template['sections'][0]['type'] == 'executive_summary'
        assert business_template['sections'][1]['type'] == 'market_analysis'
        assert 'market' in business_template['focus_keywords']
        
        # Technical template
        technical_template = self.generator.audience_templates['technical']
        assert len(technical_template['sections']) == 5
        assert technical_template['sections'][0]['type'] == 'research_highlights'
        assert technical_template['sections'][1]['type'] == 'technical_deep_dive'
        assert 'algorithm' in technical_template['focus_keywords']
        
        # Mixed template
        mixed_template = self.generator.audience_templates['mixed']
        assert len(mixed_template['sections']) == 5
        assert mixed_template['sections'][0]['type'] == 'executive_summary'
        assert mixed_template['sections'][1]['type'] == 'research_highlights'
        assert 'ai' in mixed_template['focus_keywords']
    
    def test_analyze_content_patterns(self):
        """Test content pattern analysis."""
        analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        
        assert 'top_themes' in analysis
        assert 'theme_distribution' in analysis
        assert 'entity_distribution' in analysis
        assert 'sentiment_distribution' in analysis
        assert 'total_items' in analysis
        
        assert analysis['total_items'] == 3
        assert len(analysis['top_themes']) > 0
        assert 'machine_learning' in analysis['top_themes']
        assert 'deep_learning' in analysis['top_themes']
        
        # Check entity distribution
        assert 'companies' in analysis['entity_distribution']
        assert 'OpenAI' in analysis['entity_distribution']['companies']
        assert 'Google' in analysis['entity_distribution']['companies']
        
        # Check sentiment distribution
        assert 'positive' in analysis['sentiment_distribution']
        assert 'neutral' in analysis['sentiment_distribution']
    
    def test_create_sections(self):
        """Test section creation from template."""
        template = self.generator.audience_templates['mixed']
        content_analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        target_length = 2
        
        sections = self.generator._create_sections(template, content_analysis, target_length)
        
        assert len(sections) == 5  # Mixed template has 5 sections
        
        for section in sections:
            assert isinstance(section, OutlineSection)
            assert section.id.startswith('section_')
            assert section.title
            assert section.type
            assert isinstance(section.themes, list)
            assert isinstance(section.keywords, list)
            assert section.target_length > 0
            assert section.priority in ['high', 'medium', 'low']
    
    def test_determine_section_themes(self):
        """Test theme determination for sections."""
        content_analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        
        # Research highlights section
        research_themes = self.generator._determine_section_themes('research_highlights', content_analysis)
        assert len(research_themes) > 0
        assert any(theme in ['machine_learning', 'deep_learning', 'neural_networks'] for theme in research_themes)
        
        # Industry news section - should use available themes since 'companies' not in content
        industry_themes = self.generator._determine_section_themes('industry_news', content_analysis)
        assert len(industry_themes) > 0
        # Since 'companies' theme is not in the content, it should fall back to top themes
        assert any(theme in ['machine_learning', 'deep_learning', 'business', 'market'] for theme in industry_themes)
        
        # Market analysis section
        market_themes = self.generator._determine_section_themes('market_analysis', content_analysis)
        assert len(market_themes) > 0
        assert any(theme in ['business', 'market', 'trends'] for theme in market_themes)
    
    def test_determine_section_keywords(self):
        """Test keyword determination for sections."""
        content_analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        
        # Executive summary keywords
        summary_keywords = self.generator._determine_section_keywords('executive_summary', content_analysis)
        assert 'ai' in summary_keywords
        assert 'machine learning' in summary_keywords
        assert 'breakthrough' in summary_keywords
        
        # Technical deep dive keywords
        technical_keywords = self.generator._determine_section_keywords('technical_deep_dive', content_analysis)
        assert 'implementation' in technical_keywords
        assert 'architecture' in technical_keywords
        assert 'performance' in technical_keywords
        
        # Industry news keywords
        industry_keywords = self.generator._determine_section_keywords('industry_news', content_analysis)
        assert 'company' in industry_keywords
        assert 'announcement' in industry_keywords
        assert 'product' in industry_keywords
    
    def test_determine_section_priority(self):
        """Test priority determination for sections."""
        # High priority sections
        assert self.generator._determine_section_priority('executive_summary', 0) == 'high'
        assert self.generator._determine_section_priority('research_highlights', 1) == 'high'
        
        # Medium priority sections
        assert self.generator._determine_section_priority('industry_news', 2) == 'medium'
        assert self.generator._determine_section_priority('technical_deep_dive', 3) == 'medium'
        assert self.generator._determine_section_priority('market_analysis', 4) == 'medium'
        
        # Low priority sections
        assert self.generator._determine_section_priority('trend_analysis', 5) == 'low'
        assert self.generator._determine_section_priority('future_directions', 6) == 'low'
    
    def test_optimize_outline(self):
        """Test outline optimization."""
        template = self.generator.audience_templates['mixed']
        content_analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        sections = self.generator._create_sections(template, content_analysis, 2)
        
        # Get initial total length
        initial_total = sum(section.target_length for section in sections)
        
        # Optimize outline
        optimized_sections = self.generator._optimize_outline(sections, self.sample_content_items, 2)
        
        # Check that optimization maintains reasonable total length
        optimized_total = sum(section.target_length for section in optimized_sections)
        assert abs(optimized_total - initial_total) < initial_total * 0.5  # Within 50% of original
    
    def test_count_relevant_content(self):
        """Test content relevance counting."""
        section = OutlineSection(
            id='section_01',
            title='Research Highlights',
            type='research_highlights',
            themes=['machine_learning', 'deep_learning'],
            keywords=['research', 'algorithm'],
            target_length=300,
            priority='high'
        )
        
        relevant_count = self.generator._count_relevant_content(section, self.sample_content_items)
        assert relevant_count > 0  # Should find relevant content
    
    def test_generate_outline_title(self):
        """Test outline title generation."""
        content_analysis = self.generator._analyze_content_patterns(self.sample_content_items)
        
        # Business audience title
        business_title = self.generator._generate_outline_title(content_analysis, 'business')
        assert 'Business' in business_title
        assert 'AI' in business_title
        
        # Technical audience title
        technical_title = self.generator._generate_outline_title(content_analysis, 'technical')
        assert 'Technical' in technical_title
        assert 'AI' in technical_title
        
        # Mixed audience title
        mixed_title = self.generator._generate_outline_title(content_analysis, 'mixed')
        assert 'AI' in mixed_title
        assert 'Insights' in mixed_title
    
    def test_estimate_read_time(self):
        """Test reading time estimation."""
        sections = [
            OutlineSection(
                id='section_01',
                title='Section 1',
                type='executive_summary',
                themes=[],
                keywords=[],
                target_length=200,
                priority='high'
            ),
            OutlineSection(
                id='section_02',
                title='Section 2',
                type='research_highlights',
                themes=[],
                keywords=[],
                target_length=300,
                priority='medium'
            )
        ]
        
        read_time = self.generator._estimate_read_time(sections)
        assert read_time > 0
        assert read_time == 2
    
    def test_outline_to_dict(self):
        """Test outline to dictionary conversion."""
        sections = [
            OutlineSection(
                id='section_01',
                title='Executive Summary',
                type='executive_summary',
                themes=['machine_learning'],
                keywords=['ai', 'breakthrough'],
                target_length=200,
                priority='high'
            )
        ]
        
        outline = NewsletterOutline(
            id='outline_001',
            title='AI Newsletter: Machine Learning Focus',
            target_audience='mixed',
            total_length=2,
            sections=sections,
            themes=['machine_learning'],
            estimated_read_time=2,
            created_at=datetime.now()
        )
        
        outline_dict = self.generator._outline_to_dict(outline)
        
        assert outline_dict['id'] == 'outline_001'
        assert outline_dict['title'] == 'AI Newsletter: Machine Learning Focus'
        assert outline_dict['target_audience'] == 'mixed'
        assert outline_dict['total_length'] == 2
        assert len(outline_dict['sections']) == 1
        assert outline_dict['sections'][0]['id'] == 'section_01'
        assert outline_dict['sections'][0]['title'] == 'Executive Summary'
        assert outline_dict['themes'] == ['machine_learning']
        assert outline_dict['estimated_read_time'] == 2
    
    def test_validate_outline(self):
        """Test outline validation."""
        # Valid outline
        valid_outline = {
            'id': 'outline_001',
            'title': 'Test Outline',
            'target_audience': 'mixed',
            'sections': [
                {
                    'id': 'section_01',
                    'title': 'Section 1',
                    'type': 'executive_summary',
                    'target_length': 600
                }
            ]
        }
        
        issues = self.generator.validate_outline(valid_outline)
        assert len(issues) == 0
        
        # Invalid outline - missing required field
        invalid_outline = {
            'id': 'outline_001',
            'title': 'Test Outline',
            # Missing target_audience
            'sections': []
        }
        
        issues = self.generator.validate_outline(invalid_outline)
        assert len(issues) > 0
        assert any('Missing required field' in issue for issue in issues)
        
        # Invalid outline - empty sections
        invalid_outline2 = {
            'id': 'outline_001',
            'title': 'Test Outline',
            'target_audience': 'mixed',
            'sections': []
        }
        
        issues = self.generator.validate_outline(invalid_outline2)
        assert len(issues) > 0
        assert any('at least one section' in issue for issue in issues)
    
    def test_validate_section(self):
        """Test section validation."""
        # Valid section
        valid_section = {
            'id': 'section_01',
            'title': 'Section 1',
            'type': 'executive_summary',
            'target_length': 200
        }
        
        issues = self.generator._validate_section(valid_section, 0)
        assert len(issues) == 0
        
        # Invalid section - missing required field
        invalid_section = {
            'id': 'section_01',
            'title': 'Section 1',
            # Missing type and target_length
        }
        
        issues = self.generator._validate_section(invalid_section, 0)
        assert len(issues) > 0
        assert any('Missing required field' in issue for issue in issues)
        
        # Invalid section - too short
        short_section = {
            'id': 'section_01',
            'title': 'Section 1',
            'type': 'executive_summary',
            'target_length': 25  # Too short
        }
        
        issues = self.generator._validate_section(short_section, 0)
        assert len(issues) > 0
        assert any('too short' in issue for issue in issues)
        
        # Invalid section - invalid type
        invalid_type_section = {
            'id': 'section_01',
            'title': 'Section 1',
            'type': 'invalid_type',
            'target_length': 200
        }
        
        issues = self.generator._validate_section(invalid_type_section, 0)
        assert len(issues) > 0
        assert any('Invalid section type' in issue for issue in issues)
    
    def test_get_section_template(self):
        """Test getting section templates for different audiences."""
        business_template = self.generator.get_section_template('business')
        assert business_template['sections'][0]['type'] == 'executive_summary'
        assert 'market' in business_template['focus_keywords']
        
        technical_template = self.generator.get_section_template('technical')
        assert technical_template['sections'][0]['type'] == 'research_highlights'
        assert 'algorithm' in technical_template['focus_keywords']
        
        mixed_template = self.generator.get_section_template('mixed')
        assert mixed_template['sections'][0]['type'] == 'executive_summary'
        assert 'ai' in mixed_template['focus_keywords']
        
        # Test default for unknown audience
        default_template = self.generator.get_section_template('unknown')
        assert default_template == mixed_template
    
    def test_customize_for_audience(self):
        """Test audience customization."""
        outline = {
            'total_length': 2,
            'sections': [
                {
                    'id': 'section_01',
                    'type': 'executive_summary',
                    'target_length': 200
                },
                {
                    'id': 'section_02',
                    'type': 'research_highlights',
                    'target_length': 300
                }
            ]
        }
        
        customized = self.generator.customize_for_audience(outline, 'business')
        
        # Should adjust section lengths based on business template weights
        assert 'sections' in customized
        assert len(customized['sections']) == 2
    
    @patch('src.agents.planning.outline_generator.OutlineGenerator._create_sections')
    @patch('src.agents.planning.outline_generator.OutlineGenerator._optimize_outline')
    def test_generate_outline_integration(self, mock_optimize, mock_create_sections):
        """Test complete outline generation integration."""
        # Mock dependencies
        mock_sections = [
            OutlineSection(
                id='section_01',
                title='Executive Summary',
                type='executive_summary',
                themes=['machine_learning'],
                keywords=['ai'],
                target_length=200,
                priority='high'
            )
        ]
        mock_create_sections.return_value = mock_sections
        mock_optimize.return_value = mock_sections
        
        # Test outline generation
        outline = self.generator.generate_outline(
            self.sample_content_items,
            target_audience='mixed',
            target_length=2
        )
        
        assert isinstance(outline, dict)
        assert 'id' in outline
        assert 'title' in outline
        assert 'target_audience' in outline
        assert 'sections' in outline
        assert 'themes' in outline
        assert 'estimated_read_time' in outline
        assert outline['target_audience'] == 'mixed'
        assert outline['total_length'] == 2 