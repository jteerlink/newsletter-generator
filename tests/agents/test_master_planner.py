"""
Tests for Master Planning Agent

Tests the complete newsletter planning pipeline including content analysis,
scoring, filtering, outline generation, and task assignment.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.agents.planning.master_planner import MasterPlanner, ContentItem, NewsletterPlan
from src.agents.base.communication import Message, MessageType


class TestMasterPlanner:
    """Test cases for MasterPlanner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planner = MasterPlanner("test_planner")
        
        # Sample content sources for testing
        self.sample_content_sources = [
            {
                'id': 'content_1',
                'title': 'OpenAI Releases GPT-4 Turbo with Vision',
                'content': 'OpenAI has announced GPT-4 Turbo with Vision, a breakthrough multimodal AI model that demonstrates unprecedented reasoning abilities across text, image, and code domains. This revolutionary advancement represents a significant leap forward in AI capabilities.',
                'source': 'techcrunch',
                'url': 'https://techcrunch.com/gpt4-turbo-vision',
                'published_date': datetime.now() - timedelta(hours=2)
            },
            {
                'id': 'content_2',
                'title': 'NVIDIA Blackwell B200 GPU Architecture Unveiled',
                'content': 'NVIDIA has unveiled the Blackwell B200 GPU architecture featuring 208 billion transistors and delivering 20 petaflops of AI performance. This breakthrough enables training of trillion-parameter models.',
                'source': 'nvidia',
                'url': 'https://nvidia.com/blackwell',
                'published_date': datetime.now() - timedelta(hours=1)
            },
            {
                'id': 'content_3',
                'title': 'Anthropic Claude 3.5 Sonnet Achieves New Benchmarks',
                'content': 'Anthropic\'s Claude 3.5 Sonnet shows remarkable improvements in mathematical reasoning and code generation tasks, achieving new benchmarks in AI performance.',
                'source': 'anthropic',
                'url': 'https://anthropic.com/claude-3.5',
                'published_date': datetime.now() - timedelta(hours=3)
            }
        ]
    
    def test_master_planner_initialization(self):
        """Test MasterPlanner initialization."""
        assert self.planner.agent_id == "test_planner"
        assert self.planner.max_content_items == 50
        assert self.planner.min_content_items == 10
        assert self.planner.target_newsletter_length == 2
        assert self.planner.quality_threshold == 0.7
        assert 'arxiv' in self.planner.source_weights
        assert 'techcrunch' in self.planner.source_weights
    
    def test_analyze_and_score_content(self):
        """Test content analysis and scoring."""
        content_items = self.planner._analyze_and_score_content(self.sample_content_sources)
        
        assert len(content_items) == 3
        
        # Check that ContentItem objects are created correctly
        for item in content_items:
            assert isinstance(item, ContentItem)
            assert item.id in ['content_1', 'content_2', 'content_3']
            assert item.title
            assert item.content
            assert item.source
            assert item.url
            assert isinstance(item.published_date, datetime)
            assert 0 <= item.novelty_score <= 1
            assert 0 <= item.importance_score <= 1
            assert 0 <= item.audience_relevance <= 1
            assert 0 <= item.source_reliability <= 1
            assert 0 <= item.temporal_relevance <= 1
            assert 0 <= item.overall_score <= 1
            assert isinstance(item.themes, list)
            assert isinstance(item.entities, dict)
            assert isinstance(item.sentiment, dict)
    
    def test_calculate_novelty_score(self):
        """Test novelty score calculation."""
        # High novelty content
        high_novelty_item = ContentItem(
            id='test_1',
            title='Breakthrough in Quantum AI',
            content='This revolutionary breakthrough represents unprecedented innovation in cutting-edge technology.',
            source='nature',
            url='test_url',
            published_date=datetime.now(),
            entities={'models': ['QuantumAI'], 'technologies': ['quantum computing']}
        )
        
        novelty_score = self.planner._calculate_novelty_score(high_novelty_item)
        assert novelty_score > 0.3  # Should be reasonably high due to breakthrough keywords and new entities
        
        # Low novelty content
        low_novelty_item = ContentItem(
            id='test_2',
            title='Regular AI Update',
            content='This is a regular update about existing technology.',
            source='blog',
            url='test_url',
            published_date=datetime.now(),
            entities={'models': [], 'technologies': []}
        )
        
        novelty_score = self.planner._calculate_novelty_score(low_novelty_item)
        assert novelty_score < 0.2  # Should be low due to lack of novelty indicators
    
    def test_calculate_importance_score(self):
        """Test importance score calculation."""
        # High importance content
        high_importance_item = ContentItem(
            id='test_1',
            title='OpenAI GPT-4 Breakthrough',
            content='OpenAI announced GPT-4 with improved accuracy and performance benchmarks.',
            source='techcrunch',
            url='test_url',
            published_date=datetime.now()
        )
        
        importance_score = self.planner._calculate_importance_score(high_importance_item)
        assert importance_score > 0.4  # Should be reasonably high due to major company and model
        
        # Low importance content
        low_importance_item = ContentItem(
            id='test_2',
            title='Minor AI Update',
            content='A small update about a minor algorithm improvement.',
            source='blog',
            url='test_url',
            published_date=datetime.now()
        )
        
        importance_score = self.planner._calculate_importance_score(low_importance_item)
        assert importance_score < 0.3  # Should be low due to lack of major indicators
    
    def test_calculate_audience_relevance(self):
        """Test audience relevance calculation."""
        # Business-focused content
        business_item = ContentItem(
            id='test_1',
            title='AI Market Analysis',
            content='The AI market shows strong revenue growth with significant investment in enterprise solutions.',
            source='forbes',
            url='test_url',
            published_date=datetime.now()
        )
        
        business_relevance = self.planner._calculate_audience_relevance(business_item)
        assert business_relevance > 0.3  # Should be reasonably high due to business keywords
        
        # Technical-focused content
        technical_item = ContentItem(
            id='test_2',
            title='Algorithm Implementation',
            content='The algorithm architecture uses transformer models with optimization techniques.',
            source='arxiv',
            url='test_url',
            published_date=datetime.now()
        )
        
        technical_relevance = self.planner._calculate_audience_relevance(technical_item)
        assert technical_relevance > 0.3  # Should be reasonably high due to technical keywords
    
    def test_calculate_source_reliability(self):
        """Test source reliability calculation."""
        # High reliability source
        high_reliability_item = ContentItem(
            id='test_1',
            title='Test',
            content='Test content',
            source='nature',
            url='test_url',
            published_date=datetime.now()
        )
        
        reliability = self.planner._calculate_source_reliability(high_reliability_item)
        assert reliability == 0.95  # Nature should have high reliability
        
        # Default reliability source
        default_item = ContentItem(
            id='test_2',
            title='Test',
            content='Test content',
            source='unknown_blog',
            url='test_url',
            published_date=datetime.now()
        )
        
        reliability = self.planner._calculate_source_reliability(default_item)
        assert reliability == 0.7  # Default reliability
    
    def test_calculate_temporal_relevance(self):
        """Test temporal relevance calculation."""
        # Very recent content
        recent_item = ContentItem(
            id='test_1',
            title='Test',
            content='Test content',
            source='test',
            url='test_url',
            published_date=datetime.now() - timedelta(hours=1)
        )
        
        temporal_relevance = self.planner._calculate_temporal_relevance(recent_item)
        assert temporal_relevance == 1.0  # Should be maximum for very recent content
        
        # Old content
        old_item = ContentItem(
            id='test_2',
            title='Test',
            content='Test content',
            source='test',
            url='test_url',
            published_date=datetime.now() - timedelta(days=100)
        )
        
        temporal_relevance = self.planner._calculate_temporal_relevance(old_item)
        assert temporal_relevance == 0.3  # Should be low for old content
    
    def test_filter_and_prioritize_content(self):
        """Test content filtering and prioritization."""
        # Create content items with different scores
        high_score_item = ContentItem(
            id='high_score',
            title='High Score Content',
            content='Breakthrough revolutionary innovation',
            source='nature',
            url='test_url',
            published_date=datetime.now(),
            overall_score=0.9
        )
        
        medium_score_item = ContentItem(
            id='medium_score',
            title='Medium Score Content',
            content='Regular update',
            source='techcrunch',
            url='test_url',
            published_date=datetime.now(),
            overall_score=0.7
        )
        
        low_score_item = ContentItem(
            id='low_score',
            title='Low Score Content',
            content='Minor update',
            source='blog',
            url='test_url',
            published_date=datetime.now(),
            overall_score=0.3
        )
        
        content_items = [low_score_item, high_score_item, medium_score_item]
        filtered_content = self.planner._filter_and_prioritize_content(content_items)
        
        # Should filter out low score items and prioritize by score
        assert len(filtered_content) >= 2
        assert filtered_content[0].overall_score >= filtered_content[1].overall_score
    
    def test_create_tasks(self):
        """Test task creation from content and outline."""
        # Create sample content items
        content_items = [
            ContentItem(
                id='item_1',
                title='Research Paper',
                content='Research about machine learning algorithms',
                source='arxiv',
                url='test_url',
                published_date=datetime.now(),
                themes=['machine_learning', 'deep_learning']
            ),
            ContentItem(
                id='item_2',
                title='Company News',
                content='OpenAI announces new product',
                source='techcrunch',
                url='test_url',
                published_date=datetime.now(),
                themes=['companies', 'products']
            )
        ]
        
        # Create sample outline
        outline = {
            'sections': [
                {
                    'id': 'section_01',
                    'type': 'research_summary',
                    'themes': ['machine_learning'],
                    'keywords': ['research', 'algorithm']
                },
                {
                    'id': 'section_02',
                    'type': 'industry_news',
                    'themes': ['companies'],
                    'keywords': ['company', 'announcement']
                }
            ]
        }
        
        tasks = self.planner._create_tasks(content_items, outline)
        
        assert len(tasks) == 2
        assert tasks[0]['type'] == 'research_summary'
        assert tasks[1]['type'] == 'industry_news'
        assert 'requirements' in tasks[0]
        assert 'requirements' in tasks[1]
    
    def test_filter_content_for_section(self):
        """Test content filtering for specific sections."""
        content_items = [
            ContentItem(
                id='item_1',
                title='ML Research',
                content='Machine learning algorithm research',
                source='arxiv',
                url='test_url',
                published_date=datetime.now(),
                themes=['machine_learning']
            ),
            ContentItem(
                id='item_2',
                title='Company News',
                content='Company announcement',
                source='techcrunch',
                url='test_url',
                published_date=datetime.now(),
                themes=['companies']
            )
        ]
        
        section = {
            'themes': ['machine_learning'],
            'keywords': ['research', 'algorithm']
        }
        
        relevant_content = self.planner._filter_content_for_section(content_items, section)
        
        assert len(relevant_content) == 1
        assert relevant_content[0].id == 'item_1'
    
    def test_set_deadlines(self):
        """Test deadline setting for tasks."""
        assignments = {
            'research_agent': [
                {'id': 'task_1', 'type': 'research_summary'},
                {'id': 'task_2', 'type': 'research_summary'}
            ],
            'technical_agent': [
                {'id': 'task_3', 'type': 'technical_deep_dive'}
            ]
        }
        
        deadlines = self.planner._set_deadlines(assignments)
        
        assert len(deadlines) == 3
        assert all(isinstance(deadline, datetime) for deadline in deadlines.values())
        
        # Check that deadlines are in the future
        now = datetime.now()
        assert all(deadline > now for deadline in deadlines.values())
    
    def test_create_quality_requirements(self):
        """Test quality requirements creation."""
        # Business audience
        business_requirements = self.planner._create_quality_requirements('business')
        assert business_requirements['business_focus'] is True
        assert business_requirements['citation_style'] == 'journalistic'
        
        # Technical audience
        technical_requirements = self.planner._create_quality_requirements('technical')
        assert technical_requirements['technical_depth'] == 'expert'
        assert technical_requirements['citation_style'] == 'academic'
        
        # Mixed audience
        mixed_requirements = self.planner._create_quality_requirements('mixed')
        assert mixed_requirements['technical_depth'] == 'mixed'
        assert mixed_requirements['business_focus'] is False
    
    def test_generate_title(self):
        """Test newsletter title generation."""
        content_items = [
            ContentItem(
                id='item_1',
                title='Test',
                content='Test content',
                source='test',
                url='test_url',
                published_date=datetime.now(),
                themes=['machine_learning', 'deep_learning']
            ),
            ContentItem(
                id='item_2',
                title='Test',
                content='Test content',
                source='test',
                url='test_url',
                published_date=datetime.now(),
                themes=['machine_learning', 'neural_networks']
            )
        ]
        
        title = self.planner._generate_title(content_items)
        assert 'Machine Learning' in title
        assert 'AI Newsletter' in title
    
    def test_estimate_length(self):
        """Test newsletter length estimation."""
        content_items = [
            ContentItem(
                id='item_1',
                title='Test',
                content='word ' * 300,  # 300 words
                source='test',
                url='test_url',
                published_date=datetime.now()
            ),
            ContentItem(
                id='item_2',
                title='Test',
                content='word ' * 200,  # 200 words
                source='test',
                url='test_url',
                published_date=datetime.now()
            )
        ]
        
        estimated_length = self.planner._estimate_length(content_items)
        assert estimated_length == 1  # 500 words total = 1 page
    
    @patch('src.agents.planning.master_planner.OutlineGenerator')
    @patch('src.agents.planning.master_planner.TaskAssignment')
    def test_plan_newsletter_integration(self, mock_task_assignment, mock_outline_generator):
        """Test complete newsletter planning integration."""
        # Mock dependencies
        mock_outline_generator.return_value.generate_outline.return_value = {
            'sections': [
                {
                    'id': 'section_01',
                    'type': 'research_summary',
                    'title': 'Research Summary',
                    'themes': ['machine_learning'],
                    'keywords': ['research'],
                    'target_length': 300,
                    'priority': 'high'
                }
            ]
        }
        
        mock_task_assignment.return_value.assign_tasks.return_value = {
            'research_agent': [
                {
                    'id': 'task_1',
                    'type': 'research_summary',
                    'assigned_agent': 'research_agent',
                    'status': 'assigned'
                }
            ]
        }
        
        # Test complete planning
        plan = self.planner.plan_newsletter(
            self.sample_content_sources,
            target_audience='mixed'
        )
        
        assert isinstance(plan, NewsletterPlan)
        assert plan.id.startswith('newsletter_')
        assert plan.target_audience == 'mixed'
        assert len(plan.content_items) > 0
        assert plan.outline is not None
        assert plan.assigned_tasks is not None
        assert plan.deadlines is not None
        assert plan.quality_requirements is not None
    
    def test_process_message(self):
        """Test message processing."""
        # Test plan request message
        message = Message(
            sender='test_sender',
            recipient='test_planner',
            type=MessageType.REQUEST,
            content={
                'action': 'plan_newsletter',
                'content_sources': self.sample_content_sources,
                'target_audience': 'mixed'
            }
        )
        
        with patch.object(self.planner, 'plan_newsletter') as mock_plan:
            mock_plan.return_value = Mock()
            response = self.planner.process_message(message)
            
            assert response is not None
            assert response.type == MessageType.RESPONSE
            assert 'plan' in response.content
    
    def test_process_message_error(self):
        """Test message processing with error."""
        # Test invalid message
        message = Message(
            sender='test_sender',
            recipient='test_planner',
            type=MessageType.REQUEST,
            content={
                'action': 'plan_newsletter',
                'content_sources': None  # Invalid content
            }
        )
        
        response = self.planner.process_message(message)
        
        assert response is not None
        assert response.type == MessageType.ERROR
        assert 'error' in response.content
    
    def test_get_available_agents(self):
        """Test available agents retrieval."""
        agents = self.planner._get_available_agents()
        
        expected_agents = [
            'research_summary_agent',
            'industry_news_agent',
            'technical_deep_dive_agent',
            'trend_analysis_agent',
            'interview_profile_agent'
        ]
        
        assert len(agents) == len(expected_agents)
        for agent in expected_agents:
            assert agent in agents 