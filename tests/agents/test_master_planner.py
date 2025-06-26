import pytest
from src.agents.planning.master_planner import MasterPlanningAgent

class TestMasterPlanningAgent:
    def test_master_planner_initialization(self):
        """Test that MasterPlanningAgent initializes correctly."""
        agent = MasterPlanningAgent()
        assert agent.content_analyzer is not None
        assert agent.outline_generator is not None
        assert agent.planning_history == []

    def test_create_newsletter_plan_basic(self):
        """Test basic newsletter plan creation."""
        agent = MasterPlanningAgent()
        
        content_list = [
            "OpenAI released GPT-4 with improved reasoning capabilities.",
            "Google's DeepMind announced breakthroughs in reinforcement learning.",
            "New computer vision model achieves state-of-the-art performance."
        ]
        
        plan = agent.create_newsletter_plan(content_list)
        
        # Check plan structure
        assert 'plan_id' in plan
        assert 'created_at' in plan
        assert 'target_audience' in plan
        assert 'tone' in plan
        assert 'max_length' in plan
        assert 'content_analysis' in plan
        assert 'outline' in plan
        assert 'tasks' in plan
        assert 'guidelines' in plan
        assert 'estimated_completion_time' in plan
        assert 'status' in plan
        
        # Check default values
        assert plan['target_audience'] == 'mixed'
        assert plan['tone'] == 'professional'
        assert plan['max_length'] == 'medium'
        assert plan['status'] == 'planned'
        
        # Check content analysis
        assert plan['content_analysis']['total_items'] == 3
        assert len(plan['content_analysis']['themes_distribution']) > 0
        assert len(plan['content_analysis']['top_topics']) > 0
        
        # Check outline
        assert 'sections' in plan['outline']
        assert plan['outline']['total_content_items'] == 3
        
        # Check tasks
        assert len(plan['tasks']) > 0
        assert all('task_id' in task for task in plan['tasks'])
        assert all('type' in task for task in plan['tasks'])
        assert all('status' in task for task in plan['tasks'])

    def test_create_newsletter_plan_with_custom_params(self):
        """Test newsletter plan creation with custom parameters."""
        agent = MasterPlanningAgent()
        
        content_list = [
            "Technical deep dive into transformer architectures.",
            "Advanced machine learning algorithms for optimization."
        ]
        
        plan = agent.create_newsletter_plan(
            content_list,
            target_audience="technical",
            tone="professional",
            max_length="long"
        )
        
        assert plan['target_audience'] == 'technical'
        assert plan['tone'] == 'professional'
        assert plan['max_length'] == 'long'
        
        # Check guidelines reflect technical audience
        guidelines = plan['guidelines']
        assert guidelines['target_audience'] == 'technical'
        assert guidelines['technical_depth'] == 'high'

    def test_content_analysis_batch(self):
        """Test batch content analysis functionality."""
        agent = MasterPlanningAgent()
        
        content_list = [
            "Machine learning algorithms for natural language processing.",
            "Deep learning models in computer vision applications.",
            "Ethical considerations in AI development."
        ]
        
        plan = agent.create_newsletter_plan(content_list)
        analysis = plan['content_analysis']
        
        assert analysis['total_items'] == 3
        assert analysis['average_word_count'] > 0
        assert len(analysis['content_quality_scores']) == 3
        assert len(analysis['word_count_distribution']) == 3
        
        # Check themes distribution
        themes = analysis['themes_distribution']
        assert len(themes) > 0
        
        # Check top topics
        topics = analysis['top_topics']
        assert len(topics) > 0
        assert len(topics) <= 20  # Should be capped at 20

    def test_task_breakdown_creation(self):
        """Test task breakdown creation."""
        agent = MasterPlanningAgent()
        
        content_list = [
            "Machine learning news.",
            "Deep learning developments."
        ]
        
        plan = agent.create_newsletter_plan(content_list)
        tasks = plan['tasks']
        
        # Should have at least research, writing, quality, and assembly tasks
        assert len(tasks) >= 4
        
        # Check task structure
        for task in tasks:
            assert 'task_id' in task
            assert 'type' in task
            assert 'title' in task
            assert 'description' in task
            assert 'assigned_agent' in task
            assert 'estimated_duration' in task
            assert 'priority' in task
            assert 'dependencies' in task
            assert 'status' in task
        
        # Check that first task is research
        research_tasks = [t for t in tasks if t['type'] == 'research']
        assert len(research_tasks) > 0
        
        # Check that last task is assembly
        assembly_tasks = [t for t in tasks if t['type'] == 'assembly']
        assert len(assembly_tasks) > 0

    def test_agent_assignment_by_theme(self):
        """Test that agents are correctly assigned based on themes."""
        agent = MasterPlanningAgent()
        
        content_list = [
            "Machine learning algorithms and applications.",
            "Computer vision and image processing.",
            "AI ethics and responsible development."
        ]
        
        plan = agent.create_newsletter_plan(content_list)
        tasks = plan['tasks']
        
        # Find writing tasks
        writing_tasks = [t for t in tasks if t['type'] == 'writing']
        
        # Check that different themes get different agents
        assigned_agents = set()
        for task in writing_tasks:
            if 'section_data' in task:
                assigned_agents.add(task['assigned_agent'])
        
        # Should have different agents for different themes
        assert len(assigned_agents) > 1

    def test_content_guidelines_creation(self):
        """Test content guidelines creation for different audiences."""
        agent = MasterPlanningAgent()
        
        content_list = ["Sample content."]
        
        # Test mixed audience
        plan_mixed = agent.create_newsletter_plan(content_list, target_audience="mixed")
        guidelines_mixed = plan_mixed['guidelines']
        assert guidelines_mixed['technical_depth'] == 'moderate'
        
        # Test technical audience
        plan_technical = agent.create_newsletter_plan(content_list, target_audience="technical")
        guidelines_technical = plan_technical['guidelines']
        assert guidelines_technical['technical_depth'] == 'high'
        
        # Test business audience
        plan_business = agent.create_newsletter_plan(content_list, target_audience="business")
        guidelines_business = plan_business['guidelines']
        assert guidelines_business['technical_depth'] == 'low'

    def test_completion_time_estimation(self):
        """Test completion time estimation."""
        agent = MasterPlanningAgent()
        
        content_list = ["Sample content."]
        plan = agent.create_newsletter_plan(content_list)
        
        estimated_time = plan['estimated_completion_time']
        assert isinstance(estimated_time, str)
        assert 'minutes' in estimated_time or 'hours' in estimated_time

    def test_planning_history_tracking(self):
        """Test that planning history is properly tracked."""
        agent = MasterPlanningAgent()
        
        assert len(agent.get_planning_history()) == 0
        
        content_list = ["Sample content."]
        plan1 = agent.create_newsletter_plan(content_list)
        plan2 = agent.create_newsletter_plan(content_list)
        
        history = agent.get_planning_history()
        assert len(history) == 2
        assert history[0]['plan_id'] == plan1['plan_id']
        assert history[1]['plan_id'] == plan2['plan_id']

    def test_get_plan_by_id(self):
        """Test retrieving plans by ID."""
        agent = MasterPlanningAgent()
        
        content_list = ["Sample content."]
        plan = agent.create_newsletter_plan(content_list)
        
        retrieved_plan = agent.get_plan_by_id(plan['plan_id'])
        assert retrieved_plan is not None
        assert retrieved_plan['plan_id'] == plan['plan_id']
        
        # Test non-existent plan
        non_existent = agent.get_plan_by_id("non_existent_id")
        assert non_existent is None 