"""
Tests for Enhanced Task Assignment System

Tests intelligent task distribution among specialized agents with capability
matching, workload balancing, and dependency management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.agents.tasks.task_assignment import TaskAssignment, AgentCapability, Task


class TestTaskAssignment:
    """Test cases for TaskAssignment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.task_assignment = TaskAssignment()
        
        # Sample tasks for testing
        self.sample_tasks = [
            {
                'id': 'task_1',
                'type': 'research_summary',
                'requirements': {
                    'length': 300,
                    'technical_depth': 'high',
                    'include_citations': True
                }
            },
            {
                'id': 'task_2',
                'type': 'industry_news',
                'requirements': {
                    'length': 250,
                    'business_focus': True,
                    'include_quotes': True
                }
            },
            {
                'id': 'task_3',
                'type': 'technical_deep_dive',
                'requirements': {
                    'length': 400,
                    'include_code_examples': True,
                    'technical_depth': 'expert'
                }
            }
        ]
    
    def test_task_assignment_initialization(self):
        """Test TaskAssignment initialization."""
        assert len(self.task_assignment.agents) == 5  # Default agents
        assert 'research_summary_agent' in self.task_assignment.agents
        assert 'industry_news_agent' in self.task_assignment.agents
        assert 'technical_deep_dive_agent' in self.task_assignment.agents
        assert 'trend_analysis_agent' in self.task_assignment.agents
        assert 'interview_profile_agent' in self.task_assignment.agents
    
    def test_default_agent_capabilities(self):
        """Test default agent capabilities."""
        research_agent = self.task_assignment.agents['research_summary_agent']
        assert research_agent.agent_id == 'research_summary_agent'
        assert 'research_summary' in research_agent.specializations
        assert research_agent.max_concurrent_tasks == 3
        assert research_agent.current_workload == 0
        assert research_agent.availability is True
        assert research_agent.efficiency_scores['research_summary'] == 0.95
        
        industry_agent = self.task_assignment.agents['industry_news_agent']
        assert industry_agent.agent_id == 'industry_news_agent'
        assert 'industry_news' in industry_agent.specializations
        assert industry_agent.max_concurrent_tasks == 4
        assert industry_agent.efficiency_scores['industry_news'] == 0.9
    
    def test_create_task_objects(self):
        """Test task object creation from dictionaries."""
        task_objects = self.task_assignment._create_task_objects(self.sample_tasks)
        
        assert len(task_objects) == 3
        
        for task in task_objects:
            assert isinstance(task, Task)
            assert task.id in ['task_1', 'task_2', 'task_3']
            assert task.type in ['research_summary', 'industry_news', 'technical_deep_dive']
            assert isinstance(task.priority, str)
            assert isinstance(task.estimated_duration, timedelta)
            assert isinstance(task.requirements, dict)
            assert task.status == "pending"
            assert task.assigned_agent is None
    
    def test_determine_priority(self):
        """Test task priority determination."""
        # High priority tasks
        high_priority_task = {'id': 'task_1', 'type': 'research_summary'}
        priority = self.task_assignment._determine_priority(high_priority_task)
        assert priority == 'high'
        
        # Medium priority tasks
        medium_priority_task = {'id': 'task_2', 'type': 'industry_news'}
        priority = self.task_assignment._determine_priority(medium_priority_task)
        assert priority == 'medium'
        
        # Low priority tasks
        low_priority_task = {'id': 'task_3', 'type': 'general_content'}
        priority = self.task_assignment._determine_priority(low_priority_task)
        assert priority == 'low'
    
    def test_estimate_duration(self):
        """Test task duration estimation."""
        # Research summary task
        research_task = {
            'id': 'task_1',
            'type': 'research_summary',
            'requirements': {'length': 300}
        }
        duration = self.task_assignment._estimate_duration(research_task)
        assert isinstance(duration, timedelta)
        assert duration.total_seconds() > 0
        
        # Technical deep dive task
        technical_task = {
            'id': 'task_2',
            'type': 'technical_deep_dive',
            'requirements': {'length': 400}
        }
        duration = self.task_assignment._estimate_duration(technical_task)
        assert isinstance(duration, timedelta)
        # Technical tasks should take longer
        assert duration.total_seconds() > 0
    
    def test_sort_tasks_by_priority(self):
        """Test task sorting by priority."""
        tasks = [
            Task(
                id='task_1',
                type='research_summary',
                priority='low',
                estimated_duration=timedelta(minutes=30),
                requirements={},
                created_at=datetime.now() + timedelta(minutes=1)
            ),
            Task(
                id='task_2',
                type='industry_news',
                priority='high',
                estimated_duration=timedelta(minutes=30),
                requirements={},
                created_at=datetime.now()
            ),
            Task(
                id='task_3',
                type='technical_deep_dive',
                priority='medium',
                estimated_duration=timedelta(minutes=30),
                requirements={},
                created_at=datetime.now()
            )
        ]
        
        sorted_tasks = self.task_assignment._sort_tasks_by_priority(tasks)
        
        # High priority should come first
        assert sorted_tasks[0].priority == 'high'
        # Medium priority should come second
        assert sorted_tasks[1].priority == 'medium'
        # Low priority should come last
        assert sorted_tasks[2].priority == 'low'
    
    def test_calculate_suitability_score(self):
        """Test agent suitability score calculation."""
        task = Task(
            id='task_1',
            type='research_summary',
            priority='high',
            estimated_duration=timedelta(minutes=30),
            requirements={}
        )
        
        # Perfect match agent
        perfect_agent = AgentCapability(
            agent_id='perfect_agent',
            specializations=['research_summary'],
            max_concurrent_tasks=3,
            current_workload=0,
            efficiency_scores={'research_summary': 0.95}
        )
        
        perfect_score = self.task_assignment._calculate_suitability_score(task, perfect_agent)
        
        # Good match agent
        good_agent = AgentCapability(
            agent_id='good_agent',
            specializations=['academic_papers'],
            max_concurrent_tasks=3,
            current_workload=1,
            efficiency_scores={'research_summary': 0.8}
        )
        
        good_score = self.task_assignment._calculate_suitability_score(task, good_agent)
        
        # Poor match agent
        poor_agent = AgentCapability(
            agent_id='poor_agent',
            specializations=['industry_news'],
            max_concurrent_tasks=3,
            current_workload=2,
            efficiency_scores={'research_summary': 0.5}
        )
        
        poor_score = self.task_assignment._calculate_suitability_score(task, poor_agent)
        
        # Perfect match should have highest score
        assert perfect_score > good_score
        assert good_score > poor_score
        assert perfect_score > 0.8  # Should be very high for perfect match
    
    def test_find_best_agent_for_task(self):
        """Test finding the best agent for a task."""
        # Create a task
        task = Task(
            id='task_1',
            type='research_summary',
            priority='high',
            estimated_duration=timedelta(minutes=30),
            requirements={}
        )
        
        # Add custom agents with different capabilities
        self.task_assignment.add_agent(AgentCapability(
            agent_id='custom_research_agent',
            specializations=['research_summary'],
            max_concurrent_tasks=2,
            current_workload=0,
            efficiency_scores={'research_summary': 0.9}
        ))
        
        best_agent = self.task_assignment._find_best_agent_for_task(task)
        
        # Should find an agent
        assert best_agent is not None
        # Should prefer research_summary_agent or custom_research_agent
        assert 'research' in best_agent
    
    def test_assign_tasks(self):
        """Test complete task assignment."""
        assignments = self.task_assignment.assign_tasks(self.sample_tasks)
        
        # Should have assignments
        assert len(assignments) > 0
        
        # Check that tasks are assigned to appropriate agents
        for agent_id, tasks in assignments.items():
            assert len(tasks) > 0
            for task in tasks:
                assert task['assigned_agent'] == agent_id
                assert task['status'] == 'assigned'
                
                # Check that task type matches agent specialization
                agent = self.task_assignment.agents[agent_id]
                task_type = task['type']
                assert (task_type in agent.specializations or 
                       any(spec in task_type for spec in agent.specializations))
    
    def test_workload_balancing(self):
        """Test workload balancing across agents."""
        # Create multiple tasks of the same type
        many_tasks = [
            {
                'id': f'task_{i}',
                'type': 'research_summary',
                'requirements': {'length': 200}
            }
            for i in range(5)
        ]
        
        assignments = self.task_assignment.assign_tasks(many_tasks)
        
        # Should distribute tasks across available research agents
        research_agents = ['research_summary_agent', 'trend_analysis_agent']
        assigned_agents = set(assignments.keys())
        
        # Should use multiple agents for workload balancing
        assert len(assigned_agents) > 1
        
        # Check workload distribution
        for agent_id, tasks in assignments.items():
            agent = self.task_assignment.agents[agent_id]
            assert len(tasks) <= agent.max_concurrent_tasks
    
    def test_agent_availability(self):
        """Test agent availability handling."""
        # Make an agent unavailable
        self.task_assignment.set_agent_availability('research_summary_agent', False)
        
        # Create a task that would normally go to research_summary_agent
        task = {
            'id': 'task_1',
            'type': 'research_summary',
            'requirements': {'length': 200}
        }
        
        assignments = self.task_assignment.assign_tasks([task])
        
        # Should not assign to unavailable agent
        if 'research_summary_agent' in assignments:
            assert len(assignments['research_summary_agent']) == 0
        
        # Should assign to alternative agent
        assert len(assignments) > 0
    
    def test_add_and_remove_agents(self):
        """Test adding and removing agents."""
        # Add a new agent
        new_agent = AgentCapability(
            agent_id='new_agent',
            specializations=['custom_task'],
            max_concurrent_tasks=2,
            efficiency_scores={'custom_task': 0.8}
        )
        
        self.task_assignment.add_agent(new_agent)
        assert 'new_agent' in self.task_assignment.agents
        
        # Remove the agent
        self.task_assignment.remove_agent('new_agent')
        assert 'new_agent' not in self.task_assignment.agents
    
    def test_update_agent_workload(self):
        """Test updating agent workload."""
        agent_id = 'research_summary_agent'
        initial_workload = self.task_assignment.agents[agent_id].current_workload
        
        # Increase workload
        self.task_assignment.update_agent_workload(agent_id, 1)
        assert self.task_assignment.agents[agent_id].current_workload == initial_workload + 1
        
        # Decrease workload
        self.task_assignment.update_agent_workload(agent_id, -1)
        assert self.task_assignment.agents[agent_id].current_workload == initial_workload
        
        # Ensure workload doesn't go below 0
        self.task_assignment.update_agent_workload(agent_id, -10)
        assert self.task_assignment.agents[agent_id].current_workload == 0
    
    def test_get_agent_status(self):
        """Test getting agent status."""
        agent_id = 'research_summary_agent'
        status = self.task_assignment.get_agent_status(agent_id)
        
        assert status is not None
        assert status['agent_id'] == agent_id
        assert 'specializations' in status
        assert 'current_workload' in status
        assert 'max_concurrent_tasks' in status
        assert 'availability' in status
        assert 'efficiency_scores' in status
        
        # Test non-existent agent
        status = self.task_assignment.get_agent_status('non_existent_agent')
        assert status is None
    
    def test_get_system_status(self):
        """Test getting overall system status."""
        status = self.task_assignment.get_system_status()
        
        assert 'total_agents' in status
        assert 'available_agents' in status
        assert 'total_workload' in status
        assert 'agent_details' in status
        
        assert status['total_agents'] == len(self.task_assignment.agents)
        assert status['available_agents'] >= 0
        assert status['total_workload'] >= 0
        assert len(status['agent_details']) == status['total_agents']
    
    def test_validate_dependencies(self):
        """Test dependency validation."""
        # Valid tasks with dependencies
        valid_tasks = [
            {'id': 'task_1', 'dependencies': []},
            {'id': 'task_2', 'dependencies': ['task_1']},
            {'id': 'task_3', 'dependencies': ['task_1', 'task_2']}
        ]
        
        issues = self.task_assignment.validate_dependencies(valid_tasks)
        assert len(issues) == 0
        
        # Invalid tasks with missing dependencies
        invalid_tasks = [
            {'id': 'task_1', 'dependencies': []},
            {'id': 'task_2', 'dependencies': ['task_1']},
            {'id': 'task_3', 'dependencies': ['task_1', 'missing_task']}
        ]
        
        issues = self.task_assignment.validate_dependencies(invalid_tasks)
        assert len(issues) == 1
        assert 'missing_task' in issues[0]
    
    def test_task_to_dict(self):
        """Test converting Task object to dictionary."""
        task = Task(
            id='test_task',
            type='research_summary',
            priority='high',
            estimated_duration=timedelta(minutes=45),
            requirements={'length': 300},
            dependencies=['task_1'],
            assigned_agent='research_agent',
            status='assigned'
        )
        
        task_dict = self.task_assignment._task_to_dict(task)
        
        assert task_dict['id'] == 'test_task'
        assert task_dict['type'] == 'research_summary'
        assert task_dict['priority'] == 'high'
        assert task_dict['estimated_duration'] == '0:45:00'
        assert task_dict['requirements'] == {'length': 300}
        assert task_dict['dependencies'] == ['task_1']
        assert task_dict['assigned_agent'] == 'research_agent'
        assert task_dict['status'] == 'assigned' 