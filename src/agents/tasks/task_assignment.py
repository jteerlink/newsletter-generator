"""
Enhanced Task Assignment System

Handles intelligent task distribution among specialized agents with capability
matching, workload balancing, and dependency management.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Represents an agent's capabilities and specializations."""
    agent_id: str
    specializations: List[str]
    max_concurrent_tasks: int
    current_workload: int = 0
    efficiency_scores: Dict[str, float] = None
    availability: bool = True


@dataclass
class Task:
    """Represents a task with requirements and dependencies."""
    id: str
    type: str
    priority: str
    estimated_duration: timedelta
    requirements: Dict[str, Any]
    dependencies: List[str] = None
    assigned_agent: str = None
    status: str = "pending"
    created_at: datetime = None


class TaskAssignment:
    """
    Enhanced task assignment system with intelligent agent matching.
    
    Features:
    - Agent capability matching
    - Workload balancing
    - Dependency management
    - Priority-based scheduling
    - Efficiency optimization
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.tasks: Dict[str, Task] = {}
        self.assignments: Dict[str, List[Task]] = defaultdict(list)
        
        # Initialize default agent capabilities
        self._initialize_default_agents()
        
        logger.info("TaskAssignment system initialized")
    
    def _initialize_default_agents(self):
        """Initialize default specialized agents."""
        default_agents = {
            'research_summary_agent': AgentCapability(
                agent_id='research_summary_agent',
                specializations=['research_summary', 'academic_papers', 'technical_analysis'],
                max_concurrent_tasks=3,
                efficiency_scores={
                    'research_summary': 0.95,
                    'academic_papers': 0.9,
                    'technical_analysis': 0.85
                }
            ),
            'industry_news_agent': AgentCapability(
                agent_id='industry_news_agent',
                specializations=['industry_news', 'company_announcements', 'market_analysis'],
                max_concurrent_tasks=4,
                efficiency_scores={
                    'industry_news': 0.9,
                    'company_announcements': 0.95,
                    'market_analysis': 0.85
                }
            ),
            'technical_deep_dive_agent': AgentCapability(
                agent_id='technical_deep_dive_agent',
                specializations=['technical_deep_dive', 'code_examples', 'implementation_guides'],
                max_concurrent_tasks=2,
                efficiency_scores={
                    'technical_deep_dive': 0.95,
                    'code_examples': 0.9,
                    'implementation_guides': 0.9
                }
            ),
            'trend_analysis_agent': AgentCapability(
                agent_id='trend_analysis_agent',
                specializations=['trend_analysis', 'pattern_recognition', 'predictions'],
                max_concurrent_tasks=3,
                efficiency_scores={
                    'trend_analysis': 0.9,
                    'pattern_recognition': 0.85,
                    'predictions': 0.8
                }
            ),
            'interview_profile_agent': AgentCapability(
                agent_id='interview_profile_agent',
                specializations=['interviews', 'profiles', 'conference_coverage'],
                max_concurrent_tasks=2,
                efficiency_scores={
                    'interviews': 0.9,
                    'profiles': 0.85,
                    'conference_coverage': 0.8
                }
            )
        }
        
        self.agents.update(default_agents)
    
    def assign_tasks(self, tasks: List[Dict], available_agents: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Assign tasks to agents using intelligent matching.
        
        Args:
            tasks: List of task dictionaries
            available_agents: List of available agent IDs (if None, use all agents)
            
        Returns:
            Dictionary mapping agent IDs to assigned tasks
        """
        logger.info(f"Starting task assignment for {len(tasks)} tasks")
        
        # Convert task dictionaries to Task objects
        task_objects = self._create_task_objects(tasks)
        
        # Filter available agents
        if available_agents:
            available_agent_ids = set(available_agents)
            self.agents = {k: v for k, v in self.agents.items() if k in available_agent_ids}
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_priority(task_objects)
        
        # Assign tasks
        assignments = defaultdict(list)
        
        for task in sorted_tasks:
            assigned_agent = self._find_best_agent_for_task(task)
            if assigned_agent:
                task.assigned_agent = assigned_agent
                task.status = "assigned"
                assignments[assigned_agent].append(self._task_to_dict(task))
                self.agents[assigned_agent].current_workload += 1
                logger.info(f"Assigned task {task.id} to {assigned_agent}")
            else:
                logger.warning(f"No suitable agent found for task {task.id}")
        
        return dict(assignments)
    
    def _create_task_objects(self, tasks: List[Dict]) -> List[Task]:
        """Convert task dictionaries to Task objects."""
        task_objects = []
        
        for task_dict in tasks:
            task = Task(
                id=task_dict['id'],
                type=task_dict['type'],
                priority=self._determine_priority(task_dict),
                estimated_duration=self._estimate_duration(task_dict),
                requirements=task_dict.get('requirements', {}),
                dependencies=task_dict.get('dependencies', []),
                created_at=datetime.now()
            )
            task_objects.append(task)
            self.tasks[task.id] = task
        
        return task_objects
    
    def _determine_priority(self, task_dict: Dict) -> str:
        """Determine task priority based on type and requirements."""
        task_type = task_dict['type']
        
        # High priority tasks
        if task_type in ['research_summary', 'technical_deep_dive']:
            return 'high'
        
        # Medium priority tasks
        if task_type in ['industry_news', 'trend_analysis']:
            return 'medium'
        
        # Low priority tasks
        return 'low'
    
    def _estimate_duration(self, task_dict: Dict) -> timedelta:
        """Estimate task duration based on type and requirements."""
        task_type = task_dict['type']
        requirements = task_dict.get('requirements', {})
        length = requirements.get('length', 200)
        
        # Base duration estimates (minutes)
        base_durations = {
            'research_summary': 45,
            'industry_news': 30,
            'technical_deep_dive': 60,
            'trend_analysis': 40,
            'interview_profile': 35,
            'general_content': 25
        }
        
        base_minutes = base_durations.get(task_type, 30)
        
        # Adjust for content length
        length_factor = length / 200  # Normalize to 200 words
        adjusted_minutes = int(base_minutes * length_factor)
        
        return timedelta(minutes=adjusted_minutes)
    
    def _sort_tasks_by_priority(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by priority and dependencies."""
        # Priority order
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        def task_sort_key(task):
            # Primary: priority (higher first)
            priority_score = priority_order.get(task.priority, 0)
            
            # Secondary: creation time (earlier first)
            time_score = task.created_at.timestamp()
            
            # Tertiary: dependencies (fewer dependencies first)
            dependency_count = len(task.dependencies or [])
            
            return (-priority_score, time_score, dependency_count)
        
        return sorted(tasks, key=task_sort_key)
    
    def _find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best available agent for a given task."""
        best_agent = None
        best_score = 0
        
        for agent_id, agent in self.agents.items():
            if not agent.availability:
                continue
            
            if agent.current_workload >= agent.max_concurrent_tasks:
                continue
            
            # Calculate agent suitability score
            suitability_score = self._calculate_suitability_score(task, agent)
            
            if suitability_score > best_score:
                best_score = suitability_score
                best_agent = agent_id
        
        return best_agent
    
    def _calculate_suitability_score(self, task: Task, agent: AgentCapability) -> float:
        """Calculate how suitable an agent is for a specific task."""
        # Base specialization match
        specialization_match = 0.0
        if task.type in agent.specializations:
            specialization_match = 1.0
        elif any(spec in task.type for spec in agent.specializations):
            specialization_match = 0.7
        
        # Efficiency score for this task type
        efficiency_score = agent.efficiency_scores.get(task.type, 0.5)
        
        # Workload factor (prefer less busy agents)
        workload_factor = 1.0 - (agent.current_workload / agent.max_concurrent_tasks)
        
        # Priority alignment
        priority_alignment = 1.0
        if task.priority == 'high' and agent.current_workload < 2:
            priority_alignment = 1.2  # Boost for high-priority tasks
        
        # Calculate overall score
        overall_score = (
            specialization_match * 0.4 +
            efficiency_score * 0.3 +
            workload_factor * 0.2 +
            priority_alignment * 0.1
        )
        
        return overall_score
    
    def _task_to_dict(self, task: Task) -> Dict:
        """Convert Task object back to dictionary for assignment output."""
        return {
            'id': task.id,
            'type': task.type,
            'priority': task.priority,
            'estimated_duration': str(task.estimated_duration),
            'requirements': task.requirements,
            'dependencies': task.dependencies,
            'assigned_agent': task.assigned_agent,
            'status': task.status
        }
    
    def add_agent(self, agent_capability: AgentCapability):
        """Add a new agent to the system."""
        self.agents[agent_capability.agent_id] = agent_capability
        logger.info(f"Added agent: {agent_capability.agent_id}")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent: {agent_id}")
    
    def update_agent_workload(self, agent_id: str, workload_change: int):
        """Update an agent's current workload."""
        if agent_id in self.agents:
            self.agents[agent_id].current_workload += workload_change
            self.agents[agent_id].current_workload = max(0, self.agents[agent_id].current_workload)
    
    def set_agent_availability(self, agent_id: str, available: bool):
        """Set agent availability status."""
        if agent_id in self.agents:
            self.agents[agent_id].availability = available
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get current status of an agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            return {
                'agent_id': agent.agent_id,
                'specializations': agent.specializations,
                'current_workload': agent.current_workload,
                'max_concurrent_tasks': agent.max_concurrent_tasks,
                'availability': agent.availability,
                'efficiency_scores': agent.efficiency_scores
            }
        return None
    
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        total_agents = len(self.agents)
        available_agents = sum(1 for agent in self.agents.values() if agent.availability)
        total_workload = sum(agent.current_workload for agent in self.agents.values())
        
        return {
            'total_agents': total_agents,
            'available_agents': available_agents,
            'total_workload': total_workload,
            'agent_details': {
                agent_id: self.get_agent_status(agent_id)
                for agent_id in self.agents.keys()
            }
        }
    
    def optimize_assignments(self) -> Dict[str, List[Dict]]:
        """Re-optimize existing assignments for better efficiency."""
        # Get all currently assigned tasks
        all_assigned_tasks = []
        for tasks in self.assignments.values():
            all_assigned_tasks.extend(tasks)
        
        # Reset agent workloads
        for agent in self.agents.values():
            agent.current_workload = 0
        
        # Re-assign tasks
        return self.assign_tasks(all_assigned_tasks)
    
    def validate_dependencies(self, tasks: List[Dict]) -> List[str]:
        """Validate task dependencies and return any issues."""
        issues = []
        task_ids = {task['id'] for task in tasks}
        
        for task in tasks:
            dependencies = task.get('dependencies', [])
            for dep in dependencies:
                if dep not in task_ids:
                    issues.append(f"Task {task['id']} depends on non-existent task {dep}")
        
        return issues 