"""
Management Agent for Newsletter Generation

This module provides the ManagerAgent class, which is responsible for coordinating
workflows, managing agent interactions, and overseeing the newsletter generation process.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .base import SimpleAgent, AgentType, TaskResult, TaskStatus
from src.core.core import query_llm
from src.core.template_manager import NewsletterType, AIMLTemplateManager
from src.quality import NewsletterQualityGate, QualityGateStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """Represents a step in a workflow."""
    step_id: str
    name: str
    description: str
    agent_type: str
    dependencies: List[str] = None
    estimated_time: int = 30
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class WorkflowPlan:
    """Represents a complete workflow plan."""
    workflow_id: str
    topic: str
    complexity: str
    template_type: NewsletterType
    steps: List[WorkflowStep]
    estimated_total_time: int
    quality_gates: List[str]
    created_at: float = time.time()


class ManagerAgent(SimpleAgent):
    """Agent specialized in workflow management and coordination."""
    
    def __init__(self, name: str = "ManagerAgent", **kwargs):
        super().__init__(
            name=name,
            role="Workflow Manager",
            goal="Coordinate and manage newsletter generation workflows efficiently",
            backstory="""You are an experienced project manager specializing in content 
            creation workflows. You excel at planning, coordinating, and overseeing 
            complex multi-agent processes. You understand how to break down large tasks 
            into manageable steps, assign appropriate agents to each step, and ensure 
            quality standards are met throughout the process. You can adapt workflows 
            based on complexity, requirements, and available resources.""",
            agent_type=AgentType.MANAGER,
            tools=[],  # Managers coordinate rather than execute
            **kwargs
        )
        self.template_manager = AIMLTemplateManager()
        self.quality_gate = NewsletterQualityGate()
        self.active_workflows: Dict[str, WorkflowPlan] = {}
    
    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute management task with workflow coordination."""
        logger.info(f"ManagerAgent executing management task: {task}")
        
        # Extract management parameters
        workflow_type = kwargs.get('workflow_type', 'standard')
        complexity = kwargs.get('complexity', 'standard')
        
        if 'create workflow' in task.lower() or 'plan' in task.lower():
            return self._create_workflow_plan(task, context, complexity, **kwargs)
        elif 'execute workflow' in task.lower() or 'run' in task.lower():
            return self._execute_workflow(task, context, **kwargs)
        elif 'monitor' in task.lower() or 'status' in task.lower():
            return self._monitor_workflows(task, context, **kwargs)
        else:
            return self._general_management_task(task, context, **kwargs)
    
    def _create_workflow_plan(self, task: str, context: str, complexity: str, **kwargs) -> str:
        """Create a comprehensive workflow plan."""
        try:
            # Extract topic from task
            topic = self._extract_topic_from_task(task)
            
            # Determine template type
            template_type = self._select_template_for_topic(topic)
            
            # Create workflow plan
            workflow_plan = self.create_hierarchical_workflow(topic, complexity)
            
            # Store workflow plan
            workflow_id = f"workflow_{int(time.time())}"
            self.active_workflows[workflow_id] = WorkflowPlan(
                workflow_id=workflow_id,
                topic=topic,
                complexity=complexity,
                template_type=template_type,
                steps=self._convert_to_workflow_steps(workflow_plan),
                estimated_total_time=workflow_plan.get('estimated_time', 120),
                quality_gates=workflow_plan.get('quality_gates', [])
            )
            
            return self._format_workflow_plan(workflow_plan, workflow_id)
            
        except Exception as e:
            logger.error(f"Error creating workflow plan: {e}")
            return f"Failed to create workflow plan: {e}"
    
    def _execute_workflow(self, task: str, context: str, **kwargs) -> str:
        """Execute a workflow with available agents."""
        try:
            # Extract workflow ID or create new one
            workflow_id = kwargs.get('workflow_id')
            if not workflow_id:
                # Create new workflow
                return self._create_and_execute_workflow(task, context, **kwargs)
            
            # Get workflow plan
            workflow_plan = self.active_workflows.get(workflow_id)
            if not workflow_plan:
                return f"Workflow {workflow_id} not found"
            
            # Get available agents
            available_agents = kwargs.get('available_agents', [])
            
            # Execute workflow
            results = self.execute_hierarchical_workflow(workflow_plan.__dict__, available_agents)
            
            return self._format_workflow_results(results, workflow_id)
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return f"Failed to execute workflow: {e}"
    
    def _monitor_workflows(self, task: str, context: str, **kwargs) -> str:
        """Monitor active workflows and their status."""
        try:
            if not self.active_workflows:
                return "No active workflows to monitor"
            
            monitoring_report = []
            for workflow_id, workflow in self.active_workflows.items():
                status = self._get_workflow_status(workflow)
                monitoring_report.append(f"""
                **Workflow {workflow_id}**
                - Topic: {workflow.topic}
                - Status: {status}
                - Steps: {len(workflow.steps)}
                - Estimated Time: {workflow.estimated_total_time} minutes
                - Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow.created_at))}
                """)
            
            return "\n".join(monitoring_report)
            
        except Exception as e:
            logger.error(f"Error monitoring workflows: {e}")
            return f"Failed to monitor workflows: {e}"
    
    def _general_management_task(self, task: str, context: str, **kwargs) -> str:
        """Handle general management tasks."""
        management_prompt = f"""
        You are a {self.role}.
        Your goal: {self.goal}
        Background: {self.backstory}
        
        Management Task: {task}
        Context: {context}
        
        Please provide management guidance, planning, or coordination for this task.
        Focus on:
        1. Task breakdown and planning
        2. Resource allocation
        3. Timeline estimation
        4. Risk assessment
        5. Quality assurance
        6. Process optimization
        
        Provide a structured response with clear recommendations.
        """
        
        try:
            return query_llm(management_prompt)
        except Exception as e:
            logger.error(f"Error in general management task: {e}")
            return f"Management task failed: {e}"
    
    def create_hierarchical_workflow(self, topic: str, complexity: str = "standard") -> Dict[str, Any]:
        """Create a hierarchical workflow plan for newsletter generation (simplified deep dive only)."""
        try:
            # Use unified deep dive workflow for all content
            workflow_structure = self._create_deep_dive_workflow(topic)
            
            # Add basic quality gates
            workflow_structure['quality_gates'] = ["content_quality", "technical_accuracy"]
            
            # Calculate estimated time
            workflow_structure['estimated_time'] = self._calculate_estimated_time(workflow_structure)
            
            return workflow_structure
            
        except Exception as e:
            logger.error(f"Error creating hierarchical workflow: {e}")
            return self._create_fallback_workflow(topic)
    
    def _create_deep_dive_workflow(self, topic: str) -> Dict[str, Any]:
        """Create a unified deep dive workflow for all newsletter content."""
        return {
            'topic': topic,
            'complexity': 'deep_dive',
            'streams': [
                {
                    'name': 'Research Stream',
                    'description': 'Comprehensive research and analysis',
                    'tasks': [
                        {
                            'name': 'Deep Research',
                            'description': f'Comprehensive research on {topic}',
                            'agent_type': 'research',
                            'estimated_time': 30
                        }
                    ]
                },
                {
                    'name': 'Content Creation Stream',
                    'description': 'Content writing and editing',
                    'tasks': [
                        {
                            'name': 'Write Content',
                            'description': f'Write comprehensive newsletter content about {topic}',
                            'agent_type': 'writer',
                            'estimated_time': 40
                        },
                        {
                            'name': 'Edit Content',
                            'description': f'Edit and improve newsletter content',
                            'agent_type': 'editor',
                            'estimated_time': 25
                        }
                    ]
                }
            ],
            'estimated_time': 95
        }
    
    def _create_fallback_workflow(self, topic: str) -> Dict[str, Any]:
        """Create a fallback workflow if main creation fails."""
        return {
            'topic': topic,
            'complexity': 'standard',
            'streams': [
                {
                    'name': 'Basic Workflow',
                    'description': 'Basic newsletter generation',
                    'tasks': [
                        {
                            'name': 'Research and Write',
                            'description': f'Research and write about {topic}',
                            'agent_type': 'research',
                            'estimated_time': 30
                        }
                    ]
                }
            ],
            'estimated_time': 30,
            'quality_gates': ['basic_quality']
        }
    
    def _determine_quality_gates(self, complexity: str) -> List[str]:
        """Determine appropriate quality gates (simplified for deep dive only)."""
        return ["content_quality", "technical_accuracy"]
    
    def _calculate_estimated_time(self, workflow: Dict[str, Any]) -> int:
        """Calculate estimated total time for workflow."""
        total_time = 0
        for stream in workflow.get('streams', []):
            for task in stream.get('tasks', []):
                total_time += task.get('estimated_time', 30)
        return total_time
    
    def execute_hierarchical_workflow(self, workflow_plan: Dict[str, Any], 
                                    available_agents: List[SimpleAgent]) -> Dict[str, Any]:
        """Execute a hierarchical workflow with available agents."""
        try:
            results = {
                'workflow_id': workflow_plan.get('workflow_id', 'unknown'),
                'topic': workflow_plan.get('topic', 'unknown'),
                'status': 'completed',
                'streams': {},
                'quality_gates': {},
                'total_time': 0,
                'errors': []
            }
            
            start_time = time.time()
            
            # Execute parallel streams
            stream_results = self._execute_parallel_streams(
                workflow_plan.get('streams', []), 
                available_agents
            )
            results['streams'] = stream_results
            
            # Execute quality gates
            quality_results = self._execute_quality_gates(
                workflow_plan.get('quality_gates', []),
                stream_results
            )
            results['quality_gates'] = quality_results
            
            # Calculate total time
            results['total_time'] = time.time() - start_time
            
            # Check for errors
            if any('error' in str(v).lower() for v in stream_results.values()):
                results['status'] = 'completed_with_errors'
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing hierarchical workflow: {e}")
            return {
                'workflow_id': workflow_plan.get('workflow_id', 'unknown'),
                'status': 'failed',
                'error': str(e),
                'streams': {},
                'quality_gates': {}
            }
    
    def _execute_parallel_streams(self, streams: List[Dict], 
                                available_agents: List[SimpleAgent]) -> Dict[str, Any]:
        """Execute workflow streams sequentially (simplified approach)."""
        stream_results = {}
        context = ""  # Simple context passing
        
        for stream in streams:
            stream_name = stream.get('name', 'unknown')
            try:
                stream_result = self._execute_single_stream(stream, available_agents, context)
                stream_results[stream_name] = stream_result
                # Update context with results for next stream
                if stream_result.get('status') == 'completed':
                    context += f"\n\n{stream_name} Results:\n{stream_result.get('final_result', '')}"
            except Exception as e:
                logger.error(f"Error executing stream {stream_name}: {e}")
                stream_results[stream_name] = {'status': 'failed', 'error': str(e)}
        
        return stream_results
    
    def _execute_single_stream(self, stream: Dict, available_agents: List[SimpleAgent], 
                             context: str) -> Dict[str, Any]:
        """Execute a single workflow stream (simplified sequential execution)."""
        stream_name = stream.get('name', 'unknown')
        tasks = stream.get('tasks', [])
        
        stream_result = {
            'name': stream_name,
            'status': 'completed',
            'tasks': {},
            'total_time': 0,
            'final_result': ''
        }
        
        start_time = time.time()
        accumulated_results = context
        
        for task in tasks:
            task_name = task.get('name', 'unknown')
            try:
                # Find suitable agent (simplified logic)
                agent = self._find_suitable_agent(task.get('agent_type', 'research'), available_agents)
                if not agent:
                    raise Exception(f"No suitable agent found for task {task_name}")
                
                # Execute task with accumulated context
                task_description = task.get('description', '')
                full_context = f"{task_description}\n\nContext from previous tasks:\n{accumulated_results}"
                task_result = agent.execute_task(full_context)
                
                stream_result['tasks'][task_name] = {
                    'status': 'completed',
                    'result': task_result,
                    'agent': agent.name
                }
                
                # Add to accumulated results for next task
                accumulated_results += f"\n\n{task_name}: {task_result}"
                
            except Exception as e:
                logger.error(f"Error executing task {task_name}: {e}")
                stream_result['tasks'][task_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                stream_result['status'] = 'failed'
        
        stream_result['total_time'] = time.time() - start_time
        stream_result['final_result'] = accumulated_results
        return stream_result
    
    
    def _find_suitable_agent(self, agent_type: str, available_agents: List[SimpleAgent]) -> Optional[SimpleAgent]:
        """Find a suitable agent for a task."""
        for agent in available_agents:
            if agent.agent_type.value == agent_type:
                return agent
        return None
    
    def _execute_quality_gates(self, quality_gates: List[str], 
                             stream_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality gates for the workflow."""
        quality_results = {}
        
        for gate in quality_gates:
            try:
                gate_result = self._evaluate_quality_gate(gate, stream_results)
                quality_results[gate] = gate_result
            except Exception as e:
                logger.error(f"Error executing quality gate {gate}: {e}")
                quality_results[gate] = {'status': 'failed', 'error': str(e)}
        
        return quality_results
    
    def _evaluate_quality_gate(self, gate_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific quality gate."""
        try:
            # Extract content from results
            content = self._extract_content_from_results(results)
            
            # Evaluate quality gate
            gate_result = self.quality_gate.evaluate_content(content, gate_id)
            
            return {
                'status': 'passed' if gate_result.get('status') == 'passed' else 'failed',
                'gate_id': gate_id,
                'evaluation': gate_result.get('status', 'unknown'),
                'score': gate_result.get('score', 0.0),
                'issues': gate_result.get('issues', [])
            }
            
        except Exception as e:
            logger.error(f"Error evaluating quality gate {gate_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_content_from_results(self, results: Dict[str, Any]) -> str:
        """Extract content from workflow results (simplified)."""
        content_parts = []
        
        for stream_name, stream_result in results.items():
            final_result = stream_result.get('final_result', '')
            if final_result:
                content_parts.append(final_result)
        
        return "\n\n".join(content_parts) if content_parts else ""
    
    def _extract_topic_from_task(self, task: str) -> str:
        """Extract topic from task description."""
        # Simple extraction - could be enhanced with NLP
        words = task.split()
        topic_words = []
        
        for i, word in enumerate(words):
            if word.lower() in ['about', 'on', 'regarding', 'concerning']:
                topic_words = words[i+1:]
                break
        
        return " ".join(topic_words) if topic_words else "general newsletter"
    
    def _select_template_for_topic(self, topic: str) -> NewsletterType:
        """Select appropriate template type for topic."""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ['technical', 'code', 'programming', 'development']):
            return NewsletterType.TECHNICAL_DEEP_DIVE
        elif any(word in topic_lower for word in ['business', 'market', 'industry', 'company']):
            return NewsletterType.TREND_ANALYSIS
        elif any(word in topic_lower for word in ['quick', 'brief', 'summary']):
            return NewsletterType.TUTORIAL_GUIDE
        else:
            return NewsletterType.RESEARCH_SUMMARY
    
    def _convert_to_workflow_steps(self, workflow_plan: Dict[str, Any]) -> List[WorkflowStep]:
        """Convert workflow plan to WorkflowStep objects."""
        steps = []
        step_id = 1
        
        for stream in workflow_plan.get('streams', []):
            for task in stream.get('tasks', []):
                step = WorkflowStep(
                    step_id=f"step_{step_id}",
                    name=task.get('name', 'Unknown Task'),
                    description=task.get('description', ''),
                    agent_type=task.get('agent_type', 'research'),
                    dependencies=task.get('dependencies', []),
                    estimated_time=task.get('estimated_time', 30)
                )
                steps.append(step)
                step_id += 1
        
        return steps
    
    def _get_workflow_status(self, workflow: WorkflowPlan) -> str:
        """Get current status of a workflow."""
        completed_steps = sum(1 for step in workflow.steps if step.status == TaskStatus.COMPLETED)
        total_steps = len(workflow.steps)
        
        if completed_steps == 0:
            return "Not Started"
        elif completed_steps == total_steps:
            return "Completed"
        else:
            return f"In Progress ({completed_steps}/{total_steps})"
    
    def _format_workflow_plan(self, workflow_plan: Dict[str, Any], workflow_id: str) -> str:
        """Format workflow plan for display."""
        formatted = f"""
# Workflow Plan: {workflow_id}

**Topic:** {workflow_plan.get('topic', 'Unknown')}
**Complexity:** {workflow_plan.get('complexity', 'standard')}
**Estimated Time:** {workflow_plan.get('estimated_time', 0)} minutes

## Workflow Streams
"""
        
        for stream in workflow_plan.get('streams', []):
            formatted += f"""
### {stream.get('name', 'Unknown Stream')}
{stream.get('description', '')}

**Tasks:**
"""
            for task in stream.get('tasks', []):
                dependencies = task.get('dependencies', [])
                deps_str = f" (Dependencies: {', '.join(dependencies)})" if dependencies else ""
                formatted += f"- **{task.get('name', 'Unknown Task')}** ({task.get('agent_type', 'unknown')} - {task.get('estimated_time', 0)}min){deps_str}\n"
        
        formatted += f"""
## Quality Gates
{', '.join(workflow_plan.get('quality_gates', []))}

Workflow ID: {workflow_id}
        """
        
        return formatted.strip()
    
    def _format_workflow_results(self, results: Dict[str, Any], workflow_id: str) -> str:
        """Format workflow results for display (simplified)."""
        # Extract the final content from all streams
        content_parts = []
        
        for stream_name, stream_result in results.get('streams', {}).items():
            final_result = stream_result.get('final_result', '')
            if final_result:
                content_parts.append(final_result)
        
        # Return the combined content as the main result
        return "\n\n".join(content_parts) if content_parts else "No content generated"
    
    def get_management_analytics(self) -> Dict[str, Any]:
        """Get management-specific analytics."""
        analytics = self.get_tool_usage_analytics()
        
        # Add management-specific metrics
        management_metrics = {
            "workflows_managed": len(self.active_workflows),
            "avg_workflow_time": sum(w.estimated_total_time for w in self.active_workflows.values()) / len(self.active_workflows) if self.active_workflows else 0,
            "quality_gate_performance": {
                "total_gates": sum(len(w.quality_gates) for w in self.active_workflows.values()),
                "passed_gates": 0,  # Would need to track actual results
                "failed_gates": 0   # Would need to track actual results
            }
        }
        
        analytics.update(management_metrics)
        return analytics 