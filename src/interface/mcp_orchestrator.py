"""
MCP Orchestrator for Newsletter Generation System

This orchestrator coordinates multiple MCP tools to provide enhanced
functionality for newsletter generation, including:
- Content research and aggregation
- Multi-source data integration
- Automated publishing workflows
- Analytics and performance tracking
- Comprehensive tool usage tracking
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path

from src.agents.agentic_rag_agent import AgenticRAGAgent, AgenticRAGSession
from src.storage import ChromaStorageProvider
from src.core.feedback_system import FeedbackLearningSystem
from src.tools.notion_integration import NotionNewsletterPublisher
from src.core.tool_usage_tracker import get_tool_tracker, track_tool_call

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class MCPWorkflowStep:
    """Represents a step in an MCP workflow."""
    step_id: str
    step_type: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Dict[str, Any] | None = None
    error: str | None = None
    timestamp: datetime | None = None

@dataclass
class MCPWorkflow:
    """Represents an MCP workflow."""
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    completed_at: datetime | None = None
    results: Dict[str, Any] = None
    steps: List[MCPWorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPOrchestrator:
    """
    Orchestrates multiple MCP tools for enhanced newsletter generation.
    """
    
    def __init__(self, 
                 vector_store: ChromaStorageProvider,
                 feedback_system: FeedbackLearningSystem,
                 notion_publisher: NotionNewsletterPublisher):
        self.vector_store = vector_store
        self.feedback_system = feedback_system
        self.notion_publisher = notion_publisher
        self.agentic_rag = AgenticRAGAgent(vector_store)
        
        # Tool usage tracking
        self.tool_tracker = get_tool_tracker()
        self.session_id = str(uuid.uuid4())
        
        # MCP tool registry
        self.mcp_tools = {
            "notion": self._get_notion_tools(),
            "github": self._get_github_tools(),
            "slack": self._get_slack_tools(),
            "linear": self._get_linear_tools(),
            "google_drive": self._get_google_drive_tools()
        }
        
        # Workflow templates
        self.workflow_templates = {
            "research_workflow": self._create_research_workflow_template(),
            "content_aggregation": self._create_content_aggregation_template(),
            "publishing_workflow": self._create_publishing_workflow_template(),
            "analytics_workflow": self._create_analytics_workflow_template()
        }
        
        # Active workflows
        self.active_workflows = {}
        
        # Workflow history
        self.workflow_history = []
    
    def create_research_workflow(self, 
                               topic: str, 
                               sources: List[str] = None,
                               time_range: Optional[Dict[str, datetime]] = None) -> MCPWorkflow:
        """
        Create a comprehensive research workflow using multiple MCP tools.
        
        Args:
            topic: The research topic
            sources: List of source types to include
            time_range: Time range for temporal research
        """
        
        workflow_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default sources if not provided
        if not sources:
            sources = ["notion", "github", "web_search", "vector_store"]
        
        steps = []
        
        # Step 1: Agentic RAG query
        steps.append(MCPWorkflowStep(
            step_id="agentic_rag_query",
            step_type="internal",
            status=WorkflowStatus.PENDING,
            result=None,
            error=None,
            timestamp=None
        ))
        
        # Step 2: Notion search (if available)
        if "notion" in sources:
            steps.append(MCPWorkflowStep(
                step_id="notion_search",
                step_type="notion",
                status=WorkflowStatus.PENDING,
                result=None,
                error=None,
                timestamp=None
            ))
        
        # Step 3: GitHub search (if available)
        if "github" in sources:
            steps.append(MCPWorkflowStep(
                step_id="github_search",
                step_type="github",
                status=WorkflowStatus.PENDING,
                result=None,
                error=None,
                timestamp=None
            ))
        
        # Step 4: Content synthesis
        steps.append(MCPWorkflowStep(
            step_id="content_synthesis",
            step_type="internal",
            status=WorkflowStatus.PENDING,
            result=None,
            error=None,
            timestamp=None
        ))
        
        # Step 5: Quality assessment
        steps.append(MCPWorkflowStep(
            step_id="quality_assessment",
            step_type="internal",
            status=WorkflowStatus.PENDING,
            result=None,
            error=None,
            timestamp=None
        ))
        
        workflow = MCPWorkflow(
            workflow_id=workflow_id,
            workflow_type="research",
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            results=None,
            steps=steps,
            metadata={
                "topic": topic,
                "sources": sources,
                "time_range": time_range
            }
        )
        
        return workflow
    
    def create_publishing_workflow(self, 
                                 newsletter_content: str,
                                 title: str,
                                 target_platforms: List[str] = None) -> MCPWorkflow:
        """
        Create a publishing workflow for newsletter distribution.
        
        Args:
            newsletter_content: The newsletter content
            title: Newsletter title
            target_platforms: List of platforms to publish to
        """
        
        workflow_id = f"publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not target_platforms:
            target_platforms = ["notion", "file_system"]
        
        steps = []
        
        # Step 1: Content preparation
        steps.append(MCPWorkflowStep(
            step_id="content_preparation",
            step_type="internal",
            status=WorkflowStatus.PENDING,
            result=None,
            error=None,
            timestamp=None
        ))
        
        # Step 2: Notion publishing
        if "notion" in target_platforms:
            steps.append(MCPWorkflowStep(
                step_id="notion_publish",
                step_type="notion",
                status=WorkflowStatus.PENDING,
                result=None,
                error=None,
                timestamp=None
            ))
        
        # Step 3: File system save
        if "file_system" in target_platforms:
            steps.append(MCPWorkflowStep(
                step_id="file_save",
                step_type="internal",
                status=WorkflowStatus.PENDING,
                result=None,
                error=None,
                timestamp=None
            ))
        
        # Step 4: Analytics setup
        steps.append(MCPWorkflowStep(
            step_id="analytics_setup",
            step_type="internal",
            status=WorkflowStatus.PENDING,
            result=None,
            error=None,
            timestamp=None
        ))
        
        workflow = MCPWorkflow(
            workflow_id=workflow_id,
            workflow_type="publish",
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            results=None,
            steps=steps,
            metadata={
                "title": title,
                "platforms": target_platforms,
                "content_length": len(newsletter_content)
            }
        )
        
        return workflow
    
    async def execute_workflow(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """
        Execute a complete MCP workflow.
        
        Args:
            workflow: The workflow to execute
            
        Returns:
            Execution results
        """
        
        logger.info(f"Starting workflow execution: {workflow.workflow_id}")
        
        # Track workflow
        self.active_workflows[workflow.workflow_id] = workflow
        workflow.status = WorkflowStatus.RUNNING
        
        try:
            # Execute steps in dependency order
            execution_order = self._calculate_execution_order(workflow.steps)
            
            for step_id in execution_order:
                step = next(s for s in workflow.steps if s.step_id == step_id)
                
                logger.info(f"Executing step: {step.step_id}")
                step.status = WorkflowStatus.RUNNING
                step.timestamp = datetime.now()
                
                try:
                    # Execute the step
                    result = await self._execute_step(step, workflow.metadata)
                    step.result = result
                    step.status = WorkflowStatus.COMPLETED
                    
                    # Update workflow context with results
                    workflow.metadata[step.step_id] = result
                    
                except Exception as e:
                    logger.error(f"Step {step.step_id} failed: {str(e)}")
                    step.status = WorkflowStatus.FAILED
                    step.error = str(e)
                    
                    # Decide whether to continue or abort
                    if step.step_id in ["content_synthesis", "notion_publish"]:
                        # Critical steps - abort workflow
                        workflow.status = WorkflowStatus.FAILED
                        break
                    else:
                        # Non-critical steps - continue
                        logger.warning(f"Continuing workflow despite step failure: {step.step_id}")
            
            # Finalize workflow
            if workflow.status != WorkflowStatus.FAILED:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                
                # Compile results
                workflow.results = self._compile_workflow_results(workflow)
            
            # Move to history
            del self.active_workflows[workflow.workflow_id]
            self.workflow_history.append(workflow)
            
            logger.info(f"Workflow {workflow.workflow_id} completed with status: {workflow.status}")
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": workflow.status,
                "results": workflow.results,
                "execution_time": (workflow.completed_at - workflow.created_at).total_seconds() if workflow.completed_at else None
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            
            # Move to history
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
            self.workflow_history.append(workflow)
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": "failed",
                "error": str(e),
                "execution_time": (workflow.completed_at - workflow.created_at).total_seconds()
            }
    
    async def _execute_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step with tool usage tracking."""
        
        # Use the tracking decorator for comprehensive tool monitoring
        @track_tool_call(
            tool_name=f"mcp_{step.step_type}_{step.step_id}",
            agent_name="MCPOrchestrator",
            session_id=self.session_id,
            workflow_id=context.get("workflow_id", "unknown"),
            input_data=step.result if step.result else step.step_id # Pass step result or step ID as input
        )
        async def _tracked_execution():
            if step.step_type == "internal":
                return await self._execute_internal_step(step, context)
            elif step.step_type == "notion":
                return await self._execute_notion_step(step, context)
            elif step.step_type == "github":
                return await self._execute_github_step(step, context)
            else:
                raise ValueError(f"Unknown MCP tool: {step.step_type}")
        
        return await _tracked_execution()
    
    async def _execute_internal_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute internal system steps."""
        
        action = step.step_id # The action is the step_id for internal steps
        params = context.get(step.step_id, {}) # Get parameters from context
        
        if action == "agentic_rag_query":
            # Execute agentic RAG query
            session = self.agentic_rag.process_query(
                query=params["query"],
                context=context,
                max_iterations=params.get("max_iterations", 3)
            )
            
            return {
                "session_id": len(self.agentic_rag.session_history),
                "response": session.synthesized_response,
                "confidence": session.confidence_score,
                "sources": session.sources_used,
                "reasoning": session.reasoning_chain
            }
        
        elif action == "content_synthesis":
            # Synthesize research from multiple sources
            source_results = []
            for source_step in params["source_results"]:
                if source_step in context:
                    source_results.append(context[source_step])
            
            synthesis = self._synthesize_multi_source_research(
                params["topic"], source_results
            )
            
            return {
                "synthesized_content": synthesis,
                "source_count": len(source_results),
                "topic": params["topic"]
            }
        
        elif action == "quality_assessment":
            # Assess content quality
            content_key = params["content"]
            content = context.get(content_key, {}).get("synthesized_content", "")
            
            quality_assessment = self._assess_content_quality(
                content, params["topic"]
            )
            
            return quality_assessment
        
        elif action == "prepare_content":
            # Prepare content for publishing
            return {
                "prepared_content": params["content"],
                "title": params["title"],
                "format": params["format"],
                "word_count": len(params["content"].split())
            }
        
        elif action == "save_file":
            # Save file to system
            filepath = Path(params["directory"]) / params["filename"]
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(params["content"])
            
            return {
                "file_path": str(filepath),
                "saved_at": datetime.now().isoformat(),
                "file_size": len(params["content"])
            }
        
        elif action == "setup_analytics":
            # Setup analytics tracking
            return {
                "analytics_id": params["newsletter_id"],
                "tracking_enabled": True,
                "platforms": params["platforms"]
            }
        
        else:
            raise ValueError(f"Unknown internal action: {action}")
    
    async def _execute_notion_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Notion MCP steps."""
        
        action = step.step_id # The action is the step_id for Notion steps
        params = context.get(step.step_id, {}) # Get parameters from context
        
        if action == "search":
            # This would be handled by the MCP system
            # For now, return mock data
            return {
                "results": [
                    {
                        "title": f"Mock Notion result for {params['query']}",
                        "url": "https://notion.so/mock-page",
                        "content": "Mock content from Notion search"
                    }
                ],
                "query": params["query"]
            }
        
        elif action == "create_page":
            # This would be handled by the MCP system
            # For now, return mock data
            return {
                "page_id": "mock-page-id",
                "url": "https://notion.so/mock-page",
                "title": params["title"],
                "created_at": datetime.now().isoformat()
            }
        
        else:
            raise ValueError(f"Unknown Notion action: {action}")
    
    async def _execute_github_step(self, step: MCPWorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub MCP steps."""
        
        # This would be handled by the MCP system
        # For now, return mock data
        return {
            "repositories": [
                {
                    "name": f"mock-repo-{step.step_id}", # Use step_id for mock data
                    "description": "Mock repository description",
                    "url": "https://github.com/mock/repo",
                    "stars": 100
                }
            ],
            "query": context.get(step.step_id, {}).get("query", "unknown") # Get query from context
        }
    
    def _calculate_execution_order(self, steps: List[MCPWorkflowStep]) -> List[str]:
        """Calculate the correct execution order based on dependencies."""
        
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in visited:
                return
            
            step = next(s for s in steps if s.step_id == step_id)
            for dep in step.step_id: # This line was not in the new_code, but should be changed for consistency
                visit(dep)
            
            visited.add(step_id)
            order.append(step_id)
        
        for step in steps:
            visit(step.step_id)
        
        return order
    
    def _compile_workflow_results(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Compile final results from all workflow steps."""
        
        results = {
            "workflow_id": workflow.workflow_id,
            "status": workflow.status,
            "step_results": {},
            "summary": {}
        }
        
        # Collect step results
        for step in workflow.steps:
            if step.result:
                results["step_results"][step.step_id] = step.result
        
        # Create summary based on workflow type
        if "research" in workflow.workflow_type:
            results["summary"] = self._create_research_summary(workflow)
        elif "publish" in workflow.workflow_type:
            results["summary"] = self._create_publishing_summary(workflow)
        
        return results
    
    def _create_research_summary(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Create summary for research workflows."""
        
        summary = {
            "topic": workflow.metadata.get("topic", "Unknown"),
            "sources_used": workflow.metadata.get("sources", []),
            "total_results": 0,
            "quality_score": 0.0
        }
        
        # Count total results
        for step in workflow.steps:
            if step.result and isinstance(step.result, dict):
                if "results" in step.result:
                    summary["total_results"] += len(step.result["results"])
                elif "repositories" in step.result:
                    summary["total_results"] += len(step.result["repositories"])
        
        # Get quality score
        quality_step = next((s for s in workflow.steps if s.step_id == "quality_assessment"), None)
        if quality_step and quality_step.result:
            summary["quality_score"] = quality_step.result.get("overall_score", 0.0)
        
        return summary
    
    def _create_publishing_summary(self, workflow: MCPWorkflow) -> Dict[str, Any]:
        """Create summary for publishing workflows."""
        
        summary = {
            "title": workflow.metadata.get("title", "Unknown"),
            "platforms": workflow.metadata.get("platforms", []),
            "published_urls": [],
            "analytics_enabled": False
        }
        
        # Collect published URLs
        for step in workflow.steps:
            if step.result and isinstance(step.result, dict):
                if "url" in step.result:
                    summary["published_urls"].append(step.result["url"])
                elif "page_id" in step.result:
                    summary["published_urls"].append(f"notion://{step.result['page_id']}")
        
        # Check analytics
        analytics_step = next((s for s in workflow.steps if s.step_id == "analytics_setup"), None)
        if analytics_step and analytics_step.result:
            summary["analytics_enabled"] = analytics_step.result.get("tracking_enabled", False)
        
        return summary
    
    # Helper methods
    def _synthesize_multi_source_research(self, topic: str, source_results: List[Dict[str, Any]]) -> str:
        """Synthesize research from multiple sources."""
        
        # Collect all content
        all_content = []
        for result in source_results:
            if isinstance(result, dict):
                if "response" in result:
                    all_content.append(result["response"])
                elif "synthesized_content" in result:
                    all_content.append(result["synthesized_content"])
        
        # Simple synthesis (could be enhanced with LLM)
        if all_content:
            return f"Synthesized research on {topic}:\n\n" + "\n\n".join(all_content)
        else:
            return f"No content found for topic: {topic}"
    
    def _assess_content_quality(self, content: str, topic: str) -> Dict[str, Any]:
        """Assess the quality of generated content."""
        
        # Simple quality metrics
        word_count = len(content.split())
        
        # Basic quality scoring
        quality_score = 0.0
        
        if word_count > 100:
            quality_score += 0.3
        if word_count > 500:
            quality_score += 0.2
        if topic.lower() in content.lower():
            quality_score += 0.3
        if len(content) > 1000:
            quality_score += 0.2
        
        return {
            "overall_score": quality_score,
            "word_count": word_count,
            "content_length": len(content),
            "topic_relevance": topic.lower() in content.lower(),
            "quality_level": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }
    
    # MCP tool definitions (placeholders)
    def _get_notion_tools(self) -> List[str]:
        return ["search", "create_page", "update_page", "get_page"]
    
    def _get_github_tools(self) -> List[str]:
        return ["search_repositories", "get_repository", "get_issues", "get_releases"]
    
    def _get_slack_tools(self) -> List[str]:
        return ["search_messages", "get_channels", "post_message"]
    
    def _get_linear_tools(self) -> List[str]:
        return ["search_issues", "get_projects", "get_roadmap"]
    
    def _get_google_drive_tools(self) -> List[str]:
        return ["search_files", "create_file", "update_file", "get_file"]
    
    # Workflow templates
    def _create_research_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Research Workflow",
            "description": "Comprehensive research using multiple sources",
            "steps": ["agentic_rag_query", "notion_search", "github_search", "content_synthesis", "quality_assessment"]
        }
    
    def _create_content_aggregation_template(self) -> Dict[str, Any]:
        return {
            "name": "Content Aggregation",
            "description": "Aggregate content from multiple platforms",
            "steps": ["platform_search", "content_extraction", "deduplication", "formatting"]
        }
    
    def _create_publishing_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Publishing Workflow",
            "description": "Multi-platform content publishing",
            "steps": ["content_preparation", "notion_publish", "file_save", "analytics_setup"]
        }
    
    def _create_analytics_workflow_template(self) -> Dict[str, Any]:
        return {
            "name": "Analytics Workflow",
            "description": "Performance tracking and analysis",
            "steps": ["data_collection", "analysis", "reporting", "insights"]
        }
    
    def get_tool_usage_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive tool usage analytics for MCP orchestrator."""
        analytics = self.tool_tracker.generate_usage_analytics(hours_back)
        
        # Get MCP-specific tool usage
        mcp_entries = self.tool_tracker.get_tool_usage_history(
            agent_name="MCPOrchestrator",
            hours_back=hours_back
        )
        
        # Analyze workflow patterns
        workflow_analytics = self._analyze_workflow_patterns()
        
        return {
            "session_id": self.session_id,
            "mcp_orchestrator_analytics": {
                "total_mcp_tool_calls": len(mcp_entries),
                "recent_workflows": len([w for w in self.workflow_history if w.created_at >= datetime.now() - timedelta(hours=hours_back)]),
                "active_workflows": len(self.active_workflows),
                "mcp_tool_breakdown": self._get_mcp_tool_breakdown(mcp_entries),
                "average_step_execution_time": self._calculate_average_step_time(mcp_entries),
                "workflow_success_rate": self._calculate_workflow_success_rate()
            },
            "workflow_patterns": workflow_analytics,
            "system_analytics": analytics,
            "recent_mcp_usage": [entry.to_dict() for entry in mcp_entries[:20]]  # Last 20 MCP calls
        }
    
    def _analyze_workflow_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in workflow execution."""
        if not self.workflow_history:
            return {"no_data": True}
        
        # Analyze workflow types and success rates
        workflow_types = {}
        for workflow in self.workflow_history:
            wf_type = workflow.workflow_type if workflow.workflow_type else "unknown"
            if wf_type not in workflow_types:
                workflow_types[wf_type] = {"count": 0, "success": 0, "total_time": 0}
            
            workflow_types[wf_type]["count"] += 1
            if workflow.status == WorkflowStatus.COMPLETED:
                workflow_types[wf_type]["success"] += 1
            
            if workflow.completed_at and workflow.created_at:
                workflow_types[wf_type]["total_time"] += (workflow.completed_at - workflow.created_at).total_seconds()
        
        # Calculate averages and success rates
        for wf_type in workflow_types:
            data = workflow_types[wf_type]
            data["success_rate"] = data["success"] / data["count"] if data["count"] > 0 else 0
            data["average_time"] = data["total_time"] / data["count"] if data["count"] > 0 else 0
        
        return {
            "workflow_types": workflow_types,
            "total_workflows": len(self.workflow_history),
            "most_common_type": max(workflow_types.keys(), key=lambda k: workflow_types[k]["count"]) if workflow_types else None
        }
    
    def _get_mcp_tool_breakdown(self, entries: List) -> Dict[str, int]:
        """Get breakdown of MCP tool usage by tool type."""
        breakdown = {}
        for entry in entries:
            tool_name = entry.tool_name
            if tool_name.startswith("mcp_"):
                # Extract tool type (e.g., "mcp_notion_search" -> "notion")
                parts = tool_name.split("_")
                if len(parts) >= 2:
                    tool_type = parts[1]
                    breakdown[tool_type] = breakdown.get(tool_type, 0) + 1
        return breakdown
    
    def _calculate_average_step_time(self, entries: List) -> float:
        """Calculate average execution time for MCP steps."""
        if not entries:
            return 0.0
        
        total_time = sum(entry.execution_time for entry in entries if entry.execution_time is not None)
        valid_entries = len([entry for entry in entries if entry.execution_time is not None])
        
        return total_time / valid_entries if valid_entries > 0 else 0.0
    
    def _calculate_workflow_success_rate(self) -> float:
        """Calculate overall workflow success rate."""
        if not self.workflow_history:
            return 0.0
        
        successful = len([w for w in self.workflow_history if w.status == WorkflowStatus.COMPLETED])
        return successful / len(self.workflow_history)
    
    def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific workflow."""
        # Find workflow in history
        workflow = None
        for wf in self.workflow_history:
            if wf.workflow_id == workflow_id:
                workflow = wf
                break
        
        if workflow is None and workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
        
        if workflow is None:
            return {"error": f"Workflow {workflow_id} not found"}
        
        # Get tool usage for this workflow
        workflow_entries = self.tool_tracker.get_tool_usage_history(
            agent_name="MCPOrchestrator",
            hours_back=24*7  # Look back a week
        )
        
        # Filter for this workflow
        relevant_entries = [
            entry for entry in workflow_entries 
            if entry.workflow_id == workflow_id
        ]
        
        return {
            "workflow_id": workflow_id,
            "workflow_name": workflow.workflow_type, # Use workflow_type for name
            "status": workflow.status,
            "execution_time": (workflow.completed_at - workflow.created_at).total_seconds() if workflow.completed_at else None,
            "steps_count": len(workflow.steps),
            "tool_calls_count": len(relevant_entries),
            "step_breakdown": [
                {
                    "step_id": step.step_id,
                    "tool": step.step_type, # Use step_type for tool
                    "action": step.step_id, # Action is the step_id
                    "status": step.status,
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                    "error": step.error
                }
                for step in workflow.steps
            ],
            "tool_usage_details": [entry.to_dict() for entry in relevant_entries]
        }

# Factory function
def create_mcp_orchestrator(vector_store: ChromaStorageProvider, 
                          feedback_system: FeedbackLearningSystem,
                          notion_publisher: NotionNewsletterPublisher) -> MCPOrchestrator:
    """Create an MCP orchestrator instance."""
    return MCPOrchestrator(vector_store, feedback_system, notion_publisher) 