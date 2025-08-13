"""
Enhanced Workflow Orchestrator: Orchestrates the complete newsletter generation workflow.

This module implements the enhanced workflow orchestrator that coordinates all agents
and systems to execute the complete newsletter generation workflow with campaign
context awareness, execution state management, and iterative refinement.
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import SimpleAgent
from src.agents.editing import EditorAgent
from src.agents.management import ManagerAgent
from src.agents.research import ResearchAgent
from src.agents.writing import WriterAgent
from src.agents.technical_accuracy_agent import TechnicalAccuracyAgent
from src.agents.readability_agent import ReadabilityAgent
from src.agents.continuity_manager_agent import ContinuityManagerAgent
from .agent_coordinator import AgentCoordinator, AgentExecutionSpec

from .campaign_context import CampaignContext
from .config_manager import ConfigManager
from .execution_state import ExecutionState
from .learning_system import LearningSystem
from .refinement_loop import RefinementLoop, RefinementResult
from .tool_usage_tracker import get_tool_tracker
from .template_manager import AIMLTemplateManager, NewsletterType

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    topic: str
    final_content: str
    quality_metrics: Dict[str, Any]
    execution_time: float
    phase_results: Dict[str, Any]
    learning_data: Dict[str, Any]
    status: str  # 'completed', 'failed', 'partial'
    code_generation_metrics: Dict[str, Any] = None  # Phase 3: Code generation metrics


class WorkflowOrchestrator:
    """
    Enhanced workflow orchestrator for newsletter generation.

    This class coordinates the complete workflow including campaign context
    loading, execution state management, agent coordination, and iterative
    refinement with learning integration.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize the workflow orchestrator."""
        self.config_manager = config_manager or ConfigManager()
        self.learning_system = LearningSystem()
        self.refinement_loop = RefinementLoop()

        # Initialize agents
        self.agents: Dict[str, SimpleAgent] = {}
        self._initialize_agents()

        # Execution tracking
        self.campaign_context: Optional[CampaignContext] = None
        self.execution_state: Optional[ExecutionState] = None

        # Phase 2.1: Add tool usage tracking
        self.tool_tracker = get_tool_tracker()
        
        # Phase 3: Add template manager for code generation
        self.template_manager = AIMLTemplateManager()
        
        # Code generation metrics tracking
        self.code_generation_metrics = {
            'examples_generated': 0,
            'validation_score': 0.0,
            'execution_success_rate': 0.0,
            'frameworks_used': []
        }

    def _initialize_agents(self) -> None:
        """Initialize the agent instances."""
        try:
            self.agents = {
                'manager': ManagerAgent(name="WorkflowManager"),
                'researcher': ResearchAgent(name="ResearchAgent"),
                'writer': WriterAgent(name="WriterAgent"),
                'editor': EditorAgent(name="EditorAgent"),
                # Phase 2 specialized agents
                'technical_accuracy': TechnicalAccuracyAgent(),
                'readability': ReadabilityAgent(),
                'continuity_manager': ContinuityManagerAgent(),
            }
            logger.info("Agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            # Create minimal agents for fallback
            self.agents = {}

    def execute_newsletter_generation(
            self,
            topic: str,
            context_id: str = "default",
            output_format: str = "markdown") -> WorkflowResult:
        """
        Execute the complete newsletter generation workflow.

        Args:
            topic: Newsletter topic
            context_id: Campaign context identifier
            output_format: Output format (markdown, html, etc.)

        Returns:
            WorkflowResult with complete execution details
        """
        workflow_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting workflow {workflow_id} for topic: {topic}")

        try:
            # Phase 1: Load campaign context
            self.campaign_context = self._load_campaign_context(context_id)
            logger.info(f"Loaded campaign context: {context_id}")

            # Phase 2: Initialize execution state
            self.execution_state = ExecutionState(
                workflow_id=workflow_id,
                current_phase='initialization'
            )
            logger.info(
                f"Initialized execution state for workflow: {workflow_id}")

            # Phase 3: Execute workflow phases
            phase_results = {}

            # Research Phase
            self.execution_state.update_phase('research')
            research_result = self._execute_research_phase(topic)
            phase_results['research'] = research_result
            logger.info("Research phase completed")

            # Writing Phase
            self.execution_state.update_phase('writing')
            writing_result = self._execute_writing_phase(
                research_result, topic)
            phase_results['writing'] = writing_result
            logger.info("Writing phase completed")

            # Refinement Phase (pre-multi-agent)
            self.execution_state.update_phase('refinement')
            refinement_result = self._execute_refinement_loop(
                writing_result['content'])
            phase_results['refinement'] = refinement_result
            logger.info("Refinement phase completed")

            # Phase 2: Multi-Agent Specialized Validation and Optimization (feature-flagged)
            if os.getenv('ENABLE_PHASE2', '1') == '1':
                self.execution_state.update_phase('multi_agent_validation')
                multi_agent_results = self._execute_multi_agent_phase(
                    refinement_result.final_content,
                    writing_result,
                )
                phase_results['multi_agent'] = multi_agent_results
            else:
                multi_agent_results = {'final_content': refinement_result.final_content}

            # Phase 4: Finalize workflow
            # Prefer multi-agent improved content if available
            final_improved = multi_agent_results.get('final_content', refinement_result.final_content)
            final_content = self._finalize_workflow(
                RefinementResult(
                    final_content=final_improved,
                    final_score=refinement_result.final_score,
                    revision_cycles=refinement_result.revision_cycles,
                    improvement_history=refinement_result.improvement_history,
                    quality_metrics=refinement_result.quality_metrics,
                    learning_data=refinement_result.learning_data,
                    status=refinement_result.status,
                ),
                output_format,
            )

            # Phase 5: Generate learning data and update context
            learning_data = self._generate_learning_data(
                phase_results, start_time)
            self.learning_system.update_campaign_context(
                self.campaign_context, learning_data)

            execution_time = time.time() - start_time

            result = WorkflowResult(
                workflow_id=workflow_id,
                topic=topic,
                final_content=final_content,
                quality_metrics=refinement_result.quality_metrics,
                execution_time=execution_time,
                phase_results=phase_results,
                learning_data=learning_data,
                status='completed',
                code_generation_metrics=self.code_generation_metrics.copy()  # Phase 3: Include code metrics
            )

            logger.info(
                f"Workflow {workflow_id} completed successfully in {
                    execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            execution_time = time.time() - start_time

            return WorkflowResult(
                workflow_id=workflow_id,
                topic=topic,
                final_content="",
                quality_metrics={'error': str(e)},
                execution_time=execution_time,
                phase_results={},
                learning_data={'error': str(e)},
                status='failed',
                code_generation_metrics=self.code_generation_metrics.copy()  # Phase 3: Include code metrics
            )

    def _load_campaign_context(self, context_id: str) -> CampaignContext:
        """Load campaign context from configuration manager."""
        return self.config_manager.load_campaign_context(context_id)

    def _execute_multi_agent_phase(self, content: str, writing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 2 specialized agents with coordination and fallbacks."""
        try:
            coordinator = AgentCoordinator(
                agents={k: v for k, v in self.agents.items() if k in (
                    'technical_accuracy', 'readability', 'continuity_manager')}
            )

            # Build metadata for continuity (if sections are available later, they can be passed here)
            # Derive sections from content for continuity manager
            derived_sections = self._derive_sections_from_content(content)
            metadata: Dict[str, Any] = {
                'sections': derived_sections,
                'writing': writing_result,
            }

            base_context = self._build_processing_context(content, metadata)

            specs = [
                AgentExecutionSpec(name='technical_accuracy', agent=self.agents['technical_accuracy'], depends_on=[]),
                AgentExecutionSpec(name='readability', agent=self.agents['readability'], depends_on=['technical_accuracy']),
                AgentExecutionSpec(name='continuity_manager', agent=self.agents['continuity_manager'], depends_on=['readability']),
            ]

            # Try best-effort parallel aggregator to allow independent improvements
            results = coordinator.execute_parallel_best_effort(specs, base_context)

            # Choose best improved content if any agent produced it
            final_content = content
            for key in ['technical_accuracy', 'readability', 'continuity_manager']:
                agent_dict = results.get(key)
                if agent_dict and isinstance(agent_dict, dict):
                    improved = agent_dict.get('processed_content')
                    if improved and isinstance(improved, str) and len(improved.strip()) > 0:
                        final_content = improved

            # Compile Phase 2 metrics
            metrics = self._extract_phase2_metrics(results)

            return {
                **results,
                'final_content': final_content,
                'metrics': metrics,
            }
        except Exception as e:
            logger.error(f"Multi-agent phase failed: {e}")
            return {
                'error': str(e),
                'final_content': content,
            }

    def _build_processing_context(self, content: str, metadata: Dict[str, Any]):
        from src.agents.base_agent import ProcessingContext, ProcessingMode
        # Map campaign context fields when available
        audience = None
        technical_level = None
        if self.campaign_context:
            audience = self.campaign_context.audience_persona.get('demographics', None)
            technical_level = self.campaign_context.content_style.get('technical_level', None)

        # Optional fast mode for Phase 2
        processing_mode = ProcessingMode.FULL if os.getenv('PHASE2_FAST', '0') != '1' else ProcessingMode.FAST

        return ProcessingContext(
            content=content,
            section_type='analysis',
            audience=audience,
            technical_level=technical_level,
            word_count_target=len(content.split()),
            processing_mode=processing_mode,
            metadata=metadata,
        )

    def _derive_sections_from_content(self, content: str) -> Dict[str, str]:
        """Heuristically derive sections from markdown headings for continuity analysis."""
        try:
            from src.core.section_aware_prompts import SectionType
        except Exception:
            # Fallback keys as strings
            SectionType = None

        sections: Dict[str, str] = {}
        # Split by h2 headers and label common section names
        import re
        parts = re.split(r"^##\s+(.+)$", content, flags=re.MULTILINE)
        if len(parts) <= 1:
            # No headers; return intro and conclusion slices
            head = "\n".join(content.splitlines()[:80])
            tail = "\n".join(content.splitlines()[-80:])
            if SectionType:
                sections[SectionType.INTRODUCTION] = head
                sections[SectionType.CONCLUSION] = tail
            else:
                sections['introduction'] = head
                sections['conclusion'] = tail
            return sections

        # parts layout: [pre, header1, body1, header2, body2, ...]
        preface = parts[0].strip()
        if preface:
            if SectionType:
                sections[SectionType.INTRODUCTION] = preface
            else:
                sections['introduction'] = preface
        for i in range(1, len(parts), 2):
            title = parts[i].strip().lower()
            body = parts[i+1].strip() if i+1 < len(parts) else ''
            key = 'analysis'
            if any(k in title for k in ['intro', 'welcome']):
                key = 'introduction'
            elif 'tutorial' in title or 'how to' in title:
                key = 'tutorial'
            elif 'news' in title or 'updates' in title:
                key = 'news'
            elif 'conclusion' in title or 'summary' in title:
                key = 'conclusion'
            if SectionType:
                from_map = {
                    'introduction': SectionType.INTRODUCTION,
                    'analysis': SectionType.ANALYSIS,
                    'tutorial': SectionType.TUTORIAL,
                    'news': SectionType.NEWS,
                    'conclusion': SectionType.CONCLUSION,
                }
                sec_key = from_map.get(key, SectionType.ANALYSIS)
            else:
                sec_key = key
            prev = sections.get(sec_key, '')
            sections[sec_key] = (prev + ("\n\n" if prev else "") + body) if body else prev
        return sections

    def _extract_phase2_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate key metrics from specialized agents for reporting and KPIs."""
        metrics: Dict[str, Any] = {}
        try:
            tech = results.get('technical_accuracy') or {}
            read = results.get('readability') or {}
            cont = results.get('continuity_manager') or {}

            # Technical Accuracy
            if isinstance(tech, dict):
                metrics['technical_accuracy_confidence'] = tech.get('confidence_score')
                metrics['technical_quality_score'] = tech.get('quality_score')

            # Readability
            if isinstance(read, dict):
                metrics['readability_score'] = read.get('quality_score')
                rm = ((read.get('metadata') or {}).get('readability_metrics') or {})
                if rm:
                    metrics['avg_sentence_length'] = rm.get('avg_sentence_length')
                    metrics['avg_syllables_per_word'] = rm.get('avg_syllables_per_word')

            # Continuity
            if isinstance(cont, dict):
                cr = ((cont.get('metadata') or {}).get('continuity_report') or {})
                if cr:
                    metrics['continuity_overall'] = cr.get('overall')
                    metrics['continuity_transition_quality'] = cr.get('transition_quality')
                    metrics['continuity_style_consistency'] = cr.get('style_consistency')

            # Coordinator summary
            metrics['agent_errors'] = len(results.get('__errors__', []))
        except Exception:
            # Keep metrics optional and non-fatal
            pass
        return metrics

    def _execute_research_phase(self, topic: str) -> Dict[str, Any]:
        """Execute the research phase."""
        logger.info("Starting research phase")

        try:
            if 'researcher' in self.agents:
                # Use the research agent if available with tracking
                research_task = f"Conduct comprehensive research on: {topic}"

                with self.tool_tracker.track_tool_usage(
                    tool_name="research_agent_execution",
                    agent_name="WorkflowOrchestrator",
                    workflow_id=self.execution_state.workflow_id,
                    session_id=self.execution_state.session_id,
                    input_data={"topic": topic, "research_task": research_task},
                    context={"phase": "research", "agent": "researcher"}
                ):
                    # Set agent context for downstream tracking
                    self.agents['researcher'].set_context(
                        workflow_id=self.execution_state.workflow_id,
                        session_id=self.execution_state.session_id
                    )

                    research_result = self.agents['researcher'].execute_task(
                        research_task, context=f"Campaign context: {
                            self.campaign_context.content_style if self.campaign_context else 'default'}")

                return {
                    'status': 'completed',
                    'content': research_result,
                    'sources': [],
                    'confidence': 0.8,
                    'timestamp': time.time()
                }
            else:
                # Fallback to basic research
                return self._execute_basic_research(topic)

        except Exception as e:
            logger.error(f"Research phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'content': f"Research on {topic} could not be completed due to: {e}",
                'timestamp': time.time()}

    def _execute_basic_research(self, topic: str) -> Dict[str, Any]:
        """Execute basic research without agents."""
        from core.core import query_llm

        research_prompt = f"""
        Conduct comprehensive research on the topic: {topic}

        Please provide:
        1. Key concepts and definitions
        2. Current trends and developments
        3. Important statistics or data points
        4. Expert opinions and insights
        5. Relevant examples and case studies
        6. Future outlook and implications

        Focus on factual, current information that would be valuable for a professional newsletter.
        """

        try:
            research_content = query_llm(research_prompt)
            return {
                'status': 'completed',
                'content': research_content,
                'sources': [],
                'confidence': 0.7,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Basic research failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'content': f"Basic research on {topic} failed",
                'timestamp': time.time()
            }

    def _execute_writing_phase(
            self, research_result: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Execute the writing phase with Phase 3 code generation integration."""
        logger.info("Starting enhanced writing phase with code generation")

        try:
            if 'writer' in self.agents:
                # Determine if this is a technical topic requiring code examples
                is_technical = self._is_technical_topic(topic)
                
                # Get appropriate template
                template_type = self.template_manager.suggest_template(topic)
                template = self.template_manager.get_template(template_type)
                
                if is_technical and template:
                    # Enhanced technical content with code generation
                    writing_result = self._execute_technical_writing_with_code(
                        research_result, topic, template
                    )
                else:
                    # Standard writing workflow
                    writing_result = self._execute_standard_writing(
                        research_result, topic
                    )

                return writing_result
            else:
                # Fallback to basic writing
                return self._execute_basic_writing(research_result, topic)

        except Exception as e:
            logger.error(f"Writing phase failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'content': f"Writing phase for {topic} failed: {e}",
                'timestamp': time.time()
            }
    
    def _execute_technical_writing_with_code(
            self, research_result: Dict[str, Any], topic: str, template) -> Dict[str, Any]:
        """Execute technical writing with integrated code generation."""
        logger.info("Executing technical writing with code generation")
        
        with self.tool_tracker.track_tool_usage(
            tool_name="technical_writer_agent_with_code",
            agent_name="WorkflowOrchestrator", 
            workflow_id=self.execution_state.workflow_id,
            session_id=self.execution_state.session_id,
            input_data={
                "topic": topic,
                "template_type": template.type.value,
                "research_content_length": len(research_result.get('content', ''))
            },
            context={"phase": "writing", "subphase": "technical_with_code"}
        ):
            # Set agent context
            self.agents['writer'].set_context(
                workflow_id=self.execution_state.workflow_id,
                session_id=self.execution_state.session_id
            )
            
            # Generate technical content with code examples
            writing_context = f"""
            Research findings: {research_result.get('content', '')}
            Campaign context: {self.campaign_context.content_style if self.campaign_context else 'default'}
            Target audience: {self.campaign_context.audience_persona if self.campaign_context else 'technical professionals'}
            """
            
            content = self.agents['writer'].generate_technical_content_with_code(
                topic=topic,
                context=writing_context,
                include_code=True
            )
            
            # Generate additional code examples if needed
            code_examples = self.agents['writer'].generate_code_examples(
                topic=topic,
                count=2
            )
            
            # Track code generation metrics
            self._update_code_generation_metrics(code_examples)
            
            return {
                'status': 'completed',
                'content': content,
                'word_count': len(content.split()),
                'code_examples_count': len(code_examples),
                'template_used': template.type.value,
                'technical_enhanced': True,
                'timestamp': time.time()
            }
    
    def _execute_standard_writing(
            self, research_result: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Execute standard writing workflow."""
        logger.info("Executing standard writing workflow")
        
        writing_task = f"Write a comprehensive newsletter about: {topic}"
        writing_context = f"""
        Research findings: {research_result.get('content', '')}
        Campaign context: {self.campaign_context.content_style if self.campaign_context else 'default'}
        Target audience: {self.campaign_context.audience_persona if self.campaign_context else 'general'}
        """

        with self.tool_tracker.track_tool_usage(
            tool_name="writer_agent_execution",
            agent_name="WorkflowOrchestrator",
            workflow_id=self.execution_state.workflow_id,
            session_id=self.execution_state.session_id,
            input_data={
                "topic": topic, 
                "writing_task": writing_task, 
                "research_content_length": len(research_result.get('content', ''))
            },
            context={"phase": "writing", "agent": "writer"}
        ):
            # Set agent context for downstream tracking
            self.agents['writer'].set_context(
                workflow_id=self.execution_state.workflow_id,
                session_id=self.execution_state.session_id
            )

            writing_result = self.agents['writer'].execute_task(
                writing_task,
                context=writing_context
            )

            return {
                'status': 'completed',
                'content': writing_result,
                'word_count': len(writing_result.split()),
                'timestamp': time.time()
            }
    
    def _is_technical_topic(self, topic: str) -> bool:
        """Determine if a topic requires technical treatment with code examples."""
        technical_keywords = [
            'ai', 'machine learning', 'deep learning', 'neural network',
            'algorithm', 'python', 'pytorch', 'tensorflow', 'programming',
            'data science', 'api', 'implementation', 'architecture',
            'framework', 'library', 'code', 'technical', 'engineering'
        ]
        
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in technical_keywords)
    
    def _update_code_generation_metrics(self, code_examples: List[str]) -> None:
        """Update code generation metrics for tracking."""
        if code_examples:
            self.code_generation_metrics['examples_generated'] += len(code_examples)
            
            # Extract frameworks from code examples (simplified)
            for example in code_examples:
                if 'pytorch' in example.lower():
                    if 'pytorch' not in self.code_generation_metrics['frameworks_used']:
                        self.code_generation_metrics['frameworks_used'].append('pytorch')
                elif 'tensorflow' in example.lower():
                    if 'tensorflow' not in self.code_generation_metrics['frameworks_used']:
                        self.code_generation_metrics['frameworks_used'].append('tensorflow')
                elif 'sklearn' in example.lower():
                    if 'sklearn' not in self.code_generation_metrics['frameworks_used']:
                        self.code_generation_metrics['frameworks_used'].append('sklearn')
                elif 'pandas' in example.lower():
                    if 'pandas' not in self.code_generation_metrics['frameworks_used']:
                        self.code_generation_metrics['frameworks_used'].append('pandas')

    def _execute_basic_writing(
            self, research_result: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Execute basic writing without agents."""
        from core.core import query_llm

        # Determine style based on campaign context
        style_guidance = "professional and informative"
        if self.campaign_context:
            tone = self.campaign_context.content_style.get(
                'tone', 'professional')
            style = self.campaign_context.content_style.get(
                'style', 'informative')
            style_guidance = f"{tone} and {style}"

        writing_prompt = f"""
        Write a comprehensive newsletter about: {topic}

        Research information:
        {research_result.get('content', '')}

        Style guidelines:
        - Tone: {style_guidance}
        - Target length: 1000-1500 words
        - Include clear headings and structure
        - Make it engaging and informative
        - Add practical insights and takeaways

        Please create a well-structured newsletter that provides value to the readers.
        """

        try:
            newsletter_content = query_llm(writing_prompt)
            return {
                'status': 'completed',
                'content': newsletter_content,
                'word_count': len(newsletter_content.split()),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Basic writing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'content': f"Basic writing for {topic} failed",
                'timestamp': time.time()
            }

    def _execute_refinement_loop(self, content: str) -> RefinementResult:
        """Execute the iterative refinement loop."""
        logger.info("Starting refinement loop")

        try:
            return self.refinement_loop.execute_refinement(
                content,
                self.campaign_context,
                self.execution_state
            )
        except Exception as e:
            logger.error(f"Refinement loop failed: {e}")
            # Return a basic result
            from core.refinement_loop import RefinementResult
            return RefinementResult(
                final_content=content,
                final_score=0.5,
                revision_cycles=0,
                improvement_history=[],
                quality_metrics={'error': str(e)},
                learning_data={'error': str(e)},
                status='error'
            )

    def _finalize_workflow(
            self,
            refinement_result: RefinementResult,
            output_format: str) -> str:
        """Finalize the workflow and format output."""
        logger.info("Finalizing workflow")

        final_content = refinement_result.final_content

        if output_format == "markdown":
            # Already in markdown format
            return final_content
        elif output_format == "html":
            return self._convert_to_html(final_content)
        else:
            return final_content

    def _convert_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML."""
        try:
            # Simple markdown to HTML conversion
            # In production, use a proper markdown library
            html_content = markdown_content
            html_content = html_content.replace(
                '# ', '<h1>').replace(
                '\n', '</h1>\n', 1)
            html_content = html_content.replace(
                '## ', '<h2>').replace(
                '\n', '</h2>\n', 1)
            html_content = html_content.replace(
                '### ', '<h3>').replace(
                '\n', '</h3>\n', 1)
            html_content = f"<html><body>{html_content}</body></html>"
            return html_content
        except Exception as e:
            logger.error(f"HTML conversion failed: {e}")
            return markdown_content

    def _generate_learning_data(
            self, phase_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Generate learning data from workflow execution."""
        execution_time = time.time() - start_time

        # Calculate quality metrics
        refinement_result = phase_results.get('refinement')
        quality_score = refinement_result.final_score if refinement_result else 0.5
        revision_cycles = refinement_result.revision_cycles if refinement_result else 0

        learning_data = {
            'execution_time': execution_time,
            'quality_score': quality_score,
            'revision_cycles': revision_cycles,
            'phase_count': len(phase_results),
            'success_rate': 1.0 if all(
                r.get('status') == 'completed' for r in phase_results.values() if isinstance(
                    r,
                    dict)) else 0.5,
            'workflow_metrics': {
                'research_success': phase_results.get(
                    'research',
                    {}).get('status') == 'completed',
                'writing_success': phase_results.get(
                        'writing',
                        {}).get('status') == 'completed',
                'refinement_success': refinement_result.status == 'completed' if refinement_result else False}}

        return learning_data

    def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get analytics for workflow orchestration."""
        if not self.execution_state:
            return {'status': 'no_active_workflow'}

        analytics = {
            'workflow_id': self.execution_state.workflow_id,
            'current_phase': self.execution_state.current_phase,
            'start_time': self.execution_state.start_time,
            'last_updated': self.execution_state.last_updated,
            'revision_cycles': self.execution_state.revision_cycles,
            'quality_scores': self.execution_state.quality_scores,
            'task_results_count': len(
                self.execution_state.task_results),
            'feedback_history_count': len(
                self.execution_state.feedback_history)}

        return analytics

    def create_dynamic_workflow(
            self, topic: str, context: CampaignContext) -> Dict[str, Any]:
        """Create a dynamic workflow plan based on topic and context."""
        workflow_plan = {
            'topic': topic,
            'context_id': context.created_at,  # Using timestamp as ID for now
            'phases': []
        }

        # Research phase configuration
        research_phase = {
            'name': 'research',
            'duration_estimate': 120,  # seconds
            'requirements': ['web_search', 'data_analysis'],
            'quality_gate': 0.7,
            'parallel_tasks': []
        }

        # Determine research complexity based on topic
        if any(keyword in topic.lower()
               for keyword in ['technical', 'ai', 'machine learning', 'data']):
            research_phase['duration_estimate'] = 180
            research_phase['requirements'].append('technical_analysis')

        workflow_plan['phases'].append(research_phase)

        # Writing phase configuration
        writing_phase = {
            'name': 'writing',
            'duration_estimate': 300,  # seconds
            'requirements': ['content_generation', 'style_adaptation'],
            'quality_gate': context.get_quality_threshold('minimum'),
            'dependencies': ['research']
        }

        # Adjust writing requirements based on context
        target_length = context.content_style.get('target_length', 'medium')
        if target_length == 'long':
            writing_phase['duration_estimate'] = 450
        elif target_length == 'short':
            writing_phase['duration_estimate'] = 180

        workflow_plan['phases'].append(writing_phase)

        # Refinement phase configuration
        refinement_phase = {
            'name': 'refinement',
            'duration_estimate': 240,  # seconds
            'requirements': ['quality_assessment', 'iterative_improvement'],
            'quality_gate': context.get_quality_threshold('target'),
            'dependencies': ['writing'],
            'max_cycles': 3
        }

        workflow_plan['phases'].append(refinement_phase)

        # Calculate total estimated duration
        workflow_plan['total_duration_estimate'] = sum(
            phase['duration_estimate'] for phase in workflow_plan['phases']
        )

        return workflow_plan

    def validate_workflow_requirements(
            self, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that workflow requirements can be met."""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check agent availability
        required_agents = ['researcher', 'writer', 'editor']
        for agent_type in required_agents:
            if agent_type not in self.agents:
                validation_result['warnings'].append(
                    f"Agent {agent_type} not available, using fallback")

        # Check estimated duration
        total_duration = workflow_plan.get('total_duration_estimate', 0)
        if total_duration > 1800:  # 30 minutes
            validation_result['warnings'].append(
                f"Estimated duration ({total_duration}s) is high")

        # Check quality gates
        for phase in workflow_plan.get('phases', []):
            quality_gate = phase.get('quality_gate', 0)
            if quality_gate > 0.95:
                validation_result['warnings'].append(
                    f"Quality gate for {
                        phase['name']} is very high ({quality_gate})")

        return validation_result
