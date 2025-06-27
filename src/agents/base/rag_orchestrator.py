from typing import Dict, Any, List, Optional
from .agent_base import AgentBase
from ..rag_pipeline.query_writer_agent import QueryWriterAgent
from ..rag_pipeline.context_assessment_agent import ContextAssessmentAgent
from ..rag_pipeline.source_selector_agent import SourceSelectorAgent
from ..rag_pipeline.response_evaluator_agent import ResponseEvaluatorAgent
from ..rag_pipeline.prompt_builder import PromptBuilder
from .agent_registry import AgentRegistry
from .agentic_logger import AgenticLogger
from .persistence import AgenticPersistence
from .feedback_analyzer import FeedbackAnalyzer
import time

class RAGOrchestrator:
    """
    Orchestrator for Agentic RAG workflows. Manages agent handoffs, iteration control, and workflow state.
    """
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.workflow_state: Dict[str, Any] = {}
        self.max_iterations = 5
        self.logger = AgenticLogger()  # Centralized logger
        self.persistence = None  # Will be set via set_persistence
        self.feedback_analyzer = FeedbackAnalyzer()  # Advanced feedback analysis
        # Register pipeline agents
        self.query_writer = QueryWriterAgent()
        self.context_assessor = ContextAssessmentAgent()
        self.source_selector = SourceSelectorAgent()
        self.response_evaluator = ResponseEvaluatorAgent()
        self.prompt_builder = PromptBuilder()
        self.register_agent(self.query_writer)
        self.register_agent(self.context_assessor)
        self.register_agent(self.source_selector)
        self.register_agent(self.response_evaluator)

    def register_agent(self, agent: AgentBase):
        self.agents[agent.agent_id] = agent

    def set_persistence(self, persistence: AgenticPersistence):
        """Set persistence layer for the orchestrator and all agents."""
        self.persistence = persistence
        # Set persistence for all agents
        for agent in self.agents.values():
            agent.set_persistence(persistence)
        # Set persistence for pipeline agents
        self.query_writer.set_persistence(persistence)
        self.context_assessor.set_persistence(persistence)
        self.source_selector.set_persistence(persistence)
        self.response_evaluator.set_persistence(persistence)

    def start_workflow(self, workflow_id: str, initial_input: dict):
        """Start a new RAG workflow with the given input."""
        self.workflow_state[workflow_id] = {
            'input': initial_input,
            'iterations': 0,
            'history': []
        }
        self.logger.start_workflow(workflow_id)
        self.logger.log_action('workflow_started', {'workflow_id': workflow_id, 'input': initial_input})
        # Save workflow state to persistence
        if self.persistence:
            self.persistence.save_workflow_state(workflow_id, self.workflow_state[workflow_id])

    def run_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline: query rewriting, context assessment, source selection,
        prompt building, LLM call (stub), response evaluation. Log each step.
        """
        state = self.workflow_state.get(workflow_id)
        if not state:
            raise ValueError(f"Workflow {workflow_id} not found.")
        input_data = state['input']
        log = []

        # 1. Query rewriting
        self.logger.log_action('query_rewriting_start', {'workflow_id': workflow_id, 'query': input_data['query']})
        msg = {'content': {'query': input_data['query']}, 'sender': 'user'}
        rewritten = self.query_writer.receive_message(msg)
        refined_query = rewritten['content']['refined_query']
        log.append({'step': 'query_rewriting', 'output': refined_query})
        self.logger.log_action('query_rewriting_complete', {'workflow_id': workflow_id, 'refined_query': refined_query})

        # 2. Context assessment
        self.logger.log_action('context_assessment_start', {'workflow_id': workflow_id, 'query': refined_query})
        msg = {'content': {'query': refined_query}, 'sender': self.query_writer.agent_id}
        context_assess = self.context_assessor.receive_message(msg)
        context_needed = context_assess['content']['context_needed']
        log.append({'step': 'context_assessment', 'output': context_needed})
        self.logger.log_action('context_assessment_complete', {'workflow_id': workflow_id, 'context_needed': context_needed})

        # 3. Source selection
        self.logger.log_action('source_selection_start', {'workflow_id': workflow_id, 'query': refined_query})
        msg = {'content': {'query': refined_query}, 'sender': self.context_assessor.agent_id}
        source_select = self.source_selector.receive_message(msg)
        sources = source_select['content']['sources']
        rationale = source_select['content']['rationale']
        log.append({'step': 'source_selection', 'output': {'sources': sources, 'rationale': rationale}})
        self.logger.log_action('source_selection_complete', {'workflow_id': workflow_id, 'sources': sources, 'rationale': rationale})

        # 4. Prompt building (stub context)
        self.logger.log_action('prompt_building_start', {'workflow_id': workflow_id, 'query': refined_query})
        context = input_data.get('context', 'STUB_CONTEXT')
        prompt = self.prompt_builder.build_prompt(refined_query, context)
        log.append({'step': 'prompt_building', 'output': prompt})
        self.logger.log_action('prompt_building_complete', {'workflow_id': workflow_id, 'prompt_length': len(prompt)})

        # 5. LLM call (stub)
        self.logger.log_action('llm_call_start', {'workflow_id': workflow_id})
        llm_response = f"[LLM OUTPUT for prompt: {prompt[:40]}...]"
        log.append({'step': 'llm_call', 'output': llm_response})
        self.logger.log_action('llm_call_complete', {'workflow_id': workflow_id, 'response_length': len(llm_response)})

        # 6. Response evaluation
        self.logger.log_action('response_evaluation_start', {'workflow_id': workflow_id})
        msg = {'content': {'response': llm_response}, 'sender': 'llm'}
        evaluation = self.response_evaluator.receive_message(msg)
        eval_result = evaluation['content']['evaluation']
        log.append({'step': 'response_evaluation', 'output': eval_result})
        self.logger.log_action('response_evaluation_complete', {'workflow_id': workflow_id, 'evaluation': eval_result})

        # Check for escalation
        escalated = eval_result.get('escalation', False)
        if escalated:
            self.logger.log_action('workflow_escalated', {'workflow_id': workflow_id, 'reason': eval_result.get('escalation_reason')})

        # Update workflow state
        state['history'].append(log)
        state['iterations'] += 1

        # End workflow logging
        self.logger.end_workflow(workflow_id, state['iterations'], eval_result.get('confidence'), escalated)
        self.logger.log_action('workflow_complete', {'workflow_id': workflow_id, 'iterations': state['iterations']})

        # Save updated workflow state to persistence
        if self.persistence:
            self.persistence.save_workflow_state(workflow_id, state)

        return {
            'final_output': llm_response,
            'evaluation': eval_result,
            'log': log
        }

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self.workflow_state.get(workflow_id)

    def stop_workflow(self, workflow_id: str):
        if workflow_id in self.workflow_state:
            del self.workflow_state[workflow_id]

    def set_agent_registry(self, agent_registry: AgentRegistry):
        """Attach an agent registry for delegation/collaboration."""
        self.agent_registry = agent_registry

    def delegate_to_specialist(self, agent_type: str, task: dict):
        """Delegate a task to a specialized agent via the agent registry."""
        if hasattr(self, 'agent_registry') and self.agent_registry:
            return self.agent_registry.delegate_task(agent_type, task)
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from the logger."""
        return self.logger.get_metrics()

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs from the logger."""
        return self.logger.get_logs()

    def add_user_feedback(self, workflow_id: str, feedback: dict):
        """
        Accept and store user feedback for a workflow.
        Logs feedback events and stores feedback in agent memory and persistence.
        """
        self.logger.log_action('user_feedback_received', {
            'workflow_id': workflow_id,
            'feedback': feedback
        })
        # Store feedback in response evaluator's memory
        self.response_evaluator.memory.add_feedback({
            'workflow_id': workflow_id,
            'feedback': feedback,
            'timestamp': time.time()
        })
        # Save feedback to persistence
        if self.persistence:
            self.persistence.save_user_feedback(workflow_id, feedback)
        return True

    def get_user_feedback(self, workflow_id: str = None):
        """Get user feedback from the response evaluator's memory."""
        feedback = self.response_evaluator.memory.get_feedback()
        if workflow_id:
            return [f for f in feedback if f.get('workflow_id') == workflow_id]
        return feedback

    def create_backup(self) -> str:
        """Create a backup of all data using the persistence layer."""
        if self.persistence:
            return self.persistence.create_backup()
        return None

    def load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state from persistence."""
        if self.persistence:
            return self.persistence.load_workflow_state(workflow_id)
        return None

    def get_persisted_logs(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get logs from persistence layer."""
        if self.persistence:
            return self.persistence.load_logs(limit)
        return []

    def get_persisted_user_feedback(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """Get user feedback from persistence layer."""
        if self.persistence:
            return self.persistence.load_user_feedback(workflow_id)
        return []

    def get_feedback_insights(self, workflow_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive feedback insights and analytics.
        Uses advanced analysis including sentiment, patterns, trends, and recommendations.
        """
        feedback = self.get_user_feedback(workflow_id)
        return self.feedback_analyzer.get_feedback_insights(feedback)

    def get_feedback_analytics(self, workflow_id: str = None) -> Dict[str, Any]:
        """
        Get detailed feedback analytics including sentiment analysis and trend analysis.
        """
        feedback = self.get_user_feedback(workflow_id)
        
        if not feedback:
            return {
                'sentiment_analysis': {'sentiment': 'neutral', 'score': 0.0},
                'trend_analysis': {'trend': 'stable', 'confidence': 0.0},
                'pattern_analysis': {},
                'total_feedback': 0
            }
        
        # Analyze sentiment of all feedback
        all_sentiments = []
        for f in feedback:
            feedback_text = f.get('feedback', {}).get('comment', '')
            if feedback_text:
                sentiment = self.feedback_analyzer.analyze_sentiment(feedback_text)
                all_sentiments.append(sentiment)
        
        # Calculate average sentiment
        if all_sentiments:
            avg_sentiment_score = sum(s['score'] for s in all_sentiments) / len(all_sentiments)
            avg_sentiment = 'positive' if avg_sentiment_score > 0.2 else 'negative' if avg_sentiment_score < -0.2 else 'neutral'
        else:
            avg_sentiment_score = 0.0
            avg_sentiment = 'neutral'
        
        # Analyze trends
        trend_analysis = self.feedback_analyzer.analyze_feedback_trends(feedback)
        
        # Extract patterns
        all_patterns = {}
        for f in feedback:
            feedback_text = f.get('feedback', {}).get('comment', '')
            if feedback_text:
                patterns = self.feedback_analyzer.extract_patterns(feedback_text)
                for pattern_type, matches in patterns.items():
                    if pattern_type not in all_patterns:
                        all_patterns[pattern_type] = []
                    all_patterns[pattern_type].extend(matches)
        
        return {
            'sentiment_analysis': {
                'sentiment': avg_sentiment,
                'score': avg_sentiment_score,
                'total_analyzed': len(all_sentiments)
            },
            'trend_analysis': trend_analysis,
            'pattern_analysis': all_patterns,
            'total_feedback': len(feedback)
        } 