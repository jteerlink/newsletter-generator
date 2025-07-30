"""
Agentic RAG Agent with Reasoning Capabilities

This agent can:
- Reason about what information to retrieve
- Dynamically adjust retrieval strategies
- Synthesize information from multiple sources
- Maintain context across multiple retrievals
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from src.storage import ChromaStorageProvider
from src.core.core import query_llm
from src.tools.tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

@dataclass
class RetrievalStep:
    """Represents a single retrieval step in the agentic process."""
    query: str
    strategy: str  # "semantic", "temporal", "hierarchical", "multimodal"
    results: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    timestamp: datetime

@dataclass
class AgenticRAGSession:
    """Represents a complete agentic RAG session."""
    original_query: str
    retrieval_steps: List[RetrievalStep]
    synthesized_response: str
    confidence_score: float
    sources_used: List[str]
    reasoning_chain: List[str]

class AgenticRAGAgent:
    """
    An agent that can reason about retrieval and synthesis strategies.
    """
    
    def __init__(self, vector_store: ChromaStorageProvider, llm_model: str = "deepseek-r1"):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.session_history = []
        
        # Reasoning templates
        self.reasoning_templates = {
            "query_analysis": self._get_query_analysis_template(),
            "strategy_selection": self._get_strategy_selection_template(),
            "result_evaluation": self._get_result_evaluation_template(),
            "synthesis_planning": self._get_synthesis_planning_template()
        }
    
    def process_query(self, 
                     query: str, 
                     context: Optional[Dict[str, Any]] = None,
                     max_iterations: int = 3) -> AgenticRAGSession:
        """
        Process a query using agentic RAG with reasoning.
        
        Args:
            query: The user's query
            context: Additional context from previous interactions
            max_iterations: Maximum number of retrieval iterations
        """
        
        session = AgenticRAGSession(
            original_query=query,
            retrieval_steps=[],
            synthesized_response="",
            confidence_score=0.0,
            sources_used=[],
            reasoning_chain=[]
        )
        
        # Step 1: Analyze the query
        query_analysis = self._analyze_query(query, context)
        session.reasoning_chain.append(f"Query Analysis: {query_analysis['reasoning']}")
        
        # Step 2: Plan retrieval strategy
        strategy_plan = self._plan_retrieval_strategy(query, query_analysis)
        session.reasoning_chain.append(f"Strategy Planning: {strategy_plan['reasoning']}")
        
        # Step 3: Execute retrieval iterations
        for iteration in range(max_iterations):
            # Determine what to retrieve next
            next_retrieval = self._determine_next_retrieval(
                query, session.retrieval_steps, strategy_plan
            )
            
            if not next_retrieval:
                session.reasoning_chain.append(f"Iteration {iteration + 1}: No further retrieval needed")
                break
            
            # Execute retrieval
            retrieval_step = self._execute_retrieval(next_retrieval)
            session.retrieval_steps.append(retrieval_step)
            
            # Evaluate if we have sufficient information
            evaluation = self._evaluate_information_sufficiency(
                query, session.retrieval_steps
            )
            
            session.reasoning_chain.append(
                f"Iteration {iteration + 1}: {retrieval_step.reasoning} - "
                f"Confidence: {evaluation['confidence']:.2f}"
            )
            
            if evaluation['sufficient']:
                break
        
        # Step 4: Synthesize final response
        session.synthesized_response = self._synthesize_response(query, session.retrieval_steps)
        session.confidence_score = self._calculate_session_confidence(session)
        session.sources_used = self._extract_sources(session.retrieval_steps)
        
        # Store session
        self.session_history.append(session)
        
        return session
    
    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the query to understand information needs."""
        
        analysis_prompt = self.reasoning_templates["query_analysis"].format(
            query=query,
            context=json.dumps(context or {}, indent=2)
        )
        
        response = query_llm(analysis_prompt)
        
        # Parse the response (simplified)
        return {
            "query_type": self._extract_query_type(response),
            "information_needs": self._extract_information_needs(response),
            "complexity": self._assess_complexity(response),
            "temporal_requirements": self._check_temporal_requirements(response),
            "reasoning": response
        }
    
    def _plan_retrieval_strategy(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the retrieval strategy based on query analysis."""
        
        strategy_prompt = self.reasoning_templates["strategy_selection"].format(
            query=query,
            analysis=json.dumps(analysis, indent=2)
        )
        
        response = query_llm(strategy_prompt)
        
        # Determine strategy sequence
        strategies = []
        
        if analysis.get("temporal_requirements"):
            strategies.append("temporal")
        
        if analysis.get("complexity", "low") == "high":
            strategies.append("hierarchical")
        
        strategies.append("semantic")  # Always include semantic search
        
        return {
            "strategies": strategies,
            "reasoning": response,
            "priority_order": self._determine_priority_order(analysis)
        }
    
    def _determine_next_retrieval(self, 
                                query: str, 
                                completed_steps: List[RetrievalStep],
                                strategy_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine the next retrieval step to take."""
        
        # Check if we've completed all planned strategies
        completed_strategies = [step.strategy for step in completed_steps]
        planned_strategies = strategy_plan["strategies"]
        
        for strategy in planned_strategies:
            if strategy not in completed_strategies:
                return {
                    "query": self._adapt_query_for_strategy(query, strategy, completed_steps),
                    "strategy": strategy,
                    "reasoning": f"Executing {strategy} strategy as planned"
                }
        
        # Check if we need additional information based on gaps
        gaps = self._identify_information_gaps(query, completed_steps)
        if gaps:
            return {
                "query": self._generate_gap_filling_query(gaps),
                "strategy": "semantic",
                "reasoning": f"Addressing information gaps: {', '.join(gaps)}"
            }
        
        return None
    
    def _execute_retrieval(self, retrieval_plan: Dict[str, Any]) -> RetrievalStep:
        """Execute a specific retrieval step."""
        
        query = retrieval_plan["query"]
        strategy = retrieval_plan["strategy"]
        
        # Execute based on strategy
        if strategy == "temporal":
            # Use temporal query with recent time range
            time_range = {
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            }
            result = self.vector_store.temporal_query(
                query, time_range=time_range, trend_analysis=True
            )
            results = result["results"]
        
        elif strategy == "hierarchical":
            results = self.vector_store.hierarchical_query(
                query, granularity="mixed", top_k=10
            )
        
        elif strategy == "multimodal":
            # For multimodal, we'd need to handle different content types
            results = self.vector_store.query(query, top_k=10)
        
        else:  # semantic
            results = self.vector_store.query(query, top_k=10)
        
        # Evaluate results
        evaluation = self._evaluate_retrieval_results(results, query)
        
        return RetrievalStep(
            query=query,
            strategy=strategy,
            results=results,
            reasoning=retrieval_plan["reasoning"],
            confidence=evaluation["confidence"],
            timestamp=datetime.now()
        )
    
    def _evaluate_information_sufficiency(self, 
                                        query: str, 
                                        retrieval_steps: List[RetrievalStep]) -> Dict[str, Any]:
        """Evaluate if we have sufficient information to answer the query."""
        
        # Collect all results
        all_results = []
        for step in retrieval_steps:
            all_results.extend(step.results)
        
        if not all_results:
            return {"sufficient": False, "confidence": 0.0, "reason": "No results found"}
        
        # Calculate coverage metrics
        total_results = len(all_results)
        avg_confidence = sum(step.confidence for step in retrieval_steps) / len(retrieval_steps)
        
        # Check diversity of sources
        unique_sources = set()
        for result in all_results:
            source = result.get("metadata", {}).get("source", "unknown")
            unique_sources.add(source)
        
        source_diversity = len(unique_sources) / max(total_results, 1)
        
        # Calculate overall sufficiency
        sufficiency_score = (avg_confidence + source_diversity) / 2
        
        return {
            "sufficient": sufficiency_score > 0.7,
            "confidence": sufficiency_score,
            "reason": f"Coverage: {total_results} results, {len(unique_sources)} sources, "
                     f"avg confidence: {avg_confidence:.2f}"
        }
    
    def _synthesize_response(self, query: str, retrieval_steps: List[RetrievalStep]) -> str:
        """Synthesize a comprehensive response from all retrieval steps."""
        
        # Collect all relevant information
        all_info = []
        for step in retrieval_steps:
            for result in step.results:
                all_info.append({
                    "content": result.get("document", ""),
                    "source": result.get("metadata", {}).get("source", "unknown"),
                    "confidence": step.confidence,
                    "strategy": step.strategy
                })
        
        # Sort by confidence
        all_info.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Prepare synthesis prompt
        synthesis_prompt = self.reasoning_templates["synthesis_planning"].format(
            query=query,
            information=json.dumps(all_info[:10], indent=2)  # Top 10 most confident results
        )
        
        response = query_llm(synthesis_prompt)
        return response
    
    def _calculate_session_confidence(self, session: AgenticRAGSession) -> float:
        """Calculate overall confidence for the session."""
        
        if not session.retrieval_steps:
            return 0.0
        
        # Weight by number of results and individual confidence
        total_weight = 0
        weighted_confidence = 0
        
        for step in session.retrieval_steps:
            weight = len(step.results) * step.confidence
            total_weight += weight
            weighted_confidence += weight * step.confidence
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _extract_sources(self, retrieval_steps: List[RetrievalStep]) -> List[str]:
        """Extract unique sources from all retrieval steps."""
        
        sources = set()
        for step in retrieval_steps:
            for result in step.results:
                source = result.get("metadata", {}).get("source")
                if source:
                    sources.add(source)
        
        return list(sources)
    
    # Template methods
    def _get_query_analysis_template(self) -> str:
        return """
        Analyze the following query to understand the information needs:
        
        Query: {query}
        Context: {context}
        
        Please analyze:
        1. What type of information is being requested?
        2. What are the specific information needs?
        3. How complex is this query?
        4. Are there any temporal requirements (recent vs historical)?
        5. What would constitute a complete answer?
        
        Provide a clear analysis with reasoning.
        """
    
    def _get_strategy_selection_template(self) -> str:
        return """
        Based on the query analysis, recommend the best retrieval strategy:
        
        Query: {query}
        Analysis: {analysis}
        
        Available strategies:
        - semantic: Standard similarity search
        - temporal: Time-aware search with trend analysis
        - hierarchical: Multi-granularity search
        - multimodal: Search across text, images, and documents
        
        Recommend the strategy sequence and explain your reasoning.
        """
    
    def _get_result_evaluation_template(self) -> str:
        return """
        Evaluate the quality and relevance of these retrieval results:
        
        Query: {query}
        Results: {results}
        
        Assess:
        1. Relevance to the query
        2. Completeness of information
        3. Quality of sources
        4. Gaps that need to be filled
        
        Provide evaluation with confidence score.
        """
    
    def _get_synthesis_planning_template(self) -> str:
        return """
        Synthesize a comprehensive response using the retrieved information:
        
        Query: {query}
        Information: {information}
        
        Create a well-structured response that:
        1. Directly addresses the query
        2. Incorporates information from multiple sources
        3. Provides clear citations
        4. Maintains factual accuracy
        5. Offers insights and analysis
        
        Response:
        """
    
    # Helper methods (simplified implementations)
    def _extract_query_type(self, response: str) -> str:
        """Extract query type from analysis response."""
        if "factual" in response.lower():
            return "factual"
        elif "analytical" in response.lower():
            return "analytical"
        elif "comparative" in response.lower():
            return "comparative"
        else:
            return "general"
    
    def _extract_information_needs(self, response: str) -> List[str]:
        """Extract information needs from analysis response."""
        # Simplified extraction
        needs = []
        if "definition" in response.lower():
            needs.append("definition")
        if "example" in response.lower():
            needs.append("examples")
        if "comparison" in response.lower():
            needs.append("comparison")
        return needs
    
    def _assess_complexity(self, response: str) -> str:
        """Assess query complexity."""
        if "complex" in response.lower() or "multiple" in response.lower():
            return "high"
        elif "simple" in response.lower() or "straightforward" in response.lower():
            return "low"
        else:
            return "medium"
    
    def _check_temporal_requirements(self, response: str) -> bool:
        """Check if query has temporal requirements."""
        temporal_keywords = ["recent", "latest", "current", "now", "today", "trend"]
        return any(keyword in response.lower() for keyword in temporal_keywords)
    
    def _determine_priority_order(self, analysis: Dict[str, Any]) -> List[str]:
        """Determine priority order for retrieval strategies."""
        priorities = []
        
        if analysis.get("temporal_requirements"):
            priorities.append("temporal")
        
        if analysis.get("complexity") == "high":
            priorities.append("hierarchical")
        
        priorities.append("semantic")
        
        return priorities
    
    def _adapt_query_for_strategy(self, 
                                 query: str, 
                                 strategy: str, 
                                 completed_steps: List[RetrievalStep]) -> str:
        """Adapt query for specific strategy."""
        
        if strategy == "temporal":
            return f"{query} recent developments trends"
        elif strategy == "hierarchical":
            return f"{query} comprehensive analysis detailed information"
        else:
            return query
    
    def _identify_information_gaps(self, 
                                 query: str, 
                                 completed_steps: List[RetrievalStep]) -> List[str]:
        """Identify gaps in retrieved information."""
        
        # Simplified gap identification
        gaps = []
        
        # Check if we have enough diversity
        all_results = []
        for step in completed_steps:
            all_results.extend(step.results)
        
        if len(all_results) < 5:
            gaps.append("insufficient_coverage")
        
        # Check source diversity
        sources = set()
        for result in all_results:
            source = result.get("metadata", {}).get("source", "unknown")
            sources.add(source)
        
        if len(sources) < 3:
            gaps.append("limited_sources")
        
        return gaps
    
    def _generate_gap_filling_query(self, gaps: List[str]) -> str:
        """Generate query to fill identified gaps."""
        
        if "insufficient_coverage" in gaps:
            return "additional information comprehensive coverage"
        elif "limited_sources" in gaps:
            return "alternative sources different perspectives"
        else:
            return "supplementary information"
    
    def _evaluate_retrieval_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Evaluate the quality of retrieval results."""
        
        if not results:
            return {"confidence": 0.0, "quality": "poor"}
        
        # Calculate average similarity
        similarities = []
        for result in results:
            similarity = 1 - (result.get("similarity", 1) or 1)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        
        return {
            "confidence": avg_similarity,
            "quality": "good" if avg_similarity > 0.7 else "fair" if avg_similarity > 0.5 else "poor",
            "result_count": len(results)
        }

# Factory function
def create_agentic_rag_agent(vector_store: ChromaStorageProvider) -> AgenticRAGAgent:
    """Create an agentic RAG agent instance."""
    return AgenticRAGAgent(vector_store) 