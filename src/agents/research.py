"""
Enhanced Research Agent for Newsletter Generation

This module provides the enhanced ResearchAgent class, which is responsible for gathering
information and conducting research for newsletter content with advanced context-aware capabilities.
"""

from __future__ import annotations

import json
import logging
import re  # Added for claim extraction
import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.campaign_context import CampaignContext
import src.core.core as core
from src.core.research_strategy import IntelligentResearchOrchestrator
import src.tools.tools as tools
from src.tools.enhanced_search import EnhancedSearchTool
from src.tools.query_refinement import get_query_refinement_engine

from .base import AgentType, SimpleAgent

logger = logging.getLogger(__name__)


class ResearchAgent(SimpleAgent):
    """Enhanced agent specialized in research and information gathering with advanced context awareness."""

    def __init__(self, name: str = "ResearchAgent", **kwargs):
        super().__init__(
            name=name,
            role="Research Specialist",
            goal="Gather comprehensive, accurate, and context-aware information on given topics with proactive query expansion",
            backstory="""You are an expert research specialist with years of experience in gathering
            information from various sources. You excel at finding the most relevant and recent
            information, verifying facts, and organizing research findings in a clear, structured manner.
            You have access to web search tools and knowledge bases to ensure comprehensive coverage.
            You can adapt your research approach based on audience context and content requirements.
            You are particularly skilled at expanding research queries proactively and verifying
            specific claims reactively to ensure comprehensive and accurate information gathering.""",
            agent_type=AgentType.RESEARCH,
            tools=["search_web", "search_knowledge_base"],
            **kwargs
        )
        self.campaign_context: Optional[CampaignContext] = None
        
        # Initialize enhanced research components
        try:
            self.research_orchestrator = IntelligentResearchOrchestrator()
            self.enhanced_search = EnhancedSearchTool()
            self.query_refiner = get_query_refinement_engine()
            logger.info("Enhanced research components initialized successfully")
        except Exception as e:
            logger.warning(f"Enhanced research components initialization failed: {e}")
            self.research_orchestrator = None
            self.enhanced_search = None
            self.query_refiner = None

    def conduct_context_aware_research(
            self, topic: str, context: CampaignContext) -> Dict[str, Any]:
        """Conduct research based on campaign context and audience persona with enhanced capabilities."""
        self.campaign_context = context

        # Try using enhanced research orchestrator first
        if self.research_orchestrator:
            try:
                logger.info("Using enhanced research orchestrator")
                
                # Use intelligent research orchestrator
                import asyncio
                if asyncio.iscoroutinefunction(self.research_orchestrator.conduct_research):
                    try:
                        loop = asyncio.get_event_loop()
                        enhanced_results = loop.run_until_complete(
                            self.research_orchestrator.conduct_research(topic, context)
                        )
                    except RuntimeError:
                        # If no event loop is running, create one
                        enhanced_results = asyncio.run(
                            self.research_orchestrator.conduct_research(topic, context)
                        )
                else:
                    enhanced_results = self.research_orchestrator.conduct_research(topic, context)
                
                if enhanced_results and enhanced_results.results:
                    logger.info(f"Enhanced research found {len(enhanced_results.results)} results with confidence {enhanced_results.confidence:.2f}")
                    
                    # Convert enhanced results to the expected format
                    research_results = []
                    for result in enhanced_results.results:
                        research_results.append({
                            'source': result.source,
                            'query': topic,  # Use original topic as query
                            'content': f"{result.title}\n\n{result.snippet}",
                            'title': result.title,
                            'url': result.url,
                            'relevance_to_topic': 0.8,  # High relevance from enhanced search
                            'audience_alignment': 0.7,  # Good alignment
                            'confidence_score': enhanced_results.confidence,
                            'enhanced_search': True
                        })
                    
                    # Validate and process enhanced results
                    validated_results = self.validate_sources(research_results)
                    verification_results = self._perform_reactive_verification(validated_results, context)
                    
                    # Generate structured output with enhanced metadata
                    structured_output = self.generate_structured_output({
                        'topic': topic,
                        'context': context,
                        'research_results': validated_results,
                        'verification_results': verification_results,
                        'search_queries': [topic],  # Simplified for enhanced search
                        'enhanced_metadata': {
                            'orchestrator_used': True,
                            'execution_time': enhanced_results.execution_time,
                            'sources_used': enhanced_results.sources_used,
                            'total_results': enhanced_results.total_results,
                            'synthesis_score': enhanced_results.synthesis_score
                        }
                    })
                    
                    return structured_output
                    
            except Exception as e:
                logger.warning(f"Enhanced research orchestrator failed: {e}")
        
        # Fallback to original research approach
        logger.info("Using fallback research approach")
        
        # Generate context-aware search queries with proactive expansion
        search_queries = self._generate_context_aware_queries(topic, context)
        
        # Use query refiner if available
        if self.query_refiner:
            try:
                refined_queries = self.query_refiner.refine_query(topic, context)
                search_queries.extend(refined_queries[:5])  # Add top 5 refined queries
            except Exception as e:
                logger.warning(f"Query refinement failed: {e}")
        
        expanded_queries = self._expand_queries_proactively(search_queries, context)

        # Execute research with context consideration
        research_results = self._execute_context_aware_research(
            topic, expanded_queries, context)

        # Validate sources with confidence scoring
        validated_results = self.validate_sources(research_results)

        # Perform reactive verification for critical claims
        verification_results = self._perform_reactive_verification(
            validated_results, context)

        # Generate structured output
        structured_output = self.generate_structured_output({
            'topic': topic,
            'context': context,
            'research_results': validated_results,
            'verification_results': verification_results,
            'search_queries': expanded_queries,
            'enhanced_metadata': {
                'orchestrator_used': False,
                'fallback_reason': 'Enhanced orchestrator unavailable or failed'
            }
        })

        return structured_output

    def validate_sources(self, findings: List[Dict]) -> List[Dict]:
        """Validate sources and add confidence scores with enhanced validation."""
        validated_findings = []

        for finding in findings:
            # Add confidence score based on source quality
            confidence_score = self._calculate_source_confidence(finding)

            # Add validation metadata
            validated_finding = {
                **finding,
                'confidence_score': confidence_score,
                'validation_timestamp': time.time(),
                'source_quality': self._assess_source_quality(finding),
                'relevance_score': self._calculate_relevance_score(finding),
                'freshness_score': self._calculate_freshness_score(finding),
                'authority_score': self._calculate_authority_score(finding)
            }

            validated_findings.append(validated_finding)

        # Sort by confidence score
        validated_findings.sort(
            key=lambda x: x['confidence_score'],
            reverse=True)

        return validated_findings

    def generate_structured_output(
            self, research_data: Dict) -> Dict[str, Any]:
        """Generate structured JSON output with enhanced sections."""
        topic = research_data['topic']
        context = research_data['context']
        results = research_data['research_results']
        verification_results = research_data.get('verification_results', [])

        # Organize findings by sections with enhanced categorization
        sections = {
            'key_facts': [],
            'recent_developments': [],
            'expert_opinions': [],
            'related_topics': [],
            'verified_claims': [],
            'unverified_claims': [],
            'trending_insights': [],
            'audience_specific_findings': []
        }

        # Categorize findings based on context
        for finding in results:
            category = self._categorize_finding_enhanced(finding, context)
            if category in sections:
                sections[category].append(finding)

        # Add verification results
        for verification in verification_results:
            if verification.get('verification_status') == 'verified':
                sections['verified_claims'].append(verification)
            else:
                sections['unverified_claims'].append(verification)

        # Generate research insights
        insights = self._generate_research_insights(results, context)

        # Assess research depth and coverage
        depth_assessment = self._assess_research_depth_enhanced(
            results, context)
        coverage_score = self._calculate_coverage_score_enhanced(
            results, topic, context)

        return {
            'topic': topic,
            'sections': sections,
            'insights': insights,
            'depth_assessment': depth_assessment,
            'coverage_score': coverage_score,
            'total_findings': len(results),
            'verified_claims_count': len(
                sections['verified_claims']),
            'unverified_claims_count': len(
                sections['unverified_claims']),
            'research_quality_score': self._calculate_research_quality_score(
                results,
                context),
            'recommendations': self._generate_research_recommendations_enhanced(
                results,
                context)}

    def verify_specific_claim(
            self, claim: str, context: str) -> Dict[str, Any]:
        """Verify specific claims with enhanced verification capabilities."""
        # Generate verification queries
        verification_queries = self._generate_verification_queries_enhanced(
            claim)

        # Execute verification searches
        verification_results = []
        for query in verification_queries:
            search_results = tools.search_web(query)
            verification_results.extend(
                self._analyze_verification_results_enhanced(
                    search_results, claim))

        # Analyze verification results
        analysis = self._analyze_verification_results_enhanced(
            verification_results, claim)

        # Calculate verification confidence
        confidence_score = self._calculate_verification_confidence(analysis)

        return {
            'claim': claim,
            'verification_status': 'verified' if confidence_score > 0.7 else 'unverified',
            'confidence_score': confidence_score,
            'supporting_evidence': analysis.get(
                'supporting_evidence',
                []),
            'contradicting_evidence': analysis.get(
                'contradicting_evidence',
                []),
            'verification_queries': verification_queries,
            'verification_notes': analysis.get(
                'notes',
                '')}

    def enhanced_search_with_confidence(self, query: str, context_hints: List[str] = None, max_results: int = 10) -> List[Dict]:
        """
        Use enhanced search tool with confidence scoring.
        
        Args:
            query: Search query
            context_hints: Context hints for intelligent provider selection
            max_results: Maximum number of results
            
        Returns:
            List of search results with enhanced metadata
        """
        try:
            if self.enhanced_search:
                # Use intelligent search from enhanced tool
                results = self.enhanced_search.intelligent_search(query, context_hints or [], max_results)
                
                # Convert to expected format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'title': result.title,
                        'url': result.url,
                        'snippet': result.snippet,
                        'source': result.source,
                        'confidence_score': result.confidence_score,
                        'relevance_score': result.relevance_score,
                        'authority_score': result.authority_score,
                        'overall_score': result.overall_score,
                        'enhanced_search': True
                    })
                
                return formatted_results
            else:
                logger.warning("Enhanced search tool not available, using fallback")
                return []
                
        except Exception as e:
            logger.error(f"Enhanced search with confidence failed: {e}")
            return []

    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute research task with enhanced context-aware capabilities."""
        logger.info(f"Enhanced ResearchAgent executing research task: {task}")

        # Check if we have campaign context
        campaign_context = kwargs.get('campaign_context')
        if campaign_context:
            # Use context-aware research
            research_results = self.conduct_context_aware_research(
                task, campaign_context)
            return json.dumps(research_results, indent=2)

        # Fall back to standard research (ensure tool calls for tests)
        knowledge_results = self._generate_knowledge_based_research(
            task, context)
        web_results = self._execute_web_research(task, context)
        combined_results = self._synthesize_research_results(
            knowledge_results, web_results, task)

        return combined_results

    def _generate_knowledge_based_research(
            self, task: str, context: str) -> str:
        """Generate research using knowledge base."""
        try:
            knowledge_prompt = f"""
            Research Task: {task}
            Context: {context}

            Please search the knowledge base for relevant information on this topic.
            Focus on finding:
            1. Key facts and data points
            2. Recent developments
            3. Expert opinions and analysis
            4. Related topics and connections

            Provide a structured summary of your findings.
            """

            # Use knowledge base search
            knowledge_results = tools.search_knowledge_base(task)

            if knowledge_results and knowledge_results.strip():
                return f"Knowledge Base Research:\n{knowledge_results}"
            else:
                return "No relevant information found in knowledge base."

        except Exception as e:
            logger.error(f"Error in knowledge-based research: {e}")
            return f"Error accessing knowledge base: {e}"

    def _execute_web_research(self, task: str, context: str) -> str:
        """Execute web research for the task."""
        try:
            # Extract search queries from task
            search_queries = self._generate_search_queries(task, context)

            web_results = []
            for query in search_queries:
                try:
                    result = tools.search_web(query)
                    if result and result.strip():
                        web_results.append(
                            f"Search Query: {query}\nResults:\n{result}")
                except Exception as e:
                    logger.warning(
                        f"Web search failed for query '{query}': {e}")

            return "\n\n".join(
                web_results) if web_results else "No web search results found."

        except Exception as e:
            logger.error(f"Error in web research: {e}")
            return f"Error in web research: {e}"

    def _generate_search_queries(self, task: str, context: str) -> List[str]:
        """Generate multiple search queries for comprehensive research."""
        # Create a prompt to generate search queries
        query_generation_prompt = f"""
        Task: {task}
        Context: {context}

        Generate 3-5 specific search queries to gather comprehensive information on this topic.
        Make the queries:
        1. Specific and focused
        2. Cover different aspects of the topic
        3. Include recent developments
        4. Target authoritative sources

        Return only the search queries, one per line.
        """

        try:
            response = core.query_llm(query_generation_prompt)
            queries = [line.strip()
                       for line in response.split('\n') if line.strip()]

            # Fallback to simple query if LLM fails
            if not queries:
                queries = [self._extract_search_query(task)]

            return queries[:5]  # Limit to 5 queries

        except Exception as e:
            logger.warning(f"Failed to generate search queries: {e}")
            return [self._extract_search_query(task)]

    def _synthesize_research_results(
            self,
            knowledge_results: str,
            web_results: str,
            task: str) -> str:
        """Synthesize and organize research results."""
        synthesis_prompt = f"""
        Research Task: {task}

        Knowledge Base Findings:
        {knowledge_results}

        Web Search Findings:
        {web_results}

        Please synthesize these research findings into a comprehensive, well-organized summary.
        Structure your response with:
        1. Executive Summary
        2. Key Findings
        3. Supporting Evidence
        4. Recent Developments
        5. Expert Opinions
        6. Related Topics
        7. Sources and References

        Ensure the information is:
        - Accurate and factual
        - Well-organized and easy to understand
        - Comprehensive but concise
        - Focused on the research task
        """

        try:
            return core.query_llm(synthesis_prompt)
        except Exception as e:
            logger.error(f"Error synthesizing research results: {e}")
            return f"""
            Research Synthesis Error: {e}

            Knowledge Base Results:
            {knowledge_results}

            Web Search Results:
            {web_results}
            """

    def _build_research_prompt_with_tools(
            self,
            task: str,
            context: str,
            tool_output: str) -> str:
        """Build enhanced prompt for research with tool results."""
        prompt_parts = [
            f"You are a {self.role}.",
            f"Your goal: {self.goal}",
            f"Background: {self.backstory}",
            "",
            f"Research Task: {task}"
        ]

        if context:
            prompt_parts.extend([
                "",
                f"Context: {context}"
            ])

        prompt_parts.extend([
            "",
            "Research Findings:",
            tool_output,
            "",
            "Based on the research findings above, please provide:",
            "1. A comprehensive summary of the key information",
            "2. Analysis of the findings and their implications",
            "3. Identification of trends or patterns",
            "4. Recommendations for further research if needed",
            "5. A structured format suitable for newsletter content",
            "",
            "Ensure your response is well-organized, factual, and provides valuable insights."
        ])

        return "\n".join(prompt_parts)

    def get_research_analytics(self) -> Dict[str, Any]:
        """Get research-specific analytics."""
        analytics = self.get_tool_usage_analytics()

        # Add research-specific metrics
        research_metrics = {
            "research_sessions": len(
                self.execution_history),
            "avg_research_time": sum(
                r.execution_time for r in self.execution_history) / len(
                self.execution_history) if self.execution_history else 0,
            "success_rate": sum(
                1 for r in self.execution_history if r.status.value == "completed") / len(
                self.execution_history) if self.execution_history else 0,
            "tool_usage_breakdown": {
                "search_web": sum(
                    1 for entry in analytics.get(
                        "agent_tool_usage",
                        []) if entry.get("tool_name") == "search_web"),
                "search_knowledge_base": sum(
                    1 for entry in analytics.get(
                        "agent_tool_usage",
                        []) if entry.get("tool_name") == "search_knowledge_base")}}

        analytics.update(research_metrics)
        return analytics

    def _generate_context_aware_queries(
            self, topic: str, context: CampaignContext) -> List[str]:
        """Generate search queries based on campaign context."""
        queries = []

        # Base query
        base_query = f"{topic}"

        # Add audience-specific terms
        audience = context.audience_persona.get('demographics', '')
        if 'technical' in audience.lower() or 'expert' in audience.lower():
            base_query += " technical analysis expert insights"
        elif 'business' in audience.lower():
            base_query += " business impact market analysis"
        elif 'general' in audience.lower():
            base_query += " overview guide introduction"

        # Add interest-specific terms
        interests = context.audience_persona.get('interests', '')
        if interests:
            interest_terms = interests.split(',')[:2]  # Take first 2 interests
            for interest in interest_terms:
                queries.append(f"{topic} {interest.strip()}")

        # Add pain point-specific queries
        pain_points = context.audience_persona.get('pain_points', '')
        if pain_points:
            pain_point_terms = pain_points.split(',')[:2]
            for pain_point in pain_point_terms:
                queries.append(f"{topic} solution {pain_point.strip()}")

        # Add strategic goal queries
        strategic_goals = context.strategic_goals[:2]  # Take first 2 goals
        for goal in strategic_goals:
            queries.append(f"{topic} {goal.lower()}")

        # Ensure base query is included
        if base_query not in queries:
            queries.insert(0, base_query)

        return queries[:5]  # Limit to 5 queries

    def _execute_context_aware_research(
            self,
            topic: str,
            queries: List[str],
            context: CampaignContext) -> List[Dict]:
        """Execute research with context consideration."""
        research_results = []

        for query in queries:
            try:
                # Execute web search
                web_result = search_web(query)
                if web_result:
                    research_results.append({
                        'source': 'web_search',
                        'query': query,
                        'content': web_result,
                        'relevance_to_topic': self._calculate_topic_relevance(web_result, topic),
                        'audience_alignment': self._calculate_audience_alignment(web_result, context)
                    })

                # Execute knowledge base search
                kb_result = search_knowledge_base(query)
                if kb_result:
                    research_results.append({
                        'source': 'knowledge_base',
                        'query': query,
                        'content': kb_result,
                        'relevance_to_topic': self._calculate_topic_relevance(kb_result, topic),
                        'audience_alignment': self._calculate_audience_alignment(kb_result, context)
                    })

            except Exception as e:
                logger.warning(f"Research query failed: {e}")

        return research_results

    def _calculate_source_confidence(self, finding: Dict) -> float:
        """Calculate confidence score for a source."""
        confidence_score = 0.5  # Base score

        # Source type confidence
        source_type = finding.get('source', 'unknown')
        if source_type == 'knowledge_base':
            confidence_score += 0.3
        elif source_type == 'web_search':
            confidence_score += 0.2

        # Content length confidence
        content = finding.get('content', '')
        if len(content) > 500:
            confidence_score += 0.1
        elif len(content) > 200:
            confidence_score += 0.05

        # Relevance confidence
        relevance = finding.get('relevance_to_topic', 0.5)
        confidence_score += relevance * 0.2

        # Audience alignment confidence
        alignment = finding.get('audience_alignment', 0.5)
        confidence_score += alignment * 0.1

        return min(confidence_score, 1.0)

    def _assess_source_quality(self, finding: Dict) -> str:
        """Assess the quality of a source."""
        confidence = finding.get('confidence_score', 0.5)

        if confidence > 0.8:
            return 'high'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'low'

    def _calculate_relevance_score(self, finding: Dict) -> float:
        """Calculate relevance score for a finding."""
        topic_relevance = finding.get('relevance_to_topic', 0.5)
        audience_alignment = finding.get('audience_alignment', 0.5)

        # Weighted average
        return (topic_relevance * 0.7) + (audience_alignment * 0.3)

    def _categorize_finding(
            self,
            finding: Dict,
            context: CampaignContext) -> str:
        """Categorize a finding based on content and context."""
        content = finding.get('content', '').lower()

        # Simple categorization based on keywords
        if any(
            word in content for word in [
                'data',
                'statistics',
                'numbers',
                'percent']):
            return 'data_points'
        elif any(word in content for word in ['trend', 'growing', 'increasing', 'declining']):
            return 'trends'
        elif any(word in content for word in ['expert', 'analyst', 'specialist', 'opinion']):
            return 'expert_opinions'
        elif any(word in content for word in ['recent', 'latest', 'new', 'update']):
            return 'recent_developments'
        elif any(word in content for word in ['related', 'similar', 'connected']):
            return 'related_topics'
        else:
            return 'key_facts'

    def _assess_research_depth(
            self,
            results: List[Dict],
            context: CampaignContext) -> str:
        """Assess the depth of research based on results and context."""
        if not results:
            return 'shallow'

        # Count high-confidence findings
        high_confidence_count = len(
            [r for r in results if r.get('confidence_score', 0) > 0.8])

        # Consider audience requirements
        audience = context.audience_persona.get('demographics', '')
        if 'expert' in audience.lower() and high_confidence_count < 5:
            return 'insufficient'
        elif 'general' in audience.lower() and high_confidence_count >= 3:
            return 'adequate'
        elif high_confidence_count >= 5:
            return 'comprehensive'
        else:
            return 'moderate'

    def _calculate_coverage_score(
            self,
            results: List[Dict],
            topic: str) -> float:
        """Calculate how well the research covers the topic."""
        if not results:
            return 0.0

        # Simple coverage calculation based on result count and quality
        total_score = sum(r.get('confidence_score', 0) for r in results)
        return min(total_score / len(results), 1.0)

    def _generate_research_recommendations(
            self,
            results: List[Dict],
            context: CampaignContext) -> List[str]:
        """Generate recommendations based on research results."""
        recommendations = []

        # Check research depth
        depth = self._assess_research_depth(results, context)
        if depth == 'insufficient':
            recommendations.append(
                "Consider additional research to meet expert audience requirements")
        elif depth == 'shallow':
            recommendations.append(
                "Expand research scope to provide comprehensive coverage")

        # Check source diversity
        source_types = set(r.get('source', '') for r in results)
        if len(source_types) < 2:
            recommendations.append(
                "Diversify research sources for better coverage")

        # Check recent information
        recent_findings = [
            r for r in results if 'recent_developments' in r.get(
                'content', '').lower()]
        if len(recent_findings) < 2:
            recommendations.append(
                "Include more recent developments and updates")

        return recommendations

    def _generate_verification_queries(self, claim: str) -> List[str]:
        """Generate queries to verify a specific claim."""
        queries = [
            f'"{claim}"',
            f'fact check "{claim}"',
            f'"{claim}" verification',
            f'"{claim}" evidence',
            f'"{claim}" source'
        ]
        return queries

    def _analyze_verification_results(
            self, results: List[Dict], claim: str) -> Dict[str, Any]:
        """Analyze verification results for a claim."""
        supporting_evidence = []
        contradicting_evidence = []

        for result in results:
            content = result.get('result', '').lower()
            claim_lower = claim.lower()

            # Simple analysis - could be enhanced with NLP
            if claim_lower in content:
                supporting_evidence.append(result)
            elif any(word in content for word in ['false', 'incorrect', 'wrong', 'debunked']):
                contradicting_evidence.append(result)

        # Determine confidence level
        if len(supporting_evidence) > len(contradicting_evidence):
            confidence_level = 'supported'
        elif len(contradicting_evidence) > len(supporting_evidence):
            confidence_level = 'contradicted'
        else:
            confidence_level = 'uncertain'

        return {
            'confidence_level': confidence_level,
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'total_sources': len(results)
        }

    def _calculate_topic_relevance(self, content: str, topic: str) -> float:
        """Calculate relevance of content to topic."""
        # Simple keyword-based relevance
        topic_words = topic.lower().split()
        content_lower = content.lower()

        relevance_score = 0.0
        for word in topic_words:
            if word in content_lower:
                relevance_score += 0.2

        return min(relevance_score, 1.0)

    def _calculate_audience_alignment(
            self,
            content: str,
            context: CampaignContext) -> float:
        """Calculate how well content aligns with audience."""
        audience = context.audience_persona.get('demographics', '').lower()
        content_lower = content.lower()

        alignment_score = 0.5  # Base score

        if 'technical' in audience and any(
            word in content_lower for word in [
                'technical', 'analysis', 'expert']):
            alignment_score += 0.3
        elif 'business' in audience and any(word in content_lower for word in ['business', 'market', 'strategy']):
            alignment_score += 0.3
        elif 'general' in audience and any(word in content_lower for word in ['overview', 'guide', 'introduction']):
            alignment_score += 0.3

        return min(alignment_score, 1.0)

    def _expand_queries_proactively(
            self,
            base_queries: List[str],
            context: CampaignContext) -> List[str]:
        """Expand research queries proactively based on context."""
        expanded_queries = base_queries.copy()

        # Add audience-specific queries
        audience_persona = context.audience_persona
        knowledge_level = audience_persona.get(
            'knowledge_level', 'intermediate')
        interests = audience_persona.get('interests', '')

        for query in base_queries:
            # Add knowledge level specific queries
            if knowledge_level == 'beginner':
                expanded_queries.append(f"{query} for beginners")
                expanded_queries.append(f"{query} explained simply")
            elif knowledge_level == 'advanced':
                expanded_queries.append(f"{query} advanced analysis")
                expanded_queries.append(f"{query} technical details")

            # Add interest-specific queries
            if interests:
                expanded_queries.append(f"{query} {interests}")

        # Add strategic goal specific queries
        strategic_goals = context.strategic_goals
        for goal in strategic_goals:
            for query in base_queries:
                expanded_queries.append(f"{query} {goal}")

        # Add recent developments queries
        for query in base_queries:
            expanded_queries.append(f"{query} latest news")
            expanded_queries.append(f"{query} recent developments")

        return list(set(expanded_queries))  # Remove duplicates

    def _perform_reactive_verification(
            self,
            research_results: List[Dict],
            context: CampaignContext) -> List[Dict]:
        """Perform reactive verification for critical claims in research results."""
        verification_results = []

        # Extract claims that need verification
        claims_to_verify = self._extract_claims_for_verification(
            research_results)

        for claim in claims_to_verify:
            verification_result = self.verify_specific_claim(
                claim, str(context))
            verification_results.append(verification_result)

        return verification_results

    def _extract_claims_for_verification(
            self, research_results: List[Dict]) -> List[str]:
        """Extract claims that need verification from research results."""
        claims = []

        for result in research_results:
            content = result.get('content', '') + result.get('summary', '')

            # Look for factual statements that need verification
            factual_patterns = [
                r'(\d+% of .+?)',
                r'(According to .+?, .+?)',
                r'(Research shows that .+?)',
                r'(Studies indicate that .+?)',
                r'(Experts say .+?)',
                r'(Data shows .+?)'
            ]

            for pattern in factual_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                claims.extend(matches)

        return list(set(claims))[:10]  # Limit to top 10 unique claims

    def _categorize_finding_enhanced(
            self,
            finding: Dict,
            context: CampaignContext) -> str:
        """Enhanced categorization of findings based on context."""
        content = finding.get('content', '').lower()
        title = finding.get('title', '').lower()

        # Check for trending indicators
        trending_indicators = [
            'trending',
            'latest',
            'new',
            'recent',
            'emerging']
        if any(
                indicator in title or indicator in content for indicator in trending_indicators):
            return 'trending_insights'

        # Check for audience-specific content
        audience_persona = context.audience_persona
        interests = audience_persona.get('interests', '').lower()
        if interests and any(
                interest in content for interest in interests.split(',')):
            return 'audience_specific_findings'

        # Check for expert opinions
        expert_indicators = [
            'expert',
            'specialist',
            'analyst',
            'researcher',
            'professor']
        if any(indicator in content for indicator in expert_indicators):
            return 'expert_opinions'

        # Check for recent developments
        time_indicators = [
            'today',
            'yesterday',
            'this week',
            'this month',
            '2024',
            '2025']
        if any(indicator in content for indicator in time_indicators):
            return 'recent_developments'

        # Default categorization
        return 'key_facts'

    def _assess_research_depth_enhanced(
            self, results: List[Dict], context: CampaignContext) -> Dict[str, Any]:
        """Enhanced assessment of research depth based on context."""
        total_findings = len(results)
        high_confidence_findings = len(
            [r for r in results if r.get('confidence_score', 0) > 0.8])
        verified_claims = len(
            [r for r in results if r.get('verification_status') == 'verified'])

        # Calculate depth metrics
        depth_score = min(1.0,
                          (high_confidence_findings / max(1,
                                                          total_findings)) * 0.6 + (verified_claims / max(1,
                                                                                    total_findings)) * 0.4)

        # Assess coverage of strategic goals
        goal_coverage = self._assess_goal_coverage(results, context)

        return {
            'depth_score': depth_score,
            'total_findings': total_findings,
            'high_confidence_findings': high_confidence_findings,
            'verified_claims': verified_claims,
            'goal_coverage': goal_coverage,
            'depth_level': 'comprehensive' if depth_score > 0.8 else 'moderate' if depth_score > 0.5 else 'basic'
        }

    def _calculate_coverage_score_enhanced(
            self,
            results: List[Dict],
            topic: str,
            context: CampaignContext) -> float:
        """Enhanced calculation of research coverage score."""
        if not results:
            return 0.0

        # Calculate topic relevance
        topic_relevance = sum(
            self._calculate_topic_relevance_enhanced(
                result, topic) for result in results) / len(results)

        # Calculate audience alignment
        audience_alignment = sum(
            self._calculate_audience_alignment_enhanced(
                result, context) for result in results) / len(results)

        # Calculate source diversity
        source_diversity = len(set(result.get('source', 'unknown')
                               for result in results)) / max(1, len(results))

        # Weighted average
        coverage_score = (
            topic_relevance *
            0.4 +
            audience_alignment *
            0.4 +
            source_diversity *
            0.2)

        return min(1.0, coverage_score)

    def _calculate_research_quality_score(
            self,
            results: List[Dict],
            context: CampaignContext) -> float:
        """Calculate overall research quality score."""
        if not results:
            return 0.0

        # Calculate average confidence score
        avg_confidence = sum(result.get('confidence_score', 0)
                             for result in results) / len(results)

        # Calculate source quality score
        source_quality_scores = [
            result.get(
                'source_quality_score',
                0.5) for result in results]
        avg_source_quality = sum(source_quality_scores) / \
            len(source_quality_scores)

        # Calculate freshness score
        freshness_scores = [
            result.get(
                'freshness_score',
                0.5) for result in results]
        avg_freshness = sum(freshness_scores) / len(freshness_scores)

        # Weighted quality score
        quality_score = (
            avg_confidence *
            0.4 +
            avg_source_quality *
            0.4 +
            avg_freshness *
            0.2)

        return min(1.0, quality_score)

    def _generate_research_recommendations_enhanced(
            self, results: List[Dict], context: CampaignContext) -> List[str]:
        """Generate enhanced research recommendations based on context."""
        recommendations = []

        # Check for gaps in coverage
        if len(results) < 5:
            recommendations.append(
                "Expand research to include more sources and perspectives")

        # Check for low confidence findings
        low_confidence_count = len(
            [r for r in results if r.get('confidence_score', 0) < 0.6])
        if low_confidence_count > len(results) * 0.3:
            recommendations.append(
                "Verify low-confidence findings with additional sources")

        # Check for goal alignment
        goal_coverage = self._assess_goal_coverage(results, context)
        uncovered_goals = [
            goal for goal,
            covered in goal_coverage.items() if not covered]
        if uncovered_goals:
            recommendations.append(
                f"Expand research to cover strategic goals: {
                    ', '.join(uncovered_goals)}")

        # Check for audience alignment
        audience_alignment_scores = [
            self._calculate_audience_alignment_enhanced(
                r, context) for r in results]
        avg_audience_alignment = sum(
            audience_alignment_scores) / len(audience_alignment_scores)
        if avg_audience_alignment < 0.6:
            recommendations.append(
                "Include more audience-specific research findings")

        return recommendations

    def _assess_goal_coverage(
            self, results: List[Dict], context: CampaignContext) -> Dict[str, bool]:
        """Assess coverage of strategic goals in research results."""
        strategic_goals = context.strategic_goals
        goal_coverage = {}

        for goal in strategic_goals:
            goal_keywords = self._get_goal_keywords(goal)
            goal_mentioned = any(
                any(keyword in result.get('content', '').lower() or keyword in result.get('title', '').lower()
                    for keyword in goal_keywords)
                for result in results
            )
            goal_coverage[goal] = goal_mentioned

        return goal_coverage

    def _get_goal_keywords(self, goal: str) -> List[str]:
        """Get keywords associated with a strategic goal."""
        goal_keywords = {
            'engagement': ['interactive', 'participate', 'join', 'share', 'community'],
            'education': ['learn', 'understand', 'explain', 'demonstrate', 'teach'],
            'conversion': ['sign up', 'subscribe', 'download', 'register', 'purchase'],
            'awareness': ['inform', 'announce', 'highlight', 'showcase', 'promote']
        }

        return goal_keywords.get(goal.lower(), [goal.lower()])

    def _calculate_topic_relevance_enhanced(
            self, result: Dict, topic: str) -> float:
        """Enhanced calculation of topic relevance."""
        content = result.get('content', '').lower()
        title = result.get('title', '').lower()

        # Calculate relevance based on keyword overlap
        topic_words = set(topic.lower().split())
        content_words = set(content.split())
        title_words = set(title.split())

        # Calculate overlap scores
        content_overlap = len(topic_words.intersection(
            content_words)) / max(1, len(topic_words))
        title_overlap = len(topic_words.intersection(
            title_words)) / max(1, len(topic_words))

        # Weighted relevance score
        relevance_score = (title_overlap * 0.6 + content_overlap * 0.4)

        return min(1.0, relevance_score)

    def _calculate_audience_alignment_enhanced(
            self, result: Dict, context: CampaignContext) -> float:
        """Enhanced calculation of audience alignment."""
        content = result.get('content', '').lower()
        audience_persona = context.audience_persona

        # Check knowledge level alignment
        knowledge_level = audience_persona.get(
            'knowledge_level', 'intermediate')
        technical_terms = [
            'algorithm',
            'framework',
            'protocol',
            'infrastructure',
            'architecture']
        technical_count = sum(1 for term in technical_terms if term in content)

        if knowledge_level == 'beginner' and technical_count > 2:
            alignment_score = 0.3
        elif knowledge_level == 'advanced' and technical_count < 1:
            alignment_score = 0.4
        else:
            alignment_score = 0.8

        # Check interest alignment
        interests = audience_persona.get('interests', '').lower()
        if interests:
            interest_words = set(interests.split(','))
            content_words = set(content.split())
            interest_overlap = len(interest_words.intersection(
                content_words)) / max(1, len(interest_words))
            alignment_score = (alignment_score * 0.7 + interest_overlap * 0.3)

        return min(1.0, alignment_score)

    def _generate_verification_queries_enhanced(self, claim: str) -> List[str]:
        """Generate enhanced verification queries for a claim."""
        queries = []

        # Basic verification query
        queries.append(f'"{claim}" fact check')
        queries.append(f'"{claim}" verify')

        # Add source-specific queries
        queries.append(f'"{claim}" reliable sources')
        queries.append(f'"{claim}" research studies')

        # Add time-specific queries
        queries.append(f'"{claim}" recent data')
        queries.append(f'"{claim}" latest information')

        # Add authority-specific queries
        queries.append(f'"{claim}" expert opinion')
        queries.append(f'"{claim}" official sources')

        return queries

    def _analyze_verification_results_enhanced(
            self, results: List[Dict], claim: str) -> Dict[str, Any]:
        """Enhanced analysis of verification results."""
        supporting_evidence = []
        contradicting_evidence = []

        # Handle both string and dict results
        if isinstance(results, str):
            # If results is a string (error message), return empty analysis
            return {
                'supporting_evidence': [],
                'contradicting_evidence': [],
                'confidence_score': 0.5,
                'total_evidence_count': 0,
                'notes': f"No verification results found: {results}"
            }

        for result in results:
            # Handle both dict and string results
            if isinstance(result, dict):
                content = result.get('content', '').lower()
                title = result.get('title', '').lower()
            else:
                content = str(result).lower()
                title = ''

            # Check for supporting evidence
            if any(word in content for word in claim.lower().split()):
                supporting_evidence.append(result)

            # Check for contradicting evidence
            contradicting_words = [
                'false', 'incorrect', 'wrong', 'debunked', 'myth']
            if any(word in content for word in contradicting_words):
                contradicting_evidence.append(result)

        # Calculate verification confidence
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        if total_evidence == 0:
            confidence_score = 0.5  # Neutral if no evidence
        else:
            confidence_score = len(supporting_evidence) / total_evidence

        return {
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'confidence_score': confidence_score,
            'total_evidence_count': total_evidence,
            'notes': f"Found {
                len(supporting_evidence)} supporting and {
                len(contradicting_evidence)} contradicting sources"}

    def _calculate_verification_confidence(
            self, analysis: Dict[str, Any]) -> float:
        """Calculate verification confidence score."""
        return analysis.get('confidence_score', 0.5)

    def _calculate_freshness_score(self, finding: Dict) -> float:
        """Calculate freshness score for a finding."""
        # Simple freshness calculation based on date if available
        date = finding.get('date', '')
        if not date:
            return 0.5  # Default score if no date

        # This would typically parse the date and calculate freshness
        # For now, return a default score
        return 0.7

    def _calculate_authority_score(self, finding: Dict) -> float:
        """Calculate authority score for a finding."""
        source = finding.get('source', '').lower()

        # Authority scoring based on source type
        authority_sources = [
            'journal',
            'university',
            'research',
            'study',
            'academic']
        if any(auth in source for auth in authority_sources):
            return 0.9

        # Medium authority sources
        medium_sources = ['blog', 'article', 'news']
        if any(med in source for med in medium_sources):
            return 0.6

        return 0.4  # Default authority score

    def _generate_research_insights(
            self,
            results: List[Dict],
            context: CampaignContext) -> List[str]:
        """Generate research insights based on findings and context."""
        insights = []

        if not results:
            return insights

        # Analyze trends
        if len(results) > 2:
            insights.append(
                f"Found {
                    len(results)} relevant sources on this topic")

        # Analyze confidence levels
        high_confidence = [
            r for r in results if r.get(
                'confidence_score', 0) > 0.8]
        if high_confidence:
            insights.append(
                f"{len(high_confidence)} high-confidence findings identified")

        # Analyze source diversity
        unique_sources = len(set(r.get('source', 'unknown') for r in results))
        if unique_sources > 1:
            insights.append(
                f"Research covers {unique_sources} different sources")

        return insights
