"""
Research Strategy Engine for Enhanced Newsletter Generation

This module provides intelligent research orchestration using multiple search providers
with intelligent query refinement and result synthesis.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core.campaign_context import CampaignContext
from src.tools.search_provider import (
    SearchQuery,
    SearchResult,
    get_unified_search_provider
)

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    """Represents a search strategy for a topic."""
    topic: str
    primary_queries: List[str]
    fallback_queries: List[str]
    provider_priorities: List[str]
    max_results_per_query: int = 5
    confidence_threshold: float = 0.7


@dataclass
class ResearchResults:
    """Container for research results with metadata."""
    topic: str
    results: List[SearchResult]
    confidence: float
    search_strategy: SearchStrategy
    execution_time: float
    sources_used: List[str]
    total_results: int
    synthesis_score: float


class IntelligentResearchOrchestrator:
    """
    Orchestrates real-time research using multiple search providers
    with intelligent query refinement and result synthesis.
    """

    def __init__(self):
        self.search_provider = get_unified_search_provider()
        self.query_refiner = QueryRefinementEngine()
        self.result_synthesizer = ResultSynthesizer()
        
    async def conduct_research(self, topic: str, context: CampaignContext) -> ResearchResults:
        """Execute comprehensive research with multiple strategies."""
        start_time = datetime.now()
        
        try:
            # 1. Generate search strategy based on topic and context
            search_strategy = await self._generate_search_strategy(topic, context)
            
            # 2. Execute parallel searches across providers
            raw_results = await self._execute_parallel_searches(search_strategy)
            
            # 3. Synthesize and rank results
            synthesized_results = await self._synthesize_results(raw_results, topic, context)
            
            # 4. Generate follow-up queries if needed
            if synthesized_results.confidence < search_strategy.confidence_threshold:
                logger.info(f"Confidence {synthesized_results.confidence:.2f} below threshold {search_strategy.confidence_threshold}, conducting follow-up research")
                followup_results = await self._conduct_followup_research(synthesized_results, context)
                synthesized_results = self._merge_results(synthesized_results, followup_results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ResearchResults(
                topic=topic,
                results=synthesized_results.results,
                confidence=synthesized_results.confidence,
                search_strategy=search_strategy,
                execution_time=execution_time,
                sources_used=list(set(r.source for r in synthesized_results.results)),
                total_results=len(synthesized_results.results),
                synthesis_score=synthesized_results.synthesis_score
            )
            
        except Exception as e:
            logger.error(f"Research orchestration failed for topic '{topic}': {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Return minimal results on failure
            return ResearchResults(
                topic=topic,
                results=[],
                confidence=0.0,
                search_strategy=SearchStrategy(topic, [], [], []),
                execution_time=execution_time,
                sources_used=[],
                total_results=0,
                synthesis_score=0.0
            )

    async def _generate_search_strategy(self, topic: str, context: CampaignContext) -> SearchStrategy:
        """Generate search strategy based on topic and context."""
        try:
            # Generate base queries
            primary_queries = self._generate_primary_queries(topic, context)
            
            # Generate fallback queries
            fallback_queries = self._generate_fallback_queries(topic, context)
            
            # Determine provider priorities based on context
            provider_priorities = self._determine_provider_priorities(context)
            
            return SearchStrategy(
                topic=topic,
                primary_queries=primary_queries,
                fallback_queries=fallback_queries,
                provider_priorities=provider_priorities,
                max_results_per_query=5,
                confidence_threshold=0.7
            )
            
        except Exception as e:
            logger.error(f"Failed to generate search strategy: {e}")
            # Return basic strategy on failure
            return SearchStrategy(
                topic=topic,
                primary_queries=[topic],
                fallback_queries=[f"{topic} overview", f"{topic} guide"],
                provider_priorities=["serper_search", "duckduckgo_search"],
                confidence_threshold=0.5
            )

    def _generate_primary_queries(self, topic: str, context: CampaignContext) -> List[str]:
        """Generate primary search queries based on topic and context."""
        queries = []
        
        # Base query
        base_query = topic
        queries.append(base_query)
        
        # Context-aware queries based on audience
        audience = context.audience_persona.get('demographics', '').lower()
        knowledge_level = context.audience_persona.get('knowledge_level', 'intermediate').lower()
        
        if 'technical' in audience or knowledge_level == 'advanced':
            queries.extend([
                f"{topic} technical analysis",
                f"{topic} expert insights",
                f"{topic} in-depth analysis"
            ])
        elif 'business' in audience:
            queries.extend([
                f"{topic} business impact",
                f"{topic} market analysis",
                f"{topic} industry trends"
            ])
        else:
            queries.extend([
                f"{topic} overview",
                f"{topic} guide",
                f"{topic} explained"
            ])
        
        # Add strategic goal-based queries
        for goal in context.strategic_goals[:2]:  # Take first 2 goals
            queries.append(f"{topic} {goal.lower()}")
        
        # Add interest-based queries
        interests = context.audience_persona.get('interests', '')
        if interests:
            interest_terms = [i.strip() for i in interests.split(',')][:2]
            for interest in interest_terms:
                queries.append(f"{topic} {interest}")
        
        return queries[:6]  # Limit to 6 primary queries

    def _generate_fallback_queries(self, topic: str, context: CampaignContext) -> List[str]:
        """Generate fallback queries for when primary queries fail."""
        fallback_queries = [
            f"{topic} basics",
            f"what is {topic}",
            f"{topic} tutorial",
            f"{topic} recent news",
            f"{topic} latest developments",
            f"{topic} trends 2024 2025"
        ]
        
        # Add simplified versions of complex topics
        words = topic.split()
        if len(words) > 1:
            # Create queries with individual words
            for word in words:
                if len(word) > 3:  # Skip short words
                    fallback_queries.append(word)
        
        return fallback_queries

    def _determine_provider_priorities(self, context: CampaignContext) -> List[str]:
        """Determine search provider priorities based on context."""
        # Default priority order
        priorities = ["serper_search", "duckduckgo_search", "knowledge_base_search"]
        
        # Adjust based on audience needs
        audience = context.audience_persona.get('demographics', '').lower()
        if 'academic' in audience or 'research' in audience:
            # Prioritize knowledge base for academic audiences
            priorities = ["knowledge_base_search", "serper_search", "duckduckgo_search"]
        
        return priorities

    async def _execute_parallel_searches(self, strategy: SearchStrategy) -> List[SearchResult]:
        """Execute parallel searches across multiple queries and providers."""
        all_results = []
        
        # Execute primary queries
        for query_text in strategy.primary_queries:
            try:
                query = SearchQuery(query_text, strategy.max_results_per_query)
                results = self.search_provider.search(query_text, strategy.max_results_per_query)
                all_results.extend(results)
                
            except Exception as e:
                logger.warning(f"Primary query failed: {query_text} - {e}")
        
        # If we have insufficient results, try fallback queries
        if len(all_results) < 5:
            for query_text in strategy.fallback_queries[:3]:  # Limit fallback queries
                try:
                    query = SearchQuery(query_text, strategy.max_results_per_query)
                    results = self.search_provider.search(query_text, strategy.max_results_per_query)
                    all_results.extend(results)
                    
                    if len(all_results) >= 10:  # Stop when we have enough results
                        break
                        
                except Exception as e:
                    logger.warning(f"Fallback query failed: {query_text} - {e}")
        
        return all_results

    async def _synthesize_results(self, results: List[SearchResult], topic: str, context: CampaignContext) -> 'SynthesizedResults':
        """Synthesize and rank results based on relevance and quality."""
        if not results:
            return SynthesizedResults([], 0.0, 0.0)
        
        try:
            synthesized = self.result_synthesizer.synthesize_results(results, topic, context)
            return synthesized
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            # Return unsynthesized results with low confidence
            return SynthesizedResults(results[:10], 0.3, 0.3)

    async def _conduct_followup_research(self, initial_results: 'SynthesizedResults', context: CampaignContext) -> 'SynthesizedResults':
        """Conduct follow-up research to improve confidence."""
        try:
            # Generate new queries based on initial results
            followup_queries = self._generate_followup_queries(initial_results)
            
            # Execute follow-up searches
            followup_results = []
            for query in followup_queries[:3]:  # Limit to 3 follow-up queries
                try:
                    results = self.search_provider.search(query, 3)
                    followup_results.extend(results)
                except Exception as e:
                    logger.warning(f"Follow-up query failed: {query} - {e}")
            
            # Synthesize follow-up results
            if followup_results:
                return self.result_synthesizer.synthesize_results(followup_results, "", context)
            else:
                return SynthesizedResults([], 0.0, 0.0)
                
        except Exception as e:
            logger.error(f"Follow-up research failed: {e}")
            return SynthesizedResults([], 0.0, 0.0)

    def _generate_followup_queries(self, initial_results: 'SynthesizedResults') -> List[str]:
        """Generate follow-up queries based on initial results."""
        queries = []
        
        # Extract key terms from initial results
        for result in initial_results.results[:3]:
            title_words = result.title.split()
            # Create queries with title keywords
            for word in title_words:
                if len(word) > 4 and word.lower() not in ['the', 'and', 'with', 'for']:
                    queries.append(f"{word} latest research")
                    queries.append(f"{word} expert analysis")
        
        return list(set(queries))[:5]  # Remove duplicates and limit

    def _merge_results(self, primary: ResearchResults, secondary: 'SynthesizedResults') -> ResearchResults:
        """Merge primary and secondary research results."""
        try:
            merged_results = primary.results + secondary.results
            
            # Remove duplicates based on URL
            unique_results = []
            seen_urls = set()
            
            for result in merged_results:
                if result.url not in seen_urls:
                    unique_results.append(result)
                    seen_urls.add(result.url)
            
            # Update confidence (weighted average)
            total_results = len(primary.results) + len(secondary.results)
            if total_results > 0:
                merged_confidence = (
                    (primary.confidence * len(primary.results) + 
                     secondary.confidence * len(secondary.results)) / total_results
                )
            else:
                merged_confidence = primary.confidence
            
            # Create merged research results
            return ResearchResults(
                topic=primary.topic,
                results=unique_results[:15],  # Limit to top 15 results
                confidence=merged_confidence,
                search_strategy=primary.search_strategy,
                execution_time=primary.execution_time,
                sources_used=list(set(primary.sources_used + list(set(r.source for r in secondary.results)))),
                total_results=len(unique_results),
                synthesis_score=(primary.synthesis_score + secondary.synthesis_score) / 2
            )
            
        except Exception as e:
            logger.error(f"Failed to merge results: {e}")
            return primary


@dataclass 
class SynthesizedResults:
    """Container for synthesized research results."""
    results: List[SearchResult]
    confidence: float
    synthesis_score: float


class QueryRefinementEngine:
    """Engine for refining and expanding search queries."""
    
    def __init__(self):
        self.expansion_templates = [
            "{query} latest trends",
            "{query} expert analysis", 
            "{query} case study",
            "{query} best practices",
            "{query} research findings",
            "{query} industry report"
        ]
    
    def refine_query(self, query: str, context: CampaignContext) -> List[str]:
        """Refine a query based on context."""
        refined_queries = [query]  # Always include original
        
        # Add context-based refinements
        audience = context.audience_persona.get('demographics', '').lower()
        
        if 'technical' in audience:
            refined_queries.extend([
                f"{query} technical implementation",
                f"{query} architecture",
                f"{query} performance analysis"
            ])
        elif 'business' in audience:
            refined_queries.extend([
                f"{query} ROI",
                f"{query} business case",
                f"{query} market impact"
            ])
        
        return refined_queries


class ResultSynthesizer:
    """Synthesizes and ranks search results."""
    
    def synthesize_results(self, results: List[SearchResult], topic: str, context: CampaignContext) -> SynthesizedResults:
        """Synthesize and rank results based on relevance and quality."""
        if not results:
            return SynthesizedResults([], 0.0, 0.0)
        
        try:
            # Score and rank results
            scored_results = []
            for result in results:
                score = self._calculate_result_score(result, topic, context)
                scored_results.append((result, score))
            
            # Sort by score (descending)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top results
            top_results = [result for result, score in scored_results[:10]]
            
            # Calculate overall confidence
            if scored_results:
                avg_score = sum(score for _, score in scored_results) / len(scored_results)
                confidence = min(avg_score, 1.0)
            else:
                confidence = 0.0
            
            # Calculate synthesis score (quality of synthesis)
            synthesis_score = self._calculate_synthesis_score(top_results, topic)
            
            return SynthesizedResults(top_results, confidence, synthesis_score)
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            return SynthesizedResults(results[:5], 0.3, 0.3)

    def _calculate_result_score(self, result: SearchResult, topic: str, context: CampaignContext) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        
        # Topic relevance (40% weight)
        topic_score = self._calculate_topic_relevance(result, topic)
        score += topic_score * 0.4
        
        # Source quality (30% weight)
        source_score = self._calculate_source_quality(result)
        score += source_score * 0.3
        
        # Context alignment (20% weight)
        context_score = self._calculate_context_alignment(result, context)
        score += context_score * 0.2
        
        # Freshness (10% weight)
        freshness_score = self._calculate_freshness_score(result)
        score += freshness_score * 0.1
        
        return min(score, 1.0)

    def _calculate_topic_relevance(self, result: SearchResult, topic: str) -> float:
        """Calculate how relevant a result is to the topic."""
        if not topic:
            return 0.5
        
        topic_words = set(topic.lower().split())
        title_words = set(result.title.lower().split())
        snippet_words = set(result.snippet.lower().split())
        
        # Calculate word overlap
        title_overlap = len(topic_words.intersection(title_words)) / max(len(topic_words), 1)
        snippet_overlap = len(topic_words.intersection(snippet_words)) / max(len(topic_words), 1)
        
        # Weight title overlap more heavily
        relevance = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        
        return min(relevance, 1.0)

    def _calculate_source_quality(self, result: SearchResult) -> float:
        """Calculate quality score for a source."""
        quality_score = 0.5  # Base score
        
        # High-quality domains get bonus points
        high_quality_domains = [
            'edu', 'gov', 'org',
            'ieee.org', 'acm.org', 'nature.com', 'science.org',
            'techcrunch.com', 'wired.com', 'arstechnica.com'
        ]
        
        url_lower = result.url.lower()
        for domain in high_quality_domains:
            if domain in url_lower:
                quality_score += 0.3
                break
        
        # Check for research indicators
        research_indicators = ['research', 'study', 'analysis', 'report']
        for indicator in research_indicators:
            if indicator in result.title.lower() or indicator in result.snippet.lower():
                quality_score += 0.1
                break
        
        return min(quality_score, 1.0)

    def _calculate_context_alignment(self, result: SearchResult, context: CampaignContext) -> float:
        """Calculate how well result aligns with context."""
        alignment_score = 0.5
        
        # Check audience alignment
        audience = context.audience_persona.get('demographics', '').lower()
        content_lower = (result.title + ' ' + result.snippet).lower()
        
        if 'technical' in audience:
            technical_terms = ['technical', 'implementation', 'architecture', 'engineering']
            if any(term in content_lower for term in technical_terms):
                alignment_score += 0.2
        elif 'business' in audience:
            business_terms = ['business', 'market', 'ROI', 'strategy', 'enterprise']
            if any(term in content_lower for term in business_terms):
                alignment_score += 0.2
        
        # Check strategic goal alignment
        for goal in context.strategic_goals:
            if goal.lower() in content_lower:
                alignment_score += 0.1
                break
        
        return min(alignment_score, 1.0)

    def _calculate_freshness_score(self, result: SearchResult) -> float:
        """Calculate freshness score (recent content gets higher score)."""
        # For now, return a default score since we don't have date parsing
        # In a full implementation, this would parse dates and calculate recency
        
        # Check for recent indicators in title/snippet
        recent_indicators = ['2024', '2025', 'latest', 'recent', 'new', 'update']
        content_lower = (result.title + ' ' + result.snippet).lower()
        
        for indicator in recent_indicators:
            if indicator in content_lower:
                return 0.8
        
        return 0.6  # Default freshness score

    def _calculate_synthesis_score(self, results: List[SearchResult], topic: str) -> float:
        """Calculate quality of the synthesis itself."""
        if not results:
            return 0.0
        
        # Diversity of sources
        unique_sources = len(set(r.source for r in results))
        source_diversity = min(unique_sources / 3, 1.0)  # Ideal: 3+ different sources
        
        # Content coverage (simplified)
        total_content_length = sum(len(r.snippet) for r in results)
        content_coverage = min(total_content_length / 1000, 1.0)  # Ideal: 1000+ chars
        
        # Overall synthesis quality
        synthesis_score = (source_diversity * 0.6) + (content_coverage * 0.4)
        
        return synthesis_score