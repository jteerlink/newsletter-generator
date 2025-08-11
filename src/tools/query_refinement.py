"""
Query Refinement Engine

This module provides intelligent query refinement and expansion capabilities
for enhanced research quality in newsletter generation.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from src.core.campaign_context import CampaignContext

logger = logging.getLogger(__name__)


class QueryRefinementEngine:
    """
    Engine for refining and expanding search queries based on context,
    audience, and domain-specific knowledge.
    """

    def __init__(self):
        """Initialize the query refinement engine."""
        self.domain_keywords = self._load_domain_keywords()
        self.stop_words = self._load_stop_words()
        self.synonyms = self._load_synonyms()
        self.technical_terms = self._load_technical_terms()

    def refine_query(self, query: str, context: CampaignContext) -> List[str]:
        """
        Refine a search query based on context and generate multiple query variations.

        Args:
            query: Original search query
            context: Campaign context with audience and strategic information

        Returns:
            List of refined and expanded queries
        """
        try:
            logger.info(f"Refining query: '{query}' with context")

            refined_queries = [query]  # Always include original query

            # Extract key information from context
            audience_info = self._extract_audience_info(context)
            strategic_info = self._extract_strategic_info(context)

            # Generate audience-specific variations
            audience_queries = self._generate_audience_variations(query, audience_info)
            refined_queries.extend(audience_queries)

            # Generate strategic variations
            strategic_queries = self._generate_strategic_variations(query, strategic_info)
            refined_queries.extend(strategic_queries)

            # Add domain-specific expansions
            domain_queries = self._expand_with_domain_knowledge(query, context)
            refined_queries.extend(domain_queries)

            # Add synonym variations
            synonym_queries = self._expand_with_synonyms(query)
            refined_queries.extend(synonym_queries)

            # Add technical depth variations
            technical_queries = self._add_technical_depth(query, audience_info)
            refined_queries.extend(technical_queries)

            # Remove duplicates while preserving order
            unique_queries = self._remove_duplicates(refined_queries)

            # Score and rank queries
            scored_queries = self._score_queries(unique_queries, query, context)

            logger.info(f"Generated {len(scored_queries)} refined queries")
            return scored_queries

        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return [query]  # Return original query on failure

    def _extract_audience_info(self, context: CampaignContext) -> Dict[str, str]:
        """Extract relevant audience information from context."""
        audience_persona = context.audience_persona
        
        return {
            'demographics': audience_persona.get('demographics', ''),
            'knowledge_level': audience_persona.get('knowledge_level', 'intermediate'),
            'interests': audience_persona.get('interests', ''),
            'pain_points': audience_persona.get('pain_points', ''),
            'preferred_content': audience_persona.get('preferred_content', '')
        }

    def _extract_strategic_info(self, context: CampaignContext) -> Dict[str, List[str]]:
        """Extract strategic information from context."""
        return {
            'goals': context.strategic_goals,
            'campaign_type': [context.campaign_context.get('type', 'general')],
            'target_outcomes': context.campaign_context.get('target_outcomes', [])
        }

    def _generate_audience_variations(self, query: str, audience_info: Dict[str, str]) -> List[str]:
        """Generate query variations based on audience characteristics."""
        variations = []
        
        knowledge_level = audience_info.get('knowledge_level', 'intermediate').lower()
        demographics = audience_info.get('demographics', '').lower()
        
        # Knowledge level variations
        if knowledge_level == 'beginner':
            variations.extend([
                f"{query} for beginners",
                f"{query} explained simply",
                f"{query} basic guide",
                f"introduction to {query}",
                f"{query} fundamentals"
            ])
        elif knowledge_level == 'advanced':
            variations.extend([
                f"{query} advanced",
                f"{query} expert analysis",
                f"{query} technical details",
                f"{query} deep dive",
                f"{query} professional insights"
            ])
        else:  # intermediate
            variations.extend([
                f"{query} overview",
                f"{query} comprehensive guide",
                f"{query} best practices"
            ])

        # Demographics-based variations
        if 'business' in demographics or 'executive' in demographics:
            variations.extend([
                f"{query} business impact",
                f"{query} ROI",
                f"{query} enterprise",
                f"{query} market analysis",
                f"{query} strategic implications"
            ])
        elif 'developer' in demographics or 'technical' in demographics:
            variations.extend([
                f"{query} implementation",
                f"{query} technical architecture",
                f"{query} developer guide",
                f"{query} API",
                f"{query} code examples"
            ])
        elif 'researcher' in demographics or 'academic' in demographics:
            variations.extend([
                f"{query} research",
                f"{query} studies",
                f"{query} academic papers",
                f"{query} peer reviewed",
                f"{query} methodology"
            ])

        return variations

    def _generate_strategic_variations(self, query: str, strategic_info: Dict[str, List[str]]) -> List[str]:
        """Generate query variations based on strategic goals."""
        variations = []
        
        goals = strategic_info.get('goals', [])
        
        for goal in goals[:3]:  # Limit to first 3 goals
            goal_lower = goal.lower()
            
            if 'engagement' in goal_lower:
                variations.extend([
                    f"{query} interactive",
                    f"{query} community",
                    f"{query} discussion",
                    f"{query} user experience"
                ])
            elif 'education' in goal_lower:
                variations.extend([
                    f"{query} learning",
                    f"{query} tutorial",
                    f"{query} how to",
                    f"{query} step by step"
                ])
            elif 'awareness' in goal_lower:
                variations.extend([
                    f"{query} trends",
                    f"{query} news",
                    f"{query} updates",
                    f"{query} latest developments"
                ])
            elif 'conversion' in goal_lower:
                variations.extend([
                    f"{query} benefits",
                    f"{query} advantages",
                    f"{query} comparison",
                    f"{query} success stories"
                ])

        return variations

    def _expand_with_domain_knowledge(self, query: str, context: CampaignContext) -> List[str]:
        """Expand query with domain-specific knowledge."""
        variations = []
        
        # Identify domain from query and context
        domain = self._identify_domain(query, context)
        
        if domain and domain in self.domain_keywords:
            domain_terms = self.domain_keywords[domain]
            for term in domain_terms[:3]:  # Limit to 3 terms
                variations.append(f"{query} {term}")
                variations.append(f"{term} {query}")

        return variations

    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        variations = []
        
        query_words = query.lower().split()
        
        for word in query_words:
            if word in self.synonyms:
                for synonym in self.synonyms[word][:2]:  # Limit to 2 synonyms per word
                    # Replace the word with synonym
                    synonym_query = query.lower().replace(word, synonym)
                    if synonym_query != query.lower():
                        variations.append(synonym_query)

        return variations

    def _add_technical_depth(self, query: str, audience_info: Dict[str, str]) -> List[str]:
        """Add technical depth based on audience knowledge level."""
        variations = []
        
        knowledge_level = audience_info.get('knowledge_level', 'intermediate').lower()
        
        if knowledge_level == 'advanced':
            # Add technical modifiers
            technical_modifiers = [
                'architecture', 'implementation', 'performance', 
                'scalability', 'security', 'optimization'
            ]
            
            for modifier in technical_modifiers[:3]:
                variations.append(f"{query} {modifier}")

        return variations

    def _identify_domain(self, query: str, context: CampaignContext) -> Optional[str]:
        """Identify the domain/field of the query."""
        query_lower = query.lower()
        
        # Check against known domains
        domain_patterns = {
            'technology': ['tech', 'software', 'ai', 'machine learning', 'programming', 'development'],
            'business': ['business', 'market', 'finance', 'strategy', 'corporate', 'enterprise'],
            'science': ['research', 'study', 'scientific', 'experiment', 'data', 'analysis'],
            'health': ['health', 'medical', 'wellness', 'healthcare', 'medicine', 'treatment'],
            'education': ['learning', 'education', 'teaching', 'training', 'course', 'skill']
        }
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return domain
        
        # Check context for domain hints
        audience_demographics = context.audience_persona.get('demographics', '').lower()
        for domain, patterns in domain_patterns.items():
            if any(pattern in audience_demographics for pattern in patterns):
                return domain
        
        return None

    def _score_queries(self, queries: List[str], original_query: str, context: CampaignContext) -> List[str]:
        """Score and rank queries based on relevance and utility."""
        scored_queries = []
        
        for query in queries:
            score = self._calculate_query_score(query, original_query, context)
            scored_queries.append((query, score))
        
        # Sort by score (descending) and return queries
        scored_queries.sort(key=lambda x: x[1], reverse=True)
        
        return [query for query, score in scored_queries[:10]]  # Return top 10

    def _calculate_query_score(self, query: str, original_query: str, context: CampaignContext) -> float:
        """Calculate relevance score for a query."""
        score = 0.5  # Base score
        
        # Boost score if query is the original
        if query == original_query:
            score += 0.2
        
        # Boost score for audience-relevant terms
        audience_terms = self._get_audience_terms(context)
        query_lower = query.lower()
        
        for term in audience_terms:
            if term.lower() in query_lower:
                score += 0.1
        
        # Boost score for strategic goal alignment
        strategic_terms = []
        for goal in context.strategic_goals:
            strategic_terms.extend(goal.lower().split())
        
        for term in strategic_terms:
            if term in query_lower:
                score += 0.1
        
        # Penalize overly long queries
        word_count = len(query.split())
        if word_count > 8:
            score -= 0.1
        elif word_count > 12:
            score -= 0.2
        
        # Boost score for technical depth if audience is advanced
        knowledge_level = context.audience_persona.get('knowledge_level', '').lower()
        if knowledge_level == 'advanced':
            technical_indicators = ['implementation', 'architecture', 'advanced', 'technical']
            if any(indicator in query_lower for indicator in technical_indicators):
                score += 0.15
        
        return min(score, 1.0)

    def _get_audience_terms(self, context: CampaignContext) -> List[str]:
        """Extract key terms from audience information."""
        terms = []
        
        audience_persona = context.audience_persona
        for key in ['demographics', 'interests', 'pain_points']:
            value = audience_persona.get(key, '')
            if value:
                terms.extend(value.lower().split())
        
        return terms

    def _remove_duplicates(self, queries: List[str]) -> List[str]:
        """Remove duplicate queries while preserving order."""
        seen = set()
        unique_queries = []
        
        for query in queries:
            query_normalized = ' '.join(query.lower().split())
            if query_normalized not in seen:
                seen.add(query_normalized)
                unique_queries.append(query)
        
        return unique_queries

    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for query expansion."""
        return {
            'technology': [
                'innovation', 'digital transformation', 'automation', 'cloud computing',
                'cybersecurity', 'data analytics', 'mobile technology', 'IoT'
            ],
            'business': [
                'strategy', 'growth', 'market trends', 'competitive analysis',
                'customer experience', 'digital marketing', 'operations', 'leadership'
            ],
            'science': [
                'methodology', 'peer review', 'hypothesis', 'experimental design',
                'statistical analysis', 'research findings', 'literature review'
            ],
            'health': [
                'clinical trials', 'treatment options', 'prevention', 'diagnosis',
                'healthcare delivery', 'patient outcomes', 'medical research'
            ],
            'education': [
                'pedagogy', 'curriculum', 'assessment', 'learning outcomes',
                'educational technology', 'professional development', 'skills training'
            ]
        }

    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to filter out."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        }

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonyms for common terms."""
        return {
            'news': ['updates', 'developments', 'reports', 'announcements'],
            'technology': ['tech', 'innovation', 'digital', 'computing'],
            'business': ['enterprise', 'corporate', 'commercial', 'industry'],
            'analysis': ['examination', 'study', 'review', 'assessment'],
            'development': ['progress', 'advancement', 'evolution', 'growth'],
            'implementation': ['deployment', 'execution', 'application', 'realization'],
            'strategy': ['approach', 'plan', 'method', 'framework'],
            'research': ['study', 'investigation', 'analysis', 'exploration'],
            'trend': ['pattern', 'movement', 'direction', 'tendency'],
            'impact': ['effect', 'influence', 'consequence', 'result']
        }

    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """Load technical terms for domain-specific expansion."""
        return {
            'programming': ['algorithm', 'framework', 'API', 'database', 'architecture'],
            'ai': ['machine learning', 'neural network', 'deep learning', 'NLP', 'computer vision'],
            'security': ['encryption', 'authentication', 'authorization', 'vulnerability', 'threat'],
            'data': ['analytics', 'visualization', 'mining', 'processing', 'modeling'],
            'cloud': ['scalability', 'infrastructure', 'deployment', 'microservices', 'containerization']
        }

    def generate_followup_queries(self, initial_results: List[Dict], original_query: str) -> List[str]:
        """
        Generate follow-up queries based on initial search results.

        Args:
            initial_results: List of initial search results
            original_query: Original search query

        Returns:
            List of follow-up queries
        """
        try:
            if not initial_results:
                return []

            followup_queries = []

            # Extract key terms from top results
            key_terms = self._extract_key_terms_from_results(initial_results)
            
            # Generate queries with extracted terms
            for term in key_terms[:5]:
                followup_queries.extend([
                    f"{original_query} {term}",
                    f"{term} {original_query}",
                    f"{original_query} {term} latest",
                    f"{original_query} {term} analysis"
                ])

            # Add depth queries if results seem shallow
            if len(initial_results) < 3:
                followup_queries.extend([
                    f"{original_query} comprehensive",
                    f"{original_query} detailed analysis",
                    f"{original_query} complete guide",
                    f"{original_query} expert review"
                ])

            # Add recency queries
            followup_queries.extend([
                f"{original_query} 2024 2025",
                f"{original_query} recent developments",
                f"{original_query} latest news"
            ])

            return self._remove_duplicates(followup_queries)[:8]  # Return top 8 follow-up queries

        except Exception as e:
            logger.error(f"Follow-up query generation failed: {e}")
            return []

    def _extract_key_terms_from_results(self, results: List[Dict]) -> List[str]:
        """Extract key terms from search results for follow-up queries."""
        term_frequency = {}
        
        for result in results[:5]:  # Analyze top 5 results
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            # Combine title and snippet
            content = f"{title} {snippet}".lower()
            
            # Extract meaningful terms (skip stop words)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content)  # Words with 4+ characters
            
            for word in words:
                if word not in self.stop_words:
                    term_frequency[word] = term_frequency.get(word, 0) + 1

        # Sort by frequency and return top terms
        sorted_terms = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)
        
        return [term for term, freq in sorted_terms if freq >= 2][:10]  # Terms appearing at least twice


# Global instance
_query_refinement_engine = None


def get_query_refinement_engine() -> QueryRefinementEngine:
    """Get the global query refinement engine instance."""
    global _query_refinement_engine
    if _query_refinement_engine is None:
        _query_refinement_engine = QueryRefinementEngine()
    return _query_refinement_engine


def refine_search_query(query: str, context: CampaignContext) -> List[str]:
    """
    Convenience function to refine a search query.

    Args:
        query: Original search query
        context: Campaign context

    Returns:
        List of refined queries
    """
    engine = get_query_refinement_engine()
    return engine.refine_query(query, context)