"""
Claim Validation Engine

Implements automatic claim extraction, validation, and citation generation
as specified in PRD FR2.1. Provides NLP-based claim identification and
multi-provider search validation for factual accuracy.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.core.tool_usage_tracker import get_tool_tracker
from src.tools.tools import search_web

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that can be extracted from content."""
    STATISTICAL = "statistical"
    FACTUAL = "factual"
    OPINION = "opinion"
    RESEARCH = "research"
    TEMPORAL = "temporal"


@dataclass
class Claim:
    """Represents an extracted claim with context and metadata."""
    text: str
    claim_type: ClaimType
    confidence: float
    context: str
    source_sentence: str
    position: int


@dataclass
class Source:
    """Represents a source for claim validation."""
    url: str
    title: str
    snippet: str
    domain: str
    publication_date: Optional[str] = None
    author: Optional[str] = None


@dataclass
class RankedSource(Source):
    """Source with authority and relevance scoring."""
    authority_score: float = 0.0
    relevance_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class ValidationResult:
    """Result of claim validation against sources."""
    claim: Claim
    sources: List[RankedSource]
    validation_status: str  # "supported", "contradicted", "inconclusive"
    confidence: float
    supporting_snippets: List[str]
    contradicting_snippets: List[str]


class ClaimExtractor:
    """NLP-based claim identification from content."""
    
    def __init__(self):
        self.claim_patterns = {
            ClaimType.STATISTICAL: [
                r'\b\d+(?:\.\d+)?%',  # Percentages
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|percent)',
                r'\bincreased?\s+by\s+\d+',
                r'\bdecreased?\s+by\s+\d+',
                r'\bgrew\s+\d+',
                r'\brose\s+\d+',
                r'\bfell\s+\d+',
                r'\bmarket\s+share.*\d+',
                r'\brevenue.*\$\d+',
            ],
            ClaimType.RESEARCH: [
                r'\baccording to\s+(?:a\s+)?(?:study|research|report|survey)',
                r'\bresearch\s+(?:shows|indicates|suggests|found)',
                r'\bstudy\s+(?:found|shows|indicates|suggests|reveals)',
                r'\bdata\s+(?:shows|indicates|suggests|reveals)',
                r'\bsurvey\s+(?:found|shows|indicates)',
                r'\banalysis\s+(?:shows|indicates|reveals)',
                r'\binvestigation\s+(?:found|revealed)',
                r'\bpublished\s+in\s+(?:the\s+)?journal',
            ],
            ClaimType.FACTUAL: [
                r'\bfirst\s+(?:company|time|ever)',
                r'\bonly\s+(?:company|solution|platform)',
                r'\bleading\s+(?:provider|platform|solution)',
                r'\blargest\s+(?:company|platform|provider)',
                r'\bfounded\s+in\s+\d{4}',
                r'\bheadquartered\s+in',
                r'\bavailable\s+in\s+\d+\s+countries',
                r'\bused\s+by\s+(?:over\s+)?\d+',
            ],
            ClaimType.TEMPORAL: [
                r'\bin\s+\d{4}',
                r'\bsince\s+\d{4}',
                r'\blast\s+(?:year|month|quarter)',
                r'\bthis\s+(?:year|month|quarter)',
                r'\brecently',
                r'\bcurrently',
                r'\btoday',
                r'\bnow',
                r'\bby\s+\d{4}',
                r'\bover\s+the\s+(?:past|last)',
            ]
        }
        
        # Authority indicators for claims
        self.authority_indicators = [
            r'\baccording to\s+(?:the\s+)?(?:CEO|CTO|founder)',
            r'\baccording to\s+(?:the\s+)?(?:company|organization)',
            r'\bofficial\s+(?:statement|announcement|press release)',
            r'\bpress\s+release',
            r'\bwhite\s+paper',
            r'\btechnical\s+documentation',
        ]

    def extract_claims(self, content: str) -> List[Claim]:
        """Enhanced claim extraction with context and confidence."""
        claims = []
        sentences = self._split_into_sentences(content)
        
        for i, sentence in enumerate(sentences):
            sentence_claims = self._extract_from_sentence(sentence, i, content)
            claims.extend(sentence_claims)
        
        # Remove duplicates and rank by confidence
        unique_claims = self._deduplicate_claims(claims)
        return sorted(unique_claims, key=lambda x: x.confidence, reverse=True)
    
    def classify_claim_type(self, claim: str) -> Tuple[ClaimType, float]:
        """Classify claims as statistical, factual, research, or opinion."""
        best_type = ClaimType.OPINION
        best_confidence = 0.3  # Default low confidence for opinion
        
        for claim_type, patterns in self.claim_patterns.items():
            type_confidence = self._calculate_type_confidence(claim, patterns)
            if type_confidence > best_confidence:
                best_type = claim_type
                best_confidence = type_confidence
        
        return best_type, best_confidence
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences for analysis."""
        # Simple sentence splitting - can be enhanced with nltk
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _extract_from_sentence(self, sentence: str, position: int, full_content: str) -> List[Claim]:
        """Extract claims from a single sentence."""
        claims = []
        
        # Check each claim type
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    claim_text = self._extract_claim_context(sentence, match)
                    if len(claim_text) > 20:  # Minimum meaningful claim length
                        confidence = self._calculate_claim_confidence(
                            claim_text, sentence, claim_type)
                        
                        claim = Claim(
                            text=claim_text,
                            claim_type=claim_type,
                            confidence=confidence,
                            context=self._get_surrounding_context(
                                full_content, position),
                            source_sentence=sentence,
                            position=position
                        )
                        claims.append(claim)
        
        return claims
    
    def _extract_claim_context(self, sentence: str, match: re.Match) -> str:
        """Extract meaningful claim context around a pattern match."""
        # Take the full sentence as the claim for now
        # Can be enhanced to extract more precise claim boundaries
        return sentence.strip()
    
    def _get_surrounding_context(self, content: str, position: int) -> str:
        """Get surrounding context for a claim."""
        sentences = self._split_into_sentences(content)
        start = max(0, position - 1)
        end = min(len(sentences), position + 2)
        context_sentences = sentences[start:end]
        return " ".join(context_sentences)
    
    def _calculate_claim_confidence(self, claim_text: str, sentence: str, 
                                   claim_type: ClaimType) -> float:
        """Calculate confidence score for a claim."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for authority indicators
        for indicator in self.authority_indicators:
            if re.search(indicator, sentence, re.IGNORECASE):
                confidence += 0.2
                break
        
        # Boost confidence for specific numbers/data
        if claim_type == ClaimType.STATISTICAL:
            if re.search(r'\b\d+(?:\.\d+)?%', claim_text):
                confidence += 0.15
            if re.search(r'\$\d+', claim_text):
                confidence += 0.1
        
        # Boost confidence for research indicators
        if claim_type == ClaimType.RESEARCH:
            research_terms = ['study', 'research', 'analysis', 'survey']
            for term in research_terms:
                if term.lower() in claim_text.lower():
                    confidence += 0.1
        
        # Reduce confidence for opinion words
        opinion_words = ['believe', 'think', 'feel', 'opinion', 'might', 'could', 'perhaps']
        for word in opinion_words:
            if word.lower() in claim_text.lower():
                confidence -= 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_type_confidence(self, claim: str, patterns: List[str]) -> float:
        """Calculate confidence for claim type classification."""
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                matches += 1
        
        if matches == 0:
            return 0.0
        
        # Confidence based on pattern matches and specificity
        base_confidence = matches / total_patterns
        
        # Boost for multiple matches
        if matches > 1:
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate claims based on text similarity."""
        unique_claims = []
        
        for claim in claims:
            is_duplicate = False
            for existing in unique_claims:
                if self._are_similar_claims(claim.text, existing.text):
                    is_duplicate = True
                    # Keep the higher confidence claim
                    if claim.confidence > existing.confidence:
                        unique_claims.remove(existing)
                        unique_claims.append(claim)
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
        
        return unique_claims
    
    def _are_similar_claims(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are similar enough to be considered duplicates."""
        # Simple similarity check - can be enhanced with advanced NLP
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union
        return similarity > 0.7  # 70% word overlap threshold


class SourceValidator:
    """Multi-provider search validation for claims."""
    
    def __init__(self):
        self.tool_tracker = get_tool_tracker()
        
        # Authority scores for different domains
        self.domain_authority = {
            "arxiv.org": 0.95,
            "ieee.org": 0.95,
            "acm.org": 0.90,
            "nature.com": 0.95,
            "science.org": 0.95,
            "github.com": 0.85,
            "stackoverflow.com": 0.80,
            "medium.com": 0.60,
            "techcrunch.com": 0.75,
            "venturebeat.com": 0.70,
            "wired.com": 0.80,
            "mit.edu": 0.90,
            "stanford.edu": 0.90,
            "wikipedia.org": 0.70,
        }
    
    def validate_claim(self, claim: Claim, 
                      workflow_id: Optional[str] = None,
                      session_id: Optional[str] = None) -> ValidationResult:
        """Search authoritative sources for claim verification."""
        try:
            # Construct search query from claim
            search_query = self._construct_search_query(claim)
            
            # Perform web search with tracking
            with self.tool_tracker.track_tool_usage(
                tool_name="web_search_validation",
                agent_name="ClaimValidator",
                workflow_id=workflow_id,
                session_id=session_id,
                input_data={"query": search_query, "claim_type": claim.claim_type.value},
                context={"validation": "claim_verification"}
            ):
                search_results = search_web(search_query, max_results=10)
            
            # Parse and rank sources
            sources = self._parse_search_results(search_results, claim)
            ranked_sources = self.rank_sources(sources)
            
            # Analyze validation status
            validation_status, confidence, supporting, contradicting = \
                self._analyze_validation_results(claim, ranked_sources)
            
            return ValidationResult(
                claim=claim,
                sources=ranked_sources,
                validation_status=validation_status,
                confidence=confidence,
                supporting_snippets=supporting,
                contradicting_snippets=contradicting
            )
            
        except Exception as e:
            logger.warning(f"Claim validation failed: {e}")
            return ValidationResult(
                claim=claim,
                sources=[],
                validation_status="inconclusive",
                confidence=0.0,
                supporting_snippets=[],
                contradicting_snippets=[]
            )
    
    def rank_sources(self, sources: List[Source]) -> List[RankedSource]:
        """Rank sources by authority and relevance."""
        ranked_sources = []
        
        for source in sources:
            authority_score = self._calculate_authority_score(source)
            relevance_score = self._calculate_relevance_score(source)
            combined_score = (authority_score * 0.6) + (relevance_score * 0.4)
            
            ranked_source = RankedSource(
                url=source.url,
                title=source.title,
                snippet=source.snippet,
                domain=source.domain,
                publication_date=source.publication_date,
                author=source.author,
                authority_score=authority_score,
                relevance_score=relevance_score,
                combined_score=combined_score
            )
            ranked_sources.append(ranked_source)
        
        # Sort by combined score
        return sorted(ranked_sources, key=lambda x: x.combined_score, reverse=True)
    
    def _construct_search_query(self, claim: Claim) -> str:
        """Construct an effective search query from a claim."""
        # Extract key terms from claim
        claim_text = claim.text.lower()
        
        # Remove common words and keep important terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = [w for w in claim_text.split() if w not in stop_words and len(w) > 2]
        
        # For statistical claims, preserve numbers
        if claim.claim_type == ClaimType.STATISTICAL:
            numbers = re.findall(r'\d+(?:\.\d+)?%?', claim_text)
            words.extend(numbers)
        
        # For research claims, add verification terms
        if claim.claim_type == ClaimType.RESEARCH:
            words.extend(['study', 'research', 'report'])
        
        # Take top 6 most relevant words
        query_terms = words[:6]
        return ' '.join(query_terms)
    
    def _parse_search_results(self, search_results: str, claim: Claim) -> List[Source]:
        """Parse search results into Source objects."""
        sources = []
        
        if not search_results:
            return sources
        
        # Simple parsing - assumes search_results contains URLs and snippets
        # This would need to be enhanced based on actual search tool format
        lines = search_results.split('\n')
        current_source = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('http'):
                if current_source:
                    source = self._create_source_from_dict(current_source)
                    if source:
                        sources.append(source)
                    current_source = {}
                current_source['url'] = line
            elif line and 'url' in current_source:
                if 'title' not in current_source:
                    current_source['title'] = line
                else:
                    current_source['snippet'] = line
        
        # Add the last source
        if current_source:
            source = self._create_source_from_dict(current_source)
            if source:
                sources.append(source)
        
        return sources
    
    def _create_source_from_dict(self, source_dict: Dict[str, str]) -> Optional[Source]:
        """Create Source object from dictionary."""
        try:
            url = source_dict.get('url', '')
            if not url:
                return None
            
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            return Source(
                url=url,
                title=source_dict.get('title', 'Unknown Title'),
                snippet=source_dict.get('snippet', ''),
                domain=domain,
                publication_date=source_dict.get('date'),
                author=source_dict.get('author')
            )
        except Exception as e:
            logger.warning(f"Failed to create source: {e}")
            return None
    
    def _calculate_authority_score(self, source: Source) -> float:
        """Calculate authority score for a source."""
        # Base score from domain authority
        base_score = self.domain_authority.get(source.domain, 0.5)
        
        # Adjust based on domain characteristics
        if '.edu' in source.domain:
            base_score = max(base_score, 0.85)
        elif '.gov' in source.domain:
            base_score = max(base_score, 0.90)
        elif '.org' in source.domain:
            base_score = max(base_score, 0.75)
        
        return min(1.0, base_score)
    
    def _calculate_relevance_score(self, source: Source) -> float:
        """Calculate relevance score for a source."""
        relevance = 0.5  # Base relevance
        
        # Check title relevance (simplified)
        title_words = source.title.lower().split()
        snippet_words = source.snippet.lower().split()
        
        # Boost for relevant keywords in title
        relevant_keywords = ['ai', 'machine learning', 'technology', 'research', 'study', 'data']
        title_relevance = sum(1 for word in title_words if word in relevant_keywords)
        relevance += min(0.3, title_relevance * 0.1)
        
        # Boost for detailed snippets
        if len(snippet_words) > 20:
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _analyze_validation_results(self, claim: Claim, sources: List[RankedSource]) -> Tuple[str, float, List[str], List[str]]:
        """Analyze validation results to determine claim status."""
        if not sources:
            return "inconclusive", 0.0, [], []
        
        supporting_snippets = []
        contradicting_snippets = []
        
        # Simple keyword-based analysis
        claim_keywords = set(claim.text.lower().split())
        
        for source in sources[:5]:  # Analyze top 5 sources
            snippet_words = set(source.snippet.lower().split())
            overlap = len(claim_keywords.intersection(snippet_words))
            
            if overlap > 2:  # Threshold for relevance
                supporting_snippets.append(source.snippet)
        
        # Determine validation status
        if len(supporting_snippets) >= 2:
            status = "supported"
            confidence = min(0.9, len(supporting_snippets) * 0.3)
        elif len(supporting_snippets) == 1:
            status = "inconclusive"
            confidence = 0.5
        else:
            status = "inconclusive"
            confidence = 0.2
        
        return status, confidence, supporting_snippets, contradicting_snippets


class CitationGenerator:
    """Automatic source attribution and citation formatting."""
    
    def __init__(self):
        self.citation_formats = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
            "ieee": self._format_ieee
        }
    
    def generate_citations(self, validated_claims: List[ValidationResult],
                          format_style: str = "apa") -> str:
        """Create properly formatted citations."""
        if not validated_claims:
            return ""
        
        formatter = self.citation_formats.get(format_style, self._format_apa)
        citations = []
        
        for i, result in enumerate(validated_claims, 1):
            if result.sources and result.validation_status == "supported":
                # Use the highest-ranked source for citation
                top_source = result.sources[0]
                citation = formatter(top_source, i)
                citations.append(citation)
        
        if citations:
            return "\n\n## Sources\n\n" + "\n\n".join(citations)
        return ""
    
    def generate_inline_citations(self, content: str, 
                                 validated_claims: List[ValidationResult]) -> str:
        """Add inline citations to content."""
        modified_content = content
        citation_counter = 1
        
        for result in validated_claims:
            if result.sources and result.validation_status == "supported":
                claim_text = result.claim.text
                if claim_text in modified_content:
                    citation_mark = f"[{citation_counter}]"
                    modified_content = modified_content.replace(
                        claim_text, 
                        f"{claim_text} {citation_mark}", 
                        1
                    )
                    citation_counter += 1
        
        return modified_content
    
    def _format_apa(self, source: RankedSource, number: int) -> str:
        """Format citation in APA style."""
        author = source.author or "Unknown Author"
        year = self._extract_year(source.publication_date) or "n.d."
        title = source.title
        url = source.url
        
        return f"{number}. {author} ({year}). {title}. Retrieved from {url}"
    
    def _format_mla(self, source: RankedSource, number: int) -> str:
        """Format citation in MLA style."""
        author = source.author or "Unknown Author"
        title = source.title
        domain = source.domain
        date = source.publication_date or "n.d."
        url = source.url
        
        return f'{number}. {author}. "{title}." {domain}, {date}, {url}.'
    
    def _format_chicago(self, source: RankedSource, number: int) -> str:
        """Format citation in Chicago style."""
        author = source.author or "Unknown Author"
        title = source.title
        domain = source.domain
        date = source.publication_date or "n.d."
        url = source.url
        
        return f'{number}. {author}. "{title}." {domain}. {date}. {url}.'
    
    def _format_ieee(self, source: RankedSource, number: int) -> str:
        """Format citation in IEEE style."""
        author = source.author or "Unknown Author"
        title = source.title
        domain = source.domain
        date = source.publication_date or "n.d."
        url = source.url
        
        return f"[{number}] {author}, \"{title},\" {domain}, {date}. [Online]. Available: {url}"
    
    def _extract_year(self, date_string: Optional[str]) -> Optional[str]:
        """Extract year from date string."""
        if not date_string:
            return None
        
        year_match = re.search(r'\b(20\d{2})\b', date_string)
        return year_match.group(1) if year_match else None