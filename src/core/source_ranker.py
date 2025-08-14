"""
Source Authority Ranking System

Implements authority-based source ranking for search results and claim validation
as specified in PRD FR2.1. Provides domain authority scoring, source credibility
assessment, and metadata-based ranking algorithms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of sources for authority ranking."""
    ACADEMIC = "academic"
    NEWS = "news"
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    TECHNICAL = "technical"
    SOCIAL = "social"
    BLOG = "blog"
    WIKI = "wiki"
    UNKNOWN = "unknown"


class CredibilityLevel(Enum):
    """Credibility levels for sources."""
    VERIFIED = "verified"        # High credibility, verified sources
    AUTHORITATIVE = "authoritative"  # Known authoritative sources
    RELIABLE = "reliable"        # Generally reliable sources
    QUESTIONABLE = "questionable"  # Sources requiring additional verification
    UNRELIABLE = "unreliable"    # Known unreliable sources


@dataclass
class SourceMetadata:
    """Extended metadata for source analysis."""
    domain: str
    source_type: SourceType
    credibility_level: CredibilityLevel
    authority_score: float
    freshness_score: float
    expertise_score: float
    bias_score: float
    popularity_score: float
    verification_status: str
    last_updated: Optional[datetime] = None


@dataclass
class Source:
    """Represents a source with authority and metadata."""
    url: str
    title: str
    snippet: str
    domain: str
    publication_date: Optional[str] = None
    author: Optional[str] = None
    metadata: Optional[SourceMetadata] = None


class SourceAuthorityRanker:
    """Authority-based source ranking system."""
    
    # Domain authority scores (0.0 - 1.0)
    AUTHORITY_SCORES = {
        # Academic and Research
        "arxiv.org": 0.95,
        "ieee.org": 0.95,
        "acm.org": 0.90,
        "nature.com": 0.95,
        "science.org": 0.95,
        "pubmed.ncbi.nlm.nih.gov": 0.90,
        "scholar.google.com": 0.85,
        "researchgate.net": 0.80,
        
        # Educational Institutions
        "mit.edu": 0.90,
        "stanford.edu": 0.90,
        "harvard.edu": 0.90,
        "berkeley.edu": 0.85,
        "cmu.edu": 0.85,
        
        # Technical and Development
        "github.com": 0.85,
        "stackoverflow.com": 0.80,
        "medium.com": 0.60,
        "dev.to": 0.55,
        
        # News and Media
        "reuters.com": 0.90,
        "ap.org": 0.90,
        "bbc.com": 0.85,
        "nytimes.com": 0.85,
        "washingtonpost.com": 0.80,
        "wsj.com": 0.85,
        "economist.com": 0.85,
        
        # Technology News
        "techcrunch.com": 0.75,
        "wired.com": 0.80,
        "arstechnica.com": 0.80,
        "venturebeat.com": 0.70,
        "theverge.com": 0.75,
        
        # Government and Official
        "gov": 0.90,  # All .gov domains
        "who.int": 0.95,
        "un.org": 0.90,
        
        # Reference and Wiki
        "wikipedia.org": 0.70,
        "britannica.com": 0.75,
        
        # Corporate Documentation
        "docs.microsoft.com": 0.80,
        "developers.google.com": 0.80,
        "aws.amazon.com": 0.80,
        "developer.mozilla.org": 0.85,
        
        # Default scores by domain type
        ".edu": 0.80,
        ".gov": 0.90,
        ".org": 0.70,
    }
    
    def __init__(self):
        self.domain_patterns = self._compile_domain_patterns()
        self.bias_indicators = self._load_bias_indicators()
        
    def calculate_authority_score(self, source: Source) -> float:
        """Calculate source authority based on domain and metadata."""
        if not source or not source.domain:
            return 0.0
            
        domain = source.domain.lower()
        
        # Check exact domain matches first
        if domain in self.AUTHORITY_SCORES:
            base_score = self.AUTHORITY_SCORES[domain]
        else:
            # Check domain suffix patterns
            base_score = self._get_domain_suffix_score(domain)
        
        # Apply metadata adjustments
        adjusted_score = self._apply_metadata_adjustments(source, base_score)
        
        # Apply content quality adjustments
        content_adjustment = self._assess_content_quality(source)
        
        # Final score calculation
        final_score = min(1.0, adjusted_score + content_adjustment)
        
        logger.debug(f"Authority score for {domain}: base={base_score:.3f}, "
                    f"adjusted={adjusted_score:.3f}, final={final_score:.3f}")
        
        return final_score
    
    def classify_source_type(self, source: Source) -> SourceType:
        """Classify source type based on domain and content."""
        domain = source.domain.lower()
        
        # Academic sources
        if (domain.endswith('.edu') or 
            any(academic in domain for academic in ['arxiv', 'ieee', 'acm', 'nature', 'science', 'pubmed'])):
            return SourceType.ACADEMIC
        
        # Government sources
        if domain.endswith('.gov') or domain in ['who.int', 'un.org']:
            return SourceType.GOVERNMENT
        
        # News sources
        news_domains = ['reuters', 'ap.org', 'bbc', 'nytimes', 'washingtonpost', 
                       'wsj', 'cnn', 'npr', 'economist']
        if any(news in domain for news in news_domains):
            return SourceType.NEWS
        
        # Technical sources
        tech_domains = ['github', 'stackoverflow', 'docs.microsoft', 'developers.google',
                       'developer.mozilla', 'aws.amazon']
        if any(tech in domain for tech in tech_domains):
            return SourceType.TECHNICAL
        
        # Corporate sources
        corporate_indicators = ['corp', 'inc', 'llc', 'company', 'enterprise']
        if any(indicator in domain for indicator in corporate_indicators):
            return SourceType.CORPORATE
        
        # Blog/personal sites
        blog_indicators = ['blog', 'wordpress', 'blogspot', 'medium', 'substack']
        if any(blog in domain for blog in blog_indicators):
            return SourceType.BLOG
        
        # Wiki sources
        if 'wiki' in domain:
            return SourceType.WIKI
        
        # Social media
        social_domains = ['twitter', 'facebook', 'linkedin', 'reddit', 'youtube']
        if any(social in domain for social in social_domains):
            return SourceType.SOCIAL
        
        return SourceType.UNKNOWN
    
    def assess_credibility(self, source: Source) -> CredibilityLevel:
        """Assess source credibility level."""
        authority_score = self.calculate_authority_score(source)
        source_type = self.classify_source_type(source)
        
        # High credibility thresholds
        if authority_score >= 0.90:
            return CredibilityLevel.VERIFIED
        
        # Authoritative sources
        if authority_score >= 0.80 or source_type in [SourceType.ACADEMIC, SourceType.GOVERNMENT]:
            return CredibilityLevel.AUTHORITATIVE
        
        # Reliable sources
        if authority_score >= 0.65 or source_type in [SourceType.NEWS, SourceType.TECHNICAL]:
            return CredibilityLevel.RELIABLE
        
        # Questionable sources need additional verification
        if authority_score >= 0.45 or source_type == SourceType.CORPORATE:
            return CredibilityLevel.QUESTIONABLE
        
        # Low credibility or unknown sources
        return CredibilityLevel.UNRELIABLE
    
    def calculate_freshness_score(self, source: Source) -> float:
        """Calculate freshness score based on publication date."""
        if not source.publication_date:
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Try to parse the publication date
            pub_date = self._parse_date(source.publication_date)
            if not pub_date:
                return 0.5
            
            now = datetime.now()
            age_days = (now - pub_date).days
            
            # Freshness scoring
            if age_days <= 7:          # Within a week
                return 1.0
            elif age_days <= 30:       # Within a month
                return 0.9
            elif age_days <= 90:       # Within 3 months
                return 0.8
            elif age_days <= 180:      # Within 6 months
                return 0.6
            elif age_days <= 365:      # Within a year
                return 0.4
            elif age_days <= 730:      # Within 2 years
                return 0.2
            else:                      # Older than 2 years
                return 0.1
                
        except Exception as e:
            logger.warning(f"Failed to calculate freshness for {source.domain}: {e}")
            return 0.5
    
    def calculate_expertise_score(self, source: Source) -> float:
        """Calculate expertise score based on author and content indicators."""
        score = 0.5  # Base score
        
        # Author expertise indicators
        if source.author:
            author_lower = source.author.lower()
            expertise_indicators = [
                'dr.', 'prof.', 'professor', 'phd', 'md', 'researcher',
                'scientist', 'engineer', 'analyst', 'expert', 'director',
                'ceo', 'cto', 'founder'
            ]
            
            for indicator in expertise_indicators:
                if indicator in author_lower:
                    score += 0.1
                    
        # Content expertise indicators
        if source.title or source.snippet:
            content = f"{source.title} {source.snippet}".lower()
            technical_terms = [
                'algorithm', 'implementation', 'framework', 'methodology',
                'analysis', 'research', 'study', 'evaluation', 'benchmark',
                'optimization', 'architecture', 'design pattern'
            ]
            
            term_count = sum(1 for term in technical_terms if term in content)
            score += min(0.3, term_count * 0.05)
        
        return min(1.0, score)
    
    def calculate_bias_score(self, source: Source) -> float:
        """Calculate bias score (lower = less biased)."""
        if not source.domain:
            return 0.5
            
        domain = source.domain.lower()
        
        # Known unbiased sources
        unbiased_domains = [
            'reuters.com', 'ap.org', 'bbc.com', 'npr.org',
            'arxiv.org', 'ieee.org', 'nature.com', 'science.org'
        ]
        
        if domain in unbiased_domains:
            return 0.1  # Very low bias
        
        # Academic and government sources generally less biased
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return 0.2
        
        # Check for bias indicators in content
        content = f"{source.title} {source.snippet}".lower()
        bias_keywords = self.bias_indicators.get('high_bias', [])
        
        bias_count = sum(1 for keyword in bias_keywords if keyword in content)
        bias_penalty = min(0.4, bias_count * 0.1)
        
        return 0.5 + bias_penalty  # Higher score = more biased
    
    def rank_sources(self, sources: List[Source]) -> List[Tuple[Source, float]]:
        """Rank sources by comprehensive authority scoring."""
        ranked_sources = []
        
        for source in sources:
            # Calculate individual scores
            authority_score = self.calculate_authority_score(source)
            freshness_score = self.calculate_freshness_score(source)
            expertise_score = self.calculate_expertise_score(source)
            bias_score = self.calculate_bias_score(source)
            
            # Weighted composite score
            composite_score = (
                authority_score * 0.4 +
                freshness_score * 0.2 +
                expertise_score * 0.2 +
                (1.0 - bias_score) * 0.2  # Lower bias = higher score
            )
            
            # Create enhanced metadata
            metadata = SourceMetadata(
                domain=source.domain,
                source_type=self.classify_source_type(source),
                credibility_level=self.assess_credibility(source),
                authority_score=authority_score,
                freshness_score=freshness_score,
                expertise_score=expertise_score,
                bias_score=bias_score,
                popularity_score=0.5,  # Could be enhanced with actual popularity metrics
                verification_status="automatic",
                last_updated=datetime.now()
            )
            
            # Add metadata to source
            enhanced_source = Source(
                url=source.url,
                title=source.title,
                snippet=source.snippet,
                domain=source.domain,
                publication_date=source.publication_date,
                author=source.author,
                metadata=metadata
            )
            
            ranked_sources.append((enhanced_source, composite_score))
        
        # Sort by composite score (highest first)
        ranked_sources.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_sources
    
    def get_authority_report(self, source: Source) -> Dict[str, Any]:
        """Generate detailed authority report for a source."""
        authority_score = self.calculate_authority_score(source)
        freshness_score = self.calculate_freshness_score(source)
        expertise_score = self.calculate_expertise_score(source)
        bias_score = self.calculate_bias_score(source)
        source_type = self.classify_source_type(source)
        credibility_level = self.assess_credibility(source)
        
        return {
            "source": {
                "domain": source.domain,
                "url": source.url,
                "title": source.title[:100] + "..." if len(source.title) > 100 else source.title
            },
            "scores": {
                "authority": round(authority_score, 3),
                "freshness": round(freshness_score, 3),
                "expertise": round(expertise_score, 3),
                "bias": round(bias_score, 3),
                "composite": round((authority_score * 0.4 + freshness_score * 0.2 + 
                                 expertise_score * 0.2 + (1.0 - bias_score) * 0.2), 3)
            },
            "classifications": {
                "source_type": source_type.value,
                "credibility_level": credibility_level.value
            },
            "recommendations": self._generate_recommendations(
                authority_score, freshness_score, expertise_score, bias_score, 
                source_type, credibility_level
            )
        }
    
    def _get_domain_suffix_score(self, domain: str) -> float:
        """Get authority score based on domain suffix patterns."""
        for suffix, score in self.AUTHORITY_SCORES.items():
            if suffix.startswith('.') and domain.endswith(suffix):
                return score
        
        # Default scores based on TLD
        if domain.endswith('.edu'):
            return 0.80
        elif domain.endswith('.gov'):
            return 0.90
        elif domain.endswith('.org'):
            return 0.65
        elif domain.endswith('.com'):
            return 0.50
        else:
            return 0.40  # Unknown TLD
    
    def _apply_metadata_adjustments(self, source: Source, base_score: float) -> float:
        """Apply adjustments based on source metadata."""
        adjusted_score = base_score
        
        # Boost for academic authors
        if source.author:
            author_lower = source.author.lower()
            if any(title in author_lower for title in ['dr.', 'prof.', 'phd']):
                adjusted_score += 0.05
        
        # Boost for recent content
        if source.publication_date:
            freshness = self.calculate_freshness_score(source)
            if freshness > 0.8:
                adjusted_score += 0.02
        
        return min(1.0, adjusted_score)
    
    def _assess_content_quality(self, source: Source) -> float:
        """Assess content quality and return adjustment score."""
        quality_adjustment = 0.0
        
        if source.title:
            # Check title quality
            title_lower = source.title.lower()
            
            # Negative indicators
            clickbait_words = ['shocking', 'unbelievable', 'you won\'t believe', 
                             'amazing', 'incredible', 'jaw-dropping']
            if any(word in title_lower for word in clickbait_words):
                quality_adjustment -= 0.1
            
            # Positive indicators
            quality_words = ['analysis', 'research', 'study', 'report', 
                           'investigation', 'review', 'findings']
            if any(word in title_lower for word in quality_words):
                quality_adjustment += 0.05
        
        if source.snippet:
            # Check snippet quality
            snippet_lower = source.snippet.lower()
            
            # Check for citations or references
            if any(indicator in snippet_lower for indicator in 
                   ['according to', 'study found', 'research shows', 'data indicates']):
                quality_adjustment += 0.03
        
        return quality_adjustment
    
    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_string:
            return None
        
        # Common date patterns
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_string)
            if match:
                try:
                    if '-' in date_string and len(match.group(1)) == 4:
                        # YYYY-MM-DD format
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                    else:
                        # MM/DD/YYYY or MM-DD-YYYY format
                        month, day, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def _compile_domain_patterns(self) -> Dict[str, float]:
        """Compile regex patterns for domain matching."""
        # This could be enhanced with more sophisticated pattern matching
        return {}
    
    def _load_bias_indicators(self) -> Dict[str, List[str]]:
        """Load bias indicators for content analysis."""
        return {
            'high_bias': [
                'allegedly', 'claims', 'rumors', 'reportedly', 'sources say',
                'breaking:', 'shocking', 'exposed', 'truth about', 'conspiracy'
            ],
            'low_bias': [
                'according to', 'study found', 'research indicates', 'data shows',
                'analysis reveals', 'investigation found', 'report states'
            ]
        }
    
    def _generate_recommendations(self, authority_score: float, freshness_score: float,
                                expertise_score: float, bias_score: float,
                                source_type: SourceType, credibility_level: CredibilityLevel) -> List[str]:
        """Generate recommendations for source usage."""
        recommendations = []
        
        if credibility_level == CredibilityLevel.VERIFIED:
            recommendations.append("Excellent source - use with high confidence")
        elif credibility_level == CredibilityLevel.AUTHORITATIVE:
            recommendations.append("Authoritative source - suitable for primary citations")
        elif credibility_level == CredibilityLevel.RELIABLE:
            recommendations.append("Reliable source - good for supporting evidence")
        elif credibility_level == CredibilityLevel.QUESTIONABLE:
            recommendations.append("Verify information with additional sources")
        else:
            recommendations.append("Use with caution - seek alternative sources")
        
        if freshness_score < 0.3:
            recommendations.append("Content may be outdated - check for recent updates")
        
        if bias_score > 0.7:
            recommendations.append("High bias detected - cross-reference with neutral sources")
        
        if expertise_score < 0.4:
            recommendations.append("Limited expertise indicators - verify technical claims")
        
        return recommendations