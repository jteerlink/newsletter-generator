"""
Citation Manager for Newsletter Generation

This module manages source attribution, citation formatting, and credibility scoring
for the newsletter generation system. Supports multiple citation formats and
automated bibliography generation.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class CitationRecord:
    """Represents a citation record with metadata."""
    source_id: str
    title: str
    url: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    access_date: datetime = field(default_factory=datetime.now)
    credibility_score: float = 0.0
    content_snippet: str = ""
    usage_type: str = "reference"  # reference, quote, paraphrase
    page_number: Optional[str] = None
    publisher: Optional[str] = None
    source_type: str = "web"  # web, journal, book, news, academic
    metadata: Dict[str, Any] = field(default_factory=dict)


class CitationFormatter:
    """Formats citations in various academic and professional styles."""

    def __init__(self):
        self.formats = {
            'apa': self._format_apa,
            'mla': self._format_mla,
            'chicago': self._format_chicago,
            'harvard': self._format_harvard,
            'ieee': self._format_ieee
        }

    def format_citation(self, citation: CitationRecord, format_style: str = 'apa') -> str:
        """
        Format a citation record according to the specified style.

        Args:
            citation: Citation record to format
            format_style: Citation format (apa, mla, chicago, harvard, ieee)

        Returns:
            Formatted citation string
        """
        try:
            if format_style.lower() not in self.formats:
                logger.warning(f"Unknown citation format: {format_style}, using APA")
                format_style = 'apa'

            formatter = self.formats[format_style.lower()]
            return formatter(citation)

        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return self._format_fallback(citation)

    def _format_apa(self, citation: CitationRecord) -> str:
        """Format citation in APA style."""
        parts = []

        # Author (if available)
        if citation.author:
            # Handle multiple authors
            authors = citation.author.split(',')
            if len(authors) > 1:
                author_str = ', '.join(authors[:-1]) + f", & {authors[-1].strip()}"
            else:
                author_str = citation.author.strip()
            parts.append(f"{author_str}")

        # Publication date
        year = ""
        if citation.publication_date:
            year = f"({citation.publication_date.year})"
        elif citation.access_date:
            year = f"({citation.access_date.year})"
        
        if year:
            parts.append(year)

        # Title
        if citation.title:
            # Italicize web sources, use quotes for articles
            if citation.source_type in ['web', 'news']:
                title = f"*{citation.title}*"
            else:
                title = f'"{citation.title}"'
            parts.append(title)

        # Publisher/Website name
        if citation.publisher:
            parts.append(citation.publisher)
        elif citation.url:
            domain = self._extract_domain(citation.url)
            if domain:
                parts.append(domain)

        # URL and access date for web sources
        if citation.url:
            if citation.access_date:
                access_str = citation.access_date.strftime("%B %d, %Y")
                parts.append(f"Retrieved {access_str}, from {citation.url}")
            else:
                parts.append(citation.url)

        return '. '.join(parts) + '.'

    def _format_mla(self, citation: CitationRecord) -> str:
        """Format citation in MLA style."""
        parts = []

        # Author
        if citation.author:
            # Last name first for MLA
            author = citation.author.strip()
            if ',' not in author:
                # Simple name, reverse it
                name_parts = author.split()
                if len(name_parts) >= 2:
                    author = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
            parts.append(f'{author}.')

        # Title
        if citation.title:
            if citation.source_type in ['journal', 'academic']:
                title = f'"{citation.title}"'
            else:
                title = f'*{citation.title}*'
            parts.append(title)

        # Publisher/Website
        if citation.publisher:
            parts.append(f'{citation.publisher},')
        elif citation.url:
            domain = self._extract_domain(citation.url)
            if domain:
                parts.append(f'{domain},')

        # Date
        if citation.publication_date:
            date_str = citation.publication_date.strftime("%d %b %Y")
            parts.append(f'{date_str},')
        elif citation.access_date:
            date_str = citation.access_date.strftime("%d %b %Y")
            parts.append(f'{date_str},')

        # URL
        if citation.url:
            parts.append(citation.url)

        return ' '.join(parts) + '.'

    def _format_chicago(self, citation: CitationRecord) -> str:
        """Format citation in Chicago style."""
        parts = []

        # Author
        if citation.author:
            parts.append(f'{citation.author}.')

        # Title
        if citation.title:
            if citation.source_type in ['book', 'journal']:
                title = f'*{citation.title}*'
            else:
                title = f'"{citation.title}"'
            parts.append(title)

        # Publisher and date
        if citation.publisher and citation.publication_date:
            year = citation.publication_date.year
            parts.append(f'{citation.publisher}, {year}.')
        elif citation.publication_date:
            year = citation.publication_date.year
            parts.append(f'{year}.')

        # URL and access date
        if citation.url:
            if citation.access_date:
                access_str = citation.access_date.strftime("%B %d, %Y")
                parts.append(f'Accessed {access_str}. {citation.url}.')
            else:
                parts.append(f'{citation.url}.')

        return ' '.join(parts)

    def _format_harvard(self, citation: CitationRecord) -> str:
        """Format citation in Harvard style."""
        parts = []

        # Author and date
        if citation.author and citation.publication_date:
            year = citation.publication_date.year
            parts.append(f'{citation.author} ({year})')
        elif citation.author:
            parts.append(citation.author)

        # Title
        if citation.title:
            if citation.source_type in ['book']:
                title = f'*{citation.title}*'
            else:
                title = f"'{citation.title}'"
            parts.append(title)

        # Publisher
        if citation.publisher:
            parts.append(citation.publisher)

        # URL
        if citation.url:
            if citation.access_date:
                access_str = citation.access_date.strftime("%d %B %Y")
                parts.append(f'viewed {access_str}, <{citation.url}>')
            else:
                parts.append(f'<{citation.url}>')

        return ', '.join(parts) + '.'

    def _format_ieee(self, citation: CitationRecord) -> str:
        """Format citation in IEEE style."""
        parts = []

        # Author
        if citation.author:
            # IEEE uses initials
            author = self._format_ieee_author(citation.author)
            parts.append(f'{author},')

        # Title
        if citation.title:
            parts.append(f'"{citation.title},"')

        # Publisher/Website
        if citation.publisher:
            parts.append(f'*{citation.publisher}*,')
        elif citation.url:
            domain = self._extract_domain(citation.url)
            if domain:
                parts.append(f'*{domain}*,')

        # Date
        if citation.publication_date:
            month_year = citation.publication_date.strftime("%b %Y")
            parts.append(f'{month_year}.')
        elif citation.access_date:
            month_year = citation.access_date.strftime("%b %Y")
            parts.append(f'{month_year}.')

        # URL
        if citation.url:
            parts.append(f'[Online]. Available: {citation.url}')
            if citation.access_date:
                access_str = citation.access_date.strftime("%b %d, %Y")
                parts.append(f'[Accessed: {access_str}]')

        return ' '.join(parts)

    def _format_fallback(self, citation: CitationRecord) -> str:
        """Fallback format for citations."""
        parts = []
        
        if citation.author:
            parts.append(citation.author)
        
        if citation.title:
            parts.append(f'"{citation.title}"')
        
        if citation.url:
            parts.append(citation.url)
        
        if citation.access_date:
            access_str = citation.access_date.strftime("%Y-%m-%d")
            parts.append(f"(accessed {access_str})")
        
        return '. '.join(parts) + '.'

    def _format_ieee_author(self, author: str) -> str:
        """Format author name for IEEE style."""
        # Simple implementation - convert "First Last" to "F. Last"
        parts = author.strip().split()
        if len(parts) >= 2:
            first_initials = '. '.join([name[0] for name in parts[:-1]]) + '.'
            return f"{parts[-1]}, {first_initials}"
        return author

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.title()
        except Exception:
            return ""


class CredibilityScorer:
    """Scores source credibility based on various factors."""

    def __init__(self):
        self.domain_scores = self._load_domain_scores()
        self.source_type_scores = {
            'academic': 0.9,
            'journal': 0.85,
            'news': 0.7,
            'government': 0.8,
            'organization': 0.6,
            'web': 0.5,
            'blog': 0.4,
            'social': 0.3
        }

    def score(self, citation: CitationRecord) -> float:
        """
        Calculate credibility score for a citation.

        Args:
            citation: Citation record to score

        Returns:
            Credibility score between 0.0 and 1.0
        """
        try:
            score = 0.5  # Base score

            # Domain-based scoring
            if citation.url:
                domain_score = self._get_domain_score(citation.url)
                score += domain_score * 0.3

            # Source type scoring
            source_type_score = self.source_type_scores.get(citation.source_type, 0.5)
            score += source_type_score * 0.2

            # Author presence
            if citation.author:
                score += 0.1

            # Publication date (recent is better for news, not necessarily for academic)
            if citation.publication_date:
                age_score = self._calculate_age_score(citation.publication_date, citation.source_type)
                score += age_score * 0.1

            # Content quality indicators
            content_score = self._assess_content_quality(citation)
            score += content_score * 0.1

            # Publisher credibility
            if citation.publisher:
                publisher_score = self._get_publisher_score(citation.publisher)
                score += publisher_score * 0.1

            # Normalize score
            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Credibility scoring failed: {e}")
            return 0.5  # Return neutral score on error

    def _get_domain_score(self, url: str) -> float:
        """Get credibility score based on domain."""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            return self.domain_scores.get(domain, 0.5)

        except Exception:
            return 0.5

    def _calculate_age_score(self, pub_date: datetime, source_type: str) -> float:
        """Calculate score based on publication age."""
        try:
            age_days = (datetime.now() - pub_date).days

            if source_type in ['news', 'blog']:
                # Recent is better for news
                if age_days <= 7:
                    return 0.9
                elif age_days <= 30:
                    return 0.7
                elif age_days <= 365:
                    return 0.5
                else:
                    return 0.3
            else:
                # Academic and reference sources are less time-sensitive
                if age_days <= 365:
                    return 0.8
                elif age_days <= 1825:  # 5 years
                    return 0.7
                else:
                    return 0.6

        except Exception:
            return 0.5

    def _assess_content_quality(self, citation: CitationRecord) -> float:
        """Assess content quality based on available information."""
        score = 0.0

        # Check for comprehensive information
        if citation.title and len(citation.title) > 10:
            score += 0.3

        if citation.content_snippet and len(citation.content_snippet) > 50:
            score += 0.2

        if citation.author:
            score += 0.2

        if citation.publication_date:
            score += 0.1

        # Check for quality indicators in content
        if citation.content_snippet:
            quality_indicators = [
                'research', 'study', 'analysis', 'data', 'evidence',
                'methodology', 'peer-reviewed', 'academic', 'journal'
            ]
            content_lower = citation.content_snippet.lower()
            matches = sum(1 for indicator in quality_indicators if indicator in content_lower)
            score += min(matches * 0.05, 0.2)

        return min(score, 1.0)

    def _get_publisher_score(self, publisher: str) -> float:
        """Get credibility score for publisher."""
        publisher_lower = publisher.lower()
        
        high_credibility = [
            'nature', 'science', 'cell', 'lancet', 'nejm', 'ieee', 'acm',
            'reuters', 'associated press', 'bbc', 'npr', 'pbs'
        ]
        
        medium_credibility = [
            'cnn', 'fox', 'cbc', 'guardian', 'times', 'post',
            'journal', 'magazine', 'review'
        ]

        for pub in high_credibility:
            if pub in publisher_lower:
                return 0.8

        for pub in medium_credibility:
            if pub in publisher_lower:
                return 0.6

        return 0.5

    def _load_domain_scores(self) -> Dict[str, float]:
        """Load pre-defined domain credibility scores."""
        return {
            # Academic and research
            'arxiv.org': 0.9,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'scholar.google.com': 0.85,
            'researchgate.net': 0.8,
            'ieee.org': 0.9,
            'acm.org': 0.9,

            # News organizations
            'reuters.com': 0.9,
            'ap.org': 0.9,
            'bbc.com': 0.85,
            'npr.org': 0.85,
            'pbs.org': 0.8,
            'cnn.com': 0.7,
            'nytimes.com': 0.8,
            'washingtonpost.com': 0.8,
            'wsj.com': 0.8,

            # Government
            'gov': 0.8,  # General .gov domains
            'nih.gov': 0.9,
            'cdc.gov': 0.9,
            'fda.gov': 0.85,

            # Educational
            'edu': 0.8,  # General .edu domains
            'mit.edu': 0.9,
            'harvard.edu': 0.9,
            'stanford.edu': 0.9,

            # Technology
            'github.com': 0.7,
            'stackoverflow.com': 0.6,
            'techcrunch.com': 0.6,
            'wired.com': 0.7,
            'arstechnica.com': 0.7,

            # Organizations
            'who.int': 0.85,
            'un.org': 0.8,
            'worldbank.org': 0.8,

            # Lower credibility
            'wikipedia.org': 0.5,
            'reddit.com': 0.3,
            'twitter.com': 0.2,
            'facebook.com': 0.2,
            'youtube.com': 0.3
        }


class SourceTracker:
    """Tracks source usage and manages citation records."""

    def __init__(self):
        self.citations: Dict[str, CitationRecord] = {}
        self.usage_count: Dict[str, int] = {}

    def add_citation(self, citation: CitationRecord) -> str:
        """
        Add a citation record and return its ID.

        Args:
            citation: Citation record to add

        Returns:
            Citation ID
        """
        try:
            citation_id = self._generate_citation_id(citation)
            citation.source_id = citation_id
            
            self.citations[citation_id] = citation
            self.usage_count[citation_id] = self.usage_count.get(citation_id, 0) + 1
            
            logger.debug(f"Added citation: {citation_id}")
            return citation_id

        except Exception as e:
            logger.error(f"Failed to add citation: {e}")
            return ""

    def get_citation(self, citation_id: str) -> Optional[CitationRecord]:
        """Get citation record by ID."""
        return self.citations.get(citation_id)

    def get_all_citations(self) -> List[CitationRecord]:
        """Get all citation records."""
        return list(self.citations.values())

    def get_citations_by_type(self, source_type: str) -> List[CitationRecord]:
        """Get citations filtered by source type."""
        return [citation for citation in self.citations.values() 
                if citation.source_type == source_type]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get citation usage statistics."""
        total_citations = len(self.citations)
        total_usage = sum(self.usage_count.values())
        
        type_distribution = {}
        for citation in self.citations.values():
            type_distribution[citation.source_type] = \
                type_distribution.get(citation.source_type, 0) + 1

        credibility_stats = self._calculate_credibility_stats()

        return {
            'total_citations': total_citations,
            'total_usage': total_usage,
            'type_distribution': type_distribution,
            'credibility_stats': credibility_stats,
            'most_used': self._get_most_used_citations()
        }

    def _generate_citation_id(self, citation: CitationRecord) -> str:
        """Generate unique ID for citation."""
        # Use URL and title for uniqueness
        id_string = f"{citation.url}:{citation.title}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]

    def _calculate_credibility_stats(self) -> Dict[str, float]:
        """Calculate credibility statistics."""
        if not self.citations:
            return {}

        scores = [citation.credibility_score for citation in self.citations.values()]
        
        return {
            'average_credibility': sum(scores) / len(scores),
            'min_credibility': min(scores),
            'max_credibility': max(scores),
            'high_credibility_count': sum(1 for score in scores if score > 0.8)
        }

    def _get_most_used_citations(self) -> List[Dict[str, Any]]:
        """Get most frequently used citations."""
        sorted_usage = sorted(self.usage_count.items(), 
                            key=lambda x: x[1], reverse=True)
        
        result = []
        for citation_id, usage_count in sorted_usage[:5]:
            citation = self.citations.get(citation_id)
            if citation:
                result.append({
                    'id': citation_id,
                    'title': citation.title,
                    'usage_count': usage_count,
                    'credibility_score': citation.credibility_score
                })
        
        return result


class CitationManager:
    """
    Main citation management system that orchestrates citation tracking,
    formatting, and credibility scoring.
    """

    def __init__(self, default_format: str = 'apa'):
        """
        Initialize citation manager.

        Args:
            default_format: Default citation format style
        """
        self.citation_formatter = CitationFormatter()
        self.credibility_scorer = CredibilityScorer()
        self.source_tracker = SourceTracker()
        self.default_format = default_format

    def track_source(self, content: str, source_info: Dict[str, Any]) -> CitationRecord:
        """
        Track source usage and generate citation record.

        Args:
            content: Content snippet from the source
            source_info: Dictionary containing source information

        Returns:
            Generated citation record
        """
        try:
            # Create citation record
            citation = CitationRecord(
                source_id="",  # Will be generated
                title=source_info.get('title', ''),
                url=source_info.get('url', ''),
                author=source_info.get('author'),
                publication_date=source_info.get('publication_date'),
                access_date=datetime.now(),
                content_snippet=content[:200] if content else "",
                usage_type=source_info.get('usage_type', 'reference'),
                publisher=source_info.get('publisher'),
                source_type=source_info.get('source_type', 'web'),
                metadata=source_info.get('metadata', {})
            )

            # Score credibility
            citation.credibility_score = self.credibility_scorer.score(citation)

            # Add to tracker
            citation_id = self.source_tracker.add_citation(citation)

            return citation

        except Exception as e:
            logger.error(f"Source tracking failed: {e}")
            # Return minimal citation record
            return CitationRecord(
                source_id="unknown",
                title=source_info.get('title', 'Unknown Source'),
                url=source_info.get('url', ''),
                credibility_score=0.5
            )

    def generate_inline_citation(self, citation_id: str, format_style: str = None) -> str:
        """
        Generate inline citation reference.

        Args:
            citation_id: Citation ID
            format_style: Citation format style

        Returns:
            Inline citation string
        """
        try:
            citation = self.source_tracker.get_citation(citation_id)
            if not citation:
                return "[Citation not found]"

            format_style = format_style or self.default_format

            if format_style.lower() == 'apa':
                if citation.author and citation.publication_date:
                    return f"({citation.author}, {citation.publication_date.year})"
                elif citation.author:
                    return f"({citation.author}, n.d.)"
                else:
                    domain = self.citation_formatter._extract_domain(citation.url)
                    year = citation.access_date.year if citation.access_date else "n.d."
                    return f"({domain}, {year})"
            
            elif format_style.lower() == 'mla':
                if citation.author:
                    return f"({citation.author})"
                else:
                    domain = self.citation_formatter._extract_domain(citation.url)
                    return f"({domain})"
            
            else:
                # Simple numbered reference
                return f"[{citation_id[:8]}]"

        except Exception as e:
            logger.error(f"Inline citation generation failed: {e}")
            return "[Citation error]"

    def generate_bibliography(self, format_style: str = None, 
                            filter_by_credibility: float = None) -> str:
        """
        Generate formatted bibliography from tracked sources.

        Args:
            format_style: Citation format style
            filter_by_credibility: Minimum credibility score filter

        Returns:
            Formatted bibliography string
        """
        try:
            format_style = format_style or self.default_format
            citations = self.source_tracker.get_all_citations()

            # Filter by credibility if specified
            if filter_by_credibility is not None:
                citations = [c for c in citations if c.credibility_score >= filter_by_credibility]

            if not citations:
                return "No sources to cite."

            # Sort citations (typically alphabetically by author/title)
            citations.sort(key=lambda c: (c.author or c.title or "").lower())

            # Format each citation
            formatted_citations = []
            for citation in citations:
                formatted = self.citation_formatter.format_citation(citation, format_style)
                formatted_citations.append(formatted)

            # Join with appropriate formatting
            bibliography_header = self._get_bibliography_header(format_style)
            bibliography_content = '\n\n'.join(formatted_citations)

            return f"{bibliography_header}\n\n{bibliography_content}"

        except Exception as e:
            logger.error(f"Bibliography generation failed: {e}")
            return "Error generating bibliography."

    def get_citation_analytics(self) -> Dict[str, Any]:
        """Get comprehensive citation analytics."""
        try:
            usage_stats = self.source_tracker.get_usage_stats()
            
            # Add format-specific information
            analytics = {
                **usage_stats,
                'default_format': self.default_format,
                'available_formats': list(self.citation_formatter.formats.keys())
            }

            return analytics

        except Exception as e:
            logger.error(f"Citation analytics failed: {e}")
            return {}

    def _get_bibliography_header(self, format_style: str) -> str:
        """Get appropriate header for bibliography format."""
        headers = {
            'apa': 'References',
            'mla': 'Works Cited',
            'chicago': 'Bibliography',
            'harvard': 'References',
            'ieee': 'References'
        }
        
        return headers.get(format_style.lower(), 'References')


# Global instance
_citation_manager = None


def get_citation_manager() -> CitationManager:
    """Get the global citation manager instance."""
    global _citation_manager
    if _citation_manager is None:
        _citation_manager = CitationManager()
    return _citation_manager


def track_source(content: str, source_info: Dict[str, Any]) -> CitationRecord:
    """
    Convenience function to track a source.

    Args:
        content: Content snippet from the source
        source_info: Dictionary containing source information

    Returns:
        Generated citation record
    """
    manager = get_citation_manager()
    return manager.track_source(content, source_info)