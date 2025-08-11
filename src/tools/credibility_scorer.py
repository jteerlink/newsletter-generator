"""
Credibility Scorer Tool

This module provides advanced credibility scoring capabilities for sources
in the newsletter generation system, including domain analysis, content
quality assessment, and bias detection.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AdvancedCredibilityScorer:
    """
    Advanced credibility scorer with multiple evaluation criteria
    and machine learning-inspired scoring algorithms.
    """

    def __init__(self):
        """Initialize the advanced credibility scorer."""
        self.domain_reputation = self._load_domain_reputation()
        self.bias_indicators = self._load_bias_indicators()
        self.quality_patterns = self._load_quality_patterns()
        self.spam_indicators = self._load_spam_indicators()
        self.authority_markers = self._load_authority_markers()

    def score_comprehensive(self, source_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive credibility scoring.

        Args:
            source_info: Dictionary containing source information
                - title: Source title
                - url: Source URL
                - content: Source content/snippet
                - author: Author name (optional)
                - publication_date: Publication date (optional)
                - publisher: Publisher name (optional)
                - source_type: Type of source (web, news, academic, etc.)

        Returns:
            Comprehensive scoring results
        """
        try:
            logger.debug(f"Scoring credibility for: {source_info.get('title', 'Unknown')}")

            # Extract basic information
            url = source_info.get('url', '')
            title = source_info.get('title', '')
            content = source_info.get('content', '')
            author = source_info.get('author', '')
            pub_date = source_info.get('publication_date')
            publisher = source_info.get('publisher', '')
            source_type = source_info.get('source_type', 'web')

            # Calculate individual scores
            domain_score = self._score_domain_reputation(url)
            authority_score = self._score_authority_markers(content, author, publisher)
            quality_score = self._score_content_quality(title, content)
            bias_score = self._score_bias_indicators(title, content)
            freshness_score = self._score_freshness(pub_date, source_type)
            spam_score = self._score_spam_indicators(title, content, url)
            technical_score = self._score_technical_indicators(url, content)

            # Calculate weighted overall score
            weights = self._get_scoring_weights(source_type)
            overall_score = (
                domain_score * weights['domain'] +
                authority_score * weights['authority'] +
                quality_score * weights['quality'] +
                (1 - bias_score) * weights['bias'] +  # Invert bias score
                freshness_score * weights['freshness'] +
                (1 - spam_score) * weights['spam'] +  # Invert spam score
                technical_score * weights['technical']
            )

            # Generate detailed results
            results = {
                'overall_score': round(overall_score, 3),
                'component_scores': {
                    'domain_reputation': round(domain_score, 3),
                    'authority_markers': round(authority_score, 3),
                    'content_quality': round(quality_score, 3),
                    'bias_indicators': round(bias_score, 3),
                    'freshness': round(freshness_score, 3),
                    'spam_indicators': round(spam_score, 3),
                    'technical_indicators': round(technical_score, 3)
                },
                'credibility_level': self._categorize_credibility(overall_score),
                'risk_factors': self._identify_risk_factors(
                    domain_score, authority_score, quality_score, 
                    bias_score, spam_score
                ),
                'recommendations': self._generate_recommendations(
                    overall_score, source_type, domain_score, quality_score
                ),
                'metadata': {
                    'scorer_version': '2.0',
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'source_type': source_type,
                    'weights_used': weights
                }
            }

            logger.debug(f"Credibility score: {overall_score:.3f} ({results['credibility_level']})")
            return results

        except Exception as e:
            logger.error(f"Comprehensive credibility scoring failed: {e}")
            return self._get_default_score()

    def _score_domain_reputation(self, url: str) -> float:
        """Score based on domain reputation."""
        if not url:
            return 0.3

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check exact matches first
            if domain in self.domain_reputation:
                return self.domain_reputation[domain]

            # Check for partial matches (e.g., subdomains)
            for known_domain, score in self.domain_reputation.items():
                if domain.endswith(known_domain):
                    return score * 0.9  # Slightly lower for subdomains

            # TLD-based scoring
            tld_scores = {
                '.edu': 0.8,
                '.gov': 0.85,
                '.org': 0.6,
                '.mil': 0.8,
                '.int': 0.7,
                '.com': 0.5,
                '.net': 0.45,
                '.info': 0.4,
                '.biz': 0.35,
                '.tk': 0.2,  # Often used for spam
                '.ml': 0.2   # Often used for spam
            }

            for tld, score in tld_scores.items():
                if domain.endswith(tld):
                    return score

            return 0.4  # Default for unknown domains

        except Exception as e:
            logger.warning(f"Domain reputation scoring failed: {e}")
            return 0.3

    def _score_authority_markers(self, content: str, author: str, publisher: str) -> float:
        """Score based on authority markers."""
        score = 0.0

        # Author presence and quality
        if author:
            score += 0.2
            
            # Check for academic credentials
            if any(marker in author.lower() for marker in ['ph.d', 'phd', 'dr.', 'prof', 'professor']):
                score += 0.2

        # Publisher reputation
        if publisher:
            score += 0.1
            
            # High-reputation publishers
            high_rep_publishers = [
                'nature', 'science', 'cell', 'lancet', 'nejm', 'bmj',
                'ieee', 'acm', 'springer', 'elsevier', 'wiley',
                'reuters', 'associated press', 'bbc', 'pbs', 'npr'
            ]
            
            publisher_lower = publisher.lower()
            if any(pub in publisher_lower for pub in high_rep_publishers):
                score += 0.3

        # Content authority markers
        if content:
            content_lower = content.lower()
            
            # Academic and research indicators
            authority_indicators = [
                'peer review', 'peer-reviewed', 'research study', 'clinical trial',
                'methodology', 'literature review', 'meta-analysis', 'systematic review',
                'published in', 'journal of', 'proceedings of', 'conference paper',
                'doi:', 'pmid:', 'issn:', 'isbn:', 'citation needed', 'references'
            ]
            
            matches = sum(1 for indicator in authority_indicators if indicator in content_lower)
            score += min(matches * 0.05, 0.3)

            # Expert quotes and attribution
            expert_patterns = [
                r'according to.*expert', r'.*professor.*said', r'.*researcher.*found',
                r'study.*shows', r'research.*indicates', r'data.*suggests'
            ]
            
            pattern_matches = sum(1 for pattern in expert_patterns 
                                if re.search(pattern, content_lower))
            score += min(pattern_matches * 0.03, 0.15)

        return min(score, 1.0)

    def _score_content_quality(self, title: str, content: str) -> float:
        """Score content quality based on various indicators."""
        score = 0.3  # Base score

        # Title quality
        if title:
            title_lower = title.lower()
            
            # Penalty for clickbait indicators
            clickbait_indicators = [
                'you won\'t believe', 'shocking', 'amazing', 'incredible',
                'this one trick', 'doctors hate', 'scientists hate',
                'number will shock you', 'what happens next'
            ]
            
            clickbait_count = sum(1 for indicator in clickbait_indicators 
                                if indicator in title_lower)
            score -= clickbait_count * 0.15

            # Bonus for informative titles
            if len(title) > 20 and len(title) < 200:
                score += 0.1

        # Content quality indicators
        if content:
            content_lower = content.lower()
            
            # Length indicator (longer content often more comprehensive)
            length_score = min(len(content) / 2000, 0.15)
            score += length_score

            # Quality vocabulary
            quality_indicators = [
                'analysis', 'research', 'study', 'investigation', 'examination',
                'methodology', 'findings', 'conclusion', 'evidence', 'data',
                'statistics', 'survey', 'interview', 'expert', 'professional'
            ]
            
            quality_matches = sum(1 for indicator in quality_indicators 
                                if indicator in content_lower)
            score += min(quality_matches * 0.02, 0.2)

            # Grammar and style indicators
            sentence_count = len(re.findall(r'[.!?]+', content))
            word_count = len(content.split())
            
            if sentence_count > 0:
                avg_sentence_length = word_count / sentence_count
                # Optimal sentence length is around 15-20 words
                if 10 <= avg_sentence_length <= 25:
                    score += 0.1

            # Penalty for excessive capitalization or exclamation marks
            caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
            if caps_ratio > 0.3:
                score -= 0.2

            exclamation_count = content.count('!')
            if exclamation_count > len(content) / 100:  # More than 1% exclamation marks
                score -= 0.1

        return min(max(score, 0.0), 1.0)

    def _score_bias_indicators(self, title: str, content: str) -> float:
        """Score bias indicators (higher score means more biased)."""
        bias_score = 0.0
        text = f"{title} {content}".lower()

        # Political bias indicators
        political_bias_words = [
            'liberal agenda', 'conservative conspiracy', 'mainstream media',
            'fake news', 'deep state', 'propaganda', 'brainwashing',
            'sheeple', 'wake up', 'they don\'t want you to know'
        ]
        
        bias_matches = sum(1 for phrase in political_bias_words if phrase in text)
        bias_score += min(bias_matches * 0.2, 0.6)

        # Emotional manipulation
        emotional_words = [
            'outrageous', 'disgusting', 'horrifying', 'shocking truth',
            'devastating', 'explosive', 'bombshell', 'scandal'
        ]
        
        emotional_matches = sum(1 for word in emotional_words if word in text)
        bias_score += min(emotional_matches * 0.1, 0.3)

        # Absolutist language
        absolutist_patterns = [
            r'always.*never', r'all.*none', r'every.*nothing',
            r'completely.*totally', r'absolutely.*definitely'
        ]
        
        absolutist_matches = sum(1 for pattern in absolutist_patterns 
                               if re.search(pattern, text))
        bias_score += min(absolutist_matches * 0.05, 0.2)

        return min(bias_score, 1.0)

    def _score_freshness(self, pub_date: Optional[datetime], source_type: str) -> float:
        """Score content freshness based on publication date."""
        if not pub_date:
            return 0.5  # Neutral score for unknown dates

        try:
            age_days = (datetime.now() - pub_date).days

            # Time-sensitive content types
            if source_type in ['news', 'breaking', 'alert']:
                if age_days <= 1:
                    return 1.0
                elif age_days <= 7:
                    return 0.8
                elif age_days <= 30:
                    return 0.6
                elif age_days <= 90:
                    return 0.4
                else:
                    return 0.2

            # Academic and reference content
            elif source_type in ['academic', 'journal', 'research']:
                if age_days <= 365:
                    return 0.9
                elif age_days <= 1825:  # 5 years
                    return 0.8
                elif age_days <= 3650:  # 10 years
                    return 0.7
                else:
                    return 0.6

            # General web content
            else:
                if age_days <= 30:
                    return 0.8
                elif age_days <= 180:
                    return 0.7
                elif age_days <= 365:
                    return 0.6
                elif age_days <= 1825:
                    return 0.5
                else:
                    return 0.4

        except Exception as e:
            logger.warning(f"Freshness scoring failed: {e}")
            return 0.5

    def _score_spam_indicators(self, title: str, content: str, url: str) -> float:
        """Score spam indicators (higher score means more spam-like)."""
        spam_score = 0.0
        text = f"{title} {content}".lower()

        # Common spam phrases
        spam_phrases = [
            'click here now', 'limited time offer', 'act now', 'don\'t miss out',
            'make money fast', 'work from home', 'lose weight quickly',
            'miracle cure', 'amazing results', 'secret revealed',
            'doctors hate this', 'one simple trick', 'you won\'t believe'
        ]
        
        spam_matches = sum(1 for phrase in spam_phrases if phrase in text)
        spam_score += min(spam_matches * 0.2, 0.6)

        # Excessive promotional language
        promo_indicators = [
            'free', 'discount', 'sale', 'offer', 'deal', 'save',
            'cheap', 'affordable', 'guarantee', 'risk-free'
        ]
        
        promo_count = sum(1 for word in promo_indicators if word in text)
        if promo_count > len(text.split()) * 0.05:  # More than 5% promotional words
            spam_score += 0.3

        # URL analysis for spam indicators
        if url:
            url_lower = url.lower()
            suspicious_url_patterns = [
                'bit.ly', 'tinyurl', 'goo.gl', 't.co',  # Shortened URLs
                'affiliate', 'ref=', 'tracker', 'campaign'  # Affiliate/tracking links
            ]
            
            url_spam_matches = sum(1 for pattern in suspicious_url_patterns 
                                 if pattern in url_lower)
            spam_score += min(url_spam_matches * 0.1, 0.3)

        return min(spam_score, 1.0)

    def _score_technical_indicators(self, url: str, content: str) -> float:
        """Score technical quality indicators."""
        score = 0.5  # Base score

        # HTTPS usage
        if url and url.startswith('https://'):
            score += 0.1

        # Content structure indicators
        if content:
            # Presence of structured data
            if any(marker in content for marker in ['<script type="application/ld+json">', 
                                                   'schema.org', 'microdata']):
                score += 0.1

            # Proper citation formats
            citation_patterns = [
                r'doi:\s*10\.\d+', r'pmid:\s*\d+', r'arxiv:\d+\.\d+',
                r'isbn:\s*\d{13}', r'issn:\s*\d{4}-\d{4}'
            ]
            
            citation_matches = sum(1 for pattern in citation_patterns 
                                 if re.search(pattern, content, re.IGNORECASE))
            score += min(citation_matches * 0.05, 0.2)

        return min(score, 1.0)

    def _get_scoring_weights(self, source_type: str) -> Dict[str, float]:
        """Get scoring weights based on source type."""
        if source_type in ['academic', 'journal', 'research']:
            return {
                'domain': 0.25,
                'authority': 0.30,
                'quality': 0.25,
                'bias': 0.10,
                'freshness': 0.05,
                'spam': 0.03,
                'technical': 0.02
            }
        elif source_type in ['news', 'breaking']:
            return {
                'domain': 0.30,
                'authority': 0.20,
                'quality': 0.20,
                'bias': 0.15,
                'freshness': 0.10,
                'spam': 0.03,
                'technical': 0.02
            }
        else:  # General web content
            return {
                'domain': 0.25,
                'authority': 0.15,
                'quality': 0.20,
                'bias': 0.15,
                'freshness': 0.10,
                'spam': 0.10,
                'technical': 0.05
            }

    def _categorize_credibility(self, score: float) -> str:
        """Categorize credibility score into human-readable levels."""
        if score >= 0.85:
            return "Very High"
        elif score >= 0.70:
            return "High"
        elif score >= 0.55:
            return "Moderate"
        elif score >= 0.40:
            return "Low"
        else:
            return "Very Low"

    def _identify_risk_factors(self, domain_score: float, authority_score: float,
                             quality_score: float, bias_score: float, 
                             spam_score: float) -> List[str]:
        """Identify specific risk factors based on component scores."""
        risks = []

        if domain_score < 0.4:
            risks.append("Low domain reputation")
        
        if authority_score < 0.3:
            risks.append("Lack of clear authority markers")
        
        if quality_score < 0.4:
            risks.append("Poor content quality indicators")
        
        if bias_score > 0.6:
            risks.append("High bias indicators detected")
        
        if spam_score > 0.5:
            risks.append("Spam-like characteristics present")

        return risks

    def _generate_recommendations(self, overall_score: float, source_type: str,
                                domain_score: float, quality_score: float) -> List[str]:
        """Generate recommendations based on scoring results."""
        recommendations = []

        if overall_score < 0.5:
            recommendations.append("Consider finding alternative sources for this information")
        
        if domain_score < 0.4:
            recommendations.append("Verify information through more authoritative domains")
        
        if quality_score < 0.4:
            recommendations.append("Cross-reference with higher quality sources")
        
        if source_type == 'web' and overall_score < 0.6:
            recommendations.append("Look for academic or news sources on this topic")

        if overall_score > 0.8:
            recommendations.append("This appears to be a highly credible source")

        return recommendations

    def _get_default_score(self) -> Dict[str, Any]:
        """Return default score structure on error."""
        return {
            'overall_score': 0.5,
            'component_scores': {
                'domain_reputation': 0.5,
                'authority_markers': 0.5,
                'content_quality': 0.5,
                'bias_indicators': 0.5,
                'freshness': 0.5,
                'spam_indicators': 0.5,
                'technical_indicators': 0.5
            },
            'credibility_level': 'Moderate',
            'risk_factors': ['Scoring system error'],
            'recommendations': ['Manual verification recommended'],
            'metadata': {
                'scorer_version': '2.0',
                'evaluation_timestamp': datetime.now().isoformat(),
                'error': 'Default score returned due to evaluation error'
            }
        }

    def _load_domain_reputation(self) -> Dict[str, float]:
        """Load domain reputation scores."""
        return {
            # Academic and Research (0.85-0.95)
            'arxiv.org': 0.90,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'scholar.google.com': 0.85,
            'researchgate.net': 0.80,
            'ieee.org': 0.92,
            'acm.org': 0.91,
            'nature.com': 0.95,
            'science.org': 0.94,
            'cell.com': 0.93,
            'nejm.org': 0.94,
            'thelancet.com': 0.93,
            'bmj.com': 0.92,

            # News Organizations (0.65-0.90)
            'reuters.com': 0.90,
            'ap.org': 0.89,
            'bbc.com': 0.87,
            'npr.org': 0.85,
            'pbs.org': 0.84,
            'cnn.com': 0.72,
            'nytimes.com': 0.82,
            'washingtonpost.com': 0.81,
            'wsj.com': 0.83,
            'theguardian.com': 0.80,
            'economist.com': 0.85,

            # Government (0.80-0.90)
            'nih.gov': 0.92,
            'cdc.gov': 0.91,
            'fda.gov': 0.89,
            'who.int': 0.88,
            'un.org': 0.85,

            # Technology and Industry (0.60-0.85)
            'github.com': 0.75,
            'stackoverflow.com': 0.70,
            'techcrunch.com': 0.65,
            'wired.com': 0.72,
            'arstechnica.com': 0.74,
            'ieee.org': 0.85,

            # Lower credibility sources (0.20-0.50)
            'wikipedia.org': 0.55,
            'reddit.com': 0.35,
            'twitter.com': 0.25,
            'facebook.com': 0.20,
            'youtube.com': 0.30,
            'medium.com': 0.50,

            # Known problematic domains (0.10-0.30)
            'infowars.com': 0.15,
            'breitbart.com': 0.25,
            'theonion.com': 0.10,  # Satire
            'clickhole.com': 0.10,  # Satire
        }

    def _load_bias_indicators(self) -> List[str]:
        """Load bias indicator patterns."""
        return [
            'liberal agenda', 'conservative conspiracy', 'mainstream media lies',
            'fake news', 'deep state', 'propaganda machine', 'brainwashing',
            'sheeple', 'wake up people', 'they don\'t want you to know',
            'hidden truth', 'cover up', 'conspiracy', 'secret agenda'
        ]

    def _load_quality_patterns(self) -> List[str]:
        """Load content quality patterns."""
        return [
            'peer reviewed', 'research study', 'clinical trial', 'methodology',
            'literature review', 'meta-analysis', 'systematic review', 'expert opinion',
            'data analysis', 'statistical significance', 'control group', 'sample size'
        ]

    def _load_spam_indicators(self) -> List[str]:
        """Load spam indicator patterns."""
        return [
            'click here now', 'limited time offer', 'act fast', 'don\'t miss out',
            'make money fast', 'work from home', 'lose weight quickly',
            'miracle cure', 'amazing results', 'secret revealed', 'one weird trick'
        ]

    def _load_authority_markers(self) -> List[str]:
        """Load authority marker patterns."""
        return [
            'ph.d', 'phd', 'dr.', 'prof', 'professor', 'researcher', 'scientist',
            'expert', 'specialist', 'authority', 'institute', 'university',
            'college', 'department', 'laboratory', 'center for'
        ]


# Global instance
_advanced_credibility_scorer = None


def get_advanced_credibility_scorer() -> AdvancedCredibilityScorer:
    """Get the global advanced credibility scorer instance."""
    global _advanced_credibility_scorer
    if _advanced_credibility_scorer is None:
        _advanced_credibility_scorer = AdvancedCredibilityScorer()
    return _advanced_credibility_scorer


def score_source_credibility(source_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to score source credibility.

    Args:
        source_info: Dictionary containing source information

    Returns:
        Comprehensive credibility scoring results
    """
    scorer = get_advanced_credibility_scorer()
    return scorer.score_comprehensive(source_info)