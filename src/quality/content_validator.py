"""
Content Quality Validator

This module provides comprehensive content validation functionality,
consolidating content quality assessment, repetition detection, and
fact-checking capabilities.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Union
from collections import Counter
from difflib import SequenceMatcher

from .base import QualityValidator, QualityMetrics, QualityReport, QualityStatus

logger = logging.getLogger(__name__)


class ContentQualityValidator(QualityValidator):
    """Comprehensive content quality validator for newsletter content."""
    
    def __init__(self):
        super().__init__("ContentQualityValidator")
        
        # Suspicious expert patterns for quote validation
        self.suspicious_expert_patterns = [
            r"Dr\. [A-Z][a-z]+ [A-Z][a-z]+.*?(MIT|Stanford|Harvard|Berkeley|Cambridge|Oxford)",
            r"Professor [A-Z][a-z]+ [A-Z][a-z]+.*?(Co-Founder|Chief Scientist|Director)",
            r"[A-Z][a-z]+ [A-Z][a-z]+.*?(AI Researcher|Machine Learning Expert|Technology Leader)",
        ]
        
        # Generic expert quotes that may indicate AI-generated content
        self.generic_expert_quotes = [
            "must be designed with transparency",
            "critical aspect of developing effective",
            "most important thing about",
            "key to success is",
            "future of AI depends on",
        ]
        
        # Factual claim patterns for fact-checking
        self.factual_claim_patterns = [
            r"(\d+)% of (?:companies|organizations|users|people)",
            r"studies show that",
            r"research indicates",
            r"according to (?:recent|new) (?:study|research|survey)",
            r"data reveals that",
            r"statistics show",
        ]
    
    def validate(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityReport:
        """Validate content quality and return comprehensive report."""
        start_time = self._get_current_time()
        
        # Extract text content if dictionary provided
        text_content = self._extract_text_content(content)
        
        # Perform comprehensive validation
        repetition_analysis = self._detect_repetition(text_content)
        expert_analysis = self._analyze_expert_quotes(text_content)
        factual_analysis = self._analyze_factual_claims(text_content)
        quality_metrics = self._assess_content_quality(text_content)
        
        # Calculate overall metrics
        metrics = self._calculate_content_metrics(
            quality_metrics, repetition_analysis, expert_analysis, factual_analysis
        )
        
        # Generate issues and recommendations
        issues, warnings, recommendations, blocking_issues = self._generate_content_issues(
            repetition_analysis, expert_analysis, factual_analysis, quality_metrics
        )
        
        # Determine status
        status = self._determine_content_status(issues, warnings, blocking_issues, metrics)
        
        # Create quality report
        report = QualityReport(
            status=status,
            metrics=metrics,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            blocking_issues=blocking_issues,
            strengths=self._identify_content_strengths(quality_metrics),
            detailed_analysis={
                'repetition_analysis': repetition_analysis,
                'expert_analysis': expert_analysis,
                'factual_analysis': factual_analysis,
                'quality_metrics': quality_metrics,
                'content_length': len(text_content),
                'word_count': len(text_content.split())
            }
        )
        
        return report
    
    def get_metrics(self, content: Union[str, Dict[str, Any]], **kwargs) -> QualityMetrics:
        """Extract quality metrics from content."""
        text_content = self._extract_text_content(content)
        quality_metrics = self._assess_content_quality(text_content)
        
        return self._calculate_content_metrics(quality_metrics, {}, {}, {})
    
    def _extract_text_content(self, content: Union[str, Dict[str, Any]]) -> str:
        """Extract text content from various input formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract text from common newsletter content structure
            if 'content' in content:
                return str(content['content'])
            elif 'body' in content:
                return str(content['body'])
            elif 'text' in content:
                return str(content['text'])
            else:
                # Try to find any text-like field
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 50:
                        return value
                return str(content)
        else:
            return str(content)
    
    def _detect_repetition(self, content: str) -> Dict[str, Any]:
        """Detect repetitive content patterns."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        repetitive_sentences = []
        sentence_groups = []
        
        for i, sentence1 in enumerate(sentences):
            similar_sentences = []
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                similarity = self._calculate_similarity(sentence1, sentence2)
                if similarity > 0.7:  # High similarity threshold
                    similar_sentences.append((j, sentence2, similarity))
            
            if similar_sentences:
                group = [i] + [idx for idx, _, _ in similar_sentences]
                sentence_groups.append(group)
        
        # Detect repeated concepts/phrases
        concept_repetition = self._detect_concept_repetition(content)
        
        # Calculate repetition score
        total_sentences = len(sentences)
        repetitive_count = sum(len(group) for group in sentence_groups)
        repetition_score = self._calculate_repetition_score(
            repetitive_count, len(concept_repetition), 0, total_sentences
        )
        
        return {
            'repetition_score': repetition_score,
            'repetitive_sentences': repetitive_sentences,
            'sentence_groups': sentence_groups,
            'concept_repetition': concept_repetition,
            'total_sentences': total_sentences,
            'repetitive_count': repetitive_count
        }
    
    def _analyze_expert_quotes(self, content: str) -> Dict[str, Any]:
        """Analyze expert quotes for credibility and authenticity."""
        quotes = re.findall(r'"([^"]*)"', content)
        suspicious_quotes = []
        
        for quote in quotes:
            suspicion_score = self._calculate_quote_suspicion(quote)
            if suspicion_score > 0.5:  # High suspicion threshold
                suspicious_quotes.append({
                    'quote': quote,
                    'suspicion_score': suspicion_score,
                    'reasons': self._get_suspicion_reasons(quote)
                })
        
        return {
            'total_quotes': len(quotes),
            'suspicious_quotes': suspicious_quotes,
            'suspicion_score': len(suspicious_quotes) / max(len(quotes), 1),
            'average_suspicion': sum(q['suspicion_score'] for q in suspicious_quotes) / max(len(suspicious_quotes), 1)
        }
    
    def _analyze_factual_claims(self, content: str) -> Dict[str, Any]:
        """Analyze factual claims for verification needs."""
        claims = []
        
        for pattern in self.factual_claim_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches)
        
        return {
            'total_claims': len(claims),
            'claims': claims,
            'verification_needed': len(claims) > 3,  # Flag if many claims
            'claim_density': len(claims) / max(len(content.split()), 1)
        }
    
    def _assess_content_quality(self, content: str) -> Dict[str, float]:
        """Assess various aspects of content quality."""
        return {
            'technical_accuracy': self._assess_technical_accuracy(content),
            'information_density': self._assess_information_density(content),
            'readability': self._assess_readability(content),
            'engagement': self._assess_engagement_factors(content),
            'structure': self._assess_structure_quality(content),
            'ai_ml_relevance': self._assess_ai_ml_relevance(content),
            'code_quality': self._assess_code_quality(content),
            'citation_quality': self._assess_citation_quality(content),
            'practical_value': self._assess_practical_value(content),
            'innovation_factor': self._assess_innovation_factor(content)
        }
    
    def _calculate_content_metrics(self, quality_metrics: Dict[str, float],
                                 repetition_analysis: Dict[str, Any],
                                 expert_analysis: Dict[str, Any],
                                 factual_analysis: Dict[str, Any]) -> QualityMetrics:
        """Calculate comprehensive content quality metrics."""
        # Base scores from quality assessment
        technical_accuracy = quality_metrics.get('technical_accuracy', 0.0)
        content_quality = quality_metrics.get('information_density', 0.0)
        readability = quality_metrics.get('readability', 0.0)
        engagement = quality_metrics.get('engagement', 0.0)
        structure = quality_metrics.get('structure', 0.0)
        code_quality = quality_metrics.get('code_quality', 0.0)
        
        # Adjust scores based on analysis results
        repetition_score = repetition_analysis.get('repetition_score', 0.0)
        content_quality = max(0.0, content_quality - repetition_score * 2)
        
        suspicion_score = expert_analysis.get('suspicion_score', 0.0)
        technical_accuracy = max(0.0, technical_accuracy - suspicion_score * 3)
        
        claim_density = factual_analysis.get('claim_density', 0.0)
        if claim_density > 0.1:  # High claim density
            content_quality = max(0.0, content_quality - 1.0)
        
        return QualityMetrics(
            overall_score=0.0,  # Will be calculated by post_init
            technical_accuracy_score=technical_accuracy,
            content_quality_score=content_quality,
            readability_score=readability,
            engagement_score=engagement,
            structure_score=structure,
            code_quality_score=code_quality,
            mobile_readability_score=readability * 0.9,  # Slightly lower for mobile
            source_credibility_score=max(0.0, 10.0 - suspicion_score * 10),
            content_balance_score=structure * 0.8,
            performance_score=10.0  # Placeholder for performance metrics
        )
    
    def _generate_content_issues(self, repetition_analysis: Dict[str, Any],
                               expert_analysis: Dict[str, Any],
                               factual_analysis: Dict[str, Any],
                               quality_metrics: Dict[str, float]) -> tuple:
        """Generate issues, warnings, recommendations, and blocking issues."""
        issues = []
        warnings = []
        recommendations = []
        blocking_issues = []
        
        # Repetition issues
        repetition_score = repetition_analysis.get('repetition_score', 0.0)
        if repetition_score > 0.3:
            issues.append(f"High repetition detected (score: {repetition_score:.2f})")
            recommendations.append("Consider rewriting repetitive sections to improve variety")
        
        if repetition_score > 0.6:
            blocking_issues.append("Excessive repetition detected - content needs significant revision")
        
        # Expert quote issues
        suspicious_count = len(expert_analysis.get('suspicious_quotes', []))
        if suspicious_count > 2:
            issues.append(f"Multiple suspicious expert quotes detected ({suspicious_count})")
            recommendations.append("Verify expert quotes and consider removing unverified claims")
        
        if suspicious_count > 5:
            blocking_issues.append("Too many suspicious expert quotes - content credibility compromised")
        
        # Factual claim issues
        total_claims = factual_analysis.get('total_claims', 0)
        if total_claims > 5:
            warnings.append(f"High number of factual claims ({total_claims}) - consider fact-checking")
            recommendations.append("Verify all factual claims before publishing")
        
        # Quality metric issues
        for metric, score in quality_metrics.items():
            if score < 5.0:
                issues.append(f"Low {metric.replace('_', ' ')} score ({score:.1f})")
                recommendations.append(f"Improve {metric.replace('_', ' ')} aspects of content")
        
        return issues, warnings, recommendations, blocking_issues
    
    def _determine_content_status(self, issues: List[str], warnings: List[str],
                                blocking_issues: List[str], metrics: QualityMetrics) -> QualityStatus:
        """Determine overall content quality status."""
        if blocking_issues:
            return QualityStatus.FAILED
        
        if metrics.overall_score < 5.0:
            return QualityStatus.FAILED
        
        if len(issues) > 3 or metrics.overall_score < 7.0:
            return QualityStatus.NEEDS_REVIEW
        
        if warnings or len(issues) > 0:
            return QualityStatus.WARNING
        
        return QualityStatus.PASSED
    
    def _identify_content_strengths(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Identify content strengths based on quality metrics."""
        strengths = []
        
        for metric, score in quality_metrics.items():
            if score >= 8.0:
                strengths.append(f"Excellent {metric.replace('_', ' ')}")
            elif score >= 6.0:
                strengths.append(f"Good {metric.replace('_', ' ')}")
        
        return strengths
    
    # Helper methods for quality assessment
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _detect_concept_repetition(self, content: str) -> List[Dict[str, Any]]:
        """Detect repeated concepts in content."""
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = Counter(words)
        
        repeated_concepts = []
        for word, count in word_freq.items():
            if count > 3 and len(word) > 4:  # Significant repetition
                repeated_concepts.append({
                    'concept': word,
                    'frequency': count,
                    'positions': [i for i, w in enumerate(words) if w == word]
                })
        
        return repeated_concepts
    
    def _calculate_repetition_score(self, repetitive_sentences: int, concept_repetition: int,
                                  section_repetition: int, total_sentences: int) -> float:
        """Calculate overall repetition score."""
        if total_sentences == 0:
            return 0.0
        
        sentence_repetition = repetitive_sentences / total_sentences
        concept_penalty = min(concept_repetition * 0.1, 0.5)
        
        return min(sentence_repetition + concept_penalty, 1.0)
    
    def _calculate_quote_suspicion(self, quote: str) -> float:
        """Calculate suspicion score for a quote."""
        suspicion_score = 0.0
        
        # Check for generic patterns
        for pattern in self.generic_expert_quotes:
            if pattern.lower() in quote.lower():
                suspicion_score += 0.3
        
        # Check for suspicious expert patterns
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote, re.IGNORECASE):
                suspicion_score += 0.4
        
        # Check for overly generic language
        generic_phrases = ['important', 'critical', 'essential', 'key', 'significant']
        generic_count = sum(1 for phrase in generic_phrases if phrase in quote.lower())
        suspicion_score += generic_count * 0.1
        
        return min(suspicion_score, 1.0)
    
    def _get_suspicion_reasons(self, quote: str) -> List[str]:
        """Get reasons for quote suspicion."""
        reasons = []
        
        for pattern in self.generic_expert_quotes:
            if pattern.lower() in quote.lower():
                reasons.append("Contains generic expert language")
        
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote, re.IGNORECASE):
                reasons.append("Matches suspicious expert pattern")
        
        if len(quote.split()) < 10:
            reasons.append("Quote too short for meaningful attribution")
        
        return reasons
    
    # Quality assessment methods (simplified versions)
    def _assess_technical_accuracy(self, content: str) -> float:
        """Assess technical accuracy of content."""
        technical_terms = ['algorithm', 'neural network', 'machine learning', 'AI', 'API', 'database']
        term_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        return min(term_count * 2.0, 10.0)
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density of content."""
        words = content.split()
        if len(words) == 0:
            return 0.0
        
        # Simple heuristic: more unique words = higher density
        unique_words = len(set(words))
        return min(unique_words / len(words) * 10, 10.0)
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability of content."""
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            return 8.0
        elif 5 <= avg_sentence_length <= 30:
            return 6.0
        else:
            return 4.0
    
    def _assess_engagement_factors(self, content: str) -> float:
        """Assess engagement factors in content."""
        engagement_indicators = ['you', 'imagine', 'consider', 'think about', 'suppose']
        indicator_count = sum(1 for indicator in engagement_indicators 
                            if indicator.lower() in content.lower())
        return min(indicator_count * 2.0, 10.0)
    
    def _assess_structure_quality(self, content: str) -> float:
        """Assess structural quality of content."""
        # Check for headings, lists, paragraphs
        structure_indicators = ['#', '*', '-', '\n\n']
        indicator_count = sum(1 for indicator in structure_indicators 
                            if indicator in content)
        return min(indicator_count * 2.0, 10.0)
    
    def _assess_ai_ml_relevance(self, content: str) -> float:
        """Assess AI/ML relevance of content."""
        ai_ml_terms = ['AI', 'machine learning', 'neural network', 'algorithm', 'data']
        term_count = sum(1 for term in ai_ml_terms if term.lower() in content.lower())
        return min(term_count * 2.0, 10.0)
    
    def _assess_code_quality(self, content: str) -> float:
        """Assess code quality in content."""
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        return min(len(code_blocks) * 3.0, 10.0)
    
    def _assess_citation_quality(self, content: str) -> float:
        """Assess citation quality in content."""
        citation_patterns = [r'\[.*?\]', r'\(.*?\d{4}.*?\)', r'according to']
        citation_count = sum(len(re.findall(pattern, content)) for pattern in citation_patterns)
        return min(citation_count * 2.0, 10.0)
    
    def _assess_practical_value(self, content: str) -> float:
        """Assess practical value of content."""
        practical_indicators = ['example', 'step', 'guide', 'tutorial', 'how to']
        indicator_count = sum(1 for indicator in practical_indicators 
                            if indicator.lower() in content.lower())
        return min(indicator_count * 2.0, 10.0)
    
    def _assess_innovation_factor(self, content: str) -> float:
        """Assess innovation factor of content."""
        innovation_terms = ['new', 'innovative', 'breakthrough', 'revolutionary', 'cutting-edge']
        term_count = sum(1 for term in innovation_terms if term.lower() in content.lower())
        return min(term_count * 2.0, 10.0)
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()