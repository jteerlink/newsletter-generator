"""
Content validator for newsletter quality assessment.

This module provides content validation functionality for the quality package.
"""

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ContentQualityValidator:
    """Content quality validator for newsletter assessment."""

    def __init__(self):
        self.name = "ContentQualityValidator"
        self.min_length = 100
        self.max_length = 10000
        self.quality_thresholds = {
            'min_length': 100,
            'max_repetition': 0.3,
            'min_readability': 0.6
        }

        # Suspicious expert quote patterns
        self.suspicious_expert_patterns = [
            r'Dr\.\s+\w+\s+from\s+\w+',
            r'Professor\s+\w+\s+at\s+\w+',
            r'According\s+to\s+experts',
            r'Research\s+shows',
            r'Studies\s+indicate'
        ]

        # Generic expert quotes that are suspicious
        self.generic_expert_quotes = [
            "The future of AI depends on responsible development",
            "Machine learning is critical for success",
            "AI will transform the industry",
            "Technology is advancing rapidly",
            "Innovation is key to success"
        ]

        # Factual claim patterns
        self.factual_claim_patterns = [
            r'\d+%\s+of\s+\w+',
            r'According\s+to\s+recent\s+survey',
            r'Studies\s+show\s+that',
            r'Research\s+indicates\s+that',
            r'Statistics\s+show',
            r'\d+\s+out\s+of\s+\d+',
            r'More\s+than\s+\d+%'
        ]

    def validate_content(self, content: str) -> Dict[str, Any]:
        """
        Validate newsletter content and return quality metrics.

        Args:
            content: The content to validate

        Returns:
            Dictionary with validation results
        """
        if not content:
            return self._create_empty_result()

        # Extract text content
        text_content = self._extract_text_content(content)

        # Perform various analyses
        repetition_analysis = self._detect_repetition(text_content)
        expert_analysis = self._analyze_expert_quotes(text_content)
        factual_analysis = self._analyze_factual_claims(text_content)
        quality_metrics = self._assess_content_quality(text_content)

        # Calculate overall metrics
        overall_metrics = self._calculate_content_metrics(
            quality_metrics, repetition_analysis, expert_analysis, factual_analysis)

        # Generate issues and recommendations
        issues, warnings, recommendations, blocking_issues = self._generate_content_issues(
            repetition_analysis, expert_analysis, factual_analysis, quality_metrics)

        # Determine status
        status = self._determine_content_status(
            issues, warnings, blocking_issues, overall_metrics)

        return {
            'quality_score': overall_metrics.get('overall_score', 0.0),
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'blocking_issues': blocking_issues,
            'status': status,
            'repetition_analysis': repetition_analysis,
            'expert_quote_analysis': expert_analysis,
            'fact_check_analysis': factual_analysis,
            'quality_metrics': quality_metrics
        }

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from various input formats."""
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            # Try common content keys
            for key in ['content', 'body', 'text', 'article']:
                if key in content:
                    return str(content[key])

            # Fallback to first string value
            for value in content.values():
                if isinstance(value, str) and len(value) > 10:
                    return value

        return str(content)

    def _detect_repetition(self, content: str) -> Dict[str, Any]:
        """Detect repetition in content."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                'repetition_score': 0.0,
                'repetitive_sentences': [],
                'sentence_groups': [],
                'concept_repetition': {},
                'total_sentences': 0,
                'repetitive_count': 0
            }

        # Find similar sentences
        similar_groups = []
        processed = set()

        for i, sentence1 in enumerate(sentences):
            if i in processed:
                continue

            similar_group = [i]
            for j, sentence2 in enumerate(sentences[i + 1:], i + 1):
                if j in processed:
                    continue

                similarity = self._calculate_similarity(sentence1, sentence2)
                if similarity > 0.8:  # High similarity threshold
                    similar_group.append(j)
                    processed.add(j)

            if len(similar_group) > 1:
                similar_groups.append(similar_group)
                processed.update(similar_group)

        # Calculate repetition score
        repetitive_count = sum(len(group) for group in similar_groups)
        repetition_score = repetitive_count / \
            len(sentences) if sentences else 0.0

        # Detect concept repetition
        concept_repetition = self._detect_concept_repetition(content)

        return {
            'repetition_score': repetition_score,
            'repetitive_sentences': [
                sentences[i] for group in similar_groups for i in group],
            'sentence_groups': similar_groups,
            'concept_repetition': concept_repetition,
            'total_sentences': len(sentences),
            'repetitive_count': repetitive_count}

    def _analyze_expert_quotes(self, content: str) -> Dict[str, Any]:
        """Analyze expert quotes for suspicious patterns."""
        quotes = re.findall(r'"([^"]*)"', content)

        suspicious_quotes = []
        for quote in quotes:
            suspicion_score = self._calculate_quote_suspicion(quote)
            if suspicion_score > 0.5:
                suspicious_quotes.append({
                    'quote': quote,
                    'suspicion_score': suspicion_score,
                    'reasons': self._get_suspicion_reasons(quote)
                })

        return {
            'total_quotes': len(quotes),
            'quotes': quotes,
            'has_expert_quotes': len(quotes) > 0,
            'suspicious_quotes': suspicious_quotes,
            'suspicion_score': sum(
                q['suspicion_score'] for q in suspicious_quotes) /
            len(suspicious_quotes) if suspicious_quotes else 0.0,
            'average_suspicion': sum(
                q['suspicion_score'] for q in suspicious_quotes) /
            len(suspicious_quotes) if suspicious_quotes else 0.0}

    def _analyze_factual_claims(self, content: str) -> Dict[str, Any]:
        """Analyze factual claims in content."""
        claims = []
        for pattern in self.factual_claim_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches)

        return {
            'fact_indicators': claims,
            'needs_fact_checking': len(claims) > 0,
            'fact_check_score': len(claims) / len(self.factual_claim_patterns),
            'total_claims': len(claims),
            'claim_density': len(claims) / max(len(content.split()), 1)
        }

    def _assess_content_quality(self, content: str) -> Dict[str, float]:
        """Assess various quality aspects of content."""
        return {
            'technical_accuracy': self._assess_technical_accuracy(content),
            'information_density': self._assess_information_density(content),
            'readability': self._assess_readability(content),
            'engagement': self._assess_engagement(content),
            'structure': self._assess_structure(content),
            'ai_ml_relevance': self._assess_ai_ml_relevance(content),
            'code_quality': self._assess_code_quality(content),
            'citation_quality': self._assess_citation_quality(content),
            'practical_value': self._assess_practical_value(content),
            'innovation_factor': self._assess_innovation_factor(content)
        }

    def _calculate_content_metrics(self,
                                   quality_metrics: Dict[str,
                                                         float],
                                   repetition_analysis: Dict[str,
                                                             Any],
                                   expert_analysis: Dict[str,
                                                         Any],
                                   factual_analysis: Dict[str,
                                                          Any]) -> Dict[str,
                                                                        float]:
        """Calculate overall content metrics."""
        # Weighted average of quality metrics
        weights = {
            'technical_accuracy': 0.15,
            'information_density': 0.10,
            'readability': 0.15,
            'engagement': 0.10,
            'structure': 0.10,
            'ai_ml_relevance': 0.15,
            'code_quality': 0.05,
            'citation_quality': 0.05,
            'practical_value': 0.10,
            'innovation_factor': 0.05
        }

        overall_score = sum(
            quality_metrics.get(
                key,
                0.0) *
            weights.get(
                key,
                0.0) for key in weights)

        # Adjust for repetition and suspicious quotes
        repetition_penalty = repetition_analysis.get(
            'repetition_score', 0.0) * 0.2
        suspicion_penalty = expert_analysis.get('suspicion_score', 0.0) * 0.1

        overall_score = max(
            0.0,
            overall_score -
            repetition_penalty -
            suspicion_penalty)

        return {
            'overall_score': overall_score,
            'repetition_penalty': repetition_penalty,
            'suspicion_penalty': suspicion_penalty
        }

    def _generate_content_issues(self,
                                 repetition_analysis: Dict[str,
                                                           Any],
                                 expert_analysis: Dict[str,
                                                       Any],
                                 factual_analysis: Dict[str,
                                                        Any],
                                 quality_metrics: Dict[str,
                                                       float]) -> Tuple[List[str],
                                                                        List[str],
                                                                        List[str],
                                                                        List[str]]:
        """Generate content issues, warnings, recommendations, and blocking issues."""
        issues = []
        warnings = []
        recommendations = []
        blocking_issues = []

        # Repetition issues
        if repetition_analysis.get('repetition_score', 0.0) > 0.3:
            issues.append("Content has too much repetition")
            recommendations.append("Vary sentence structure and use synonyms")

        # Suspicious quote issues
        if expert_analysis.get('suspicion_score', 0.0) > 0.5:
            warnings.append("Some quotes may be generic or suspicious")
            recommendations.append("Use more specific and credible quotes")

        # Quality metric issues
        for metric, score in quality_metrics.items():
            if score < 5.0:
                issues.append(f"Low {metric.replace('_', ' ')} score")
                recommendations.append(f"Improve {metric.replace('_', ' ')}")

        # Blocking issues for very low scores
        if quality_metrics.get('readability', 10.0) < 3.0:
            blocking_issues.append("Content is not readable")

        if quality_metrics.get('technical_accuracy', 10.0) < 3.0:
            blocking_issues.append("Content has technical inaccuracies")

        return issues, warnings, recommendations, blocking_issues

    def _determine_content_status(self,
                                  issues: List[str],
                                  warnings: List[str],
                                  blocking_issues: List[str],
                                  metrics: Dict[str,
                                                float]) -> str:
        """Determine overall content status."""
        if blocking_issues:
            return "BLOCKED"
        elif issues:
            return "NEEDS_REVISION"
        elif warnings:
            return "WARNING"
        else:
            return "PASSED"

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _detect_concept_repetition(self, content: str) -> Dict[str, int]:
        """Detect repetition of concepts/terms."""
        words = re.findall(r'\b\w+\b', content.lower())
        word_counts = {}

        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_counts[word] = word_counts.get(word, 0) + 1

        # Return words that appear more than twice
        return {
            word: count for word,
            count in word_counts.items() if count > 2}

    def _calculate_quote_suspicion(self, quote: str) -> float:
        """Calculate suspicion score for a quote."""
        suspicion_score = 0.0

        # Check for generic patterns
        for generic_quote in self.generic_expert_quotes:
            similarity = self._calculate_similarity(quote, generic_quote)
            suspicion_score = max(suspicion_score, similarity)

        # Check for suspicious patterns
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote, re.IGNORECASE):
                suspicion_score += 0.3

        return min(1.0, suspicion_score)

    def _get_suspicion_reasons(self, quote: str) -> List[str]:
        """Get reasons why a quote is suspicious."""
        reasons = []

        # Check for generic patterns
        for generic_quote in self.generic_expert_quotes:
            if self._calculate_similarity(quote, generic_quote) > 0.7:
                reasons.append("Similar to generic expert quote")

        # Check for suspicious patterns
        for pattern in self.suspicious_expert_patterns:
            if re.search(pattern, quote, re.IGNORECASE):
                reasons.append("Matches suspicious expert pattern")

        if not reasons:
            reasons.append("Generic or non-specific content")

        return reasons

    # Quality assessment methods
    def _assess_technical_accuracy(self, content: str) -> float:
        """Assess technical accuracy of content."""
        # Simple heuristic based on technical terms
        technical_terms = [
            'algorithm',
            'neural network',
            'machine learning',
            'AI',
            'data',
            'model']
        term_count = sum(
            1 for term in technical_terms if term.lower() in content.lower())
        return min(10.0, term_count * 2.0)

    def _assess_information_density(self, content: str) -> float:
        """Assess information density of content."""
        words = content.split()
        if not words:
            return 0.0
        return min(10.0, len(words) / 50.0)  # Normalize to 10-point scale

    def _assess_readability(self, content: str) -> float:
        """Assess readability of content."""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        avg_sentence_length = sum(len(s.split())
                                  for s in sentences) / len(sentences)

        if avg_sentence_length <= 15:
            return 10.0
        elif avg_sentence_length <= 25:
            return 8.0
        elif avg_sentence_length <= 35:
            return 6.0
        else:
            return 4.0

    def _assess_engagement(self, content: str) -> float:
        """Assess engagement level of content."""
        engagement_indicators = ['?', '!', '"', 'http', 'www']
        indicator_count = sum(
            1 for indicator in engagement_indicators if indicator in content)
        return min(10.0, indicator_count * 2.0)

    def _assess_structure(self, content: str) -> float:
        """Assess content structure."""
        structure_elements = content.count(
            '#') + content.count('\n\n') + content.count('- ')
        return min(10.0, structure_elements * 1.5)

    def _assess_ai_ml_relevance(self, content: str) -> float:
        """Assess AI/ML relevance of content."""
        ai_ml_terms = [
            'AI',
            'ML',
            'machine learning',
            'neural network',
            'algorithm',
            'data science']
        term_count = sum(
            1 for term in ai_ml_terms if term.lower() in content.lower())
        return min(10.0, term_count * 2.0)

    def _assess_code_quality(self, content: str) -> float:
        """Assess code quality in content."""
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        return min(10.0, code_blocks * 3.0)

    def _assess_citation_quality(self, content: str) -> float:
        """Assess citation quality."""
        citations = len(
            re.findall(
                r'according to|study shows|research indicates',
                content,
                re.IGNORECASE))
        return min(10.0, citations * 2.0)

    def _assess_practical_value(self, content: str) -> float:
        """Assess practical value of content."""
        practical_indicators = [
            'how to',
            'step by step',
            'example',
            'tutorial',
            'guide']
        indicator_count = sum(
            1 for indicator in practical_indicators if indicator.lower() in content.lower())
        return min(10.0, indicator_count * 3.0)

    def _assess_innovation_factor(self, content: str) -> float:
        """Assess innovation factor of content."""
        innovation_terms = [
            'breakthrough',
            'innovation',
            'revolutionary',
            'cutting-edge',
            'novel']
        term_count = sum(
            1 for term in innovation_terms if term.lower() in content.lower())
        return min(10.0, term_count * 2.0)

    def validate(self, content: str) -> Dict[str, Any]:
        """Main validation method."""
        return self.validate_content(content)

    def get_metrics(self, content: str) -> Dict[str, float]:
        """Get quality metrics for content."""
        result = self.validate_content(content)
        return result.get('quality_metrics', {})

    def validate_batch(self, contents: List[str]) -> List[Dict[str, Any]]:
        """Validate multiple content items."""
        return [self.validate_content(content) for content in contents]

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty content."""
        return {
            'quality_score': 0.0,
            'issues': ['Content is empty'],
            'warnings': [],
            'recommendations': ['Add content to the newsletter'],
            'blocking_issues': ['Content is empty'],
            'status': 'BLOCKED',
            'repetition_analysis': {
                'repetition_score': 0.0,
                'repetitive_sentences': [],
                'sentence_groups': [],
                'concept_repetition': {},
                'total_sentences': 0,
                'repetitive_count': 0},
            'expert_quote_analysis': {
                'quote_count': 0,
                'quotes': [],
                'has_expert_quotes': False,
                'suspicious_quotes': [],
                'suspicion_score': 0.0},
            'fact_check_analysis': {
                'fact_indicators': [],
                'needs_fact_checking': False,
                'fact_check_score': 0.0,
                'total_claims': 0,
                'claim_density': 0.0},
            'quality_metrics': {}}
