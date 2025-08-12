"""
Section-Level Quality Metrics System

This module implements granular quality assessment per section as specified 
in Phase 1 FR1.3 of the Multi-Agent Enhancement PRD.

Features:
- Track quality metrics for each newsletter section
- Aggregate section metrics into overall quality score
- Support section-specific quality thresholds
- Detailed quality breakdown in reports
- Integration with existing QualityAssuranceSystem
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .section_aware_prompts import SectionType

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    ENGAGEMENT = "engagement"
    STRUCTURE = "structure"
    CONSISTENCY = "consistency"
    READABILITY = "readability"


@dataclass
class QualityMetric:
    """Individual quality metric with score and details."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    details: str = ""
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SectionQualityMetrics:
    """Quality metrics for a specific newsletter section."""
    section_type: SectionType
    section_content: str
    overall_score: float
    metrics: Dict[QualityDimension, QualityMetric] = field(default_factory=dict)
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    readability_score: float = 0.0
    technical_accuracy_score: float = 0.0
    engagement_score: float = 0.0
    timestamp: Optional[str] = None
    
    def get_weighted_score(self) -> float:
        """Calculate weighted overall score from individual metrics."""
        if not self.metrics:
            return self.overall_score
        
        total_weight = sum(metric.weight for metric in self.metrics.values())
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            metric.score * metric.weight 
            for metric in self.metrics.values()
        )
        
        return weighted_sum / total_weight
    
    def get_issues(self) -> List[str]:
        """Get all issues from all metrics."""
        all_issues = []
        for metric in self.metrics.values():
            all_issues.extend(metric.issues)
        return all_issues
    
    def get_suggestions(self) -> List[str]:
        """Get all suggestions from all metrics."""
        all_suggestions = []
        for metric in self.metrics.values():
            all_suggestions.extend(metric.suggestions)
        return all_suggestions


@dataclass
class AggregatedQualityReport:
    """Aggregated quality report for entire newsletter."""
    overall_score: float
    section_scores: Dict[SectionType, SectionQualityMetrics]
    total_word_count: int
    average_readability: float
    consistency_score: float
    flow_score: float
    completion_time: float = 0.0
    
    def get_section_breakdown(self) -> Dict[str, float]:
        """Get section scores breakdown."""
        return {
            section_type.value: metrics.overall_score
            for section_type, metrics in self.section_scores.items()
        }
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary for reporting."""
        return {
            'overall_score': self.overall_score,
            'total_sections': len(self.section_scores),
            'word_count': self.total_word_count,
            'readability': self.average_readability,
            'consistency': self.consistency_score,
            'flow': self.flow_score,
            'section_breakdown': self.get_section_breakdown()
        }


class SectionQualityAnalyzer:
    """Analyzes quality metrics for individual newsletter sections."""
    
    def __init__(self):
        """Initialize section quality analyzer."""
        # Section-specific quality weights
        self.section_weights = {
            SectionType.INTRODUCTION: {
                QualityDimension.ENGAGEMENT: 2.0,
                QualityDimension.CLARITY: 1.5,
                QualityDimension.RELEVANCE: 1.5,
                QualityDimension.STRUCTURE: 1.0,
            },
            SectionType.ANALYSIS: {
                QualityDimension.ACCURACY: 2.0,
                QualityDimension.COMPLETENESS: 1.8,
                QualityDimension.CLARITY: 1.5,
                QualityDimension.STRUCTURE: 1.2,
            },
            SectionType.TUTORIAL: {
                QualityDimension.CLARITY: 2.0,
                QualityDimension.COMPLETENESS: 1.8,
                QualityDimension.STRUCTURE: 1.5,
                QualityDimension.ACCURACY: 1.5,
            },
            SectionType.NEWS: {
                QualityDimension.ACCURACY: 2.0,
                QualityDimension.RELEVANCE: 1.8,
                QualityDimension.CLARITY: 1.5,
                QualityDimension.ENGAGEMENT: 1.2,
            },
            SectionType.CONCLUSION: {
                QualityDimension.COMPLETENESS: 1.8,
                QualityDimension.ENGAGEMENT: 1.5,
                QualityDimension.CLARITY: 1.2,
                QualityDimension.RELEVANCE: 1.0,
            }
        }
        
        # Default weights for unknown sections
        self.default_weights = {dim: 1.0 for dim in QualityDimension}
        
        logger.info("Section quality analyzer initialized")

    def analyze_section(self, content: str, section_type: SectionType,
                       context: Optional[Dict[str, Any]] = None) -> SectionQualityMetrics:
        """
        Analyze quality metrics for a newsletter section.
        
        Args:
            content: Section content to analyze
            section_type: Type of section being analyzed
            context: Optional context for analysis
            
        Returns:
            SectionQualityMetrics: Detailed quality analysis
        """
        context = context or {}
        
        # Basic content statistics
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Get section-specific weights
        weights = self.section_weights.get(section_type, self.default_weights)
        
        # Analyze individual quality dimensions
        metrics = {}
        for dimension in QualityDimension:
            weight = weights.get(dimension, 1.0)
            metric = self._analyze_dimension(content, dimension, section_type, context)
            metric.weight = weight
            metrics[dimension] = metric
        
        # Calculate overall scores
        readability_score = self._calculate_readability(content)
        technical_accuracy = self._assess_technical_accuracy(content, section_type)
        engagement_score = self._assess_engagement(content, section_type)
        
        # Create section quality metrics
        section_metrics = SectionQualityMetrics(
            section_type=section_type,
            section_content=content,
            overall_score=0.0,  # Will be calculated
            metrics=metrics,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            readability_score=readability_score,
            technical_accuracy_score=technical_accuracy,
            engagement_score=engagement_score
        )
        
        # Calculate weighted overall score
        section_metrics.overall_score = section_metrics.get_weighted_score()
        
        logger.debug("Analyzed section %s: score %.3f (%d words)",
                    section_type.value, section_metrics.overall_score, word_count)
        
        return section_metrics

    def _analyze_dimension(self, content: str, dimension: QualityDimension,
                          section_type: SectionType, context: Dict[str, Any]) -> QualityMetric:
        """Analyze a specific quality dimension."""
        if dimension == QualityDimension.CLARITY:
            return self._assess_clarity(content, section_type)
        elif dimension == QualityDimension.RELEVANCE:
            return self._assess_relevance(content, section_type, context)
        elif dimension == QualityDimension.COMPLETENESS:
            return self._assess_completeness(content, section_type)
        elif dimension == QualityDimension.ACCURACY:
            return self._assess_accuracy(content, section_type)
        elif dimension == QualityDimension.ENGAGEMENT:
            return self._assess_engagement_dimension(content, section_type)
        elif dimension == QualityDimension.STRUCTURE:
            return self._assess_structure(content, section_type)
        elif dimension == QualityDimension.CONSISTENCY:
            return self._assess_consistency(content, section_type)
        elif dimension == QualityDimension.READABILITY:
            return self._assess_readability_dimension(content, section_type)
        else:
            return QualityMetric(dimension=dimension, score=0.5)

    def _assess_clarity(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess content clarity."""
        score = 0.0
        issues = []
        suggestions = []
        
        # Check sentence length
        sentences = re.findall(r'[^.!?]*[.!?]', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_sentence_length <= 20:
            score += 0.3
        elif avg_sentence_length <= 30:
            score += 0.2
        else:
            issues.append("Average sentence length too long")
            suggestions.append("Break down long sentences for better clarity")
        
        # Check for clarity indicators
        clear_indicators = ['for example', 'specifically', 'in other words', 'that is']
        if any(indicator in content.lower() for indicator in clear_indicators):
            score += 0.2
        
        # Check for jargon without explanation
        technical_terms = re.findall(r'\b[A-Z]{2,}\b', content)
        if len(technical_terms) > 5:
            issues.append("Many technical terms without explanation")
            suggestions.append("Define technical terms for broader accessibility")
        else:
            score += 0.2
        
        # Check paragraph structure
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if all(len(p.split()) <= 100 for p in paragraphs):
            score += 0.3
        else:
            issues.append("Some paragraphs are too long")
            suggestions.append("Break long paragraphs into smaller chunks")
        
        return QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=min(1.0, score),
            details=f"Average sentence length: {avg_sentence_length:.1f} words",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_relevance(self, content: str, section_type: SectionType,
                         context: Dict[str, Any]) -> QualityMetric:
        """Assess content relevance to topic and audience."""
        score = 0.0
        issues = []
        suggestions = []
        
        topic = context.get('topic', '').lower()
        audience = context.get('audience', '').lower()
        content_lower = content.lower()
        
        # Check topic relevance
        if topic and any(word in content_lower for word in topic.split()):
            score += 0.4
        else:
            issues.append("Content doesn't clearly relate to stated topic")
            suggestions.append("Ensure content directly addresses the main topic")
        
        # Check audience appropriateness
        audience_indicators = {
            'engineers': ['implementation', 'code', 'technical', 'architecture'],
            'scientists': ['analysis', 'data', 'research', 'methodology'],
            'developers': ['development', 'programming', 'tools', 'frameworks'],
            'business': ['impact', 'strategy', 'roi', 'implementation'],
            'general': ['overview', 'introduction', 'basics', 'examples']
        }
        
        for aud_type, indicators in audience_indicators.items():
            if aud_type in audience:
                if any(indicator in content_lower for indicator in indicators):
                    score += 0.3
                    break
        else:
            suggestions.append("Consider adding content more relevant to target audience")
        
        # Section-specific relevance
        if section_type == SectionType.NEWS:
            recent_indicators = ['recent', 'latest', 'new', 'announced', 'released']
            if any(indicator in content_lower for indicator in recent_indicators):
                score += 0.3
            else:
                issues.append("News section lacks recent/current indicators")
        
        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=min(1.0, score),
            details="Topic and audience relevance assessment",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_completeness(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess content completeness for section type."""
        score = 0.0
        issues = []
        suggestions = []
        
        word_count = len(content.split())
        
        # Section-specific completeness requirements
        if section_type == SectionType.INTRODUCTION:
            required_elements = ['hook', 'overview', 'preview']
            expected_length = (100, 300)
        elif section_type == SectionType.ANALYSIS:
            required_elements = ['data', 'insights', 'implications']
            expected_length = (400, 800)
        elif section_type == SectionType.TUTORIAL:
            required_elements = ['steps', 'examples', 'instructions']
            expected_length = (300, 600)
        elif section_type == SectionType.NEWS:
            required_elements = ['updates', 'details', 'sources']
            expected_length = (200, 400)
        elif section_type == SectionType.CONCLUSION:
            required_elements = ['summary', 'takeaways', 'action']
            expected_length = (100, 250)
        else:
            required_elements = []
            expected_length = (200, 500)
        
        # Check word count appropriateness
        min_words, max_words = expected_length
        if min_words <= word_count <= max_words:
            score += 0.4
        elif word_count < min_words:
            issues.append(f"Content too short ({word_count} words, expected {min_words}-{max_words})")
            suggestions.append("Expand content to provide more comprehensive coverage")
        else:
            issues.append(f"Content too long ({word_count} words, expected {min_words}-{max_words})")
            suggestions.append("Consider condensing content or splitting into subsections")
        
        # Check for required elements (simplified heuristic)
        content_lower = content.lower()
        elements_found = 0
        for element in required_elements:
            element_indicators = {
                'hook': ['?', 'imagine', 'consider', 'what if'],
                'overview': ['overview', 'introduce', 'cover', 'discuss'],
                'preview': ['will explore', 'will examine', 'will discuss'],
                'data': ['data', 'statistics', 'numbers', 'research'],
                'insights': ['insight', 'finding', 'discovery', 'analysis'],
                'implications': ['means', 'impact', 'result', 'consequence'],
                'steps': ['step', '1.', 'first', 'next', 'then'],
                'examples': ['example', 'instance', 'case', 'demonstration'],
                'instructions': ['how to', 'procedure', 'method', 'process'],
                'updates': ['update', 'news', 'announcement', 'release'],
                'details': ['detail', 'specific', 'particular', 'information'],
                'sources': ['source', 'according', 'reported', 'stated'],
                'summary': ['summary', 'conclusion', 'recap', 'overall'],
                'takeaways': ['takeaway', 'key point', 'important', 'remember'],
                'action': ['try', 'implement', 'apply', 'use', 'start']
            }
            
            indicators = element_indicators.get(element, [element])
            if any(indicator in content_lower for indicator in indicators):
                elements_found += 1
        
        if required_elements:
            element_score = elements_found / len(required_elements)
            score += element_score * 0.6
            
            if element_score < 0.5:
                missing = len(required_elements) - elements_found
                issues.append(f"Missing {missing} key elements for {section_type.value} section")
                suggestions.append(f"Include essential elements: {', '.join(required_elements)}")
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=min(1.0, score),
            details=f"Word count: {word_count}, Elements found: {elements_found}/{len(required_elements)}",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_accuracy(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess content accuracy (basic heuristics)."""
        score = 0.8  # Default assumption of accuracy
        issues = []
        suggestions = []
        
        # Check for claims without support
        claim_indicators = ['studies show', 'research indicates', 'proven', 'definitely']
        unsupported_claims = 0
        for indicator in claim_indicators:
            if indicator in content.lower():
                # Look for nearby citations or sources
                indicator_pos = content.lower().find(indicator)
                nearby_text = content[max(0, indicator_pos-50):indicator_pos+150].lower()
                if not any(source in nearby_text for source in ['source', 'study', 'research', 'according']):
                    unsupported_claims += 1
        
        if unsupported_claims > 0:
            score -= unsupported_claims * 0.1
            issues.append(f"{unsupported_claims} claims lack supporting evidence")
            suggestions.append("Provide sources or citations for factual claims")
        
        # Check for outdated information indicators
        outdated_indicators = ['last year', 'recently', 'current', 'now']
        if any(indicator in content.lower() for indicator in outdated_indicators):
            if not any(year in content for year in ['2023', '2024', '2025']):
                score -= 0.1
                issues.append("Temporal references may be outdated")
                suggestions.append("Include specific dates for time-sensitive information")
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=max(0.0, score),
            details="Basic accuracy assessment using heuristics",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_engagement_dimension(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess content engagement level."""
        score = 0.0
        issues = []
        suggestions = []
        
        # Check for engaging elements
        engagement_indicators = {
            'questions': len(re.findall(r'\?', content)),
            'examples': len(re.findall(r'(?i)\bfor example\b|\bfor instance\b', content)),
            'personal_pronouns': len(re.findall(r'(?i)\byou\b|\byour\b', content)),
            'action_words': len(re.findall(r'(?i)\b(discover|explore|learn|master|achieve)\b', content))
        }
        
        # Score based on engagement elements
        if engagement_indicators['questions'] > 0:
            score += 0.2
        if engagement_indicators['examples'] > 0:
            score += 0.3
        if engagement_indicators['personal_pronouns'] > 2:
            score += 0.2
        if engagement_indicators['action_words'] > 0:
            score += 0.2
        
        # Check for variety in sentence structure
        sentences = re.findall(r'[^.!?]*[.!?]', content)
        sentence_starts = [s.strip()[:10] for s in sentences if s.strip()]
        unique_starts = len(set(sentence_starts))
        if unique_starts > len(sentence_starts) * 0.7:
            score += 0.1
        else:
            suggestions.append("Vary sentence structures for better engagement")
        
        if score < 0.5:
            issues.append("Content lacks engaging elements")
            suggestions.append("Add questions, examples, or direct address to readers")
        
        return QualityMetric(
            dimension=QualityDimension.ENGAGEMENT,
            score=min(1.0, score),
            details=f"Questions: {engagement_indicators['questions']}, Examples: {engagement_indicators['examples']}",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_structure(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess content structure and organization."""
        score = 0.0
        issues = []
        suggestions = []
        
        # Check for headers/subheadings
        headers = re.findall(r'^#+\s+.+$', content, re.MULTILINE)
        if headers:
            score += 0.3
        elif len(content.split()) > 200:
            issues.append("Long content lacks structural headers")
            suggestions.append("Add headers to break up long sections")
        
        # Check for lists or bullet points
        lists = re.findall(r'^[\-\*\+]\s+.+$', content, re.MULTILINE)
        numbered_lists = re.findall(r'^\d+\.\s+.+$', content, re.MULTILINE)
        if lists or numbered_lists:
            score += 0.2
        
        # Check paragraph structure
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if 2 <= len(paragraphs) <= 10:
            score += 0.3
        elif len(paragraphs) == 1 and len(content.split()) > 100:
            issues.append("Content should be broken into multiple paragraphs")
            suggestions.append("Organize content into logical paragraphs")
        
        # Section-specific structure checks
        if section_type == SectionType.TUTORIAL:
            if numbered_lists:
                score += 0.2
            else:
                issues.append("Tutorial lacks numbered steps")
                suggestions.append("Use numbered lists for step-by-step instructions")
        
        return QualityMetric(
            dimension=QualityDimension.STRUCTURE,
            score=min(1.0, score),
            details=f"Headers: {len(headers)}, Paragraphs: {len(paragraphs)}",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_consistency(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess internal consistency."""
        score = 0.8  # Default assumption of consistency
        issues = []
        suggestions = []
        
        # Check for consistent terminology
        # This is a simplified check - in practice would be more sophisticated
        technical_terms = re.findall(r'\b[A-Z][a-z]*[A-Z][a-z]*\b', content)
        if len(set(technical_terms)) != len(technical_terms):
            # Some terms are repeated, which is good for consistency
            score += 0.2
        
        # Check for consistent tone (basic heuristic)
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'thus']
        informal_indicators = ['so', 'well', 'basically', 'just']
        
        formal_count = sum(1 for ind in formal_indicators if ind in content.lower())
        informal_count = sum(1 for ind in informal_indicators if ind in content.lower())
        
        if formal_count > 0 and informal_count > 0:
            score -= 0.1
            issues.append("Mixed formal and informal tone")
            suggestions.append("Maintain consistent tone throughout")
        
        return QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=min(1.0, score),
            details="Basic consistency assessment",
            issues=issues,
            suggestions=suggestions
        )

    def _assess_readability_dimension(self, content: str, section_type: SectionType) -> QualityMetric:
        """Assess readability as a quality dimension."""
        readability_score = self._calculate_readability(content)
        
        issues = []
        suggestions = []
        
        if readability_score < 0.6:
            issues.append("Content may be difficult to read")
            suggestions.append("Simplify language and sentence structure")
        
        return QualityMetric(
            dimension=QualityDimension.READABILITY,
            score=readability_score,
            details=f"Readability score: {readability_score:.2f}",
            issues=issues,
            suggestions=suggestions
        )

    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using simplified Flesch-Kincaid."""
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        syllables = self._count_syllables(content)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Convert to 0-1 scale (0 = very difficult, 1 = very easy)
        # Flesch scores typically range from 0-100
        normalized_score = max(0.0, min(1.0, flesch_score / 100.0))
        
        return normalized_score

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified heuristic)."""
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = 0
        
        for word in words:
            # Simple syllable counting heuristic
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            # Minimum of 1 syllable per word
            total_syllables += max(1, syllable_count)
        
        return total_syllables

    def _assess_technical_accuracy(self, content: str, section_type: SectionType) -> float:
        """Assess technical accuracy (placeholder for more sophisticated analysis)."""
        # This would integrate with external fact-checking APIs in a real implementation
        # For now, return a base score based on content indicators
        
        accuracy_indicators = ['according to', 'research shows', 'study found', 'data indicates']
        if any(indicator in content.lower() for indicator in accuracy_indicators):
            return 0.85
        else:
            return 0.75

    def _assess_engagement(self, content: str, section_type: SectionType) -> float:
        """Assess engagement level."""
        return self._assess_engagement_dimension(content, section_type).score


class SectionAwareQualitySystem:
    """
    Enhanced quality assurance system with section-aware capabilities.
    
    Extends existing QualityAssuranceSystem with section-level quality tracking,
    aggregation, and reporting as specified in FR1.3.
    """
    
    def __init__(self, quality_thresholds: Optional[Dict[SectionType, float]] = None):
        """
        Initialize section-aware quality system.
        
        Args:
            quality_thresholds: Section-specific quality thresholds
        """
        self.analyzer = SectionQualityAnalyzer()
        
        # Default quality thresholds per section type
        self.quality_thresholds = quality_thresholds or {
            SectionType.INTRODUCTION: 0.8,
            SectionType.ANALYSIS: 0.85,
            SectionType.TUTORIAL: 0.8,
            SectionType.NEWS: 0.75,
            SectionType.CONCLUSION: 0.7
        }
        
        logger.info("Section-aware quality system initialized")

    def analyze_newsletter_quality(self, newsletter_content: str,
                                  section_boundaries: Optional[List[Tuple[SectionType, int, int]]] = None,
                                  context: Optional[Dict[str, Any]] = None) -> AggregatedQualityReport:
        """
        Analyze quality of complete newsletter with section breakdown.
        
        Args:
            newsletter_content: Complete newsletter content
            section_boundaries: Optional section boundary definitions
            context: Context for quality analysis
            
        Returns:
            AggregatedQualityReport: Comprehensive quality analysis
        """
        context = context or {}
        
        # If no boundaries provided, detect them
        if not section_boundaries:
            from .section_aware_refinement import SectionBoundaryDetector
            detector = SectionBoundaryDetector()
            boundaries = detector.detect_boundaries(newsletter_content)
            section_boundaries = [
                (b.section_type, b.start_index, b.end_index) 
                for b in boundaries
            ]
        
        # Analyze each section
        section_scores = {}
        total_word_count = 0
        readability_scores = []
        
        for section_type, start_idx, end_idx in section_boundaries:
            section_content = newsletter_content[start_idx:end_idx].strip()
            
            if len(section_content) > 10:  # Skip very short sections
                section_metrics = self.analyzer.analyze_section(
                    section_content, section_type, context
                )
                section_scores[section_type] = section_metrics
                total_word_count += section_metrics.word_count
                readability_scores.append(section_metrics.readability_score)
        
        # Calculate aggregate metrics
        if section_scores:
            overall_score = sum(metrics.overall_score for metrics in section_scores.values()) / len(section_scores)
            average_readability = sum(readability_scores) / len(readability_scores)
        else:
            overall_score = 0.0
            average_readability = 0.0
        
        # Calculate consistency score across sections
        consistency_score = self._calculate_cross_section_consistency(section_scores)
        
        # Calculate narrative flow score
        flow_score = self._calculate_narrative_flow_score(section_scores, newsletter_content)
        
        return AggregatedQualityReport(
            overall_score=overall_score,
            section_scores=section_scores,
            total_word_count=total_word_count,
            average_readability=average_readability,
            consistency_score=consistency_score,
            flow_score=flow_score
        )

    def validate_section_thresholds(self, quality_report: AggregatedQualityReport) -> Tuple[bool, List[str]]:
        """
        Validate that sections meet quality thresholds.
        
        Args:
            quality_report: Quality analysis report
            
        Returns:
            Tuple[bool, List[str]]: (all_passed, list_of_issues)
        """
        issues = []
        all_passed = True
        
        for section_type, metrics in quality_report.section_scores.items():
            threshold = self.quality_thresholds.get(section_type, 0.8)
            
            if metrics.overall_score < threshold:
                all_passed = False
                issues.append(
                    f"{section_type.value} section quality {metrics.overall_score:.2f} "
                    f"below threshold {threshold:.2f}"
                )
        
        return all_passed, issues

    def get_improvement_recommendations(self, quality_report: AggregatedQualityReport) -> List[str]:
        """
        Get prioritized improvement recommendations.
        
        Args:
            quality_report: Quality analysis report
            
        Returns:
            List[str]: Prioritized recommendations
        """
        recommendations = []
        
        # Global recommendations
        if quality_report.overall_score < 0.8:
            recommendations.append("Overall quality needs improvement - review all sections")
        
        if quality_report.consistency_score < 0.8:
            recommendations.append("Improve consistency across sections")
        
        if quality_report.flow_score < 0.8:
            recommendations.append("Enhance narrative flow between sections")
        
        # Section-specific recommendations
        for section_type, metrics in quality_report.section_scores.items():
            threshold = self.quality_thresholds.get(section_type, 0.8)
            
            if metrics.overall_score < threshold:
                section_suggestions = metrics.get_suggestions()
                for suggestion in section_suggestions[:2]:  # Top 2 suggestions
                    recommendations.append(f"{section_type.value}: {suggestion}")
        
        return recommendations

    def _calculate_cross_section_consistency(self, section_scores: Dict[SectionType, SectionQualityMetrics]) -> float:
        """Calculate consistency score across sections."""
        if len(section_scores) < 2:
            return 1.0
        
        # Check consistency of quality scores
        scores = [metrics.overall_score for metrics in section_scores.values()]
        score_variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
        consistency_score = max(0.0, 1.0 - score_variance)
        
        return consistency_score

    def _calculate_narrative_flow_score(self, section_scores: Dict[SectionType, SectionQualityMetrics],
                                      content: str) -> float:
        """Calculate narrative flow score."""
        # This is a simplified implementation
        # In practice, would analyze transitions, coherence, etc.
        
        # Check for transitions between sections
        transition_indicators = ['however', 'furthermore', 'in addition', 'meanwhile', 'next']
        sections = content.split('\n\n')
        
        transitions_found = 0
        for i in range(1, len(sections)):
            section_start = sections[i][:100].lower()
            if any(indicator in section_start for indicator in transition_indicators):
                transitions_found += 1
        
        # Basic flow score based on transitions
        if len(sections) <= 1:
            return 1.0
        
        transition_ratio = transitions_found / (len(sections) - 1)
        return min(1.0, 0.5 + transition_ratio * 0.5)


# Export main classes and functions
__all__ = [
    'SectionQualityMetrics',
    'SectionQualityAnalyzer', 
    'SectionAwareQualitySystem',
    'AggregatedQualityReport',
    'QualityMetric',
    'QualityDimension'
]