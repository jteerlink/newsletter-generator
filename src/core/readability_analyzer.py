"""
Mobile Readability Analysis System

This module implements the MobileReadabilityAnalyzer for real-time mobile readability
assessment and optimization recommendations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReadabilityMetrics:
    """Comprehensive readability metrics for mobile content."""
    flesch_kincaid_grade: float
    flesch_reading_ease: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    avg_words_per_paragraph: float
    complex_words_ratio: float
    passive_voice_ratio: float
    mobile_readability_score: float


@dataclass
class MobileFriendlinessMetrics:
    """Mobile-specific friendliness metrics."""
    paragraph_length_score: float
    sentence_complexity_score: float
    navigation_clarity_score: float
    touch_target_accessibility: float
    visual_hierarchy_score: float
    code_block_mobile_score: float


@dataclass
class MobileReadabilityReport:
    """Comprehensive mobile readability analysis report."""
    overall_mobile_score: float
    readability_metrics: ReadabilityMetrics
    mobile_metrics: MobileFriendlinessMetrics
    recommendations: List[str]
    content_analysis: Dict[str, Any]
    improvement_priorities: List[str]


class MobileReadabilityAnalyzer:
    """
    Real-time mobile readability assessment system.
    
    Analyzes content for mobile readability across multiple dimensions including
    traditional readability metrics, mobile-specific factors, and accessibility.
    """
    
    def __init__(self):
        """Initialize the mobile readability analyzer."""
        self.mobile_thresholds = self._initialize_mobile_thresholds()
        self.readability_weights = self._initialize_readability_weights()
        
        logger.info("MobileReadabilityAnalyzer initialized")
    
    def _initialize_mobile_thresholds(self) -> Dict[str, float]:
        """Initialize mobile readability thresholds."""
        return {
            # Traditional readability thresholds
            'flesch_reading_ease_min': 60.0,      # Minimum for good readability
            'flesch_reading_ease_target': 80.0,   # Target for mobile
            'flesch_kincaid_max': 8.0,            # Maximum grade level
            'avg_sentence_length_max': 20.0,      # Maximum words per sentence
            'complex_words_ratio_max': 0.15,      # Maximum 15% complex words
            
            # Mobile-specific thresholds
            'paragraph_words_max': 75,            # Maximum words per paragraph
            'paragraph_sentences_max': 4,         # Maximum sentences per paragraph
            'heading_frequency_min': 0.1,         # Minimum heading frequency
            'list_usage_min': 0.05,              # Minimum list usage ratio
            
            # Mobile navigation thresholds
            'touch_target_min_size': 44,          # Minimum touch target (px)
            'line_height_mobile_min': 1.4,       # Minimum line height
            'contrast_ratio_min': 4.5,           # WCAG AA contrast ratio
            
            # Overall mobile scores
            'mobile_readability_min': 0.85,      # 85% minimum mobile score
            'mobile_readability_target': 0.95    # 95% target mobile score
        }
    
    def _initialize_readability_weights(self) -> Dict[str, float]:
        """Initialize weights for readability score calculation."""
        return {
            'traditional_readability': 0.35,      # Traditional metrics weight
            'mobile_friendliness': 0.40,          # Mobile-specific weight
            'accessibility': 0.25                 # Accessibility weight
        }
    
    def analyze_mobile_readability(self, content: str, 
                                 context: Optional[Dict[str, Any]] = None) -> MobileReadabilityReport:
        """
        Analyze content for mobile readability and generate comprehensive report.
        
        Args:
            content: Text content to analyze
            context: Additional context for analysis
            
        Returns:
            MobileReadabilityReport with detailed analysis and recommendations
        """
        context = context or {}
        
        logger.info("Starting mobile readability analysis")
        
        try:
            # Step 1: Calculate traditional readability metrics
            readability_metrics = self._calculate_readability_metrics(content)
            
            # Step 2: Calculate mobile-specific metrics
            mobile_metrics = self._calculate_mobile_metrics(content)
            
            # Step 3: Analyze content structure and characteristics
            content_analysis = self._analyze_content_structure(content)
            
            # Step 4: Calculate overall mobile readability score
            overall_score = self._calculate_overall_mobile_score(
                readability_metrics, mobile_metrics, content_analysis
            )
            
            # Step 5: Generate recommendations and priorities
            recommendations = self._generate_recommendations(
                readability_metrics, mobile_metrics, content_analysis
            )
            priorities = self._prioritize_improvements(
                readability_metrics, mobile_metrics
            )
            
            report = MobileReadabilityReport(
                overall_mobile_score=overall_score,
                readability_metrics=readability_metrics,
                mobile_metrics=mobile_metrics,
                recommendations=recommendations,
                content_analysis=content_analysis,
                improvement_priorities=priorities
            )
            
            logger.info(f"Mobile readability analysis completed: {overall_score:.1%} score")
            return report
            
        except Exception as e:
            logger.error(f"Mobile readability analysis failed: {e}")
            # Return baseline report on failure
            return self._create_fallback_report(content)
    
    def _calculate_readability_metrics(self, content: str) -> ReadabilityMetrics:
        """Calculate traditional readability metrics."""
        sentences = self._extract_sentences(content)
        words = self._extract_words(content)
        paragraphs = self._extract_paragraphs(content)
        
        # Basic counts
        total_sentences = len(sentences)
        total_words = len(words)
        total_paragraphs = len(paragraphs)
        
        if total_sentences == 0 or total_words == 0:
            return self._create_baseline_metrics()
        
        # Calculate syllables
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Average metrics
        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words
        avg_words_per_paragraph = total_words / max(1, total_paragraphs)
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp to 0-100
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)
        
        # Complex words ratio (words with 3+ syllables)
        complex_words = [word for word in words if self._count_syllables(word) >= 3]
        complex_words_ratio = len(complex_words) / total_words
        
        # Passive voice ratio (simplified detection)
        passive_voice_ratio = self._calculate_passive_voice_ratio(sentences)
        
        # Mobile readability score (adjusted for mobile consumption)
        mobile_readability_score = self._calculate_mobile_adjusted_score(
            flesch_reading_ease, avg_sentence_length, complex_words_ratio
        )
        
        return ReadabilityMetrics(
            flesch_kincaid_grade=flesch_kincaid_grade,
            flesch_reading_ease=flesch_reading_ease,
            avg_sentence_length=avg_sentence_length,
            avg_syllables_per_word=avg_syllables_per_word,
            avg_words_per_paragraph=avg_words_per_paragraph,
            complex_words_ratio=complex_words_ratio,
            passive_voice_ratio=passive_voice_ratio,
            mobile_readability_score=mobile_readability_score
        )
    
    def _calculate_mobile_metrics(self, content: str) -> MobileFriendlinessMetrics:
        """Calculate mobile-specific friendliness metrics."""
        # Paragraph length analysis
        paragraph_score = self._analyze_paragraph_mobile_friendliness(content)
        
        # Sentence complexity for mobile
        sentence_score = self._analyze_sentence_mobile_complexity(content)
        
        # Navigation clarity
        navigation_score = self._analyze_mobile_navigation_clarity(content)
        
        # Touch target accessibility (estimated from content structure)
        touch_target_score = self._analyze_touch_target_accessibility(content)
        
        # Visual hierarchy for mobile
        visual_hierarchy_score = self._analyze_mobile_visual_hierarchy(content)
        
        # Code block mobile-friendliness
        code_block_score = self._analyze_code_block_mobile_friendliness(content)
        
        return MobileFriendlinessMetrics(
            paragraph_length_score=paragraph_score,
            sentence_complexity_score=sentence_score,
            navigation_clarity_score=navigation_score,
            touch_target_accessibility=touch_target_score,
            visual_hierarchy_score=visual_hierarchy_score,
            code_block_mobile_score=code_block_score
        )
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure characteristics."""
        sentences = self._extract_sentences(content)
        paragraphs = self._extract_paragraphs(content)
        words = self._extract_words(content)
        
        # Content structure analysis
        analysis = {
            'total_words': len(words),
            'total_sentences': len(sentences),
            'total_paragraphs': len(paragraphs),
            'avg_words_per_sentence': len(words) / max(1, len(sentences)),
            'avg_sentences_per_paragraph': len(sentences) / max(1, len(paragraphs)),
            
            # Mobile-specific structure
            'heading_count': len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE)),
            'list_items': content.count('- ') + content.count('* ') + len(re.findall(r'^\d+\.', content, re.MULTILINE)),
            'code_blocks': content.count('```') // 2,
            'images': len(re.findall(r'!\[.*?\]\(.*?\)', content)),
            'links': len(re.findall(r'\[.*?\]\(.*?\)', content)),
            
            # Formatting characteristics
            'bold_text': content.count('**') // 2,
            'italic_text': content.count('*') - content.count('**'),
            'line_breaks': content.count('\n\n'),
            
            # Mobile navigation aids
            'section_breaks': len(re.findall(r'^#{2,6}\s+', content, re.MULTILINE)),
            'bullet_lists': content.count('- ') + content.count('* '),
            'numbered_lists': len(re.findall(r'^\d+\.', content, re.MULTILINE))
        }
        
        # Calculate content density metrics
        analysis['content_density'] = analysis['total_words'] / max(1, analysis['total_paragraphs'])
        analysis['formatting_density'] = (analysis['bold_text'] + analysis['italic_text']) / max(1, analysis['total_words'])
        analysis['navigation_density'] = analysis['heading_count'] / max(1, analysis['total_words'] / 100)  # Per 100 words
        
        return analysis
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extract sentences from content."""
        # Remove code blocks to avoid interference
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', content_no_code)
        
        # Filter out empty sentences and headers
        sentences = [
            s.strip() for s in sentences 
            if s.strip() and not s.strip().startswith('#') and len(s.strip()) > 5
        ]
        
        return sentences
    
    def _extract_words(self, content: str) -> List[str]:
        """Extract words from content."""
        # Remove code blocks and special formatting
        content_clean = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content_clean = re.sub(r'`[^`]+`', '', content_clean)  # Remove inline code
        content_clean = re.sub(r'[#*\[\]()]', '', content_clean)  # Remove markdown
        
        # Extract words (alphabetic characters only)
        words = re.findall(r'\b[a-zA-Z]+\b', content_clean.lower())
        
        return words
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extract paragraphs from content."""
        # Split on double line breaks
        paragraphs = content.split('\n\n')
        
        # Filter out empty paragraphs and headers
        paragraphs = [
            p.strip() for p in paragraphs 
            if p.strip() and not p.strip().startswith('#') and len(p.split()) > 3
        ]
        
        return paragraphs
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using basic heuristics."""
        word = word.lower()
        
        # Handle special cases
        if len(word) <= 2:
            return 1
        
        # Count vowel groups
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
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def _calculate_passive_voice_ratio(self, sentences: List[str]) -> float:
        """Calculate ratio of passive voice sentences."""
        if not sentences:
            return 0.0
        
        passive_indicators = [
            r'\bis\s+\w+ed\b', r'\bare\s+\w+ed\b', r'\bwas\s+\w+ed\b',
            r'\bwere\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bbeing\s+\w+ed\b'
        ]
        
        passive_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(re.search(pattern, sentence_lower) for pattern in passive_indicators):
                passive_count += 1
        
        return passive_count / len(sentences)
    
    def _calculate_mobile_adjusted_score(self, flesch_ease: float, 
                                       avg_sentence_length: float,
                                       complex_words_ratio: float) -> float:
        """Calculate mobile-adjusted readability score."""
        # Base score from Flesch Reading Ease
        base_score = flesch_ease / 100
        
        # Mobile penalties
        sentence_penalty = max(0, (avg_sentence_length - 15) / 20)  # Penalty for long sentences
        complexity_penalty = complex_words_ratio * 2  # Penalty for complex words
        
        # Calculate mobile score
        mobile_score = base_score - sentence_penalty - complexity_penalty
        
        return max(0.0, min(1.0, mobile_score))
    
    def _analyze_paragraph_mobile_friendliness(self, content: str) -> float:
        """Analyze paragraph length for mobile friendliness."""
        paragraphs = self._extract_paragraphs(content)
        
        if not paragraphs:
            return 1.0
        
        thresholds = self.mobile_thresholds
        optimal_scores = []
        
        for paragraph in paragraphs:
            word_count = len(paragraph.split())
            sentence_count = len(re.split(r'[.!?]+', paragraph))
            
            # Score based on word count (optimal: 30-75 words)
            if word_count <= 75:
                word_score = 1.0
            elif word_count <= 100:
                word_score = 0.8
            elif word_count <= 150:
                word_score = 0.5
            else:
                word_score = 0.2
            
            # Score based on sentence count (optimal: 2-4 sentences)
            if 2 <= sentence_count <= 4:
                sentence_score = 1.0
            elif sentence_count <= 6:
                sentence_score = 0.7
            else:
                sentence_score = 0.4
            
            optimal_scores.append((word_score + sentence_score) / 2)
        
        return sum(optimal_scores) / len(optimal_scores)
    
    def _analyze_sentence_mobile_complexity(self, content: str) -> float:
        """Analyze sentence complexity for mobile reading."""
        sentences = self._extract_sentences(content)
        
        if not sentences:
            return 1.0
        
        complexity_scores = []
        
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            # Score based on length (optimal: 8-20 words)
            if 8 <= word_count <= 20:
                length_score = 1.0
            elif word_count <= 25:
                length_score = 0.8
            elif word_count <= 30:
                length_score = 0.5
            else:
                length_score = 0.2
            
            # Check for complex structures
            complex_patterns = [
                r',.*,',  # Multiple commas
                r'[;:]',  # Semicolons and colons
                r'\b(?:although|however|nevertheless|furthermore|moreover)\b'  # Complex connectors
            ]
            
            complexity_penalty = sum(
                1 for pattern in complex_patterns 
                if re.search(pattern, sentence.lower())
            ) * 0.2
            
            structure_score = max(0.0, 1.0 - complexity_penalty)
            
            complexity_scores.append((length_score + structure_score) / 2)
        
        return sum(complexity_scores) / len(complexity_scores)
    
    def _analyze_mobile_navigation_clarity(self, content: str) -> float:
        """Analyze navigation clarity for mobile devices."""
        analysis = self._analyze_content_structure(content)
        
        # Check heading frequency (should have headings regularly)
        words_per_heading = analysis['total_words'] / max(1, analysis['heading_count'])
        heading_score = 1.0 if words_per_heading <= 200 else max(0.2, 1.0 - (words_per_heading - 200) / 300)
        
        # Check for proper heading hierarchy
        headings = re.findall(r'^(#{1,6})', content, re.MULTILINE)
        if headings:
            heading_levels = [len(h) for h in headings]
            # Check for logical progression
            hierarchy_score = 1.0 if all(
                level <= prev_level + 1 
                for prev_level, level in zip(heading_levels, heading_levels[1:])
            ) else 0.6
        else:
            hierarchy_score = 0.0
        
        # Check for navigation aids (lists, breaks)
        navigation_aids = (analysis['bullet_lists'] + analysis['numbered_lists']) / max(1, analysis['total_paragraphs'])
        aids_score = min(1.0, navigation_aids * 3)  # Boost score for good list usage
        
        return (heading_score + hierarchy_score + aids_score) / 3
    
    def _analyze_touch_target_accessibility(self, content: str) -> float:
        """Analyze touch target accessibility for mobile."""
        # This is a simplified analysis for markdown content
        # In a real implementation, this would analyze rendered HTML
        
        lines = content.split('\n')
        lines_with_links = [line for line in lines if '[' in line and '](' in line]
        
        if not lines_with_links:
            return 1.0  # No links means no touch target issues
        
        # Check for lines with too many links (poor touch targets)
        dense_link_lines = [line for line in lines_with_links if line.count('](') > 2]
        
        # Check for very short link text (harder to tap)
        short_links = re.findall(r'\[(.{1,3})\]', content)
        
        # Calculate score
        density_score = max(0.0, 1.0 - len(dense_link_lines) / len(lines_with_links))
        length_score = max(0.0, 1.0 - len(short_links) / max(1, content.count('](')))
        
        return (density_score + length_score) / 2
    
    def _analyze_mobile_visual_hierarchy(self, content: str) -> float:
        """Analyze visual hierarchy for mobile devices."""
        # Check heading distribution
        h1_count = len(re.findall(r'^# ', content, re.MULTILINE))
        h2_count = len(re.findall(r'^## ', content, re.MULTILINE))
        h3_count = len(re.findall(r'^### ', content, re.MULTILINE))
        
        # Optimal: One H1, multiple H2s, some H3s
        hierarchy_score = 0.0
        if h1_count == 1:
            hierarchy_score += 0.4
        if h2_count >= 2:
            hierarchy_score += 0.4
        if h3_count >= 1:
            hierarchy_score += 0.2
        
        # Check for visual breaks (empty lines, formatting)
        visual_breaks = content.count('\n\n')
        paragraphs = len(self._extract_paragraphs(content))
        break_ratio = visual_breaks / max(1, paragraphs)
        
        break_score = min(1.0, break_ratio)  # Good spacing between elements
        
        # Check for emphasis usage (bold, italic)
        emphasis_count = content.count('**') // 2 + content.count('*') - content.count('**')
        words = len(self._extract_words(content))
        emphasis_ratio = emphasis_count / max(1, words / 50)  # Per 50 words
        
        emphasis_score = min(1.0, emphasis_ratio)
        
        return (hierarchy_score + break_score + emphasis_score) / 3
    
    def _analyze_code_block_mobile_friendliness(self, content: str) -> float:
        """Analyze code block mobile-friendliness."""
        code_blocks = re.findall(r'```(.*?)```', content, re.DOTALL)
        
        if not code_blocks:
            return 1.0  # No code blocks means no mobile issues
        
        mobile_scores = []
        
        for block in code_blocks:
            lines = block.split('\n')
            max_line_length = max(len(line) for line in lines if line.strip())
            
            # Score based on line length (mobile screens are narrow)
            if max_line_length <= 50:
                length_score = 1.0
            elif max_line_length <= 80:
                length_score = 0.7
            elif max_line_length <= 120:
                length_score = 0.4
            else:
                length_score = 0.1
            
            # Score based on block size
            line_count = len([line for line in lines if line.strip()])
            if line_count <= 20:
                size_score = 1.0
            elif line_count <= 40:
                size_score = 0.7
            else:
                size_score = 0.4
            
            mobile_scores.append((length_score + size_score) / 2)
        
        return sum(mobile_scores) / len(mobile_scores)
    
    def _calculate_overall_mobile_score(self, readability: ReadabilityMetrics,
                                      mobile: MobileFriendlinessMetrics,
                                      analysis: Dict[str, Any]) -> float:
        """Calculate overall mobile readability score."""
        weights = self.readability_weights
        
        # Traditional readability component
        traditional_score = (
            readability.mobile_readability_score * 0.6 +
            min(1.0, readability.flesch_reading_ease / 80) * 0.4
        )
        
        # Mobile friendliness component
        mobile_score = (
            mobile.paragraph_length_score * 0.25 +
            mobile.sentence_complexity_score * 0.25 +
            mobile.navigation_clarity_score * 0.20 +
            mobile.visual_hierarchy_score * 0.15 +
            mobile.code_block_mobile_score * 0.15
        )
        
        # Accessibility component
        accessibility_score = (
            mobile.touch_target_accessibility * 0.6 +
            (1.0 - min(1.0, readability.passive_voice_ratio * 2)) * 0.4  # Lower passive voice is better
        )
        
        # Combined score
        overall_score = (
            traditional_score * weights['traditional_readability'] +
            mobile_score * weights['mobile_friendliness'] +
            accessibility_score * weights['accessibility']
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _generate_recommendations(self, readability: ReadabilityMetrics,
                                mobile: MobileFriendlinessMetrics,
                                analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for mobile optimization."""
        recommendations = []
        thresholds = self.mobile_thresholds
        
        # Readability recommendations
        if readability.flesch_reading_ease < thresholds['flesch_reading_ease_min']:
            recommendations.append("Simplify vocabulary and sentence structure to improve readability")
        
        if readability.avg_sentence_length > thresholds['avg_sentence_length_max']:
            recommendations.append(f"Reduce average sentence length from {readability.avg_sentence_length:.1f} to under {thresholds['avg_sentence_length_max']} words")
        
        if readability.complex_words_ratio > thresholds['complex_words_ratio_max']:
            recommendations.append(f"Reduce complex words ratio from {readability.complex_words_ratio:.1%} to under {thresholds['complex_words_ratio_max']:.1%}")
        
        # Mobile-specific recommendations
        if mobile.paragraph_length_score < 0.7:
            recommendations.append("Break long paragraphs into shorter, mobile-friendly chunks (30-75 words)")
        
        if mobile.sentence_complexity_score < 0.7:
            recommendations.append("Simplify sentence structure for easier mobile reading")
        
        if mobile.navigation_clarity_score < 0.7:
            recommendations.append("Add more headings and navigation aids for mobile users")
        
        if mobile.visual_hierarchy_score < 0.7:
            recommendations.append("Improve visual hierarchy with better heading structure and formatting")
        
        if mobile.code_block_mobile_score < 0.7:
            recommendations.append("Optimize code blocks for mobile display (shorter lines, smaller blocks)")
        
        # Content structure recommendations
        if analysis['navigation_density'] < 0.8:
            recommendations.append("Add more section headings to improve mobile navigation")
        
        if analysis['content_density'] > 100:
            recommendations.append("Reduce content density by adding more paragraph breaks")
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def _prioritize_improvements(self, readability: ReadabilityMetrics,
                               mobile: MobileFriendlinessMetrics) -> List[str]:
        """Prioritize improvement areas based on impact."""
        priorities = []
        
        # Calculate improvement potential for each area
        improvement_areas = [
            ("Paragraph length optimization", 1.0 - mobile.paragraph_length_score, 0.8),
            ("Sentence complexity reduction", 1.0 - mobile.sentence_complexity_score, 0.7),
            ("Navigation clarity improvement", 1.0 - mobile.navigation_clarity_score, 0.6),
            ("Visual hierarchy enhancement", 1.0 - mobile.visual_hierarchy_score, 0.5),
            ("Readability optimization", 1.0 - readability.mobile_readability_score, 0.9),
            ("Code block mobile formatting", 1.0 - mobile.code_block_mobile_score, 0.4),
            ("Touch target accessibility", 1.0 - mobile.touch_target_accessibility, 0.5)
        ]
        
        # Sort by impact potential (gap * weight)
        improvement_areas.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Return top priorities
        priorities = [area[0] for area in improvement_areas[:5] if area[1] > 0.2]
        
        return priorities
    
    def _create_baseline_metrics(self) -> ReadabilityMetrics:
        """Create baseline readability metrics for error cases."""
        return ReadabilityMetrics(
            flesch_kincaid_grade=8.0,
            flesch_reading_ease=60.0,
            avg_sentence_length=15.0,
            avg_syllables_per_word=1.5,
            avg_words_per_paragraph=50.0,
            complex_words_ratio=0.1,
            passive_voice_ratio=0.1,
            mobile_readability_score=0.6
        )
    
    def _create_fallback_report(self, content: str) -> MobileReadabilityReport:
        """Create fallback report when analysis fails."""
        baseline_readability = self._create_baseline_metrics()
        baseline_mobile = MobileFriendlinessMetrics(
            paragraph_length_score=0.6,
            sentence_complexity_score=0.6,
            navigation_clarity_score=0.6,
            touch_target_accessibility=0.8,
            visual_hierarchy_score=0.6,
            code_block_mobile_score=0.8
        )
        
        return MobileReadabilityReport(
            overall_mobile_score=0.6,
            readability_metrics=baseline_readability,
            mobile_metrics=baseline_mobile,
            recommendations=["Analysis failed - manual review recommended"],
            content_analysis={'error': 'Analysis failed'},
            improvement_priorities=["Manual review required"]
        )


# Export main classes
__all__ = [
    'MobileReadabilityAnalyzer',
    'MobileReadabilityReport',
    'ReadabilityMetrics',
    'MobileFriendlinessMetrics'
]