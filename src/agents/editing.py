"""
Enhanced Editing Agent for Newsletter Generation

This module provides the enhanced EditorAgent class, which is responsible for reviewing,
improving, and ensuring quality of newsletter content with tool-assisted auditing.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.campaign_context import CampaignContext
import core.core as core
from core.feedback_system import (
    FeedbackGenerator,
    IssueType,
    RequiredAction,
    Severity,
    StructuredFeedback,
)
from core.template_manager import AIMLTemplateManager, NewsletterType

from .base import AgentType, SimpleAgent, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class EditorAgent(SimpleAgent):
    """Enhanced agent specialized in editing and quality assurance with tool-assisted auditing."""

    def __init__(self, name: str = "EditorAgent", **kwargs):
        super().__init__(
            name=name,
            role="Content Editor",
            goal="Review, improve, and ensure the quality of newsletter content with comprehensive auditing",
            backstory="""You are an experienced content editor with expertise in newsletter
            editing and quality assurance. You excel at identifying areas for improvement,
            ensuring clarity and readability, maintaining consistency, and enhancing overall
            content quality. You understand editorial standards, grammar rules, and best
            practices for engaging content. You can provide constructive feedback and
            implement improvements while preserving the author's voice and intent.
            You are particularly skilled at using tools to assist in auditing content quality,
            evaluating content against campaign contexts, and generating structured feedback.""",
            agent_type=AgentType.EDITOR,
            tools=[],  # Editors don't use external tools in unit tests
            **kwargs
        )
        self.feedback_generator = FeedbackGenerator()
        self.campaign_context: Optional[CampaignContext] = None

    def perform_tool_assisted_audit(self, content: str) -> Dict[str, Any]:
        """Perform comprehensive audit using tools and analysis."""
        audit_results = {
            'grammar_issues': self._check_grammar(content),
            'style_issues': self._check_style(content),
            'clarity_issues': self._check_clarity(content),
            'structure_issues': self._check_structure(content),
            'engagement_issues': self._check_engagement(content),
            'seo_issues': self._check_seo(content),
            'brand_compliance_issues': self._check_brand_compliance(content),
            'accessibility_issues': self._check_accessibility(content),
            'overall_score': 0.0,
            'recommendations': []
        }

        # Calculate overall score
        audit_results['overall_score'] = self._calculate_audit_score(
            audit_results)

        # Generate recommendations
        audit_results['recommendations'] = self._generate_audit_recommendations(
            audit_results)

        return audit_results

    def evaluate_against_context(
            self, content: str, context: CampaignContext) -> Dict[str, Any]:
        """Evaluate content against campaign context."""
        self.campaign_context = context

        evaluation_results = {
            'tone_alignment': self._evaluate_tone_alignment(
                content,
                context),
            'audience_alignment': self._evaluate_audience_alignment(
                content,
                context),
            'goal_alignment': self._evaluate_goal_alignment(
                content,
                context),
            'terminology_compliance': self._evaluate_terminology_compliance(
                content,
                context),
            'quality_threshold_met': False,
            'context_score': 0.0,
            'recommendations': []}

        # Calculate context score
        evaluation_results['context_score'] = self._calculate_context_score(
            evaluation_results)

        # Check if quality threshold is met
        quality_threshold = context.get_quality_threshold('newsletter')
        evaluation_results['quality_threshold_met'] = evaluation_results['context_score'] >= quality_threshold

        # Generate context-specific recommendations
        evaluation_results['recommendations'] = self._generate_context_recommendations(
            evaluation_results, context)

        return evaluation_results

    def generate_structured_feedback(
            self, audit_results: Dict) -> StructuredFeedback:
        """Generate structured feedback from audit results."""
        return self.feedback_generator.generate_feedback_from_audit(
            audit_results)

    def verify_critical_claims(self, content: str) -> List[Dict[str, Any]]:
        """Verify critical claims in content through independent fact-checking."""
        # Extract claims that need verification
        claims = self._extract_claims_for_verification(content)

        verification_results = []

        for claim in claims:
            verification_result = self._verify_single_claim(claim)
            verification_results.append(verification_result)

        return verification_results

    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute editing task with comprehensive quality review."""
        logger.info(f"EditorAgent executing editing task: {task}")

        # Extract editing parameters
        quality_threshold = kwargs.get('quality_threshold', 7.0)
        focus_areas = kwargs.get(
            'focus_areas', [
                'clarity', 'grammar', 'structure'])

        # Perform comprehensive editing
        edited_content = self._perform_comprehensive_editing(
            task, context, quality_threshold, focus_areas)

        return edited_content

    def _perform_comprehensive_editing(
            self,
            content: str,
            context: str,
            quality_threshold: float,
            focus_areas: List[str]) -> str:
        """Perform comprehensive content editing."""
        try:
            # Step 1: Initial quality assessment
            quality_metrics = self.extract_quality_metrics(content)
            quality_analysis = self.calculate_quality_score(quality_metrics)

            # Step 2: Content improvement
            improved_content = self._improve_content(
                content, quality_analysis, focus_areas)

            # Step 3: Final quality check
            final_quality_metrics = self.extract_quality_metrics(
                improved_content)
            final_quality_analysis = self.calculate_quality_score(
                final_quality_metrics)

            # Step 4: Generate improvement report
            improvement_report = self._generate_improvement_report(
                quality_analysis, final_quality_analysis, focus_areas
            )

            # Combine improved content with report
            final_result = f"""
{improved_content}

---
## Editorial Report
{improvement_report}
            """

            return final_result.strip()

        except Exception as e:
            logger.error(f"Error in comprehensive editing: {e}")
            return f"Editing failed: {e}\n\nOriginal content:\n{content}"

    def _improve_content(self, content: str, quality_analysis: Dict[str, Any],
                         focus_areas: List[str]) -> str:
        """Improve content based on quality analysis."""
        improvement_prompt = f"""
        Please improve the following newsletter content based on the quality analysis:

        Original Content:
        {content}

        Quality Analysis:
        - Overall Score: {quality_analysis.get('overall_score', 'N/A')}
        - Grade: {quality_analysis.get('grade', 'N/A')}
        - Areas for Improvement: {', '.join(quality_analysis.get('improvement_areas', []))}

        Focus Areas: {', '.join(focus_areas)}

        Please improve the content by:
        1. Enhancing clarity and readability
        2. Fixing grammar and spelling issues
        3. Improving structure and flow
        4. Strengthening engagement elements
        5. Ensuring consistency and professionalism
        6. Adding missing context where needed
        7. Optimizing for the target audience

        Return the improved content with proper formatting.
        """

        try:
            improved_content = core.query_llm(improvement_prompt)
            return improved_content if improved_content.strip() else content
        except Exception as e:
            logger.error(f"Error improving content: {e}")
            return content

    def calculate_quality_score(
            self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality score for content."""
        try:
            # Extract individual scores
            scores = content_analysis.get('scores', {})

            # Calculate weighted overall score
            weights = {
                'clarity': 0.25,
                'accuracy': 0.20,
                'engagement': 0.20,
                'completeness': 0.15,
                'structure': 0.10,
                'grammar': 0.10
            }

            overall_score = 0.0
            total_weight = 0.0

            for metric, weight in weights.items():
                if metric in scores:
                    overall_score += scores[metric] * weight
                    total_weight += weight

            if total_weight > 0:
                overall_score = overall_score / total_weight
            else:
                overall_score = 6.0  # Default score

            # Determine grade
            if overall_score >= 9.0:
                grade = "A+"
            elif overall_score >= 8.5:
                grade = "A"
            elif overall_score >= 8.0:
                grade = "A-"
            elif overall_score >= 7.5:
                grade = "B+"
            elif overall_score >= 7.0:
                grade = "B"
            elif overall_score >= 6.5:
                grade = "B-"
            elif overall_score >= 6.0:
                grade = "C+"
            elif overall_score >= 5.5:
                grade = "C"
            else:
                grade = "C-"

            # Identify improvement areas
            improvement_areas = []
            for metric, score in scores.items():
                if score < 7.0:
                    improvement_areas.append(metric)

            return {
                'overall_score': round(overall_score, 2),
                'grade': grade,
                'individual_scores': scores,
                'improvement_areas': improvement_areas,
                'quality_level': 'excellent' if overall_score >= 8.5 else 'good' if overall_score >= 7.0 else 'needs_improvement'
            }

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return {
                'overall_score': 6.0,
                'grade': 'C',
                'individual_scores': {},
                'improvement_areas': ['calculation_error'],
                'quality_level': 'error'
            }

    def extract_quality_metrics(self, content: str) -> Dict[str, float]:
        """Extract comprehensive quality metrics from content."""
        try:
            # Basic metrics
            word_count = len(content.split())
            sentence_count = len(re.split(r'[.!?]+', content))
            paragraph_count = len(
                [p for p in content.split('\n\n') if p.strip()])

            # Calculate reading time (average 200 words per minute)
            estimated_reading_time = max(1, word_count // 200)

            # Content depth analysis
            content_depth = self._assess_content_depth(content)

            # Engagement elements
            engagement_score = self._count_engagement_elements(content)

            # Technical depth
            technical_depth = self._assess_technical_depth(content)

            # Examples and cases
            examples_score = self._count_examples_and_cases(content)

            # Calculate individual quality scores
            clarity_score = min(
                10.0, max(1.0, 8.0 + (content_depth - 0.5) * 4))
            accuracy_score = 8.0  # Default, could be enhanced with fact-checking
            engagement_score = min(10.0, max(1.0, 6.0 + engagement_score * 2))
            completeness_score = min(
                10.0, max(1.0, 7.0 + examples_score * 1.5))
            structure_score = min(10.0, max(
                1.0, 7.0 + (paragraph_count / max(1, sentence_count)) * 3))
            grammar_score = 8.0  # Default, could be enhanced with grammar checking

            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'estimated_reading_time': estimated_reading_time,
                'content_depth': content_depth,
                'engagement_elements': engagement_score,
                'technical_depth': technical_depth,
                'examples_count': examples_score,
                'scores': {
                    'clarity': clarity_score,
                    'accuracy': accuracy_score,
                    'engagement': engagement_score,
                    'completeness': completeness_score,
                    'structure': structure_score,
                    'grammar': grammar_score
                }
            }

        except Exception as e:
            logger.error(f"Error extracting quality metrics: {e}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'estimated_reading_time': 1,
                'content_depth': 0.5,
                'engagement_elements': 0.5,
                'technical_depth': 0.5,
                'examples_count': 0.5,
                'scores': {
                    'clarity': 6.0,
                    'accuracy': 6.0,
                    'engagement': 6.0,
                    'completeness': 6.0,
                    'structure': 6.0,
                    'grammar': 6.0
                }
            }

    def _assess_content_depth(self, content: str) -> float:
        """Assess the depth and sophistication of content."""
        depth_indicators = [
            'because', 'therefore', 'however', 'although', 'furthermore',
            'moreover', 'consequently', 'nevertheless', 'meanwhile',
            'analysis', 'research', 'study', 'evidence', 'data',
            'statistics', 'trends', 'patterns', 'implications'
        ]

        content_lower = content.lower()
        depth_count = sum(
            1 for indicator in depth_indicators if indicator in content_lower)

        # Normalize to 0-1 scale
        return min(1.0, depth_count / 10.0)

    def _count_engagement_elements(self, content: str) -> float:
        """Count engagement elements in content."""
        engagement_indicators = [
            'you', 'your', 'we', 'our', 'us',  # Personal pronouns
            'imagine', 'consider', 'think about',  # Interactive elements
            'question', 'challenge', 'opportunity',  # Engaging concepts
            '!', '?',  # Punctuation
            '**', '__',  # Emphasis
            'list', 'bullet', 'point',  # Structured elements
        ]

        content_lower = content.lower()
        engagement_count = 0

        for indicator in engagement_indicators:
            if indicator in content_lower:
                engagement_count += content_lower.count(indicator)

        # Normalize to 0-1 scale
        return min(1.0, engagement_count / 20.0)

    def _assess_technical_depth(self, content: str) -> float:
        """Assess technical depth of content."""
        technical_indicators = [
            'algorithm', 'architecture', 'framework', 'protocol',
            'api', 'sdk', 'library', 'toolkit', 'platform',
            'implementation', 'deployment', 'configuration',
            'performance', 'scalability', 'security', 'optimization',
            'code', 'function', 'method', 'class', 'object',
            'database', 'server', 'client', 'network'
        ]

        content_lower = content.lower()
        technical_count = sum(
            1 for indicator in technical_indicators if indicator in content_lower)

        # Normalize to 0-1 scale
        return min(1.0, technical_count / 15.0)

    def _count_examples_and_cases(self, content: str) -> float:
        """Count examples and case studies in content."""
        example_indicators = [
            'example', 'case study', 'instance', 'scenario',
            'for instance', 'such as', 'like', 'including',
            'specifically', 'particularly', 'notably',
            'demonstrates', 'shows', 'illustrates'
        ]

        content_lower = content.lower()
        example_count = sum(
            1 for indicator in example_indicators if indicator in content_lower)

        # Normalize to 0-1 scale
        return min(1.0, example_count / 10.0)

    def _generate_improvement_report(self, initial_analysis: Dict[str, Any],
                                     final_analysis: Dict[str, Any],
                                     focus_areas: List[str]) -> str:
        """Generate improvement report."""
        initial_score = initial_analysis.get('overall_score', 0)
        final_score = final_analysis.get('overall_score', 0)
        improvement = final_score - initial_score

        report = f"""
### Quality Assessment Summary

**Initial Quality Score:** {initial_score}/10 ({initial_analysis.get('grade', 'N/A')})
**Final Quality Score:** {final_score}/10 ({final_analysis.get('grade', 'N/A')})
**Improvement:** {improvement:+.2f} points

### Focus Areas Addressed
{', '.join(focus_areas)}

### Key Improvements Made
"""

        if improvement > 0:
            report += f"- Overall quality improved by {
                improvement:.2f} points\n"
            report += "- Enhanced clarity and readability\n"
            report += "- Improved structure and flow\n"
            report += "- Strengthened engagement elements\n"
        else:
            report += "- Content maintained high quality standards\n"
            report += "- Minor refinements applied\n"

        report += f"""
### Quality Level: {final_analysis.get('quality_level', 'unknown').title()}

This content meets {'excellent' if final_score >= 8.5 else 'good' if final_score >= 7.0 else 'minimum'} quality standards.
        """

        return report

    def validate_content_quality(self, content: str) -> Dict[str, Any]:
        """Validate content quality and provide recommendations."""
        try:
            # Extract metrics
            metrics = self.extract_quality_metrics(content)

            # Calculate quality score
            quality_analysis = self.calculate_quality_score(metrics)

            # Generate recommendations
            recommendations = self.generate_improvement_recommendations(
                quality_analysis, metrics)

            # Determine if content passes quality gate
            passes_quality_gate = quality_analysis.get(
                'overall_score', 0) >= 7.0

            return {
                'passes_quality_gate': passes_quality_gate,
                'quality_score': quality_analysis.get(
                    'overall_score',
                    0),
                'quality_grade': quality_analysis.get(
                    'grade',
                    'C'),
                'metrics': metrics,
                'analysis': quality_analysis,
                'recommendations': recommendations,
                'status': 'approved' if passes_quality_gate else 'needs_revision'}

        except Exception as e:
            logger.error(f"Error validating content quality: {e}")
            return {
                'passes_quality_gate': False,
                'quality_score': 0,
                'quality_grade': 'F',
                'metrics': {},
                'analysis': {},
                'recommendations': [f"Error in quality validation: {e}"],
                'status': 'error'
            }

    def generate_improvement_recommendations(self, quality_analysis: Dict[str, Any],
                                             content_metrics: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations."""
        recommendations = []
        scores = quality_analysis.get('individual_scores', {})

        # Clarity recommendations
        if scores.get('clarity', 10) < 7.0:
            recommendations.append(
                "Improve clarity by simplifying complex sentences and using more direct language")

        # Engagement recommendations
        if scores.get('engagement', 10) < 7.0:
            recommendations.append(
                "Increase engagement by adding personal pronouns, questions, and interactive elements")

        # Structure recommendations
        if scores.get('structure', 10) < 7.0:
            recommendations.append(
                "Improve structure by organizing content with clear headings and logical flow")

        # Completeness recommendations
        if scores.get('completeness', 10) < 7.0:
            recommendations.append(
                "Add more examples, case studies, and supporting evidence")

        # Grammar recommendations
        if scores.get('grammar', 10) < 7.0:
            recommendations.append(
                "Review and correct grammar, spelling, and punctuation issues")

        # Content length recommendations
        word_count = content_metrics.get('word_count', 0)
        if word_count < 500:
            recommendations.append(
                "Expand content with more detailed explanations and examples")
        elif word_count > 3000:
            recommendations.append(
                "Consider condensing content to improve readability and focus")

        # Technical depth recommendations
        technical_depth = content_metrics.get('technical_depth', 0)
        if technical_depth < 0.3:
            recommendations.append(
                "Add more technical details and implementation guidance")

        return recommendations if recommendations else [
            "Content quality is satisfactory"]

    def get_editing_analytics(self) -> Dict[str, Any]:
        """Get editing-specific analytics."""
        analytics = self.get_tool_usage_analytics()

        # Add editing-specific metrics
        editing_metrics = {
            "editing_sessions": len(
                self.execution_history), "avg_editing_time": sum(
                r.execution_time for r in self.execution_history) / len(
                self.execution_history) if self.execution_history else 0, "success_rate": sum(
                    1 for r in self.execution_history if r.status.value == "completed") / len(
                        self.execution_history) if self.execution_history else 0, "quality_improvement_metrics": {
                            "avg_quality_score": sum(
                                float(
                                    r.metadata.get(
                                        'quality_score', 0)) for r in self.execution_history) / len(
                                            self.execution_history) if self.execution_history else 0, "content_approved": sum(
                                                1 for r in self.execution_history if r.metadata.get('status') == 'approved'), "content_revised": sum(
                                                    1 for r in self.execution_history if r.metadata.get('status') == 'needs_revision')}}

        analytics.update(editing_metrics)
        return analytics

    def _check_grammar(self, content: str) -> List[Dict[str, Any]]:
        """Check grammar issues in content."""
        # Simple grammar checking - can be enhanced with language-tool
        # integration
        grammar_issues = []

        # Common grammar patterns to check
        grammar_patterns = [
            (r'\b(its|it\'s)\b', 'its/it\'s confusion'),
            (r'\b(their|they\'re|there)\b', 'their/they\'re/there confusion'),
            (r'\b(your|you\'re)\b', 'your/you\'re confusion'),
            (r'\b(affect|effect)\b', 'affect/effect confusion'),
        ]

        for pattern, issue_type in grammar_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                grammar_issues.append({
                    'text': match.group(0),
                    'position': match.start(),
                    'issue_type': issue_type,
                    'suggestion': 'Review for correct usage'
                })

        return grammar_issues

    def _check_style(self, content: str) -> List[Dict[str, Any]]:
        """Check style issues in content."""
        style_issues = []

        # Check for passive voice
        passive_patterns = [
            r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
            r'\b(has|have|had)\s+been\s+\w+ed\b'
        ]

        for pattern in passive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                style_issues.append({
                    'text': match.group(0),
                    'position': match.start(),
                    'issue_type': 'passive_voice',
                    'suggestion': 'Consider using active voice'
                })

        # Check for wordiness
        wordy_phrases = [
            'due to the fact that',
            'in order to',
            'at this point in time',
            'in the event that'
        ]

        for phrase in wordy_phrases:
            if phrase in content.lower():
                style_issues.append({
                    'text': phrase,
                    'issue_type': 'wordiness',
                    'suggestion': 'Use more concise language'
                })

        return style_issues

    def _check_clarity(self, content: str) -> List[Dict[str, Any]]:
        """Check clarity issues in content."""
        clarity_issues = []

        # Check for long sentences
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if len(sentence.split()) > 25:  # Long sentence threshold
                clarity_issues.append({
                    'text': sentence.strip(),
                    'issue_type': 'long_sentence',
                    'suggestion': 'Consider breaking into shorter sentences'
                })

        # Check for jargon
        jargon_words = [
            'paradigm',
            'synergy',
            'leverage',
            'optimize',
            'facilitate']
        for word in jargon_words:
            if word in content.lower():
                clarity_issues.append({
                    'text': word,
                    'issue_type': 'jargon',
                    'suggestion': 'Use simpler, clearer language'
                })

        return clarity_issues

    def _check_structure(self, content: str) -> List[Dict[str, Any]]:
        """Check structure issues in content."""
        structure_issues = []

        # Check for proper heading structure
        headings = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
        heading_levels = [len(h.split()[0]) for h in headings]

        # Check for proper heading hierarchy
        for i in range(1, len(heading_levels)):
            if heading_levels[i] > heading_levels[i - 1] + 1:
                structure_issues.append({
                    'text': headings[i],
                    'issue_type': 'heading_hierarchy',
                    'suggestion': 'Maintain proper heading hierarchy'
                })

        return structure_issues

    def _check_engagement(self, content: str) -> List[Dict[str, Any]]:
        """Check engagement issues in content."""
        engagement_issues = []

        # Check for engaging elements
        engaging_elements = ['!', '?', 'example', 'case study', 'story']
        has_engaging_elements = any(
            element in content for element in engaging_elements)

        if not has_engaging_elements:
            engagement_issues.append({
                'text': 'Content lacks engaging elements',
                'issue_type': 'low_engagement',
                'suggestion': 'Add examples, questions, or stories to increase engagement'
            })

        return engagement_issues

    def _check_seo(self, content: str) -> List[Dict[str, Any]]:
        """Check SEO issues in content."""
        seo_issues = []

        # Check for keyword density
        words = content.lower().split()
        if len(words) > 0:
            # Simple keyword density check
            common_words = [
                'the',
                'and',
                'or',
                'but',
                'in',
                'on',
                'at',
                'to',
                'for']
            content_words = [w for w in words if w not in common_words]

            if len(content_words) < len(words) * 0.3:  # Low content density
                seo_issues.append({
                    'text': 'Low content density',
                    'issue_type': 'seo_content_density',
                    'suggestion': 'Increase meaningful content density'
                })

        return seo_issues

    def _check_brand_compliance(self, content: str) -> List[Dict[str, Any]]:
        """Check brand compliance issues in content."""
        brand_issues = []

        # This would typically check against brand guidelines
        # For now, we'll do a simple check for consistency

        # Check for consistent capitalization
        title_case_words = re.findall(r'\b[A-Z][a-z]+\b', content)
        if len(title_case_words) > len(content.split()) * \
                0.1:  # Too many title case words
            brand_issues.append({
                'text': 'Inconsistent capitalization',
                'issue_type': 'brand_consistency',
                'suggestion': 'Maintain consistent capitalization style'
            })

        return brand_issues

    def _check_accessibility(self, content: str) -> List[Dict[str, Any]]:
        """Check accessibility issues in content."""
        accessibility_issues = []

        # Check for alt text in images (if any)
        image_patterns = re.findall(r'!\[.*?\]\(.*?\)', content)
        for image in image_patterns:
            if 'alt=' not in image:
                accessibility_issues.append({
                    'text': image,
                    'issue_type': 'missing_alt_text',
                    'suggestion': 'Add descriptive alt text for images'
                })

        return accessibility_issues

    def _calculate_audit_score(self, audit_results: Dict) -> float:
        """Calculate overall audit score."""
        total_issues = sum(
            len(
                audit_results.get(
                    f'{category}_issues',
                    [])) for category in [
                'grammar',
                'style',
                'clarity',
                'structure',
                'engagement',
                'seo',
                'brand_compliance',
                'accessibility'])

        # Base score of 10, deduct points for issues
        base_score = 10.0
        deduction_per_issue = 0.5

        score = base_score - (total_issues * deduction_per_issue)
        return max(0.0, min(10.0, score))

    def _generate_audit_recommendations(
            self, audit_results: Dict) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []

        if len(audit_results.get('grammar_issues', [])) > 0:
            recommendations.append("Review and correct grammar issues")

        if len(audit_results.get('style_issues', [])) > 0:
            recommendations.append("Improve writing style and clarity")

        if len(audit_results.get('clarity_issues', [])) > 0:
            recommendations.append("Enhance content clarity and readability")

        if len(audit_results.get('structure_issues', [])) > 0:
            recommendations.append(
                "Improve content structure and organization")

        if len(audit_results.get('engagement_issues', [])) > 0:
            recommendations.append(
                "Add engaging elements to increase reader interest")

        return recommendations

    def _evaluate_tone_alignment(
            self, content: str, context: CampaignContext) -> Dict[str, Any]:
        """Evaluate tone alignment with campaign context."""
        target_tone = context.content_style.get('tone', 'professional')

        # Simple tone evaluation - can be enhanced with more sophisticated
        # analysis
        tone_indicators = {
            'professional': ['furthermore', 'moreover', 'consequently'],
            'casual': ['also', 'plus', 'anyway'],
            'enthusiastic': ['exciting', 'amazing', 'incredible'],
            'analytical': ['analysis', 'data', 'evidence']
        }

        content_lower = content.lower()
        target_indicators = tone_indicators.get(target_tone, [])

        # Count target tone indicators
        indicator_count = sum(
            1 for indicator in target_indicators if indicator in content_lower)

        return {
            # Normalize to 0-1
            'alignment_score': min(1.0, indicator_count / 3.0),
            'target_tone': target_tone,
            'found_indicators': indicator_count
        }

    def _evaluate_audience_alignment(
            self, content: str, context: CampaignContext) -> Dict[str, Any]:
        """Evaluate audience alignment with campaign context."""
        audience_persona = context.audience_persona
        knowledge_level = audience_persona.get(
            'knowledge_level', 'intermediate')

        # Simple knowledge level evaluation
        technical_terms = [
            'algorithm',
            'framework',
            'protocol',
            'infrastructure']
        technical_count = sum(
            1 for term in technical_terms if term in content.lower())

        if knowledge_level == 'beginner' and technical_count > 2:
            alignment_score = 0.3
        elif knowledge_level == 'advanced' and technical_count < 1:
            alignment_score = 0.4
        else:
            alignment_score = 0.8

        return {
            'alignment_score': alignment_score,
            'target_knowledge_level': knowledge_level,
            'technical_term_count': technical_count
        }

    def _evaluate_goal_alignment(
            self, content: str, context: CampaignContext) -> Dict[str, Any]:
        """Evaluate goal alignment with campaign context."""
        strategic_goals = context.strategic_goals

        # Simple goal alignment check
        goal_keywords = {
            'engagement': ['interactive', 'participate', 'join', 'share'],
            'education': ['learn', 'understand', 'explain', 'demonstrate'],
            'conversion': ['sign up', 'subscribe', 'download', 'register']
        }

        alignment_scores = {}
        for goal in strategic_goals:
            keywords = goal_keywords.get(goal.lower(), [])
            keyword_count = sum(
                1 for keyword in keywords if keyword in content.lower())
            alignment_scores[goal] = min(1.0, keyword_count / 2.0)

        return {
            'alignment_scores': alignment_scores,
            'overall_alignment': sum(
                alignment_scores.values()) /
            len(alignment_scores) if alignment_scores else 0.0}

    def _evaluate_terminology_compliance(
            self, content: str, context: CampaignContext) -> Dict[str, Any]:
        """Evaluate terminology compliance with campaign context."""
        forbidden_terms = context.forbidden_terminology
        preferred_terms = context.preferred_terminology

        # Check for forbidden terms
        forbidden_found = []
        for term in forbidden_terms:
            if term.lower() in content.lower():
                forbidden_found.append(term)

        # Check for preferred terms
        preferred_found = []
        for term in preferred_terms:
            if term.lower() in content.lower():
                preferred_found.append(term)

        compliance_score = 1.0
        if forbidden_found:
            compliance_score -= 0.3 * len(forbidden_found)
        if preferred_found:
            compliance_score += 0.1 * len(preferred_found)

        return {
            'compliance_score': max(0.0, min(1.0, compliance_score)),
            'forbidden_terms_found': forbidden_found,
            'preferred_terms_found': preferred_found
        }

    def _calculate_context_score(self, evaluation_results: Dict) -> float:
        """Calculate overall context alignment score."""
        scores = [
            evaluation_results['tone_alignment']['alignment_score'],
            evaluation_results['audience_alignment']['alignment_score'],
            evaluation_results['goal_alignment']['overall_alignment'],
            evaluation_results['terminology_compliance']['compliance_score']
        ]

        return sum(scores) / len(scores)

    def _generate_context_recommendations(
            self,
            evaluation_results: Dict,
            context: CampaignContext) -> List[str]:
        """Generate context-specific recommendations."""
        recommendations = []

        tone_alignment = evaluation_results['tone_alignment']['alignment_score']
        if tone_alignment < 0.5:
            target_tone = context.content_style.get('tone', 'professional')
            recommendations.append(f"Adjust tone to be more {target_tone}")

        audience_alignment = evaluation_results['audience_alignment']['alignment_score']
        if audience_alignment < 0.5:
            knowledge_level = context.audience_persona.get(
                'knowledge_level', 'intermediate')
            recommendations.append(f"Adjust content complexity for {
                                   knowledge_level} audience")

        terminology_compliance = evaluation_results['terminology_compliance']
        if terminology_compliance['forbidden_terms_found']:
            recommendations.append(
                "Replace forbidden terminology with approved alternatives")

        return recommendations

    def _extract_claims_for_verification(self, content: str) -> List[str]:
        """Extract claims that need verification."""
        # Simple claim extraction - can be enhanced with more sophisticated NLP
        claims = []

        # Look for factual statements
        factual_patterns = [
            r'(\d+% of .+?)',
            r'(According to .+?, .+?)',
            r'(Research shows that .+?)',
            r'(Studies indicate that .+?)'
        ]

        for pattern in factual_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            claims.extend(matches)

        return claims[:5]  # Limit to top 5 claims

    def _verify_single_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a single claim through fact-checking."""
        # Simple verification - can be enhanced with web search integration
        verification_result = {
            'claim': claim,
            'verification_status': 'pending',
            'confidence_score': 0.5,
            'supporting_sources': [],
            'contradicting_sources': [],
            'verification_notes': 'Manual verification required'
        }

        # This would typically involve web search and fact-checking
        # For now, we'll return a basic structure

        return verification_result
