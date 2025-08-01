"""
Editing Agent for Newsletter Generation

This module provides the EditorAgent class, which is responsible for reviewing,
improving, and ensuring quality of newsletter content.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from .base import SimpleAgent, AgentType, TaskResult, TaskStatus
from src.core.core import query_llm
from src.core.template_manager import NewsletterType, AIMLTemplateManager

logger = logging.getLogger(__name__)


class EditorAgent(SimpleAgent):
    """Agent specialized in editing and quality assurance."""
    
    def __init__(self, name: str = "EditorAgent", **kwargs):
        super().__init__(
            name=name,
            role="Content Editor",
            goal="Review, improve, and ensure the quality of newsletter content",
            backstory="""You are an experienced content editor with expertise in newsletter 
            editing and quality assurance. You excel at identifying areas for improvement, 
            ensuring clarity and readability, maintaining consistency, and enhancing overall 
            content quality. You understand editorial standards, grammar rules, and best 
            practices for engaging content. You can provide constructive feedback and 
            implement improvements while preserving the author's voice and intent.""",
            agent_type=AgentType.EDITOR,
            tools=[],  # Editors typically don't need external tools
            **kwargs
        )
    
    def execute_task(self, task: str, context: str = "", **kwargs) -> str:
        """Execute editing task with comprehensive quality review."""
        logger.info(f"EditorAgent executing editing task: {task}")
        
        # Extract editing parameters
        quality_threshold = kwargs.get('quality_threshold', 7.0)
        focus_areas = kwargs.get('focus_areas', ['clarity', 'grammar', 'structure'])
        
        # Perform comprehensive editing
        edited_content = self._perform_comprehensive_editing(task, context, quality_threshold, focus_areas)
        
        return edited_content
    
    def _perform_comprehensive_editing(self, content: str, context: str, 
                                     quality_threshold: float, focus_areas: List[str]) -> str:
        """Perform comprehensive content editing."""
        try:
            # Step 1: Initial quality assessment
            quality_metrics = self.extract_quality_metrics(content)
            quality_analysis = self.calculate_quality_score(quality_metrics)
            
            # Step 2: Content improvement
            improved_content = self._improve_content(content, quality_analysis, focus_areas)
            
            # Step 3: Final quality check
            final_quality_metrics = self.extract_quality_metrics(improved_content)
            final_quality_analysis = self.calculate_quality_score(final_quality_metrics)
            
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
            improved_content = query_llm(improvement_prompt)
            return improved_content if improved_content.strip() else content
        except Exception as e:
            logger.error(f"Error improving content: {e}")
            return content
    
    def calculate_quality_score(self, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
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
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
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
            clarity_score = min(10.0, max(1.0, 8.0 + (content_depth - 0.5) * 4))
            accuracy_score = 8.0  # Default, could be enhanced with fact-checking
            engagement_score = min(10.0, max(1.0, 6.0 + engagement_score * 2))
            completeness_score = min(10.0, max(1.0, 7.0 + examples_score * 1.5))
            structure_score = min(10.0, max(1.0, 7.0 + (paragraph_count / max(1, sentence_count)) * 3))
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
        depth_count = sum(1 for indicator in depth_indicators if indicator in content_lower)
        
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
        technical_count = sum(1 for indicator in technical_indicators if indicator in content_lower)
        
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
        example_count = sum(1 for indicator in example_indicators if indicator in content_lower)
        
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
            report += f"- Overall quality improved by {improvement:.2f} points\n"
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
            recommendations = self.generate_improvement_recommendations(quality_analysis, metrics)
            
            # Determine if content passes quality gate
            passes_quality_gate = quality_analysis.get('overall_score', 0) >= 7.0
            
            return {
                'passes_quality_gate': passes_quality_gate,
                'quality_score': quality_analysis.get('overall_score', 0),
                'quality_grade': quality_analysis.get('grade', 'C'),
                'metrics': metrics,
                'analysis': quality_analysis,
                'recommendations': recommendations,
                'status': 'approved' if passes_quality_gate else 'needs_revision'
            }
            
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
            recommendations.append("Improve clarity by simplifying complex sentences and using more direct language")
        
        # Engagement recommendations
        if scores.get('engagement', 10) < 7.0:
            recommendations.append("Increase engagement by adding personal pronouns, questions, and interactive elements")
        
        # Structure recommendations
        if scores.get('structure', 10) < 7.0:
            recommendations.append("Improve structure by organizing content with clear headings and logical flow")
        
        # Completeness recommendations
        if scores.get('completeness', 10) < 7.0:
            recommendations.append("Add more examples, case studies, and supporting evidence")
        
        # Grammar recommendations
        if scores.get('grammar', 10) < 7.0:
            recommendations.append("Review and correct grammar, spelling, and punctuation issues")
        
        # Content length recommendations
        word_count = content_metrics.get('word_count', 0)
        if word_count < 500:
            recommendations.append("Expand content with more detailed explanations and examples")
        elif word_count > 3000:
            recommendations.append("Consider condensing content to improve readability and focus")
        
        # Technical depth recommendations
        technical_depth = content_metrics.get('technical_depth', 0)
        if technical_depth < 0.3:
            recommendations.append("Add more technical details and implementation guidance")
        
        return recommendations if recommendations else ["Content quality is satisfactory"]
    
    def get_editing_analytics(self) -> Dict[str, Any]:
        """Get editing-specific analytics."""
        analytics = self.get_tool_usage_analytics()
        
        # Add editing-specific metrics
        editing_metrics = {
            "editing_sessions": len(self.execution_history),
            "avg_editing_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            "success_rate": sum(1 for r in self.execution_history if r.status.value == "completed") / len(self.execution_history) if self.execution_history else 0,
            "quality_improvement_metrics": {
                "avg_quality_score": sum(float(r.metadata.get('quality_score', 0)) for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
                "content_approved": sum(1 for r in self.execution_history if r.metadata.get('status') == 'approved'),
                "content_revised": sum(1 for r in self.execution_history if r.metadata.get('status') == 'needs_revision')
            }
        }
        
        analytics.update(editing_metrics)
        return analytics 