"""
Technical validator for newsletter quality assessment.

This module provides technical validation functionality for the quality package.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TechnicalQualityValidator:
    """Technical quality validator for newsletter assessment."""

    def __init__(self):
        self.technical_thresholds = {
            'min_code_blocks': 0,
            'max_code_blocks': 10,
            'min_links': 0,
            'max_links': 20,
            'min_images': 0,
            'max_images': 10
        }

    def validate_content(self, content: str) -> Dict[str, Any]:
        """
        Validate newsletter content for technical quality.

        Args:
            content: The content to validate

        Returns:
            Dictionary with technical validation results
        """
        if not content:
            return self._create_empty_result()

        # Technical validation
        code_analysis = self._analyze_code_blocks(content)
        link_analysis = self._analyze_links(content)
        image_analysis = self._analyze_images(content)
        formatting_analysis = self._analyze_formatting(content)

        # Calculate overall technical score
        technical_score = (
            code_analysis['score'] +
            link_analysis['score'] +
            image_analysis['score'] +
            formatting_analysis['score']
        ) / 4

        # Identify technical issues
        issues = self._identify_technical_issues(
            code_analysis, link_analysis, image_analysis, formatting_analysis
        )

        # Generate technical recommendations
        recommendations = self._generate_technical_recommendations(
            issues, technical_score)

        return {
            'technical_score': technical_score,
            'issues': issues,
            'recommendations': recommendations,
            'code_analysis': code_analysis,
            'link_analysis': link_analysis,
            'image_analysis': image_analysis,
            'formatting_analysis': formatting_analysis
        }

    def _analyze_code_blocks(self, content: str) -> Dict[str, Any]:
        """Analyze code blocks in content."""
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        inline_code = re.findall(r'`[^`]+`', content)

        code_block_count = len(code_blocks)
        inline_code_count = len(inline_code)

        # Calculate score based on thresholds
        if code_block_count < self.technical_thresholds['min_code_blocks']:
            score = 0.3
        elif code_block_count > self.technical_thresholds['max_code_blocks']:
            score = 0.5
        else:
            score = 1.0

        return {
            'score': score,
            'code_blocks': code_block_count,
            'inline_code': inline_code_count,
            'total_code': code_block_count + inline_code_count,
            'within_thresholds': (
                self.technical_thresholds['min_code_blocks'] <= code_block_count <=
                self.technical_thresholds['max_code_blocks']
            )
        }

    def _analyze_links(self, content: str) -> Dict[str, Any]:
        """Analyze links in content."""
        # Find links
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)',
                           content)  # Markdown links
        urls = re.findall(r'https?://[^\s]+', content)  # Plain URLs

        link_count = len(links) + len(urls)

        # Calculate score based on thresholds
        if link_count < self.technical_thresholds['min_links']:
            score = 0.3
        elif link_count > self.technical_thresholds['max_links']:
            score = 0.5
        else:
            score = 1.0

        return {
            'score': score,
            'markdown_links': len(links),
            'plain_urls': len(urls),
            'total_links': link_count,
            'within_thresholds': (
                self.technical_thresholds['min_links'] <= link_count <=
                self.technical_thresholds['max_links']
            )
        }

    def _analyze_images(self, content: str) -> Dict[str, Any]:
        """Analyze images in content."""
        # Find image references
        markdown_images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        html_images = re.findall(r'<img[^>]+>', content)

        image_count = len(markdown_images) + len(html_images)

        # Calculate score based on thresholds
        if image_count < self.technical_thresholds['min_images']:
            score = 0.3
        elif image_count > self.technical_thresholds['max_images']:
            score = 0.5
        else:
            score = 1.0

        return {
            'score': score,
            'markdown_images': len(markdown_images),
            'html_images': len(html_images),
            'total_images': image_count,
            'within_thresholds': (
                self.technical_thresholds['min_images'] <= image_count <=
                self.technical_thresholds['max_images']
            )
        }

    def _analyze_formatting(self, content: str) -> Dict[str, Any]:
        """Analyze content formatting."""
        # Check for proper markdown formatting
        headers = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
        lists = re.findall(r'^[\s]*[-*+]\s+', content, re.MULTILINE)
        bold_text = re.findall(r'\*\*[^*]+\*\*', content)
        italic_text = re.findall(r'\*[^*]+\*', content)

        # Calculate formatting score
        formatting_elements = len(headers) + len(lists) + \
            len(bold_text) + len(italic_text)

        if formatting_elements == 0:
            score = 0.3  # No formatting
        elif formatting_elements < 3:
            score = 0.6  # Minimal formatting
        elif formatting_elements < 10:
            score = 0.8  # Good formatting
        else:
            score = 1.0  # Excellent formatting

        return {
            'score': score,
            'headers': len(headers),
            'lists': len(lists),
            'bold_text': len(bold_text),
            'italic_text': len(italic_text),
            'total_formatting_elements': formatting_elements
        }

    def _identify_technical_issues(self,
                                   code_analysis: Dict[str,
                                                       Any],
                                   link_analysis: Dict[str,
                                                       Any],
                                   image_analysis: Dict[str,
                                                        Any],
                                   formatting_analysis: Dict[str,
                                                             Any]) -> List[str]:
        """Identify technical issues."""
        issues = []

        if not code_analysis['within_thresholds']:
            if code_analysis['code_blocks'] < self.technical_thresholds['min_code_blocks']:
                issues.append("Too few code examples")
            else:
                issues.append("Too many code blocks")

        if not link_analysis['within_thresholds']:
            if link_analysis['total_links'] < self.technical_thresholds['min_links']:
                issues.append("Too few references/links")
            else:
                issues.append("Too many links")

        if not image_analysis['within_thresholds']:
            if image_analysis['total_images'] < self.technical_thresholds['min_images']:
                issues.append("Could benefit from more visual elements")
            else:
                issues.append("Too many images")

        if formatting_analysis['score'] < 0.5:
            issues.append("Poor content formatting")

        return issues

    def _generate_technical_recommendations(
            self,
            issues: List[str],
            technical_score: float) -> List[str]:
        """Generate technical improvement recommendations."""
        recommendations = []

        if technical_score < 0.5:
            recommendations.append("Improve technical content structure")

        if "Too few code examples" in issues:
            recommendations.append("Add code examples to illustrate concepts")

        if "Too many code blocks" in issues:
            recommendations.append("Reduce the number of code blocks")

        if "Too few references/links" in issues:
            recommendations.append("Add more references and external links")

        if "Too many links" in issues:
            recommendations.append("Reduce the number of links")

        if "Could benefit from more visual elements" in issues:
            recommendations.append("Add relevant images or diagrams")

        if "Too many images" in issues:
            recommendations.append("Reduce the number of images")

        if "Poor content formatting" in issues:
            recommendations.append(
                "Improve markdown formatting with headers, lists, and emphasis")

        if not recommendations and technical_score >= 0.8:
            recommendations.append("Technical quality is excellent")

        return recommendations

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for empty content."""
        return {
            'technical_score': 0.0,
            'issues': ['Content is empty'],
            'recommendations': ['Add content to the newsletter'],
            'code_analysis': {
                'score': 0.0,
                'code_blocks': 0,
                'inline_code': 0,
                'total_code': 0,
                'within_thresholds': False},
            'link_analysis': {
                'score': 0.0,
                'markdown_links': 0,
                'plain_urls': 0,
                'total_links': 0,
                'within_thresholds': False},
            'image_analysis': {
                'score': 0.0,
                'markdown_images': 0,
                'html_images': 0,
                'total_images': 0,
                'within_thresholds': False},
            'formatting_analysis': {
                'score': 0.0,
                'headers': 0,
                'lists': 0,
                'bold_text': 0,
                'italic_text': 0,
                'total_formatting_elements': 0}}
