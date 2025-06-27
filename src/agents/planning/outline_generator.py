"""
Enhanced Newsletter Outline Generator

Generates structured newsletter outlines based on content analysis, target audience,
and length requirements with intelligent section organization and audience adaptation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import Counter

from .content_analyzer import ContentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OutlineSection:
    """Represents a section in the newsletter outline."""
    id: str
    title: str
    type: str
    themes: List[str]
    keywords: List[str]
    target_length: int
    priority: str
    content_items: List[Any] = None
    subsections: List['OutlineSection'] = None


@dataclass
class NewsletterOutline:
    """Complete newsletter outline structure."""
    id: str
    title: str
    target_audience: str
    total_length: int
    sections: List[OutlineSection]
    themes: List[str]
    estimated_read_time: int
    created_at: datetime
    metadata: Dict[str, Any] = None


class OutlineGenerator:
    """
    Enhanced outline generator with audience adaptation and intelligent organization.
    
    Features:
    - Audience-specific section templates
    - Theme-based content organization
    - Length optimization
    - Priority-based section ordering
    - Content balance validation
    """
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        
        # Section templates for different audiences
        self.audience_templates = {
            'business': {
                'sections': [
                    {'type': 'executive_summary', 'title': 'Executive Summary', 'weight': 0.15},
                    {'type': 'market_analysis', 'title': 'Market Analysis', 'weight': 0.25},
                    {'type': 'company_news', 'title': 'Company Updates', 'weight': 0.30},
                    {'type': 'investment_trends', 'title': 'Investment Trends', 'weight': 0.20},
                    {'type': 'strategic_insights', 'title': 'Strategic Insights', 'weight': 0.10}
                ],
                'focus_keywords': ['market', 'revenue', 'investment', 'business', 'commercial', 'enterprise']
            },
            'technical': {
                'sections': [
                    {'type': 'research_highlights', 'title': 'Research Highlights', 'weight': 0.25},
                    {'type': 'technical_deep_dive', 'title': 'Technical Deep Dive', 'weight': 0.30},
                    {'type': 'implementation_guides', 'title': 'Implementation Guides', 'weight': 0.20},
                    {'type': 'performance_analysis', 'title': 'Performance Analysis', 'weight': 0.15},
                    {'type': 'future_directions', 'title': 'Future Directions', 'weight': 0.10}
                ],
                'focus_keywords': ['algorithm', 'implementation', 'performance', 'architecture', 'optimization']
            },
            'mixed': {
                'sections': [
                    {'type': 'executive_summary', 'title': 'Executive Summary', 'weight': 0.15},
                    {'type': 'research_highlights', 'title': 'Research Highlights', 'weight': 0.20},
                    {'type': 'industry_news', 'title': 'Industry News', 'weight': 0.25},
                    {'type': 'technical_insights', 'title': 'Technical Insights', 'weight': 0.25},
                    {'type': 'trend_analysis', 'title': 'Trend Analysis', 'weight': 0.15}
                ],
                'focus_keywords': ['ai', 'machine learning', 'technology', 'business', 'innovation']
            }
        }
        
        # Section type mappings
        self.section_type_mapping = {
            'executive_summary': 'summary',
            'research_highlights': 'research_summary',
            'technical_deep_dive': 'technical_deep_dive',
            'industry_news': 'industry_news',
            'market_analysis': 'trend_analysis',
            'company_news': 'industry_news',
            'investment_trends': 'trend_analysis',
            'strategic_insights': 'trend_analysis',
            'implementation_guides': 'technical_deep_dive',
            'performance_analysis': 'technical_deep_dive',
            'future_directions': 'trend_analysis',
            'technical_insights': 'technical_deep_dive',
            'trend_analysis': 'trend_analysis'
        }
        
        logger.info("OutlineGenerator initialized")
    
    def generate_outline(self, content_items: List[Any], target_audience: str = "mixed", 
                        target_length: int = 2) -> Dict[str, Any]:
        """
        Generate a complete newsletter outline.
        
        Args:
            content_items: List of analyzed content items
            target_audience: Target audience type ('business', 'technical', 'mixed')
            target_length: Target newsletter length in pages
            
        Returns:
            Dictionary containing the complete outline structure
        """
        logger.info(f"Generating outline for {len(content_items)} content items, audience: {target_audience}")
        
        # Step 1: Analyze content themes and patterns
        content_analysis = self._analyze_content_patterns(content_items)
        
        # Step 2: Select appropriate template
        template = self.audience_templates.get(target_audience, self.audience_templates['mixed'])
        
        # Step 3: Create sections based on template and content
        sections = self._create_sections(template, content_analysis, target_length)
        
        # Step 4: Validate and optimize outline
        optimized_sections = self._optimize_outline(sections, content_items, target_length)
        
        # Step 5: Create final outline
        outline = NewsletterOutline(
            id=f"outline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=self._generate_outline_title(content_analysis, target_audience),
            target_audience=target_audience,
            total_length=target_length,
            sections=optimized_sections,
            themes=content_analysis['top_themes'],
            estimated_read_time=self._estimate_read_time(optimized_sections),
            created_at=datetime.now(),
            metadata={
                'content_count': len(content_items),
                'theme_distribution': content_analysis['theme_distribution'],
                'audience_focus': template['focus_keywords']
            }
        )
        
        logger.info(f"Generated outline with {len(optimized_sections)} sections")
        return self._outline_to_dict(outline)
    
    def _analyze_content_patterns(self, content_items: List[Any]) -> Dict[str, Any]:
        """Analyze content to identify themes, patterns, and distribution."""
        all_themes = []
        all_entities = {'companies': [], 'models': [], 'technologies': []}
        all_sentiments = []
        
        for item in content_items:
            # Collect themes
            if hasattr(item, 'themes'):
                all_themes.extend(item.themes)
            
            # Collect entities
            if hasattr(item, 'entities'):
                for entity_type, entities in item.entities.items():
                    if entity_type in all_entities:
                        all_entities[entity_type].extend(entities)
            
            # Collect sentiment
            if hasattr(item, 'sentiment'):
                all_sentiments.append(item.sentiment.get('label', 'neutral'))
        
        # Analyze theme distribution
        theme_counter = Counter(all_themes)
        top_themes = [theme for theme, count in theme_counter.most_common(10)]
        
        # Analyze entity distribution
        entity_distribution = {}
        for entity_type, entities in all_entities.items():
            entity_counter = Counter(entities)
            entity_distribution[entity_type] = dict(entity_counter.most_common(5))
        
        # Analyze sentiment distribution
        sentiment_counter = Counter(all_sentiments)
        
        return {
            'top_themes': top_themes,
            'theme_distribution': dict(theme_counter),
            'entity_distribution': entity_distribution,
            'sentiment_distribution': dict(sentiment_counter),
            'total_items': len(content_items)
        }
    
    def _create_sections(self, template: Dict, content_analysis: Dict, target_length: int) -> List[OutlineSection]:
        """Create outline sections based on template and content analysis."""
        sections = []
        total_words = target_length * 500  # Assume 500 words per page
        
        for i, section_template in enumerate(template['sections']):
            # Calculate section length based on weight
            section_length = int(total_words * section_template['weight'])
            
            # Determine themes for this section
            section_themes = self._determine_section_themes(section_template['type'], content_analysis)
            
            # Determine keywords for this section
            section_keywords = self._determine_section_keywords(section_template['type'], content_analysis)
            
            # Determine priority
            priority = self._determine_section_priority(section_template['type'], i)
            
            section = OutlineSection(
                id=f"section_{i+1:02d}",
                title=section_template['title'],
                type=section_template['type'],
                themes=section_themes,
                keywords=section_keywords,
                target_length=section_length,
                priority=priority
            )
            
            sections.append(section)
        
        return sections
    
    def _determine_section_themes(self, section_type: str, content_analysis: Dict) -> List[str]:
        """Determine appropriate themes for a section based on its type."""
        theme_mapping = {
            'executive_summary': ['machine_learning', 'deep_learning', 'ai_models'],
            'research_highlights': ['machine_learning', 'deep_learning', 'neural_networks'],
            'technical_deep_dive': ['deep_learning', 'neural_networks', 'algorithms'],
            'industry_news': ['companies', 'products', 'announcements'],
            'market_analysis': ['business', 'market', 'trends'],
            'company_news': ['companies', 'business', 'products'],
            'investment_trends': ['business', 'investment', 'market'],
            'strategic_insights': ['business', 'strategy', 'trends'],
            'implementation_guides': ['algorithms', 'implementation', 'code'],
            'performance_analysis': ['performance', 'benchmarks', 'optimization'],
            'future_directions': ['research', 'trends', 'predictions'],
            'technical_insights': ['algorithms', 'architecture', 'optimization'],
            'trend_analysis': ['trends', 'predictions', 'analysis']
        }
        
        # Get themes from mapping
        mapped_themes = theme_mapping.get(section_type, [])
        
        # Filter to themes that exist in content
        available_themes = content_analysis['top_themes']
        relevant_themes = [theme for theme in mapped_themes if theme in available_themes]
        
        # If no mapped themes are available, use top themes
        if not relevant_themes:
            relevant_themes = available_themes[:3]
        
        return relevant_themes
    
    def _determine_section_keywords(self, section_type: str, content_analysis: Dict) -> List[str]:
        """Determine appropriate keywords for a section based on its type."""
        keyword_mapping = {
            'executive_summary': ['ai', 'machine learning', 'breakthrough', 'innovation'],
            'research_highlights': ['research', 'paper', 'algorithm', 'model'],
            'technical_deep_dive': ['implementation', 'architecture', 'performance', 'optimization'],
            'industry_news': ['company', 'announcement', 'product', 'launch'],
            'market_analysis': ['market', 'trend', 'growth', 'investment'],
            'company_news': ['company', 'business', 'revenue', 'funding'],
            'investment_trends': ['investment', 'funding', 'startup', 'venture'],
            'strategic_insights': ['strategy', 'business', 'competitive', 'advantage'],
            'implementation_guides': ['code', 'implementation', 'tutorial', 'guide'],
            'performance_analysis': ['benchmark', 'performance', 'speed', 'efficiency'],
            'future_directions': ['future', 'prediction', 'trend', 'development'],
            'technical_insights': ['technical', 'algorithm', 'architecture', 'design'],
            'trend_analysis': ['trend', 'pattern', 'analysis', 'prediction']
        }
        
        return keyword_mapping.get(section_type, ['ai', 'technology'])
    
    def _determine_section_priority(self, section_type: str, position: int) -> str:
        """Determine priority for a section based on type and position."""
        # High priority sections
        if section_type in ['executive_summary', 'research_highlights']:
            return 'high'
        
        # Medium priority sections
        if section_type in ['industry_news', 'technical_deep_dive', 'market_analysis']:
            return 'medium'
        
        # Low priority sections (usually at the end)
        return 'low'
    
    def _optimize_outline(self, sections: List[OutlineSection], content_items: List[Any], 
                         target_length: int) -> List[OutlineSection]:
        """Optimize outline based on content availability and balance."""
        # Calculate total target length
        total_target_length = sum(section.target_length for section in sections)
        
        # Adjust section lengths based on content availability
        for section in sections:
            relevant_content = self._count_relevant_content(section, content_items)
            
            # Adjust length based on content availability
            if relevant_content < 2:
                # Reduce length if little relevant content
                section.target_length = int(section.target_length * 0.7)
            elif relevant_content > 8:
                # Increase length if lots of relevant content
                section.target_length = int(section.target_length * 1.2)
        
        # Rebalance total length
        new_total_length = sum(section.target_length for section in sections)
        if new_total_length != total_target_length:
            scale_factor = total_target_length / new_total_length
            for section in sections:
                section.target_length = int(section.target_length * scale_factor)
        
        return sections
    
    def _count_relevant_content(self, section: OutlineSection, content_items: List[Any]) -> int:
        """Count content items relevant to a section."""
        relevant_count = 0
        
        for item in content_items:
            # Check theme overlap
            if hasattr(item, 'themes'):
                theme_overlap = any(theme in item.themes for theme in section.themes)
                if theme_overlap:
                    relevant_count += 1
                    continue
            
            # Check keyword presence
            if hasattr(item, 'content'):
                keyword_presence = any(keyword.lower() in item.content.lower() 
                                     for keyword in section.keywords)
                if keyword_presence:
                    relevant_count += 1
        
        return relevant_count
    
    def _generate_outline_title(self, content_analysis: Dict, target_audience: str) -> str:
        """Generate a title for the outline based on content analysis."""
        top_themes = content_analysis['top_themes']
        
        if top_themes:
            primary_theme = top_themes[0].replace('_', ' ').title()
            
            if target_audience == 'business':
                return f"AI Business Newsletter: {primary_theme} Focus"
            elif target_audience == 'technical':
                return f"AI Technical Newsletter: {primary_theme} Developments"
            else:
                return f"AI Newsletter: {primary_theme} Insights"
        
        return "AI Newsletter: Latest Developments"
    
    def _estimate_read_time(self, sections: List[OutlineSection]) -> int:
        """Estimate reading time in minutes."""
        total_words = sum(section.target_length for section in sections)
        # Assume 200 words per minute reading speed
        return max(1, total_words // 200)
    
    def _outline_to_dict(self, outline: NewsletterOutline) -> Dict[str, Any]:
        """Convert NewsletterOutline to dictionary format."""
        return {
            'id': outline.id,
            'title': outline.title,
            'target_audience': outline.target_audience,
            'total_length': outline.total_length,
            'sections': [
                {
                    'id': section.id,
                    'title': section.title,
                    'type': section.type,
                    'themes': section.themes,
                    'keywords': section.keywords,
                    'target_length': section.target_length,
                    'priority': section.priority
                }
                for section in outline.sections
            ],
            'themes': outline.themes,
            'estimated_read_time': outline.estimated_read_time,
            'created_at': outline.created_at.isoformat(),
            'metadata': outline.metadata
        }
    
    def validate_outline(self, outline: Dict[str, Any]) -> List[str]:
        """Validate outline structure and return any issues."""
        issues = []
        
        # Check required fields
        required_fields = ['id', 'title', 'target_audience', 'sections']
        for field in required_fields:
            if field not in outline:
                issues.append(f"Missing required field: {field}")
        
        # Check sections
        if 'sections' in outline:
            sections = outline['sections']
            if not sections:
                issues.append("Outline must have at least one section")
            
            for i, section in enumerate(sections):
                section_issues = self._validate_section(section, i)
                issues.extend(section_issues)
        
        # Check length balance
        if 'sections' in outline and outline['sections']:
            total_length = sum(section.get('target_length', 0) for section in outline['sections'])
            if total_length < 500:
                issues.append("Total outline length is too short (minimum 500 words)")
            elif total_length > 5000:
                issues.append("Total outline length is too long (maximum 5000 words)")
        
        return issues
    
    def _validate_section(self, section: Dict, index: int) -> List[str]:
        """Validate individual section structure."""
        issues = []
        
        # Check required section fields
        required_fields = ['id', 'title', 'type', 'target_length']
        for field in required_fields:
            if field not in section:
                issues.append(f"Section {index}: Missing required field: {field}")
        
        # Check target length
        if 'target_length' in section:
            length = section['target_length']
            if length < 50:
                issues.append(f"Section {index}: Target length too short (minimum 50 words)")
            elif length > 1000:
                issues.append(f"Section {index}: Target length too long (maximum 1000 words)")
        
        # Check type validity
        if 'type' in section:
            valid_types = list(self.section_type_mapping.keys())
            if section['type'] not in valid_types:
                issues.append(f"Section {index}: Invalid section type: {section['type']}")
        
        return issues
    
    def get_section_template(self, audience: str) -> Dict[str, Any]:
        """Get section template for a specific audience."""
        return self.audience_templates.get(audience, self.audience_templates['mixed'])
    
    def customize_for_audience(self, outline: Dict[str, Any], audience: str) -> Dict[str, Any]:
        """Customize outline for specific audience preferences."""
        template = self.audience_templates.get(audience, self.audience_templates['mixed'])
        
        # Adjust section weights based on audience
        for section in outline['sections']:
            section_type = section['type']
            
            # Find corresponding template section
            for template_section in template['sections']:
                if template_section['type'] == section_type:
                    # Adjust target length based on template weight
                    new_length = int(outline['total_length'] * 500 * template_section['weight'])
                    section['target_length'] = new_length
                    break
        
        return outline 