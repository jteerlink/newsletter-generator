from typing import List, Dict, Any
from .content_analyzer import ContentAnalyzer

class OutlineGenerator:
    """
    Generates newsletter outlines based on content analysis.
    """
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        
        # Define section templates based on themes
        self.section_templates = {
            'machine_learning': {
                'title': 'Machine Learning & AI',
                'description': 'Latest developments in machine learning algorithms and applications'
            },
            'deep_learning': {
                'title': 'Deep Learning Insights',
                'description': 'Advances in neural networks and deep learning architectures'
            },
            'nlp': {
                'title': 'Natural Language Processing',
                'description': 'Breakthroughs in language models and text processing'
            },
            'computer_vision': {
                'title': 'Computer Vision',
                'description': 'Innovations in image processing and visual AI'
            },
            'robotics': {
                'title': 'Robotics & Automation',
                'description': 'Latest in robotics and autonomous systems'
            },
            'ethics': {
                'title': 'AI Ethics & Responsibility',
                'description': 'Discussions on AI safety, bias, and ethical considerations'
            }
        }

    def generate_outline(self, content_list: List[str]) -> Dict[str, Any]:
        """
        Generate a newsletter outline from a list of content items.
        """
        # Analyze all content
        all_themes = set()
        all_topics = set()
        content_summary = []
        
        for i, content in enumerate(content_list):
            analysis = self.content_analyzer.analyze_content(content)
            all_themes.update(analysis['themes'])
            all_topics.update(analysis['key_topics'])
            content_summary.append({
                'id': i,
                'themes': analysis['themes'],
                'key_topics': analysis['key_topics'][:5],  # Top 5 topics
                'word_count': analysis['word_count']
            })
        
        # Generate sections based on identified themes
        sections = []
        for theme in all_themes:
            if theme in self.section_templates:
                sections.append({
                    'theme': theme,
                    'title': self.section_templates[theme]['title'],
                    'description': self.section_templates[theme]['description'],
                    'content_items': [item for item in content_summary if theme in item['themes']]
                })
        
        # Add a general section for unclassified content
        unclassified_items = [item for item in content_summary if not item['themes']]
        if unclassified_items:
            sections.append({
                'theme': 'general',
                'title': 'General AI News',
                'description': 'Other AI-related developments and insights',
                'content_items': unclassified_items
            })
        
        return {
            'sections': sections,
            'total_themes': list(all_themes),
            'key_topics': list(all_topics)[:15],  # Top 15 topics
            'total_content_items': len(content_list),
            'estimated_length': self._estimate_length(content_summary)
        }

    def _estimate_length(self, content_summary: List[Dict[str, Any]]) -> str:
        """
        Estimate the newsletter length based on content.
        """
        total_words = sum(item['word_count'] for item in content_summary)
        
        if total_words < 1000:
            return 'Short (1-2 pages)'
        elif total_words < 3000:
            return 'Medium (2-3 pages)'
        else:
            return 'Long (3+ pages)' 