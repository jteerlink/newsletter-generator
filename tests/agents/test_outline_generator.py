import pytest
from src.agents.planning.outline_generator import OutlineGenerator

class TestOutlineGenerator:
    def test_outline_generator_initialization(self):
        """Test that OutlineGenerator initializes correctly."""
        generator = OutlineGenerator()
        assert generator.content_analyzer is not None
        assert 'machine_learning' in generator.section_templates
        assert 'deep_learning' in generator.section_templates
        assert 'nlp' in generator.section_templates

    def test_generate_outline_with_ml_content(self):
        """Test outline generation with machine learning content."""
        generator = OutlineGenerator()
        
        content_list = [
            "OpenAI released GPT-4 with improved reasoning capabilities. The new model shows significant advances in machine learning and natural language processing.",
            "Google's DeepMind announced breakthroughs in reinforcement learning algorithms for game playing and optimization problems."
        ]
        
        outline = generator.generate_outline(content_list)
        
        assert 'sections' in outline
        assert 'total_themes' in outline
        assert 'key_topics' in outline
        assert 'total_content_items' in outline
        assert 'estimated_length' in outline
        
        assert outline['total_content_items'] == 2
        assert 'machine_learning' in outline['total_themes']
        assert 'nlp' in outline['total_themes']
        
        # Check that sections were created
        assert len(outline['sections']) > 0
        
        # Verify section structure
        for section in outline['sections']:
            assert 'theme' in section
            assert 'title' in section
            assert 'description' in section
            assert 'content_items' in section

    def test_generate_outline_with_mixed_content(self):
        """Test outline generation with mixed AI content."""
        generator = OutlineGenerator()
        
        content_list = [
            "New computer vision model achieves state-of-the-art performance on ImageNet",
            "Ethical concerns raised about AI bias in facial recognition systems",
            "Robotics company develops autonomous delivery robots for urban environments",
            "Transformer architecture continues to dominate natural language processing tasks"
        ]
        
        outline = generator.generate_outline(content_list)
        
        assert outline['total_content_items'] == 4
        assert len(outline['sections']) >= 3  # Should have multiple sections
        
        # Check that different themes are identified
        themes = outline['total_themes']
        assert len(themes) >= 3

    def test_generate_outline_with_empty_content(self):
        """Test outline generation with empty content list."""
        generator = OutlineGenerator()
        
        outline = generator.generate_outline([])
        
        assert outline['total_content_items'] == 0
        assert outline['sections'] == []
        assert outline['total_themes'] == []
        assert outline['key_topics'] == []
        assert outline['estimated_length'] == 'Short (1-2 pages)'

    def test_estimate_length_calculation(self):
        """Test newsletter length estimation."""
        generator = OutlineGenerator()
        
        # Short content
        short_content = ["Brief AI news."]
        outline = generator.generate_outline(short_content)
        assert 'Short' in outline['estimated_length']
        
        # Medium content - create longer text
        medium_content = [
            "This is a longer article about machine learning developments. " * 100,  # ~2000 words
            "Another substantial piece about deep learning advances. " * 100  # ~2000 words
        ]
        outline = generator.generate_outline(medium_content)
        assert 'Medium' in outline['estimated_length'] or 'Long' in outline['estimated_length']

    def test_section_templates_coverage(self):
        """Test that all section templates are properly defined."""
        generator = OutlineGenerator()
        
        expected_themes = ['machine_learning', 'deep_learning', 'nlp', 'computer_vision', 'robotics', 'ethics']
        
        for theme in expected_themes:
            assert theme in generator.section_templates
            template = generator.section_templates[theme]
            assert 'title' in template
            assert 'description' in template
            assert len(template['title']) > 0
            assert len(template['description']) > 0 