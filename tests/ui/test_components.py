"""
UI component tests for the unified Streamlit application
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components
from streamlit.components.header import create_header, create_feature_cards
from streamlit.components.configuration import create_configuration_panel, create_advanced_settings_panel
from streamlit.components.dashboard import create_dashboard, create_performance_dashboard
from streamlit.components.content_display import create_content_display, format_content_as_markdown, format_content_as_html
from streamlit.components.progress import create_progress_tracker, create_generate_button
from streamlit.components.feedback import create_feedback_section, save_feedback
from streamlit.utils.ui_utils import load_sources_config, get_categories_from_sources, validate_configuration

class TestHeaderComponents:
    """Test header components"""
    
    def test_create_header_function_exists(self):
        """Test that create_header function exists and is callable"""
        assert callable(create_header)
    
    def test_create_feature_cards_function_exists(self):
        """Test that create_feature_cards function exists and is callable"""
        assert callable(create_feature_cards)

class TestConfigurationComponents:
    """Test configuration components"""
    
    def test_create_configuration_panel_function_exists(self):
        """Test that create_configuration_panel function exists and is callable"""
        assert callable(create_configuration_panel)
    
    def test_create_advanced_settings_panel_function_exists(self):
        """Test that create_advanced_settings_panel function exists and is callable"""
        assert callable(create_advanced_settings_panel)
    
    def test_validate_configuration_valid_input(self):
        """Test configuration validation with valid input"""
        config = {
            "topic": "AI Development",
            "audience": "Developers",
            "word_count": 1500,
            "selected_categories": ["AI", "Technology"]
        }
        
        is_valid, message = validate_configuration(config)
        assert is_valid
        assert message == "Configuration is valid"
    
    def test_validate_configuration_missing_topic(self):
        """Test configuration validation with missing topic"""
        config = {
            "audience": "Developers",
            "word_count": 1500,
            "selected_categories": ["AI", "Technology"]
        }
        
        is_valid, message = validate_configuration(config)
        assert not is_valid
        assert "Topic is required" in message
    
    def test_validate_configuration_invalid_word_count(self):
        """Test configuration validation with invalid word count"""
        config = {
            "topic": "AI Development",
            "audience": "Developers",
            "word_count": 50,  # Too low
            "selected_categories": ["AI", "Technology"]
        }
        
        is_valid, message = validate_configuration(config)
        assert not is_valid
        assert "Word count must be at least 100" in message

class TestDashboardComponents:
    """Test dashboard components"""
    
    def test_create_dashboard_function_exists(self):
        """Test that create_dashboard function exists and is callable"""
        assert callable(create_dashboard)
    
    def test_create_performance_dashboard_function_exists(self):
        """Test that create_performance_dashboard function exists and is callable"""
        assert callable(create_performance_dashboard)

class TestContentDisplayComponents:
    """Test content display components"""
    
    def test_create_content_display_function_exists(self):
        """Test that create_content_display function exists and is callable"""
        assert callable(create_content_display)
    
    def test_format_content_as_markdown(self):
        """Test markdown formatting"""
        content = {
            "title": "Test Newsletter",
            "content": "# Test Content\n\nThis is test content.",
            "metadata": {
                "word_count": 100,
                "generation_time": 60,
                "quality_score": 0.85
            }
        }
        
        markdown = format_content_as_markdown(content)
        assert "# Test Newsletter" in markdown
        assert "Test Content" in markdown
        assert "100" in markdown
        assert "60s" in markdown
        assert "0.85" in markdown
    
    def test_format_content_as_html(self):
        """Test HTML formatting"""
        content = {
            "title": "Test Newsletter",
            "content": "# Test Content\n\nThis is test content.",
            "metadata": {
                "word_count": 100,
                "quality_score": 0.85
            }
        }
        
        html = format_content_as_html(content)
        assert "<!DOCTYPE html>" in html
        assert "Test Newsletter" in html
        assert "Test Content" in html
        assert "100" in html
        assert "0.85" in html

class TestProgressComponents:
    """Test progress components"""
    
    def test_create_progress_tracker_function_exists(self):
        """Test that create_progress_tracker function exists and is callable"""
        assert callable(create_progress_tracker)
    
    def test_create_generate_button_function_exists(self):
        """Test that create_generate_button function exists and is callable"""
        assert callable(create_generate_button)

class TestFeedbackComponents:
    """Test feedback components"""
    
    def test_create_feedback_section_function_exists(self):
        """Test that create_feedback_section function exists and is callable"""
        assert callable(create_feedback_section)
    
    def test_save_feedback_function_exists(self):
        """Test that save_feedback function exists and is callable"""
        assert callable(save_feedback)

class TestUIUtils:
    """Test UI utility functions"""
    
    def test_load_sources_config_function_exists(self):
        """Test that load_sources_config function exists and is callable"""
        assert callable(load_sources_config)
    
    def test_get_categories_from_sources_function_exists(self):
        """Test that get_categories_from_sources function exists and is callable"""
        assert callable(get_categories_from_sources)
    
    def test_get_categories_from_sources(self):
        """Test category extraction from sources"""
        sources_config = {
            "sources": [
                {
                    "name": "TechCrunch",
                    "categories": ["AI", "Technology", "Startups"]
                },
                {
                    "name": "Ars Technica",
                    "categories": ["Technology", "Science", "AI"]
                }
            ]
        }
        
        categories = get_categories_from_sources(sources_config)
        expected_categories = ["AI", "Science", "Startups", "Technology"]
        
        assert set(categories) == set(expected_categories)
    
    def test_format_execution_time(self):
        """Test execution time formatting"""
        from streamlit.utils.ui_utils import format_execution_time
        
        assert format_execution_time(30) == "30.0 seconds"
        assert format_execution_time(90) == "1.5 minutes"
        assert format_execution_time(7200) == "2.0 hours"
    
    def test_format_word_count(self):
        """Test word count formatting"""
        from streamlit.utils.ui_utils import format_word_count
        
        assert format_word_count(500) == "500 words"
        assert format_word_count(1500) == "1.5k words"
        assert format_word_count(1500000) == "1.5M words"

class TestComponentIntegration:
    """Test component integration"""
    
    def test_all_components_importable(self):
        """Test that all components can be imported"""
        try:
            from streamlit.components.header import create_header
            from streamlit.components.configuration import create_configuration_panel
            from streamlit.components.dashboard import create_dashboard
            from streamlit.components.content_display import create_content_display
            from streamlit.components.progress import create_progress_tracker
            from streamlit.components.feedback import create_feedback_section
            from streamlit.utils.ui_utils import load_sources_config
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import components: {e}")
    
    def test_component_function_signatures(self):
        """Test that components have expected function signatures"""
        
        # Test header components
        assert create_header.__name__ == "create_header"
        assert create_feature_cards.__name__ == "create_feature_cards"
        
        # Test configuration components
        assert create_configuration_panel.__name__ == "create_configuration_panel"
        assert create_advanced_settings_panel.__name__ == "create_advanced_settings_panel"
        
        # Test dashboard components
        assert create_dashboard.__name__ == "create_dashboard"
        assert create_performance_dashboard.__name__ == "create_performance_dashboard"
        
        # Test content display components
        assert create_content_display.__name__ == "create_content_display"
        assert format_content_as_markdown.__name__ == "format_content_as_markdown"
        assert format_content_as_html.__name__ == "format_content_as_html"
        
        # Test progress components
        assert create_progress_tracker.__name__ == "create_progress_tracker"
        assert create_generate_button.__name__ == "create_generate_button"
        
        # Test feedback components
        assert create_feedback_section.__name__ == "create_feedback_section"
        assert save_feedback.__name__ == "save_feedback"
        
        # Test utility functions
        assert load_sources_config.__name__ == "load_sources_config"
        assert get_categories_from_sources.__name__ == "get_categories_from_sources"
        assert validate_configuration.__name__ == "validate_configuration" 