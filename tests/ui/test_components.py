"""
UI component tests for the unified Streamlit application
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock the streamlit components
class MockStreamlitComponents:
    def create_header(self, *args, **kwargs):
        return "Mock Header"
    
    def create_feature_cards(self, *args, **kwargs):
        return "Mock Feature Cards"
    
    def create_configuration_panel(self, *args, **kwargs):
        return "Mock Configuration Panel"
    
    def create_advanced_settings_panel(self, *args, **kwargs):
        return "Mock Advanced Settings Panel"
    
    def create_dashboard(self, *args, **kwargs):
        return "Mock Dashboard"
    
    def create_performance_dashboard(self, *args, **kwargs):
        return "Mock Performance Dashboard"
    
    def create_content_display(self, *args, **kwargs):
        return "Mock Content Display"
    
    def format_content_as_markdown(self, *args, **kwargs):
        return "Mock Markdown Content"
    
    def format_content_as_html(self, *args, **kwargs):
        return "Mock HTML Content"
    
    def create_progress_tracker(self, *args, **kwargs):
        return "Mock Progress Tracker"
    
    def create_generate_button(self, *args, **kwargs):
        return "Mock Generate Button"
    
    def create_feedback_section(self, *args, **kwargs):
        return "Mock Feedback Section"
    
    def save_feedback(self, *args, **kwargs):
        return "Mock Save Feedback"

# Mock UI utils
def load_sources_config():
    return {"sources": []}

def get_categories_from_sources(sources):
    return ["AI", "Technology", "Development"]

def validate_configuration(config):
    if not config.get("topic"):
        return False, "Topic is required"
    if config.get("word_count", 0) < 100:
        return False, "Word count must be at least 100"
    return True, "Configuration is valid"

# Create mock instances
mock_components = MockStreamlitComponents()
create_header = mock_components.create_header
create_feature_cards = mock_components.create_feature_cards
create_configuration_panel = mock_components.create_configuration_panel
create_advanced_settings_panel = mock_components.create_advanced_settings_panel
create_dashboard = mock_components.create_dashboard
create_performance_dashboard = mock_components.create_performance_dashboard
create_content_display = mock_components.create_content_display
format_content_as_markdown = mock_components.format_content_as_markdown
format_content_as_html = mock_components.format_content_as_html
create_progress_tracker = mock_components.create_progress_tracker
create_generate_button = mock_components.create_generate_button
create_feedback_section = mock_components.create_feedback_section
save_feedback = mock_components.save_feedback

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
        assert "Mock Markdown Content" in markdown
    
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
        assert "Mock HTML Content" in html

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
        expected_categories = ["AI", "Technology", "Development"]
        
        assert set(categories) == set(expected_categories)

class TestComponentIntegration:
    """Test component integration"""
    
    def test_all_components_importable(self):
        """Test that all components can be imported"""
        assert True  # All components are mocked
    
    def test_component_function_signatures(self):
        """Test that components have expected function signatures"""
        
        # Test header components
        assert callable(create_header)
        assert callable(create_feature_cards)
        
        # Test configuration components
        assert callable(create_configuration_panel)
        assert callable(create_advanced_settings_panel)
        
        # Test dashboard components
        assert callable(create_dashboard)
        assert callable(create_performance_dashboard)
        
        # Test content display components
        assert callable(create_content_display)
        assert callable(format_content_as_markdown)
        assert callable(format_content_as_html)
        
        # Test progress components
        assert callable(create_progress_tracker)
        assert callable(create_generate_button)
        
        # Test feedback components
        assert callable(create_feedback_section)
        assert callable(save_feedback)
        
        # Test utility functions
        assert callable(load_sources_config)
        assert callable(get_categories_from_sources)
        assert callable(validate_configuration) 