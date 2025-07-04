# Streamlit Web Interface

This directory contains the Streamlit web interface for the newsletter generator:

## Main Interface Files
- **streamlit_app.py** - Basic Streamlit application with core functionality
- **streamlit_app_enhanced.py** - Enhanced version with advanced UI components and styling
- **ui_components.py** - Reusable UI components and styling functions
- **launch_streamlit.py** - Smart launcher script with dependency checking

## Documentation
- **README_STREAMLIT.md** - Comprehensive documentation for the Streamlit interface

## Quick Start
1. Install dependencies:
   ```bash
   pip install streamlit plotly pyyaml
   ```

2. Launch the application:
   ```bash
   python streamlit/launch_streamlit.py
   ```
   
   Or run directly:
   ```bash
   streamlit run streamlit/streamlit_app_enhanced.py
   ```

## Features
- **Topic Input & Validation** - Real-time topic validation and suggestions
- **Audience Selection** - 7+ audience types for targeted content
- **Workflow Options** - Standard multi-agent vs hierarchical workflows
- **Content Controls** - Length settings, quality focus areas, source selection
- **Progress Tracking** - Real-time generation progress with animations
- **Output Display** - Multi-format output with statistics and download options
- **Performance Analytics** - Execution time tracking and visualization
- **Feedback System** - Multi-dimensional feedback collection

## Interface Versions
- **streamlit_app.py** - Basic interface for simple use cases
- **streamlit_app_enhanced.py** - Recommended version with full feature set and modern styling

The enhanced version includes advanced CSS styling, better progress tracking, and comprehensive feedback analytics. 