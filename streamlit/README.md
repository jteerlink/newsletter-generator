# Streamlit Web Interface

This directory contains the Streamlit web interface for the newsletter generator:

## Main Interface Files
- **app.py** - Main Streamlit application with simplified interface
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
   streamlit run streamlit/app.py
   ```

## Features
- **Hierarchical Deep-Dive Pipeline** - Comprehensive newsletter generation with ManagerAgent orchestration
- **Content Pillar Selection** - Choose from News & Breakthroughs, Tools & Tutorials, or Deep Dives & Analysis
- **Topic Input & Validation** - Real-time topic validation and suggestions
- **Audience Selection** - 7+ audience types for targeted content
- **Configuration Controls** - Length settings, quality focus areas, special requirements
- **Progress Tracking** - Real-time generation progress with animations
- **Quality Assurance** - Real-time quality monitoring with technical accuracy validation
- **Output Display** - Multi-format output with statistics and download options
- **Performance Analytics** - Execution time tracking and visualization

## Simplified Interface
The interface has been streamlined to focus on the hierarchical deep-dive pipeline:
- **Single Pipeline Mode** - Only deep dive pipeline with hierarchical execution
- **Content Pillars** - Three focused content areas for targeted generation
- **Quality Monitoring** - Real-time quality metrics and validation
- **Modern UI** - Clean, professional interface with consistent design system

## Architecture
- **ManagerAgent** - Orchestrates the hierarchical workflow
- **Specialized Agents** - Research, Writer, Editor, and Planner agents
- **Quality Assurance** - Multi-gate validation system
- **Notion Integration** - Direct publishing to Notion workspace

The interface provides a modern, user-friendly way to generate comprehensive technical newsletters using the hierarchical multi-agent system. 