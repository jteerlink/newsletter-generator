# Python 3.10 and CrewAI with SerperDevTool Setup Guide

## Overview

This guide explains how to update the newsletter generator environment to use Python 3.10 with modern CrewAI versions and the SerperDevTool for web search functionality.

## Updated Dependencies

### Python Version Requirements
- **Python 3.10+** (required for modern CrewAI versions)
- Compatible with Python 3.10, 3.11, 3.12, and 3.13

### Key Package Updates

| Package | Old Version | New Version | Notes |
|---------|-------------|-------------|-------|
| `crewai` | 0.82.0 | 0.95.0+ | Major update with improved APIs |
| `crewai-tools` | N/A | 0.25.8+ | New separate package for tools |
| `langchain` | 0.1.0 | 0.2.0+ | Updated for compatibility |
| `chromadb` | 0.4.15 | 1.0.13+ | Stable release |
| `aiohttp` | 3.8.0 | 3.9.0+ | Better async support |

## Installation Steps

### 1. Update Python Environment

```bash
# Using conda
conda create -n news_env python=3.10 -y
conda activate news_env

# Using pyenv
pyenv install 3.10.12
pyenv local 3.10.12

# Using python-version manager
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Updated Dependencies

```bash
# Install core packages
pip install -r requirements.txt

# Or install individually
pip install crewai>=0.95.0
pip install crewai-tools>=0.25.8
pip install serper-dev>=0.1.0
pip install google-search-results>=2.4.2
```

### 3. Set Up API Keys

Create a `.env` file in your project root:

```env
# Serper API Key (for web search)
SERPER_API_KEY=your-serper-api-key-here

# Other API keys
OPENAI_API_KEY=your-openai-key-here
GROQ_API_KEY=your-groq-key-here
```

To get a Serper API key:
1. Visit [serper.dev](https://serper.dev)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Free tier includes 2,500 searches per month

### 4. Verify Installation

Run the demo script to verify everything works:

```bash
cd dev
python demo_crewai_serper_tool.py
```

## Code Changes

### Import Structure Updates

#### Old Import Style (deprecated)
```python
# Old - may not work in newer versions
from crewai.tools import SerperDevTool, ScrapeWebsiteTool
```

#### New Import Style (recommended)
```python
# New - use separate crewai-tools package
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew
```

### Error Handling for Compatibility

```python
try:
    from crewai_tools import SerperDevTool
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    SerperDevTool = None
```

## SerperDevTool Usage

### Basic Usage

```python
from crewai_tools import SerperDevTool

# Initialize
search_tool = SerperDevTool()

# Search
results = search_tool.run(search_query="AI news 2024")
```

### With CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Create search tool
search_tool = SerperDevTool()

# Create agent with search capability
researcher = Agent(
    role='Research Specialist',
    goal='Find and analyze information',
    backstory='Expert researcher focused on AI trends',
    tools=[search_tool],
    verbose=True
)

# Create task
research_task = Task(
    description='Search for recent AI developments',
    agent=researcher,
    expected_output='Summary of AI developments'
)

# Create and run crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

# Execute
result = crew.kickoff()
```

## Benefits of the Update

### 1. Better Performance
- Modern CrewAI versions are more efficient
- Improved error handling and logging
- Better async support

### 2. Enhanced Search Capabilities
- SerperDevTool provides Google search results
- More reliable than DuckDuckGo search
- Better structured results

### 3. Improved Compatibility
- Works with latest Python versions
- Better integration with other tools
- More stable API

### 4. Enhanced Features
- Better agent orchestration
- Improved task management
- More sophisticated tool integration

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Clear cache and reinstall
   pip cache purge
   pip uninstall crewai crewai-tools
   pip install crewai>=0.95.0 crewai-tools>=0.25.8
   ```

2. **API Key Issues**
   ```bash
   # Check environment variables
   echo $SERPER_API_KEY
   
   # Set temporarily
   export SERPER_API_KEY=your-key-here
   ```

3. **Version Conflicts**
   ```bash
   # Check installed versions
   pip list | grep crew
   
   # Update specific package
   pip install --upgrade crewai-tools
   ```

### Memory Issues with Large Projects

If you encounter memory issues:

```python
# Use smaller batch sizes
search_tool = SerperDevTool(max_results=3)

# Enable result caching
from tools.crewai_tools import CrewAISearchTool
search_tool = CrewAISearchTool(enable_caching=True)
```

## Migration Checklist

- [ ] Update Python to 3.10+
- [ ] Update requirements.txt with new versions
- [ ] Update pyproject.toml Python version
- [ ] Update setup.sh script
- [ ] Add SERPER_API_KEY to environment
- [ ] Update import statements
- [ ] Test search functionality
- [ ] Update any custom tools
- [ ] Run integration tests

## Next Steps

1. **Test the Environment**: Run the demo script to verify everything works
2. **Update Existing Code**: Migrate any old CrewAI tool usage
3. **Configure API Keys**: Set up Serper API key for search functionality
4. **Run Tests**: Execute the project's test suite to ensure compatibility

## Resources

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI Tools Repository](https://github.com/joaomdmoura/crewai-tools)
- [Serper API Documentation](https://serper.dev/docs)
- [Python 3.10 Features](https://docs.python.org/3.10/whatsnew/3.10.html)

This update provides a more robust and modern foundation for the newsletter generation system with better search capabilities and improved performance. 