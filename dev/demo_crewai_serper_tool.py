#!/usr/bin/env python3
"""
Demo script for CrewAI SerperDevTool integration with Python 3.10

This script demonstrates how to use the updated CrewAI tools with SerperDevTool
for web search functionality in the newsletter generation system.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up environment variables for demo
os.environ['SERPER_API_KEY'] = 'your-serper-api-key-here'  # Replace with actual key

def demo_serper_tool():
    """Demonstrate SerperDevTool functionality"""
    print("=== CrewAI SerperDevTool Demo ===")
    print("Python version:", sys.version)
    print()
    
    # Check if required packages are available
    try:
        from crewai_tools import SerperDevTool
        print("✅ crewai-tools package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import crewai-tools: {e}")
        print("Install with: pip install crewai-tools")
        return False
    
    try:
        from crewai import Agent, Task, Crew
        print("✅ crewai package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import crewai: {e}")
        print("Install with: pip install crewai")
        return False
    
    # Initialize SerperDevTool
    try:
        serper_tool = SerperDevTool()
        print("✅ SerperDevTool initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SerperDevTool: {e}")
        print("Make sure SERPER_API_KEY is set in your environment")
        return False
    
    # Demo search queries
    demo_queries = [
        "artificial intelligence news 2024",
        "machine learning breakthroughs",
        "Python 3.10 new features",
        "CrewAI tutorial examples"
    ]
    
    print("\n=== Running Demo Searches ===")
    for query in demo_queries:
        print(f"\nSearching for: {query}")
        try:
            results = serper_tool.run(search_query=query)
            if isinstance(results, dict):
                organic_results = results.get('organic', [])
                print(f"Found {len(organic_results)} results")
                if organic_results:
                    for i, result in enumerate(organic_results[:3], 1):
                        print(f"  {i}. {result.get('title', 'No title')}")
                        print(f"     {result.get('snippet', 'No snippet')}")
                        print(f"     {result.get('link', 'No link')}")
            else:
                print(f"Raw results: {results}")
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    return True

def demo_crewai_agents():
    """Demonstrate CrewAI agents with SerperDevTool"""
    print("\n=== CrewAI Agents Demo ===")
    
    try:
        from crewai import Agent, Task, Crew
        from crewai_tools import SerperDevTool
        
        # Create search tool
        search_tool = SerperDevTool()
        
        # Create a research agent
        researcher = Agent(
            role='Research Specialist',
            goal='Find and analyze information about AI and technology trends',
            backstory='You are an expert researcher with a focus on AI and technology.',
            tools=[search_tool],
            verbose=True
        )
        
        # Create a research task
        research_task = Task(
            description='Search for recent AI developments and summarize key findings',
            agent=researcher,
            expected_output='A summary of recent AI developments with key insights'
        )
        
        # Create and run crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )
        
        print("✅ CrewAI agents configured successfully")
        print("Note: To run the crew, you would call: crew.kickoff()")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to set up CrewAI agents: {e}")
        return False

def demo_enhanced_search_tool():
    """Demonstrate the enhanced search tool from the project"""
    print("\n=== Enhanced Search Tool Demo ===")
    
    try:
        from tools.crewai_tools import CrewAISearchTool
        
        # Create enhanced search tool
        search_tool = CrewAISearchTool(max_results_per_search=5)
        
        # Test search
        query = "latest AI research papers"
        results = search_tool.run(query)
        
        print(f"✅ Enhanced search tool worked successfully")
        print(f"Results preview: {results[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced search tool failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("=== Python Version Check ===")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 10):
        print("✅ Python 3.10+ detected - Compatible with CrewAI")
        return True
    else:
        print("❌ Python 3.10+ required for modern CrewAI")
        return False

def main():
    """Main demo function"""
    print("CrewAI SerperDevTool Demo - Python 3.10 Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Demo SerperDevTool
    if not demo_serper_tool():
        return
    
    # Demo CrewAI agents
    demo_crewai_agents()
    
    # Demo enhanced search tool
    demo_enhanced_search_tool()
    
    print("\n=== Demo Complete ===")
    print("To use SerperDevTool in your project:")
    print("1. Set SERPER_API_KEY environment variable")
    print("2. Import from crewai_tools: from crewai_tools import SerperDevTool")
    print("3. Initialize: tool = SerperDevTool()")
    print("4. Search: results = tool.run(search_query='your query')")

if __name__ == "__main__":
    main() 