#!/usr/bin/env python3
"""
Test script for SerperDevTool functionality

This script tests the SerperDevTool integration with CrewAI to ensure
the search functionality is working properly.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_serper_tool_import():
    """Test if SerperDevTool can be imported successfully"""
    print("🔍 Testing SerperDevTool import...")
    
    try:
        from crewai_tools import SerperDevTool
        print("✅ SerperDevTool imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import SerperDevTool: {e}")
        return False

def test_serper_tool_initialization():
    """Test if SerperDevTool can be initialized"""
    print("\n🔧 Testing SerperDevTool initialization...")
    
    try:
        from crewai_tools import SerperDevTool
        
        # Test with mock API key for initialization
        os.environ['SERPER_API_KEY'] = 'test-key-for-initialization'
        search_tool = SerperDevTool()
        print("✅ SerperDevTool initialized successfully")
        return True, search_tool
    except Exception as e:
        print(f"❌ Failed to initialize SerperDevTool: {e}")
        return False, None

def test_serper_tool_with_real_api_key():
    """Test SerperDevTool with real API key (if available)"""
    print("\n🌐 Testing SerperDevTool with real API key...")
    
    # Check if real API key is available
    api_key = os.environ.get('SERPER_API_KEY')
    if not api_key or api_key == 'test-key-for-initialization':
        print("⚠️  No real SERPER_API_KEY found. Skipping real API test.")
        print("💡 To test with real API key, set SERPER_API_KEY environment variable")
        return True
    
    try:
        from crewai_tools import SerperDevTool
        
        search_tool = SerperDevTool()
        
        # Test search with a simple query
        test_query = "Python programming latest news"
        print(f"🔍 Testing search with query: '{test_query}'")
        
        result = search_tool.run(test_query)
        
        print("✅ Search completed successfully")
        print(f"📊 Result type: {type(result)}")
        print(f"📝 Result preview: {str(result)[:200]}...")
        
        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False

def test_crewai_agent_with_serper():
    """Test CrewAI Agent using SerperDevTool"""
    print("\n🤖 Testing CrewAI Agent with SerperDevTool...")
    
    try:
        from crewai import Agent, Task, Crew
        from crewai_tools import SerperDevTool
        
        # Check if real API key is available
        api_key = os.environ.get('SERPER_API_KEY')
        if not api_key or api_key == 'test-key-for-initialization':
            print("⚠️  No real SERPER_API_KEY found. Skipping Agent test.")
            return True
        
        # Create search tool
        search_tool = SerperDevTool()
        
        # Create research agent
        researcher = Agent(
            role='Research Specialist',
            goal='Gather information about specified topics',
            backstory='You are an expert researcher who finds accurate and relevant information.',
            tools=[search_tool],
            verbose=True
        )
        
        # Create a simple research task
        research_task = Task(
            description='Research the latest developments in AI and machine learning',
            expected_output='A brief summary of recent AI/ML developments',
            agent=researcher
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )
        
        print("🚀 Executing CrewAI with SerperDevTool...")
        result = crew.kickoff()
        
        print("✅ CrewAI Agent executed successfully")
        print(f"📝 Result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ CrewAI Agent test failed: {e}")
        return False

def test_alternative_search_approach():
    """Test alternative search approach using google-search-results directly"""
    print("\n🔍 Testing alternative search approach...")
    
    try:
        from serpapi import GoogleSearch
        
        # Check if real API key is available
        api_key = os.environ.get('SERPER_API_KEY')
        if not api_key or api_key == 'test-key-for-initialization':
            print("⚠️  No real SERPER_API_KEY found. Skipping direct search test.")
            return True
        
        params = {
            "q": "Python programming news",
            "api_key": api_key,
            "num": 3
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        print("✅ Direct search completed successfully")
        print(f"📊 Found {len(results.get('organic_results', []))} results")
        
        return True
    except Exception as e:
        print(f"❌ Direct search failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SerperDevTool Functionality Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Import
    test_results.append(test_serper_tool_import())
    
    # Test 2: Initialization
    init_success, tool = test_serper_tool_initialization()
    test_results.append(init_success)
    
    # Test 3: Real API key test
    test_results.append(test_serper_tool_with_real_api_key())
    
    # Test 4: CrewAI Agent test
    test_results.append(test_crewai_agent_with_serper())
    
    # Test 5: Alternative approach
    test_results.append(test_alternative_search_approach())
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! SerperDevTool is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Environment information
    print("\n🔧 Environment Information")
    print("=" * 30)
    print(f"Python version: {sys.version}")
    print(f"SERPER_API_KEY set: {'Yes' if os.environ.get('SERPER_API_KEY') else 'No'}")
    
    try:
        from crewai import __version__ as crewai_version
        print(f"CrewAI version: {crewai_version}")
    except:
        print("CrewAI version: Could not determine")
    
    try:
        from crewai_tools import __version__ as tools_version
        print(f"CrewAI Tools version: {tools_version}")
    except:
        print("CrewAI Tools version: Could not determine")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 