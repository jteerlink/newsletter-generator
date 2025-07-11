#!/usr/bin/env python3
"""
SerperDevTool Demo Script

This script demonstrates how to use SerperDevTool with a real API key
for web search functionality in the newsletter generation system.

To use this script:
1. Get a Serper API key from https://serper.dev/
2. Set the SERPER_API_KEY environment variable
3. Run this script

Example usage:
    export SERPER_API_KEY="your-api-key-here"
    python dev/serper_tool_demo.py
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_basic_search():
    """Demonstrate basic SerperDevTool search functionality"""
    print("🔍 Basic SerperDevTool Search Demo")
    print("=" * 40)
    
    # Check for API key
    api_key = os.environ.get('SERPER_API_KEY')
    if not api_key:
        print("❌ SERPER_API_KEY not found!")
        print("💡 Get your API key from https://serper.dev/")
        print("💡 Set it with: export SERPER_API_KEY='your-key-here'")
        return False
    
    try:
        from crewai_tools import SerperDevTool
        
        # Initialize the search tool
        search_tool = SerperDevTool()
        
        # Test queries
        test_queries = [
            "latest AI developments 2024",
            "Python programming best practices",
            "machine learning research papers"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Searching: '{query}'")
            result = search_tool.run(query)
            
            print(f"📊 Result type: {type(result)}")
            print(f"📝 Result preview: {str(result)[:300]}...")
            print("-" * 50)
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_crewai_research_agent():
    """Demonstrate CrewAI Agent using SerperDevTool for research"""
    print("\n🤖 CrewAI Research Agent Demo")
    print("=" * 40)
    
    # Check for API key
    api_key = os.environ.get('SERPER_API_KEY')
    if not api_key:
        print("❌ SERPER_API_KEY not found!")
        return False
    
    try:
        from crewai import Agent, Task, Crew
        from crewai_tools import SerperDevTool
        
        # Initialize the search tool
        search_tool = SerperDevTool()
        
        # Create a research agent
        researcher = Agent(
            role='Technical Research Specialist',
            goal='Research and analyze technical topics for newsletter content',
            backstory='''You are an expert technical researcher who specializes in 
            finding the latest developments in AI, machine learning, and software development.
            You provide concise, accurate summaries of technical topics.''',
            tools=[search_tool],
            verbose=True
        )
        
        # Create research tasks
        research_task = Task(
            description='''Research the latest developments in AI and machine learning 
            for this week. Focus on new research papers, tool releases, and industry news.
            Provide a structured summary with key points and sources.''',
            expected_output='''A structured summary containing:
            - 3-5 key developments
            - Brief description of each
            - Relevance to the tech community
            - Sources/links where applicable''',
            agent=researcher
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )
        
        print("🚀 Executing research task...")
        result = crew.kickoff()
        
        print("\n✅ Research completed!")
        print("📝 Research Results:")
        print("=" * 50)
        print(result)
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demo_newsletter_content_search():
    """Demonstrate newsletter-specific content search"""
    print("\n📰 Newsletter Content Search Demo")
    print("=" * 40)
    
    # Check for API key
    api_key = os.environ.get('SERPER_API_KEY')
    if not api_key:
        print("❌ SERPER_API_KEY not found!")
        return False
    
    try:
        from crewai_tools import SerperDevTool
        
        # Initialize the search tool
        search_tool = SerperDevTool()
        
        # Newsletter-specific queries
        newsletter_queries = [
            "site:github.com trending AI projects this week",
            "site:arxiv.org machine learning papers 2024",
            "site:techcrunch.com AI startup news",
            "site:news.ycombinator.com AI discussion",
            "Python new features releases 2024"
        ]
        
        search_results = {}
        
        for query in newsletter_queries:
            print(f"\n🔍 Newsletter search: '{query}'")
            try:
                result = search_tool.run(query)
                search_results[query] = result
                print(f"✅ Found content for: {query}")
            except Exception as e:
                print(f"❌ Failed search for '{query}': {e}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"dev/newsletter_search_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(search_results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all demos"""
    print("🌐 SerperDevTool Demonstration Suite")
    print("=" * 50)
    
    # Check environment
    print(f"Python version: {sys.version}")
    print(f"SERPER_API_KEY set: {'Yes' if os.environ.get('SERPER_API_KEY') else 'No'}")
    
    try:
        from crewai import __version__ as crewai_version
        print(f"CrewAI version: {crewai_version}")
    except:
        print("CrewAI version: Could not determine")
    
    if not os.environ.get('SERPER_API_KEY'):
        print("\n⚠️  No SERPER_API_KEY found!")
        print("🔗 Get your API key from: https://serper.dev/")
        print("💡 Set it with: export SERPER_API_KEY='your-key-here'")
        print("🚀 Then run this script again")
        return False
    
    # Run demos
    demos = [
        ("Basic Search", demo_basic_search),
        ("CrewAI Research Agent", demo_crewai_research_agent),
        ("Newsletter Content Search", demo_newsletter_content_search),
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            results.append(demo_func())
        except Exception as e:
            print(f"❌ Demo '{name}' failed: {e}")
            results.append(False)
    
    # Summary
    print("\n📋 Demo Summary")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    print(f"✅ Successful demos: {passed}/{total}")
    
    if passed == total:
        print("🎉 All demos completed successfully!")
        print("🚀 SerperDevTool is fully functional and ready for use!")
    else:
        print("⚠️  Some demos failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 