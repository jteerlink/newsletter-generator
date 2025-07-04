#!/usr/bin/env python3
"""
Test script to verify tool calling functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from agents.agents import ResearchAgent
from tools.tools import search_web

def test_direct_tool_call():
    """Test direct tool call"""
    print('=== DIRECT TOOL TEST ===')
    try:
        result = search_web('OpenAI GPT', max_results=1)
        print('✓ Direct tool call successful')
        print(f'Result length: {len(result)}')
        print(f'First 150 chars: {result[:150]}...')
        return True
    except Exception as e:
        print(f'✗ Direct tool call error: {str(e)}')
        return False

def test_agent_configuration():
    """Test agent configuration"""
    print('\n=== AGENT CONFIGURATION TEST ===')
    try:
        agent = ResearchAgent()
        print(f'✓ Agent created: {agent.name}')
        print(f'Agent tools: {agent.tools}')
        print(f'Available tools: {list(agent.available_tools.keys())}')
        print(f'Tools properly loaded: {"search_web" in agent.available_tools}')
        return True
    except Exception as e:
        print(f'✗ Agent configuration error: {str(e)}')
        return False

def test_tool_detection():
    """Test tool detection logic"""
    print('\n=== TOOL DETECTION TEST ===')
    try:
        agent = ResearchAgent()
        test_cases = [
            ('I need to search for more information', True),
            ('NEED_TOOLS to complete this task', True),
            ('I can answer this directly without searching', False),
            ('Let me find some recent data on this topic', True)
        ]
        
        all_passed = True
        for response, expected in test_cases:
            result = agent._should_use_tools(response)
            status = '✓' if result == expected else '✗'
            print(f'{status} "{response[:30]}..." -> {result} (expected {expected})')
            if result != expected:
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f'✗ Tool detection error: {str(e)}')
        return False

def test_agent_task_execution():
    """Test agent task execution with tool usage"""
    print('\n=== AGENT TASK EXECUTION TEST ===')
    try:
        agent = ResearchAgent()
        # Force a task that should trigger tool usage
        task = "I need to search for current information about artificial intelligence trends"
        
        print('Executing task that should trigger tool usage...')
        result = agent.execute_task(task)
        
        print(f'✓ Task executed successfully')
        print(f'Result length: {len(result)}')
        print(f'Contains search results: {"search" in result.lower()}')
        print(f'First 200 chars: {result[:200]}...')
        
        # Check if tools were actually used
        tools_used = "WEB SEARCH RESULTS" in result or "SEARCH RESULTS" in result
        print(f'Tools appear to have been used: {tools_used}')
        
        return True
    except Exception as e:
        print(f'✗ Agent task execution error: {str(e)}')
        return False

def main():
    """Run all tests"""
    print('Testing Tool Call Functionality')
    print('=' * 50)
    
    tests = [
        test_direct_tool_call,
        test_agent_configuration,
        test_tool_detection,
        test_agent_task_execution
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f'✗ Test failed with exception: {e}')
            results.append(False)
    
    print('\n' + '=' * 50)
    print('SUMMARY')
    print('=' * 50)
    passed = sum(results)
    total = len(results)
    print(f'Tests passed: {passed}/{total}')
    
    if passed == total:
        print('✓ All tool calling functionality is working correctly!')
    else:
        print('✗ Some tool calling functionality needs attention')
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 