#!/usr/bin/env python3
"""
Phase 3 Tools Demo

This script demonstrates the enhanced tools implemented in Phase 3:
- Grammar and Style Linter
- Enhanced Search Tool

These tools provide advanced content validation and intelligent search capabilities
for the newsletter generation system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools.grammar_linter import GrammarAndStyleLinter
from tools.enhanced_search import EnhancedSearchTool
import json

def demo_grammar_linter():
    """Demonstrate the grammar and style linter functionality."""
    print("=" * 60)
    print("GRAMMAR AND STYLE LINTER DEMO")
    print("=" * 60)
    
    linter = GrammarAndStyleLinter()
    
    # Test content with various issues
    test_content = """
    The project was optimized to leverage synergies and think outside the box.  
    Its very important that we utilize the latest technology.  The team was 
    working very hard to achieve their goals.  This sentence has  double  spaces.
    """
    
    print("Testing content with various issues:")
    print(f"Content: {test_content.strip()}")
    print()
    
    # Get detailed analysis
    result = linter.check_content(test_content)
    
    print("Linter Results:")
    print(f"- Overall Score: {result.overall_score:.2f}")
    print(f"- Grammar Score: {result.grammar_score:.2f}")
    print(f"- Style Score: {result.style_score:.2f}")
    print(f"- Total Issues: {result.total_issues}")
    print()
    
    # Show grammar issues
    if result.grammar_issues:
        print("Grammar Issues:")
        for i, issue in enumerate(result.grammar_issues[:3], 1):
            print(f"  {i}. {issue.message} (Severity: {issue.severity})")
            if issue.suggestion:
                print(f"     Suggestion: {issue.suggestion}")
        print()
    
    # Show style issues
    if result.style_issues:
        print("Style Issues:")
        for i, issue in enumerate(result.style_issues[:3], 1):
            print(f"  {i}. {issue.message} (Severity: {issue.severity})")
            if issue.suggestion:
                print(f"     Suggestion: {issue.suggestion}")
        print()
    
    # Show suggestions
    if result.suggestions:
        print("Improvement Suggestions:")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()
    
    # Show summary
    print("Content Summary:")
    summary = result.summary
    print(f"- Word Count: {summary.get('word_count', 0)}")
    print(f"- Sentence Count: {summary.get('sentence_count', 0)}")
    print(f"- Average Sentence Length: {summary.get('average_sentence_length', 0):.1f}")
    print(f"- High Severity Issues: {summary.get('high_severity_issues', 0)}")
    print(f"- Medium Severity Issues: {summary.get('medium_severity_issues', 0)}")
    print(f"- Low Severity Issues: {summary.get('low_severity_issues', 0)}")
    print()
    
    # Quick feedback demo
    print("Quick Feedback:")
    quick_feedback = linter.get_quick_feedback(test_content)
    print(f"- Score: {quick_feedback['score']:.2f}")
    print(f"- Needs Revision: {quick_feedback['needs_revision']}")
    print(f"- High Priority Issues: {quick_feedback['high_priority_issues']}")
    print()

def demo_enhanced_search():
    """Demonstrate the enhanced search tool functionality."""
    print("=" * 60)
    print("ENHANCED SEARCH TOOL DEMO")
    print("=" * 60)
    
    search_tool = EnhancedSearchTool()
    
    # Test query expansion
    query = "artificial intelligence"
    context = "latest developments in machine learning and neural networks"
    
    print("Query Expansion Demo:")
    print(f"Original Query: {query}")
    print(f"Context: {context}")
    
    expanded_query = search_tool._expand_query(query, context)
    print(f"Expanded Query: {expanded_query}")
    print()
    
    # Test key term extraction
    text = "The latest developments in artificial intelligence and machine learning show promising results."
    key_terms = search_tool._extract_key_terms(text)
    print("Key Term Extraction:")
    print(f"Text: {text}")
    print(f"Key Terms: {key_terms}")
    print()
    
    # Test confidence scoring
    test_result = {
        'title': 'Artificial Intelligence Latest News',
        'snippet': 'Latest developments in AI and machine learning technology',
        'url': 'https://example.com/ai-news'
    }
    
    confidence_score = search_tool._calculate_confidence_score(test_result, query)
    relevance_score = search_tool._calculate_relevance_score(test_result, query, context)
    authority_score = search_tool._calculate_authority_score(test_result)
    
    print("Result Scoring Demo:")
    print(f"Result: {test_result['title']}")
    print(f"- Confidence Score: {confidence_score:.2f}")
    print(f"- Relevance Score: {relevance_score:.2f}")
    print(f"- Authority Score: {authority_score:.2f}")
    print()
    
    # Test search analytics (with mock data)
    print("Search Analytics Demo:")
    mock_results = [
        {
            'title': 'AI News 1',
            'url': 'https://reuters.com/ai-news-1',
            'snippet': 'Latest AI developments',
            'source': 'duckduckgo',
            'confidence_score': 0.8,
            'relevance_score': 0.7,
            'freshness_score': 0.6,
            'authority_score': 0.5,
            'overall_score': 0.65,
            'metadata': {},
            'timestamp': '2024-01-01T00:00:00'
        },
        {
            'title': 'AI News 2',
            'url': 'https://techcrunch.com/ai-news-2',
            'snippet': 'Machine learning updates',
            'source': 'duckduckgo',
            'confidence_score': 0.9,
            'relevance_score': 0.8,
            'freshness_score': 0.7,
            'authority_score': 0.6,
            'overall_score': 0.75,
            'metadata': {},
            'timestamp': '2024-01-01T00:00:00'
        }
    ]
    
    # Convert to SearchResult objects
    from tools.enhanced_search import SearchResult
    from datetime import datetime
    
    search_results = []
    for result in mock_results:
        search_results.append(SearchResult(
            title=result['title'],
            url=result['url'],
            snippet=result['snippet'],
            source=result['source'],
            confidence_score=result['confidence_score'],
            relevance_score=result['relevance_score'],
            freshness_score=result['freshness_score'],
            authority_score=result['authority_score'],
            overall_score=result['overall_score'],
            metadata=result['metadata'],
            timestamp=datetime.now()
        ))
    
    # Calculate analytics
    avg_confidence = sum(r.confidence_score for r in search_results) / len(search_results)
    avg_relevance = sum(r.relevance_score for r in search_results) / len(search_results)
    avg_freshness = sum(r.freshness_score for r in search_results) / len(search_results)
    avg_authority = sum(r.authority_score for r in search_results) / len(search_results)
    
    source_counts = {}
    for result in search_results:
        source_counts[result.source] = source_counts.get(result.source, 0) + 1
    
    analytics = {
        'query': query,
        'total_results': len(search_results),
        'average_scores': {
            'confidence': round(avg_confidence, 3),
            'relevance': round(avg_relevance, 3),
            'freshness': round(avg_freshness, 3),
            'authority': round(avg_authority, 3)
        },
        'source_distribution': source_counts,
        'top_results': [
            {
                'title': r.title,
                'url': r.url,
                'score': round(r.overall_score, 3)
            }
            for r in search_results[:5]
        ]
    }
    
    print("Analytics Results:")
    print(f"- Query: {analytics['query']}")
    print(f"- Total Results: {analytics['total_results']}")
    print(f"- Average Scores:")
    for score_type, score in analytics['average_scores'].items():
        print(f"  * {score_type}: {score}")
    print(f"- Source Distribution: {analytics['source_distribution']}")
    print(f"- Top Results:")
    for i, result in enumerate(analytics['top_results'], 1):
        print(f"  {i}. {result['title']} (Score: {result['score']})")
    print()

def demo_integration():
    """Demonstrate integration between the tools."""
    print("=" * 60)
    print("TOOL INTEGRATION DEMO")
    print("=" * 60)
    
    linter = GrammarAndStyleLinter()
    search_tool = EnhancedSearchTool()
    
    # Simulate a content creation workflow
    content = """
    The latest artificial intelligence developments show that AI is becoming 
    very important in our daily lives.  The technology was optimized to 
    leverage synergies and think outside the box.  Its very exciting to see 
    these developments.
    """
    
    print("Content Creation Workflow:")
    print(f"Original Content: {content.strip()}")
    print()
    
    # Step 1: Check content quality
    print("Step 1: Content Quality Check")
    linter_result = linter.check_content(content)
    print(f"- Quality Score: {linter_result.overall_score:.2f}")
    print(f"- Issues Found: {linter_result.total_issues}")
    print()
    
    # Step 2: If issues found, search for verification
    if linter_result.total_issues > 0:
        print("Step 2: Search for Verification")
        print("Searching for 'artificial intelligence developments' to verify claims...")
        
        # In a real scenario, this would perform an actual search
        # For demo purposes, we'll show the query expansion
        query = "artificial intelligence developments"
        context = "latest AI technology news and research"
        expanded_query = search_tool._expand_query(query, context)
        
        print(f"- Original Query: {query}")
        print(f"- Expanded Query: {expanded_query}")
        print(f"- Context: {context}")
        print()
    
    # Step 3: Generate improvement suggestions
    print("Step 3: Improvement Suggestions")
    if linter_result.suggestions:
        for i, suggestion in enumerate(linter_result.suggestions, 1):
            print(f"  {i}. {suggestion}")
    print()
    
    print("Integration Complete!")
    print()

def main():
    """Run the Phase 3 tools demo."""
    print("PHASE 3 TOOLS DEMONSTRATION")
    print("Enhanced Agent Architecture Implementation")
    print()
    
    try:
        # Demo each tool
        demo_grammar_linter()
        demo_enhanced_search()
        demo_integration()
        
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Phase 3 Tools Summary:")
        print("✓ Grammar and Style Linter - Content validation with scoring")
        print("✓ Enhanced Search Tool - Multi-provider search with confidence scoring")
        print("✓ Tool Integration - Seamless workflow between tools")
        print()
        print("These tools provide the foundation for enhanced agent capabilities")
        print("in the newsletter generation system.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 