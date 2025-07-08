#!/usr/bin/env python3
"""
Test script for Content Validator - Phase 1 Implementation
Tests the new content validation system for newsletter generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.content_validator import ContentValidator

def test_repetition_detection():
    """Test repetition detection functionality."""
    print("=== Testing Repetition Detection ===")
    
    # Test with repetitive content
    repetitive_content = """
    This is a test paragraph about AI development. AI development is crucial for the future.
    AI development has many applications in various fields. The importance of AI development 
    cannot be overstated. AI development continues to advance rapidly. AI development is 
    transforming industries across the globe. AI development requires skilled professionals.
    """
    
    validator = ContentValidator()
    result = validator.detect_repetition(repetitive_content)
    
    print(f"Repetition Score: {result['repetition_score']:.2f}")
    print(f"Concept Repetitions: {len(result['concept_repetition'])}")
    for concept in result['concept_repetition']:
        print(f"  - '{concept['phrase']}' (count: {concept['count']})")
    
    print(f"Section Repetitions: {len(result['section_repetition'])}")
    for section in result['section_repetition']:
        print(f"  - Sections {section['section1_index']} and {section['section2_index']} (similarity: {section['similarity']:.2f})")
    
    # Test with good content
    good_content = """
    Machine learning algorithms have revolutionized data analysis. These sophisticated 
    techniques enable computers to learn patterns from large datasets. Neural networks,
    a subset of machine learning, mimic the human brain's structure. Deep learning 
    architectures have achieved remarkable success in image recognition tasks.
    """
    
    good_result = validator.detect_repetition(good_content)
    print(f"\nGood Content Repetition Score: {good_result['repetition_score']:.2f}")

def test_expert_quote_validation():
    """Test expert quote validation functionality."""
    print("\n=== Testing Expert Quote Validation ===")
    
    # Test with suspicious quotes
    suspicious_content = """
    According to Dr. John Smith from MIT, "AI will change everything by 2025."
    Professor Jane Doe, Co-Founder of TechCorp, stated, "Machine learning is the future."
    AI researcher Michael Johnson from Stanford believes, "Deep learning will solve all problems."
    """
    
    validator = ContentValidator()
    result = validator.validate_expert_quotes(suspicious_content)
    
    print(f"Suspicious Quotes Found: {len(result['suspicious_quotes'])}")
    for quote in result['suspicious_quotes']:
        print(f"  - {quote}")
    
    # Test with good content
    good_content = """
    Recent research published in Nature shows promising results for AI applications.
    Industry reports indicate significant growth in machine learning adoption.
    Academic studies demonstrate the effectiveness of neural networks in pattern recognition.
    """
    
    good_result = validator.validate_expert_quotes(good_content)
    print(f"\nGood Content Suspicious Quotes: {len(good_result['suspicious_quotes'])}")

def test_fact_checking():
    """Test basic fact-checking functionality."""
    print("\n=== Testing Fact Checking ===")
    
    # Test with unverifiable claims
    questionable_content = """
    Studies show that 87% of companies will adopt AI by 2025.
    Research indicates that machine learning increases productivity by 340%.
    Experts predict that AI will create 15 million jobs next year.
    According to recent data, 95% of developers use AI tools daily.
    """
    
    validator = ContentValidator()
    result = validator.basic_fact_check(questionable_content)
    
    print(f"Unverifiable Claims Found: {len(result['unverifiable_claims'])}")
    for claim in result['unverifiable_claims']:
        print(f"  - {claim}")
    
    # Test with better content
    better_content = """
    Machine learning has gained significant adoption in recent years.
    Many companies are exploring AI applications for business optimization.
    The technology sector continues to invest heavily in artificial intelligence.
    Developers increasingly incorporate AI tools into their workflows.
    """
    
    better_result = validator.basic_fact_check(better_content)
    print(f"\nBetter Content Unverifiable Claims: {len(better_result['unverifiable_claims'])}")

def test_content_quality_assessment():
    """Test overall content quality assessment."""
    print("\n=== Testing Content Quality Assessment ===")
    
    # Test with poor quality content
    poor_content = """
    AI is good. AI is very good. AI helps people. AI is the future.
    AI will change everything. AI is important. AI is everywhere.
    AI is growing. AI is useful. AI is amazing.
    """
    
    validator = ContentValidator()
    result = validator.assess_content_quality(poor_content)
    
    print(f"Poor Content Overall Score: {result['overall_score']:.2f}")
    print(f"Structure Score: {result['structure_score']:.2f}")
    print(f"Readability Score: {result['readability_score']:.2f}")
    print(f"Engagement Score: {result['engagement_score']:.2f}")
    
    # Test with good quality content
    good_content = """
    Machine learning represents a paradigm shift in how we approach complex problem-solving.
    Unlike traditional programming, where developers explicitly code solutions, machine learning
    enables systems to learn patterns from data and make predictions or decisions autonomously.
    
    This revolutionary approach has found applications across diverse industries, from healthcare
    diagnostics to financial fraud detection. The technology's ability to process vast amounts
    of information and identify subtle patterns that humans might miss has made it invaluable
    for tackling previously intractable challenges.
    
    Consider the example of medical imaging analysis. Traditional methods required radiologists
    to manually examine thousands of images, a time-consuming process prone to human error.
    Modern machine learning systems can now analyze medical scans in seconds, often with
    accuracy rates exceeding human specialists. This capability has not only improved diagnostic
    speed but has also enabled early detection of conditions that might otherwise go unnoticed.
    """
    
    good_result = validator.assess_content_quality(good_content)
    print(f"\nGood Content Overall Score: {good_result['overall_score']:.2f}")
    print(f"Structure Score: {good_result['structure_score']:.2f}")
    print(f"Readability Score: {good_result['readability_score']:.2f}")
    print(f"Engagement Score: {good_result['engagement_score']:.2f}")

def test_comprehensive_validation():
    """Test comprehensive validation on a sample newsletter excerpt."""
    print("\n=== Testing Comprehensive Validation ===")
    
    # Sample newsletter content with various quality issues
    sample_content = """
    **The Future of AI Development**
    
    AI development is crucial for the future. AI development has many applications in various fields. 
    The importance of AI development cannot be overstated. According to Dr. John Smith from MIT, 
    "AI will change everything by 2025." Studies show that 87% of companies will adopt AI by 2025.
    
    AI development continues to advance rapidly. AI development is transforming industries across 
    the globe. Professor Jane Doe, Co-Founder of TechCorp, stated, "Machine learning is the future."
    Research indicates that machine learning increases productivity by 340%.
    
    AI development requires skilled professionals. AI development is everywhere. AI development
    will create new opportunities. Experts predict that AI will create 15 million jobs next year.
    According to recent data, 95% of developers use AI tools daily.
    """
    
    validator = ContentValidator()
    
    # Run all validation checks
    repetition_result = validator.detect_repetition(sample_content)
    expert_result = validator.validate_expert_quotes(sample_content)
    fact_result = validator.basic_fact_check(sample_content)
    quality_result = validator.assess_content_quality(sample_content)
    
    print(f"Repetition Score: {repetition_result['repetition_score']:.2f}")
    print(f"Suspicious Quotes: {len(expert_result['suspicious_quotes'])}")
    print(f"Unverifiable Claims: {len(fact_result['unverifiable_claims'])}")
    print(f"Content Quality Score: {quality_result['overall_score']:.2f}")
    
    # Calculate overall validation score
    quality_factors = {
        'repetition': max(0, 1 - repetition_result['repetition_score']),
        'expert_credibility': max(0, 1 - len(expert_result['suspicious_quotes']) / 10),
        'fact_accuracy': max(0, 1 - len(fact_result['unverifiable_claims']) / 10),
        'content_quality': quality_result['overall_score']
    }
    
    overall_score = sum(quality_factors.values()) / len(quality_factors)
    print(f"\nOverall Validation Score: {overall_score:.2f}")
    
    print("\nQuality Factors:")
    for factor, score in quality_factors.items():
        print(f"  {factor}: {score:.2f}")

if __name__ == "__main__":
    print("Content Validator Test Suite - Phase 1 Implementation")
    print("=" * 60)
    
    test_repetition_detection()
    test_expert_quote_validation()
    test_fact_checking()
    test_content_quality_assessment()
    test_comprehensive_validation()
    
    print("\n" + "=" * 60)
    print("Content Validator testing completed!")
    print("Phase 1 improvements are now active in the newsletter generation system.") 