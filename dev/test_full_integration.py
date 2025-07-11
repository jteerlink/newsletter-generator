#!/usr/bin/env python3

import sys
import os
import json
from datetime import datetime
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.quality_assurance_system import QualityAssuranceSystem

def create_realistic_newsletter_content():
    """Create realistic newsletter content for testing"""
    return {
        'subject_line': 'New AI Models & Dev Tools This Week 🚀',
        'preview_text': 'Claude 3.5 Sonnet updates, new VS Code features, and Python 3.13 beta',
        'news_breakthroughs': [
            {
                'headline': 'Claude 3.5 Sonnet Gets Computer Use',
                'content': '''Anthropic released Claude 3.5 Sonnet with computer use capabilities. 
                The model can now interact with desktop applications and web browsers through screenshots. 
                This represents a significant advancement in AI agent capabilities.
                
                Key features include:
                • Screenshot analysis and interpretation
                • Mouse and keyboard automation
                • Multi-step task completion
                • Integration with popular applications'''
            },
            {
                'headline': 'PyTorch 2.5 Performance Improvements',
                'content': '''PyTorch 2.5 introduces torch.compile optimizations that deliver 2x speedup 
                for transformer models. The new release includes improved memory management and 
                better CUDA kernel fusion.
                
                Performance benchmarks show:
                • 40% faster training for large language models
                • 60% reduction in memory usage
                • Improved distributed training efficiency'''
            }
        ],
        'tools_tutorials': [
            {
                'headline': 'VS Code Dev Containers Tutorial',
                'content': '''Learn how to use VS Code Dev Containers for consistent development environments.
                This tutorial covers setup, configuration, and best practices.
                
                ```json
                {
                    "name": "Python 3.11",
                    "image": "mcr.microsoft.com/devcontainers/python:3.11",
                    "features": {
                        "ghcr.io/devcontainers/features/docker-in-docker:2": {}
                    }
                }
                ```
                
                Benefits include:
                • Consistent environments across teams
                • Isolated dependencies
                • Easy onboarding for new developers'''
            }
        ],
        'quick_hits': [
            'Python 3.13 beta released with improved error messages',
            'GitHub Copilot Chat now supports voice commands',
            'New Rust async runtime tokio 1.40 improves performance'
        ],
        'footer': 'Follow us for more tech updates • Unsubscribe anytime'
    }

def run_full_integration_test():
    """Run comprehensive integration test of the quality assurance system"""
    print("=" * 80)
    print("🧪 FULL INTEGRATION TEST: Quality Assurance System")
    print("=" * 80)
    
    # Initialize the QA system
    qa_system = QualityAssuranceSystem()
    
    # Create realistic newsletter content
    newsletter_content = create_realistic_newsletter_content()
    
    print("\n📄 Testing with realistic newsletter content...")
    print(f"   Subject: {newsletter_content['subject_line']}")
    print(f"   Preview: {newsletter_content['preview_text']}")
    print(f"   Sections: {len(newsletter_content['news_breakthroughs']) + len(newsletter_content['tools_tutorials'])}")
    
    # Test comprehensive quality assessment
    print("\n🔍 Running comprehensive quality assessment...")
    start_time = time.time()
    
    ready_for_publish, validation_report = qa_system.validate_newsletter_ready_for_publish(
        newsletter_content, "daily"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Display results
    print(f"\n📊 QUALITY ASSESSMENT RESULTS")
    print("=" * 50)
    
    metrics = validation_report['quality_metrics']
    print(f"✅ Technical Accuracy Score: {metrics.technical_accuracy_score:.3f}")
    print(f"📱 Mobile Readability Score: {metrics.mobile_readability_score:.3f}")
    print(f"💻 Code Validation Score: {metrics.code_validation_score:.3f}")
    print(f"🏆 Overall Quality Score: {metrics.overall_quality_score:.3f}")
    print(f"⏱️  Processing Time: {processing_time:.3f}s")
    print(f"🚀 Ready for Publish: {'YES' if ready_for_publish else 'NO'}")
    
    # Check individual components
    print(f"\n🔍 DETAILED COMPONENT RESULTS")
    print("=" * 50)
    
    # Technical accuracy details
    tech_details = validation_report.get('technical_validation', {})
    print(f"📋 Technical Claims Found: {tech_details.get('technical_claims_count', 0)}")
    print(f"✅ Technical Accuracy: {tech_details.get('passes_threshold', False)}")
    
    # Mobile readability details
    mobile_details = validation_report.get('mobile_validation', {})
    print(f"📱 Mobile Compliance: {mobile_details.get('passes_mobile_compliance', False)}")
    print(f"📏 Subject Line Length: {len(newsletter_content['subject_line'])} chars")
    print(f"📄 Preview Text Length: {len(newsletter_content['preview_text'])} chars")
    
    # Code validation details
    code_details = validation_report.get('code_validation', {})
    print(f"💻 Code Blocks Found: {code_details.get('code_blocks_found', 0)}")
    print(f"✅ Code Validation: {code_details.get('passes_code_validation', False)}")
    
    # Issues and recommendations
    if validation_report.get('issues_found'):
        print(f"\n⚠️  ISSUES FOUND ({len(validation_report['issues_found'])})")
        for issue in validation_report['issues_found']:
            print(f"   • {issue}")
    
    if validation_report.get('recommendations'):
        print(f"\n💡 RECOMMENDATIONS ({len(validation_report['recommendations'])})")
        for rec in validation_report['recommendations'][:5]:  # Show first 5
            print(f"   • {rec}")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    performance_thresholds = {
        'processing_time': 2.0,  # 2 seconds max
        'technical_accuracy': 0.8,
        'mobile_readability': 0.8,
        'code_validation': 0.8,
        'overall_quality': 0.8
    }
    
    performance_results = {
        'processing_time': processing_time,
        'technical_accuracy': metrics.technical_accuracy_score,
        'mobile_readability': metrics.mobile_readability_score,
        'code_validation': metrics.code_validation_score,
        'overall_quality': metrics.overall_quality_score
    }
    
    all_passed = True
    for metric, value in performance_results.items():
        threshold = performance_thresholds[metric]
        if metric == 'processing_time':
            passed = value <= threshold
            print(f"⏱️  {metric}: {value:.3f}s (threshold: ≤{threshold}s) {'✅' if passed else '❌'}")
        else:
            passed = value >= threshold
            print(f"📊 {metric}: {value:.3f} (threshold: ≥{threshold}) {'✅' if passed else '❌'}")
        
        if not passed:
            all_passed = False
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT")
    print("=" * 50)
    
    if all_passed and ready_for_publish:
        status = "✅ PASSED"
        color = "🟢"
    else:
        status = "❌ FAILED"
        color = "🔴"
    
    print(f"{color} Integration Test Status: {status}")
    print(f"📊 Overall Performance: {(sum(performance_results.values()) / len(performance_results)):.3f}")
    print(f"🚀 Production Ready: {'YES' if ready_for_publish else 'NO'}")
    
    # Save detailed results
    results_file = "dev/integration_test_results.json"
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'processing_time': processing_time,
        'ready_for_publish': ready_for_publish,
        'quality_metrics': {
            'technical_accuracy_score': metrics.technical_accuracy_score,
            'mobile_readability_score': metrics.mobile_readability_score,
            'code_validation_score': metrics.code_validation_score,
            'overall_quality_score': metrics.overall_quality_score
        },
        'performance_results': performance_results,
        'all_tests_passed': all_passed,
        'issues_count': len(validation_report.get('issues_found', [])),
        'recommendations_count': len(validation_report.get('recommendations', []))
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return all_passed

if __name__ == "__main__":
    success = run_full_integration_test()
    sys.exit(0 if success else 1) 