"""
Comprehensive Test Runner for Newsletter Enhancement System

Main test runner that executes all test suites and provides comprehensive
reporting for the newsletter enhancement system.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all test modules
from test_content_expansion import (
    TestIntelligentContentExpander,
    TestSectionExpansionOrchestrator,
    TestContentExpansionIntegration
)
from test_mobile_optimization import (
    TestMobileContentOptimizer,
    TestMobileReadabilityAnalyzer,
    TestResponsiveTypographyManager
)
from test_integration_comprehensive import (
    TestComprehensiveNewsletterEnhancement,
    TestQualityGateIntegration
)
from test_quality_gates_enhanced import (
    TestEnhancedQualityGates,
    TestQualityDimensionAssessment
)


class NewsletterTestResult(unittest.TestResult):
    """Custom test result class for detailed reporting."""
    
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__()
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
    
    def stopTest(self, test):
        super().stopTest(test)
        self.end_time = time.time()
        
        duration = self.end_time - self.start_time
        status = "PASS"
        
        if test in [failure[0] for failure in self.failures]:
            status = "FAIL"
        elif test in [error[0] for error in self.errors]:
            status = "ERROR"
        
        self.test_results.append({
            'test': str(test),
            'status': status,
            'duration': duration
        })


class ComprehensiveTestRunner:
    """Main test runner for the newsletter enhancement system."""
    
    def __init__(self):
        self.test_suites = {
            'Content Expansion': [
                TestIntelligentContentExpander,
                TestSectionExpansionOrchestrator,
                TestContentExpansionIntegration
            ],
            'Mobile Optimization': [
                TestMobileContentOptimizer,
                TestMobileReadabilityAnalyzer,
                TestResponsiveTypographyManager
            ],
            'Quality Gates': [
                TestEnhancedQualityGates,
                TestQualityDimensionAssessment
            ],
            'Integration': [
                TestComprehensiveNewsletterEnhancement,
                TestQualityGateIntegration
            ]
        }
        
        self.results = {}
        self.total_start_time = None
        self.total_end_time = None
    
    def run_all_tests(self, verbosity=2):
        """Run all test suites and collect results."""
        print("=" * 100)
        print("NEWSLETTER ENHANCEMENT SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 100)
        print(f"Testing Phase 1: Content Expansion System")
        print(f"Testing Phase 2: Mobile Optimization System")
        print(f"Testing Quality Gates Integration")
        print(f"Testing End-to-End Workflows")
        print("=" * 100)
        
        self.total_start_time = time.time()
        overall_success = True
        
        for suite_name, test_classes in self.test_suites.items():
            print(f"\n🧪 RUNNING {suite_name.upper()} TESTS")
            print("-" * 60)
            
            suite_result = self._run_test_suite(suite_name, test_classes, verbosity)
            self.results[suite_name] = suite_result
            
            if not suite_result['success']:
                overall_success = False
        
        self.total_end_time = time.time()
        
        # Generate comprehensive report
        self._generate_final_report(overall_success)
        
        return overall_success
    
    def _run_test_suite(self, suite_name, test_classes, verbosity):
        """Run a specific test suite."""
        suite = unittest.TestSuite()
        
        # Add all test classes to suite
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests with custom result collector
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=verbosity,
            resultclass=NewsletterTestResult
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Collect results
        suite_result = {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'duration': end_time - start_time,
            'details': result.test_results if hasattr(result, 'test_results') else [],
            'output': stream.getvalue()
        }
        
        # Print immediate results
        if suite_result['success']:
            print(f"✅ {suite_name}: {suite_result['tests_run']} tests PASSED in {suite_result['duration']:.2f}s")
        else:
            print(f"❌ {suite_name}: {suite_result['failures']} failures, {suite_result['errors']} errors")
            
        return suite_result
    
    def _generate_final_report(self, overall_success):
        """Generate comprehensive final report."""
        total_duration = self.total_end_time - self.total_start_time
        
        print("\n" + "=" * 100)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 100)
        
        # Summary statistics
        total_tests = sum(result['tests_run'] for result in self.results.values())
        total_failures = sum(result['failures'] for result in self.results.values())
        total_errors = sum(result['errors'] for result in self.results.values())
        
        print(f"📊 OVERALL STATISTICS")
        print(f"   Total Test Suites: {len(self.test_suites)}")
        print(f"   Total Tests Run: {total_tests}")
        print(f"   Total Duration: {total_duration:.2f} seconds")
        print(f"   Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
        
        # Suite-by-suite breakdown
        print(f"\n📋 SUITE BREAKDOWN")
        for suite_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"   {suite_name:20} | {status} | {result['tests_run']:3d} tests | {result['duration']:6.2f}s")
            
            if not result['success']:
                if result['failures'] > 0:
                    print(f"      └─ {result['failures']} failures")
                if result['errors'] > 0:
                    print(f"      └─ {result['errors']} errors")
        
        # System readiness assessment
        print(f"\n🎯 SYSTEM READINESS ASSESSMENT")
        
        content_expansion_ready = self.results['Content Expansion']['success']
        mobile_optimization_ready = self.results['Mobile Optimization']['success']
        quality_gates_ready = self.results['Quality Gates']['success']
        integration_ready = self.results['Integration']['success']
        
        print(f"   Phase 1 (Content Expansion): {'✅ Ready' if content_expansion_ready else '❌ Issues detected'}")
        print(f"   Phase 2 (Mobile Optimization): {'✅ Ready' if mobile_optimization_ready else '❌ Issues detected'}")
        print(f"   Quality Gate System: {'✅ Ready' if quality_gates_ready else '❌ Issues detected'}")
        print(f"   End-to-End Integration: {'✅ Ready' if integration_ready else '❌ Issues detected'}")
        
        # Overall status
        print(f"\n🏆 FINAL STATUS")
        if overall_success:
            print("   ✅ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
            print("   🚀 Newsletter enhancement system is fully operational")
            print("   📈 Both Phase 1 and Phase 2 implementations are validated")
        else:
            print("   ❌ SOME TESTS FAILED - REVIEW REQUIRED")
            print("   🔧 Please address failing tests before deployment")
            
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"   Average test duration: {total_duration/total_tests:.3f}s per test")
        print(f"   System throughput: {total_tests/total_duration:.1f} tests per second")
        
        # Feature coverage
        print(f"\n🎯 FEATURE COVERAGE")
        print("   ✅ Content expansion and enhancement")
        print("   ✅ Section-aware orchestration")
        print("   ✅ Mobile-first optimization")
        print("   ✅ Responsive typography management")
        print("   ✅ Advanced readability analysis")
        print("   ✅ Enhanced quality gate validation")
        print("   ✅ End-to-end workflow integration")
        
        print("=" * 100)
    
    def run_specific_suite(self, suite_name, verbosity=2):
        """Run a specific test suite only."""
        if suite_name not in self.test_suites:
            print(f"❌ Test suite '{suite_name}' not found")
            print(f"Available suites: {list(self.test_suites.keys())}")
            return False
        
        print(f"🧪 Running {suite_name} Tests Only")
        print("-" * 50)
        
        test_classes = self.test_suites[suite_name]
        result = self._run_test_suite(suite_name, test_classes, verbosity)
        
        return result['success']
    
    def run_quick_validation(self):
        """Run a quick validation suite for rapid feedback."""
        print("🚀 QUICK VALIDATION SUITE")
        print("-" * 40)
        
        # Select key tests for quick validation
        quick_tests = [
            ('Content Expansion', [TestIntelligentContentExpander]),
            ('Mobile Optimization', [TestMobileContentOptimizer]),
            ('Integration', [TestQualityGateIntegration])
        ]
        
        all_passed = True
        for suite_name, test_classes in quick_tests:
            result = self._run_test_suite(f"Quick {suite_name}", test_classes, 1)
            if not result['success']:
                all_passed = False
        
        print("\n🎯 QUICK VALIDATION RESULT")
        if all_passed:
            print("✅ Quick validation PASSED - Core functionality working")
        else:
            print("❌ Quick validation FAILED - Core issues detected")
        
        return all_passed


def main():
    """Main test runner entry point."""
    runner = ComprehensiveTestRunner()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'quick':
            success = runner.run_quick_validation()
        elif command in ['content', 'mobile', 'quality', 'integration']:
            suite_map = {
                'content': 'Content Expansion',
                'mobile': 'Mobile Optimization', 
                'quality': 'Quality Gates',
                'integration': 'Integration'
            }
            success = runner.run_specific_suite(suite_map[command])
        elif command == 'help':
            print("Newsletter Enhancement Test Runner")
            print("\nUsage:")
            print("  python test_runner.py [command]")
            print("\nCommands:")
            print("  (no args)    - Run all tests")
            print("  quick        - Run quick validation")
            print("  content      - Run content expansion tests only")
            print("  mobile       - Run mobile optimization tests only")
            print("  quality      - Run quality gates tests only")
            print("  integration  - Run integration tests only")
            print("  help         - Show this help message")
            return
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python test_runner.py help' for usage information")
            return
    else:
        # Run all tests
        success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()