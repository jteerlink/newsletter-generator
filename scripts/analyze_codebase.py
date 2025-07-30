#!/usr/bin/env python3
"""Codebase analysis script for baseline metrics."""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebaseAnalyzer:
    """Analyze codebase for metrics and issues."""
    
    def __init__(self, src_path: str = "src"):
        self.src_path = Path(src_path)
        self.metrics = defaultdict(int)
        self.issues = []
        self.file_metrics = {}
        
    def analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Count lines
            lines = content.split('\n')
            total_lines = len(lines)
            code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            comment_lines = len([l for l in lines if l.strip().startswith('#')])
            blank_lines = len([l for l in lines if not l.strip()])
            
            # Count functions and classes
            functions = 0
            classes = 0
            imports = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1
                elif isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports += 1
            
            # Store file metrics
            file_metric = {
                'total_lines': total_lines,
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'complexity': self._calculate_complexity(tree)
            }
            
            self.file_metrics[str(file_path)] = file_metric
            
            # Update global metrics
            self.metrics['total_lines'] += total_lines
            self.metrics['code_lines'] += code_lines
            self.metrics['comment_lines'] += comment_lines
            self.metrics['blank_lines'] += blank_lines
            self.metrics['functions'] += functions
            self.metrics['classes'] += classes
            self.metrics['imports'] += imports
            self.metrics['files'] += 1
            
            # Check for issues
            if total_lines > 500:
                self.issues.append(f"Large file: {file_path} ({total_lines} lines)")
            
            if file_metric['complexity'] > 10:
                self.issues.append(f"High complexity: {file_path} (complexity: {file_metric['complexity']})")
                
        except Exception as e:
            self.issues.append(f"Error analyzing {file_path}: {e}")
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of a file."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def analyze_codebase(self):
        """Analyze entire codebase."""
        logger.info(f"Analyzing codebase in {self.src_path}")
        
        if not self.src_path.exists():
            logger.error(f"Source path {self.src_path} does not exist")
            return None
        
        for file_path in self.src_path.rglob("*.py"):
            logger.debug(f"Analyzing {file_path}")
            self.analyze_file(file_path)
        
        # Calculate additional metrics
        self._calculate_additional_metrics()
        
        return {
            'metrics': dict(self.metrics),
            'issues': self.issues,
            'files_analyzed': len(list(self.src_path.rglob("*.py"))),
            'file_metrics': self.file_metrics,
            'summary': self._generate_summary()
        }
    
    def _calculate_additional_metrics(self):
        """Calculate additional metrics."""
        if self.metrics['files'] > 0:
            self.metrics['avg_lines_per_file'] = self.metrics['total_lines'] / self.metrics['files']
            self.metrics['avg_functions_per_file'] = self.metrics['functions'] / self.metrics['files']
            self.metrics['avg_classes_per_file'] = self.metrics['classes'] / self.metrics['files']
            self.metrics['code_comment_ratio'] = self.metrics['code_lines'] / max(self.metrics['comment_lines'], 1)
    
    def _generate_summary(self):
        """Generate a summary of the analysis."""
        return {
            'total_files': self.metrics['files'],
            'total_lines': self.metrics['total_lines'],
            'code_lines': self.metrics['code_lines'],
            'comment_lines': self.metrics['comment_lines'],
            'blank_lines': self.metrics['blank_lines'],
            'functions': self.metrics['functions'],
            'classes': self.metrics['classes'],
            'imports': self.metrics['imports'],
            'avg_lines_per_file': self.metrics.get('avg_lines_per_file', 0),
            'avg_functions_per_file': self.metrics.get('avg_functions_per_file', 0),
            'avg_classes_per_file': self.metrics.get('avg_classes_per_file', 0),
            'code_comment_ratio': self.metrics.get('code_comment_ratio', 0),
            'issues_found': len(self.issues)
        }

def analyze_dependencies():
    """Analyze project dependencies."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.warning("requirements.txt not found")
        return {}
    
    dependencies = {}
    try:
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==', 1)
                        dependencies[package] = version
                    elif '>=' in line:
                        package, version = line.split('>=', 1)
                        dependencies[package] = f">={version}"
                    else:
                        dependencies[line] = "latest"
    except Exception as e:
        logger.error(f"Error reading requirements.txt: {e}")
    
    return dependencies

def generate_report(results, dependencies):
    """Generate a comprehensive analysis report."""
    if not results:
        logger.error("No analysis results to report")
        return
    
    print("=" * 60)
    print("NEWSLETTER GENERATOR - CODEBASE ANALYSIS REPORT")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\n📊 SUMMARY METRICS:")
    print(f"   Total Files: {summary['total_files']}")
    print(f"   Total Lines: {summary['total_lines']:,}")
    print(f"   Code Lines: {summary['code_lines']:,}")
    print(f"   Comment Lines: {summary['comment_lines']:,}")
    print(f"   Blank Lines: {summary['blank_lines']:,}")
    print(f"   Functions: {summary['functions']:,}")
    print(f"   Classes: {summary['classes']:,}")
    print(f"   Imports: {summary['imports']:,}")
    
    print(f"\n📈 AVERAGE METRICS:")
    print(f"   Lines per File: {summary['avg_lines_per_file']:.1f}")
    print(f"   Functions per File: {summary['avg_functions_per_file']:.1f}")
    print(f"   Classes per File: {summary['avg_classes_per_file']:.1f}")
    print(f"   Code/Comment Ratio: {summary['code_comment_ratio']:.2f}")
    
    print(f"\n🔧 DEPENDENCIES:")
    print(f"   Total Dependencies: {len(dependencies)}")
    for package, version in sorted(dependencies.items()):
        print(f"   - {package}: {version}")
    
    if results['issues']:
        print(f"\n⚠️  ISSUES FOUND ({len(results['issues'])}):")
        for issue in results['issues']:
            print(f"   - {issue}")
    else:
        print(f"\n✅ No issues found!")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    quality_score = 100
    
    if summary['avg_lines_per_file'] > 500:
        quality_score -= 20
        print("   ⚠️  Some files are very large (>500 lines)")
    
    if summary['code_comment_ratio'] < 0.1:
        quality_score -= 15
        print("   ⚠️  Low comment coverage")
    
    if len(results['issues']) > 10:
        quality_score -= 25
        print("   ⚠️  Many issues detected")
    
    print(f"   Overall Quality Score: {quality_score}/100")
    
    if quality_score >= 80:
        print("   🎉 Excellent code quality!")
    elif quality_score >= 60:
        print("   👍 Good code quality with room for improvement")
    else:
        print("   🔧 Significant improvements needed")

def main():
    """Run codebase analysis."""
    logger.info("Starting codebase analysis...")
    
    # Analyze codebase
    analyzer = CodebaseAnalyzer()
    results = analyzer.analyze_codebase()
    
    if not results:
        logger.error("Analysis failed")
        sys.exit(1)
    
    # Analyze dependencies
    dependencies = analyze_dependencies()
    
    # Generate report
    generate_report(results, dependencies)
    
    # Save results
    output_file = 'codebase_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'analysis_results': results,
            'dependencies': dependencies,
            'timestamp': str(Path().cwd()),
            'version': '1.0'
        }, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_file}")
    
    # Return exit code based on issues
    if results['issues']:
        logger.warning(f"Found {len(results['issues'])} issues")
        return 1
    else:
        logger.info("No issues found")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 