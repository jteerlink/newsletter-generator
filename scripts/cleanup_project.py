#!/usr/bin/env python3
"""
Project Cleanup Script

Automated cleanup script for the newsletter enhancement system.
Removes unused files, optimizes imports, and validates project structure.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
from typing import List, Dict, Set
import argparse


class ProjectCleanup:
    """Main cleanup orchestrator for the newsletter enhancement system."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.stats = {
            'files_cleaned': 0,
            'imports_optimized': 0,
            'unused_files_removed': 0,
            'lines_saved': 0
        }
    
    def run_cleanup(self, dry_run: bool = False) -> Dict:
        """Run complete project cleanup."""
        print("🧹 Starting Newsletter Enhancement System Cleanup")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
            print("=" * 60)
        
        # Step 1: Clean Python cache files
        self.clean_python_cache(dry_run)
        
        # Step 2: Remove empty __pycache__ directories
        self.remove_empty_pycache_dirs(dry_run)
        
        # Step 3: Clean test artifacts
        self.clean_test_artifacts(dry_run)
        
        # Step 4: Optimize imports (if not dry run)
        if not dry_run:
            self.optimize_imports()
        
        # Step 5: Generate cleanup report
        self.generate_report()
        
        return self.stats
    
    def clean_python_cache(self, dry_run: bool = False):
        """Remove Python cache files and directories."""
        print("🗑️  Cleaning Python cache files...")
        
        cache_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.pyd',
            '**/*$py.class'
        ]
        
        removed_count = 0
        
        for pattern in cache_patterns:
            cache_files = list(self.project_root.glob(pattern))
            for cache_file in cache_files:
                # Skip .venv directory
                if '.venv' in str(cache_file):
                    continue
                    
                if dry_run:
                    print(f"  Would remove: {cache_file}")
                else:
                    if cache_file.is_dir():
                        subprocess.run(['rm', '-rf', str(cache_file)], check=True)
                    else:
                        cache_file.unlink()
                    print(f"  Removed: {cache_file}")
                
                removed_count += 1
        
        self.stats['unused_files_removed'] += removed_count
        print(f"✅ Cleaned {removed_count} cache files/directories")
    
    def remove_empty_pycache_dirs(self, dry_run: bool = False):
        """Remove empty __pycache__ directories."""
        print("📁 Removing empty directories...")
        
        empty_dirs = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip .venv directory
            if '.venv' in root:
                continue
                
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if dir_name == '__pycache__' or (
                    dir_path.is_dir() and 
                    len(list(dir_path.iterdir())) == 0
                ):
                    empty_dirs.append(dir_path)
        
        for empty_dir in empty_dirs:
            if dry_run:
                print(f"  Would remove empty dir: {empty_dir}")
            else:
                empty_dir.rmdir()
                print(f"  Removed empty dir: {empty_dir}")
        
        print(f"✅ Removed {len(empty_dirs)} empty directories")
    
    def clean_test_artifacts(self, dry_run: bool = False):
        """Clean test artifacts and temporary files."""
        print("🧪 Cleaning test artifacts...")
        
        test_artifacts = [
            'tests/.pytest_cache',
            'tests/htmlcov',
            'tests/coverage_html',
            'tests/.coverage',
            'tests/test.log',
            'tests/*.pyc',
            '.coverage',
            'htmlcov',
            '*.log'
        ]
        
        cleaned_count = 0
        
        for pattern in test_artifacts:
            artifacts = list(self.project_root.glob(pattern))
            for artifact in artifacts:
                if dry_run:
                    print(f"  Would remove: {artifact}")
                else:
                    if artifact.is_dir():
                        subprocess.run(['rm', '-rf', str(artifact)], check=True)
                        print(f"  Removed directory: {artifact}")
                    else:
                        artifact.unlink()
                        print(f"  Removed file: {artifact}")
                
                cleaned_count += 1
        
        self.stats['files_cleaned'] += cleaned_count
        print(f"✅ Cleaned {cleaned_count} test artifacts")
    
    def optimize_imports(self):
        """Optimize imports in Python files (requires isort)."""
        print("📦 Optimizing imports...")
        
        try:
            # Check if isort is available
            subprocess.run(['isort', '--version'], 
                         capture_output=True, check=True)
            
            # Run isort on source and test directories
            directories = ['src', 'tests']
            for directory in directories:
                dir_path = self.project_root / directory
                if dir_path.exists():
                    result = subprocess.run([
                        'isort', 
                        str(dir_path),
                        '--profile', 'black',
                        '--line-length', '100',
                        '--multi-line', '3'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"  ✅ Optimized imports in {directory}/")
                        self.stats['imports_optimized'] += 1
                    else:
                        print(f"  ⚠️  Warning: Issues optimizing {directory}/")
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ℹ️  isort not available, skipping import optimization")
            print("  💡 Install with: pip install isort")
    
    def generate_report(self):
        """Generate cleanup completion report."""
        print("\n" + "=" * 60)
        print("📊 CLEANUP COMPLETION REPORT")
        print("=" * 60)
        
        print(f"📁 Files cleaned: {self.stats['files_cleaned']}")
        print(f"🗑️  Unused files removed: {self.stats['unused_files_removed']}")
        print(f"📦 Import optimizations: {self.stats['imports_optimized']}")
        
        # Calculate project size
        total_files = len(list(self.project_root.rglob('*.py')))
        test_files = len(list((self.project_root / 'tests').rglob('*.py')))
        src_files = len(list((self.project_root / 'src').rglob('*.py')))
        
        print(f"\n📈 PROJECT METRICS")
        print(f"   Total Python files: {total_files}")
        print(f"   Source files: {src_files}")
        print(f"   Test files: {test_files}")
        
        # Estimate size savings
        estimated_savings = (
            self.stats['files_cleaned'] * 50 +  # Assume 50 lines per artifact
            self.stats['unused_files_removed'] * 20  # Cache files
        )
        
        if estimated_savings > 0:
            print(f"   Estimated lines saved: ~{estimated_savings}")
            self.stats['lines_saved'] = estimated_savings
        
        print(f"\n✅ PROJECT CLEANUP COMPLETED SUCCESSFULLY")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("   - Run 'python tests/test_runner.py quick' to validate")
        print("   - Consider setting up pre-commit hooks")
        print("   - Schedule regular cleanup (monthly recommended)")


def main():
    """Main entry point for cleanup script."""
    parser = argparse.ArgumentParser(description='Clean up the newsletter enhancement project')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be cleaned without making changes')
    parser.add_argument('--project-root', default='.', 
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Resolve project root
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        print(f"❌ Error: Project root '{project_root}' does not exist")
        sys.exit(1)
    
    # Verify this is a newsletter project
    required_dirs = ['src', 'tests']
    missing_dirs = [d for d in required_dirs if not (project_root / d).exists()]
    
    if missing_dirs:
        print(f"❌ Error: Missing required directories: {missing_dirs}")
        print(f"   Make sure you're in the newsletter-generator project root")
        sys.exit(1)
    
    # Run cleanup
    cleanup = ProjectCleanup(project_root)
    
    try:
        stats = cleanup.run_cleanup(dry_run=args.dry_run)
        
        # Exit with success
        print(f"\n🎉 Cleanup completed successfully!")
        
        if args.dry_run:
            print("   Run without --dry-run to apply changes")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()