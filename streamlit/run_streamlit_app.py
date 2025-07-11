#!/usr/bin/env python3
"""
Streamlit App Runner for Hybrid Newsletter System
Simplifies starting the Streamlit app with proper configuration
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for running the Streamlit app"""
    # Get the script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Add src to Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(src_path)
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Source path: {src_path}")
    print(f"🐍 Python path: {sys.path[:3]}...")

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_streamlit_app():
    """Run the Streamlit app"""
    script_dir = Path(__file__).parent
    app_path = script_dir / "app_hybrid_minimal.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        return False
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--theme.base=light",
        "--theme.primaryColor=#FFA600",
        "--theme.backgroundColor=#F8F9FA",
        "--theme.secondaryBackgroundColor=#FFFFFF",
        "--theme.textColor=#212529"
    ]
    
    print(f"🚀 Starting Streamlit app...")
    print(f"📍 Command: {' '.join(cmd)}")
    print(f"🌐 The app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, cwd=str(script_dir))
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Hybrid Newsletter System - Streamlit App Runner")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing packages.")
        return False
    
    # Run the app
    return run_streamlit_app()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 