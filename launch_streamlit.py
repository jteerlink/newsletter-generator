#!/usr/bin/env python3
"""
Launch script for AI Newsletter Generator Streamlit Interface
Handles environment setup and application launching
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'PyYAML',
        'ollama',
        'crewai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def setup_environment():
    """Setup the environment for the newsletter generator"""
    
    # Create necessary directories
    dirs_to_create = [
        'logs',
        'output',
        'src/scrapers/output',
        'src/scrapers/data'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check if sources.yaml exists
    sources_file = Path('src/sources.yaml')
    if not sources_file.exists():
        print("⚠️  Warning: sources.yaml not found. Some features may not work properly.")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  Warning: .env file not found. You may need to configure environment variables.")
    
    print("✅ Environment setup complete!")
    return True

def launch_streamlit(interface_type="enhanced", port=8501, host="localhost"):
    """Launch the Streamlit interface"""
    
    # Determine which interface to launch
    if interface_type == "enhanced":
        app_file = "streamlit_app_enhanced.py"
    else:
        app_file = "streamlit_app.py"
    
    if not Path(app_file).exists():
        print(f"❌ Interface file {app_file} not found!")
        return False
    
    print(f"🚀 Launching {interface_type} interface...")
    print(f"📊 Interface: {app_file}")
    print(f"🌐 URL: http://{host}:{port}")
    print(f"🔧 To stop the server, press Ctrl+C")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", host,
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped by user")
        return True
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Launch AI Newsletter Generator Streamlit Interface")
    
    parser.add_argument(
        "--interface", 
        choices=["basic", "enhanced"], 
        default="enhanced",
        help="Choose interface type (default: enhanced)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the server on (default: 8501)"
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )
    
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip dependency and environment checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 AI Newsletter Generator - Streamlit Interface Launcher")
    print("=" * 60)
    
    if not args.skip_checks:
        # Check dependencies
        print("📦 Checking dependencies...")
        if not check_dependencies():
            print("❌ Dependency check failed. Exiting.")
            return 1
        
        # Setup environment
        print("⚙️  Setting up environment...")
        if not setup_environment():
            print("❌ Environment setup failed. Exiting.")
            return 1
    
    # Launch the interface
    if not launch_streamlit(args.interface, args.port, args.host):
        print("❌ Failed to launch interface. Exiting.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 