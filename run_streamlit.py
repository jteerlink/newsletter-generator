#!/usr/bin/env python3
"""
Streamlit App Launcher
This script properly sets up the Python path and launches the Streamlit app.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add src directory to Python path
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        print(f"Added {src_path} to Python path")
    
    # Set environment variable for other processes
    os.environ['PYTHONPATH'] = f"{src_path}:{os.environ.get('PYTHONPATH', '')}"
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Check if streamlit app exists
    streamlit_app = project_root / "streamlit" / "app.py"
    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)
    
    # Prepare streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_app),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--logger.level", "info",
        "--theme.base", "light",
        "--theme.primaryColor", "#003F5C",
        "--theme.backgroundColor", "#F8F9FA",
        "--theme.secondaryBackgroundColor", "#FFFFFF",
        "--theme.textColor", "#212529",
        "--theme.font", "sans serif"
    ]
    
    print("Starting Streamlit app...")
    print(f"Command: {' '.join(cmd)}")
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    
    try:
        # Run streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 