#!/bin/bash

# Newsletter Generator - Streamlit App Launcher
# This script starts the Streamlit app from the root directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "streamlit" ]; then
    print_error "This script must be run from the newsletter-generator root directory"
    print_error "Please navigate to the project root and try again"
    exit 1
fi

print_status "Starting Newsletter Generator Streamlit App..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ -d "venv" ] || [ -d ".venv" ]; then
    print_status "Virtual environment detected"
    if [ -d "venv" ]; then
        print_status "Activating virtual environment (venv)..."
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        print_status "Activating virtual environment (.venv)..."
        source .venv/bin/activate
    fi
else
    print_warning "No virtual environment detected"
    print_warning "Consider creating one with: python3 -m venv venv"
fi

# Check if required packages are installed
print_status "Checking dependencies..."

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    print_error "Streamlit is not installed"
    print_status "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ -f "streamlit/requirements.txt" ]; then
        print_status "Installing Streamlit-specific dependencies..."
        pip install -r streamlit/requirements.txt
    fi
else
    print_success "Streamlit is installed"
fi

# Check if the streamlit app file exists
if [ ! -f "streamlit/app.py" ]; then
    print_error "Streamlit app file not found: streamlit/app.py"
    exit 1
fi

print_success "All dependencies are satisfied"

# Set environment variables if .env file exists
if [ -f ".env" ]; then
    print_status "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Set PYTHONPATH to include the src directory
print_status "Setting up Python path..."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
print_status "PYTHONPATH set to: $PYTHONPATH"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/streamlit_${TIMESTAMP}.log"

print_status "Starting Streamlit app..."
print_status "Log file: $LOG_FILE"
print_status "App will be available at: http://localhost:8501"
print_status "Press Ctrl+C to stop the app"

# Start Streamlit with proper configuration
streamlit run streamlit/app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --logger.level info \
    --theme.base light \
    --theme.primaryColor "#003F5C" \
    --theme.backgroundColor "#F8F9FA" \
    --theme.secondaryBackgroundColor "#FFFFFF" \
    --theme.textColor "#212529" \
    --theme.font "sans serif" \
    2>&1 | tee "$LOG_FILE"

print_success "Streamlit app stopped" 