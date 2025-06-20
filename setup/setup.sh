#!/bin/bash

# Script to create conda environment from requirements.txt
# Usage: ./create_conda_env.sh [environment_name]

# Set default environment name if not provided
ENV_NAME=${1:-"news_env"}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating conda environment: $ENV_NAME${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found in current directory${NC}"
    exit 1
fi

# Create conda environment with Python
echo -e "${YELLOW}Creating conda environment with Python...${NC}"
conda create -n "$ENV_NAME" python=3.9 -y

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create conda environment${NC}"
    exit 1
fi

# Activate the environment
echo -e "${YELLOW}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate conda environment${NC}"
    exit 1
fi

# Install pip packages from requirements.txt
echo -e "${YELLOW}Installing packages from requirements.txt...${NC}"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully created conda environment '$ENV_NAME' with all packages!${NC}"
    echo -e "${GREEN}To activate the environment, run:${NC}"
    echo -e "${GREEN}conda activate $ENV_NAME${NC}"
else
    echo -e "${RED}Error: Some packages failed to install${NC}"
    exit 1
fi

# Display installed packages
echo -e "${YELLOW}Installed packages:${NC}"
conda list