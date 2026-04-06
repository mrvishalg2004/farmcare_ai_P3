#!/bin/bash

# FarmCare AI - Complete Setup and Test Script for macOS/Linux
# This script sets up everything needed to run the app locally

echo "🌱 FarmCare AI - Complete Setup"
echo "================================"
echo ""

# Color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ $PYTHON_VERSION is installed${NC}"
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Do you want to delete and recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf venv
        echo -e "${GREEN}✅ Old environment removed${NC}"
    fi
fi

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ Pip upgraded${NC}"

echo ""

# Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
echo ""

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Failed to install some dependencies${NC}"
    echo "Please check requirements.txt and try again"
    exit 1
fi

echo ""
echo "================================"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "================================"
echo ""
echo "To run the app:"
echo "  1. Activate the virtual environment:"
echo "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "  2. Run the Streamlit app:"
echo "     ${YELLOW}streamlit run app.py${NC}"
echo ""
echo "  Or simply run:"
echo "     ${YELLOW}./run_local.sh${NC}"
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
