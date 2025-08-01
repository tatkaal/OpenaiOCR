#!/bin/bash

# Atelier-Scrapper Startup Script

echo "🎨 Starting Atelier-Scrapper Setup..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p images processed

# Check if test images exist, if not create them
if [ ! "$(ls -A images)" ]; then
    echo "🖼️ No images found. Creating test images..."
    python create_test_images.py
else
    echo "✅ Images folder contains files"
fi

echo ""
echo "🚀 Setup complete! Ready to start Atelier-Scrapper"
echo ""
echo "To run the application:"
echo "1. Make sure you have your OpenAI API key ready"
echo "2. Run: streamlit run app.py"
echo ""
echo "The application will open in your default web browser."
echo ""

# Optionally start the app immediately
read -p "Would you like to start the application now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🎨 Starting Atelier-Scrapper..."
    streamlit run app.py
fi
