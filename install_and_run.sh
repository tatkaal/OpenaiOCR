#!/bin/bash

# Atelier-Scrapper Auto Installer for Mac
# Fully automated installation - no technical knowledge required!

set -e  # Exit on any error

echo "🎨 Welcome to Atelier-Scrapper Auto Installer!"
echo "=============================================="
echo ""
echo "This script will automatically:"
echo "• Check and install Python 3 (if needed)"
echo "• Check and install pip (if needed)"
echo "• Install required Python packages"
echo "• Set up the application"
echo "• Launch the image processing app"
echo ""
echo "No technical knowledge required - just wait for completion!"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew if not present
install_homebrew() {
    if ! command_exists brew; then
        echo "📦 Installing Homebrew (package manager for Mac)..."
        echo "   This may take a few minutes..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for the current session
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        elif [[ -f "/usr/local/bin/brew" ]]; then
            eval "$(/usr/local/bin/brew shellenv)"
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        fi
        echo "✅ Homebrew installed successfully!"
    else
        echo "✅ Homebrew is already installed"
        # Update Homebrew
        echo "🔄 Updating Homebrew..."
        brew update || echo "⚠️ Homebrew update failed, continuing..."
    fi
}

# Function to install Python 3
install_python() {
    if ! command_exists python3; then
        echo "🐍 Installing Python 3..."
        install_homebrew
        brew install python
        echo "✅ Python 3 installed successfully!"
    else
        echo "✅ Python 3 is already installed"
        python3 --version
    fi
}

# Function to install pip
install_pip() {
    if ! command_exists pip3; then
        echo "📦 Installing pip..."
        python3 -m ensurepip --upgrade
        echo "✅ pip installed successfully!"
    else
        echo "✅ pip is already installed"
    fi
}

# Function to create virtual environment (optional but recommended)
setup_virtual_env() {
    if [ ! -d "venv" ]; then
        echo "🏗️ Creating isolated Python environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created!"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo "✅ Virtual environment activated!"
}

# Main installation process
main() {
    echo "🔍 Checking system requirements..."
    
    # Check and install Python
    install_python
    
    # Check and install pip
    install_pip
    
    # Set up virtual environment
    setup_virtual_env
    
    # Update pip to latest version
    echo "⬆️ Updating pip to latest version..."
    python3 -m pip install --upgrade pip
    
    # Install required packages with better error handling
    echo "📥 Installing required packages..."
    echo "   This may take a few minutes depending on your internet connection..."
    
    # Install packages one by one for better error reporting
    packages=("streamlit" "openai" "Pillow" "aiohttp" "aiofiles")
    for package in "${packages[@]}"; do
        echo "   Installing $package..."
        python3 -m pip install --upgrade "$package" || {
            echo "⚠️ Failed to install $package, trying alternative method..."
            pip3 install --upgrade "$package"
        }
    done
    
    echo "✅ All packages installed successfully!"
    
    # Check if images folder exists and has content
    if [ ! -d "images" ]; then
        echo "📁 Creating images folder..."
        mkdir -p images
    fi
    
    if [ ! "$(ls -A images 2>/dev/null)" ]; then
        echo "🖼️ Creating sample test images for demonstration..."
        python3 create_test_images.py || {
            echo "⚠️ Could not create test images, but this won't prevent the app from working"
            echo "   You can add your own images to the 'images' folder"
        }
        if [ "$(ls -A images 2>/dev/null)" ]; then
            echo "✅ Sample images created in images/ folder"
            echo "   You can replace these with your own images anytime"
        fi
    else
        image_count=$(ls images/ | wc -l | tr -d ' ')
        echo "✅ Found $image_count images ready for processing"
    fi
    
    # Create necessary directories
    mkdir -p processed logs
    
    echo ""
    echo "🎉 Installation Complete!"
    echo "========================"
    echo ""
    echo "✅ All requirements installed successfully"
    echo "✅ Application is ready to use"
    echo "✅ Images folder is prepared"
    echo ""
    echo "🚀 Launching Atelier-Scrapper..."
    echo ""
    echo "📝 Quick Start Instructions:"
    echo "1. The app will open in your web browser"
    echo "2. Enter your OpenAI API key in the sidebar"
    echo "3. Select your preferred AI model"
    echo "4. Click 'Start Processing Images'"
    echo "5. Monitor progress and costs in real-time"
    echo ""
    echo "💡 Tip: Have your OpenAI API key ready!"
    echo "   Get it from: https://platform.openai.com/api-keys"
    echo ""
    
    # Launch the application
    echo "🌐 Opening Atelier-Scrapper in your browser..."
    echo "   If it doesn't open automatically, go to: http://localhost:8501"
    echo ""
    
    # Start Streamlit with better settings for non-technical users
    python3 -m streamlit run app.py \
        --server.headless false \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false \
        --theme.base light \
        || {
            echo ""
            echo "❌ Failed to start the application"
            echo "💡 Try running this command manually:"
            echo "   python3 -m streamlit run app.py"
            echo ""
            read -p "Press Enter to close this window..."
        }
}

# Error handling with user-friendly messages
handle_error() {
    echo ""
    echo "❌ Installation encountered an error!"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "1. Make sure you have an internet connection"
    echo "2. Try running the installer again"
    echo "3. If problems persist, try installing manually:"
    echo "   - Open Terminal"
    echo "   - Navigate to this folder"
    echo "   - Run: pip3 install streamlit openai Pillow"
    echo "   - Run: python3 app.py"
    echo ""
    echo "📞 Need help? Check USER_GUIDE.md for detailed instructions"
    echo ""
    read -p "Press Enter to close this window..."
    exit 1
}

# Set up error handling
trap 'handle_error' ERR

# Run main function
main
