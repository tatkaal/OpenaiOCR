#!/bin/bash

# Atelier-Scrapper - Self-Fixing Double-click installer for Mac
# This script fixes its own permissions and then runs the installation
# Works even if downloaded without execute permissions!

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Function to show GUI dialog
show_dialog() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\" with icon note with title \"Atelier-Scrapper\""
}

show_error() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\" with icon stop with title \"Atelier-Scrapper - Error\""
}

# Self-fix permissions for all scripts in this folder
# This ensures the installer works even if downloaded without proper permissions
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null
chmod +x "$SCRIPT_DIR"/*.command 2>/dev/null

# Verify we can execute the main installer
if [ ! -x "$SCRIPT_DIR/install_and_run.sh" ]; then
    show_error "Could not set up installer permissions. 

Please try:
1. Right-click this file
2. Choose 'Open With' → 'Terminal'
3. Or contact support for help"
    exit 1
fi

# Welcome message
show_dialog "Welcome to Atelier-Scrapper! 🎨

This will automatically install and set up everything you need:
• Python and required tools
• AI image processing system  
• Launch the app in your browser

The installation takes about 3-5 minutes.

Click OK to begin..."

# Check if install script exists
if [ ! -f "$SCRIPT_DIR/install_and_run.sh" ]; then
    show_error "Installation files are missing! 

Please make sure all files are in the same folder:
• Double-Click-to-Install.command
• install_and_run.sh
• app.py
• And other system files

Download the complete package and try again."
    exit 1
fi

# Open Terminal and run the installation with better error handling
osascript -e "
tell application \"Terminal\"
    activate
    set newTab to do script \"cd '$SCRIPT_DIR' && echo '🎨 Welcome to Atelier-Scrapper Installation!' && echo '================================================' && echo '' && ./install_and_run.sh; echo ''; echo '✅ Installation complete! You can close this Terminal window.'; echo ''; read -p 'Press Enter to close this window...'\"
    set custom title of newTab to \"Atelier-Scrapper Auto-Installer\"
end tell
"

# Show final instructions
sleep 3
show_dialog "Installation Started! 🚀

What's happening:
• Terminal window shows installation progress
• All required software installs automatically
• App will open in your web browser when ready

Next steps:
• Have your OpenAI API key ready
• The app will guide you through the rest

Tip: You can get an API key from platform.openai.com"
