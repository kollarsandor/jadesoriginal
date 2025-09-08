#!/bin/bash

echo "🚀 JADED Platform Setup Script"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Python detected: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check for optional dependencies
echo "🔍 Checking optional dependencies..."

# Julia
if command -v julia &> /dev/null; then
    echo "✓ Julia found"
else
    echo "⚠️ Julia not found - AlphaFold service will use simulation mode"
fi

# Nim
if command -v nim &> /dev/null; then
    echo "✓ Nim found"
else
    echo "⚠️ Nim not found - Native performance service will use simulation mode"
fi

# Check for API keys
if [[ -z "${CEREBRAS_API_KEY}" ]]; then
    echo "⚠️ CEREBRAS_API_KEY not set - AI chat will be limited"
    echo "   Set it with: export CEREBRAS_API_KEY='your_key_here'"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the JADED Platform:"
echo "  python3 coordinator.py"
echo ""
echo "Then open: http://localhost:5000"
