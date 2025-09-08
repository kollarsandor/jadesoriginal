#!/bin/bash

echo "🚀 Starting JADED Platform..."

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the platform
python coordinator.py
