#!/bin/bash

# Build script for Image-to-Image Editor Application

set -e

echo "====================================="
echo "Building Image-to-Image Editor App"
echo "====================================="

# Navigate to frontend directory
echo ""
echo "Step 1: Installing frontend dependencies..."
cd frontend

if [ ! -d "node_modules" ]; then
    npm install
else
    echo "Frontend dependencies already installed."
fi

# Build frontend
echo ""
echo "Step 2: Building frontend..."
npm run build

# Check if build was successful
if [ -d "../static" ]; then
    echo "✓ Frontend built successfully!"
    echo "✓ Static files created in static/"
else
    echo "✗ Frontend build failed!"
    exit 1
fi

# Navigate back to root
cd ..

# Install backend dependencies
echo ""
echo "Step 3: Installing backend dependencies..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing Python packages..."
pip install -q --upgrade pip setuptools wheel

# Check if pyproject.toml exists and install accordingly
if [ -f "pyproject.toml" ]; then
    echo "Installing from pyproject.toml..."
    pip install -q -e .
else
    echo "Installing from requirements.txt..."
    pip install -q -r requirements.txt
fi

echo "✓ Backend dependencies installed!"

echo ""
echo "====================================="
echo "Build complete!"
echo "====================================="
echo ""
echo "To run in production mode:"
echo "  source venv/bin/activate"
echo "  export ENV=production"
echo "  python main.py"
echo ""
echo "Then visit: http://localhost:8000"
echo "====================================="
