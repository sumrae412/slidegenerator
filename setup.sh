#!/bin/bash
# Setup script for Script to Slides Generator

echo "🚀 Setting up Script to Slides Generator..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads exports temp_sketches static

# Set permissions
chmod +x setup.sh

echo "✅ Setup complete!"
echo ""
echo "To run locally:"
echo "  Development: python file_to_slides.py"
echo "  Production:  python wsgi.py"
echo "  Gunicorn:    gunicorn wsgi:app --bind 0.0.0.0:8000"
echo ""
echo "To deploy to Heroku, see DEPLOYMENT.md"