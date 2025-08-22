#!/bin/bash
# Run Script to Slides Generator in production mode locally

echo "🚀 Starting Script to Slides Generator (Production Mode)"
echo ""

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "📦 Installing gunicorn..."
    pip install gunicorn
fi

# Create necessary directories
mkdir -p uploads exports temp_sketches static

# Start the production server
echo "🌟 Starting production server..."
echo "📱 Your app will be available at: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run with gunicorn
gunicorn wsgi:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 300 \
    --max-requests 1000 \
    --preload \
    --log-level info