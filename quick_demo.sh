#!/bin/bash

# Quick Demo Script for UI Testing
# Launches the demo server and opens browser

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Document to Slides - UI Demo Launcher               ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing..."
    pip install flask
fi

echo "🚀 Starting demo server..."
echo ""

# Start server in background
python ui_demo_server.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 3

# Check if server is running
if ! curl -s http://localhost:5001/health > /dev/null; then
    echo "❌ Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Server running on http://localhost:5001"
echo ""
echo "📊 Opening comparison page in browser..."
echo ""

# Try to open browser (works on macOS, Linux, Windows)
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:5001/compare
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:5001/compare
elif command -v start &> /dev/null; then
    # Windows
    start http://localhost:5001/compare
else
    echo "📋 Could not auto-open browser. Please visit:"
    echo "   http://localhost:5001/compare"
fi

echo ""
echo "Available URLs:"
echo "  • http://localhost:5001/         - Side-by-side comparison"
echo "  • http://localhost:5001/old      - Original UI"
echo "  • http://localhost:5001/new      - Streamlined UI"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for user to stop
trap "kill $SERVER_PID 2>/dev/null; echo ''; echo '👋 Server stopped'; exit 0" INT TERM

# Keep script running
wait $SERVER_PID
