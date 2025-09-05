#!/usr/bin/env python3
"""
WSGI entry point for Script to Slides Generator
Production-ready Flask application for Heroku deployment
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_to_slides_enhanced import app

# Configure for production
app.config['DEBUG'] = False
app.config['ENV'] = 'production'

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )