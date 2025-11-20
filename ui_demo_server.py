#!/usr/bin/env python3
"""
Simple demo server to compare old vs new UI.

This lightweight server serves both UI versions for comparison
without requiring full Flask app dependencies.

Usage:
    python ui_demo_server.py

Then visit:
    - http://localhost:5001/old - Original UI
    - http://localhost:5001/new - Streamlined UI
    - http://localhost:5001/compare - Side-by-side comparison
"""

from flask import Flask, send_from_directory, render_template
import os

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    """Show comparison page by default."""
    return send_from_directory('templates', 'ui_comparison.html')


@app.route('/compare')
def compare():
    """Side-by-side comparison."""
    return send_from_directory('templates', 'ui_comparison.html')


@app.route('/old')
def old_ui():
    """Serve original UI."""
    return send_from_directory('templates', 'file_to_slides.html.backup')


@app.route('/new')
def new_ui():
    """Serve streamlined UI."""
    return send_from_directory('templates', 'file_to_slides_streamlined.html')


@app.route('/file_to_slides.html')
def old_ui_iframe():
    """For iframe embedding in comparison page."""
    return send_from_directory('templates', 'file_to_slides.html.backup')


@app.route('/file_to_slides_streamlined.html')
def new_ui_iframe():
    """For iframe embedding in comparison page."""
    return send_from_directory('templates', 'file_to_slides_streamlined.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'message': 'UI Demo Server running'}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║   UI Comparison Demo Server                   ║
    ╚═══════════════════════════════════════════════╝

    🌐 Server starting on http://localhost:{port}

    Available routes:
    ─────────────────────────────────────────────────
    📊 http://localhost:{port}/           - Side-by-side comparison
    📊 http://localhost:{port}/compare    - Side-by-side comparison
    📜 http://localhost:{port}/old        - Original UI (full page)
    ✨ http://localhost:{port}/new        - Streamlined UI (full page)

    Press Ctrl+C to stop the server
    """)

    app.run(host='0.0.0.0', port=port, debug=False)
