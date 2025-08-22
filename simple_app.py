#!/usr/bin/env python3
"""
Simple Multi-Agent AI Code Collaboration
A lightweight web interface for Claude + ChatGPT collaboration
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import json
import os
import threading
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent AI Code Collaboration</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .fade-in { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .progress-bar { transition: width 0.5s ease-in-out; }
        .code-block { 
            background: #1f2937; 
            color: #f9fafb; 
            padding: 1rem; 
            border-radius: 0.5rem; 
            white-space: pre-wrap; 
            font-family: 'Courier New', monospace; 
            overflow-x: auto;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-4 py-4">
            <h1 class="text-2xl font-bold text-gray-900">🤖 Multi-Agent AI Code Collaboration</h1>
            <p class="text-sm text-gray-600">Claude + ChatGPT → Reliable Project Generation</p>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-8">
        <!-- Input Section -->
        <div class="bg-white rounded-lg shadow-sm border mb-8">
            <div class="p-6">
                <h2 class="text-lg font-semibold text-gray-900 mb-4">💭 Describe Your Coding Project</h2>
                
                <form id="generateForm" onsubmit="generateProject(event)">
                    <div class="space-y-4">
                        <div>
                            <label for="prompt" class="block text-sm font-medium text-gray-700 mb-2">Project Description</label>
                            <textarea
                                id="prompt"
                                rows="4"
                                class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                placeholder="Example: Create a web scraper that extracts product information from e-commerce sites with error handling and CSV export"
                                required
                            ></textarea>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label for="project-name" class="block text-sm font-medium text-gray-700 mb-2">Project Name</label>
                                <input
                                    type="text"
                                    id="project-name"
                                    class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="my_awesome_project"
                                    required
                                />
                            </div>
                            
                            <div class="flex items-end">
                                <button
                                    type="submit"
                                    id="generate-btn"
                                    class="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors font-medium"
                                >
                                    🚀 Generate Project
                                </button>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <!-- Status Section -->
        <div id="status-section" class="bg-white rounded-lg shadow-sm border mb-8 hidden">
            <div class="p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">⚡ Generation Status</h3>
                <div id="status-messages" class="space-y-2">
                    <!-- Status messages will appear here -->
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results-section" class="bg-white rounded-lg shadow-sm border hidden">
            <div class="p-6">
                <h3 class="text-lg font-semibold text-gray-900 mb-4">🎉 Generated Project</h3>
                <div id="results-content">
                    <!-- Results will appear here -->
                </div>
            </div>
        </div>

        <!-- Error Section -->
        <div id="error-section" class="hidden">
            <div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                <div class="flex">
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">Error</h3>
                        <div class="mt-2 text-sm text-red-700" id="error-message"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        function addStatusMessage(message, type = 'info') {
            const statusSection = document.getElementById('status-section');
            const messagesDiv = document.getElementById('status-messages');
            
            statusSection.classList.remove('hidden');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `flex items-start space-x-2 text-sm fade-in p-3 rounded-md ${
                type === 'success' ? 'bg-green-50 text-green-800' : 
                type === 'error' ? 'bg-red-50 text-red-800' : 
                'bg-blue-50 text-blue-800'
            }`;
            
            const timestamp = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `
                <span class="flex-shrink-0 text-xs text-gray-500">${timestamp}</span>
                <span>${message}</span>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function showError(message) {
            const errorSection = document.getElementById('error-section');
            const errorMessage = document.getElementById('error-message');
            
            errorMessage.textContent = message;
            errorSection.classList.remove('hidden');
        }

        function hideError() {
            document.getElementById('error-section').classList.add('hidden');
        }

        function showResults(data) {
            const resultsSection = document.getElementById('results-section');
            const resultsContent = document.getElementById('results-content');
            
            let html = '';
            
            if (data.success) {
                html += `<div class="space-y-6">`;
                
                // Summary
                html += `<div class="grid grid-cols-3 gap-4 mb-6">`;
                html += `<div class="bg-blue-50 rounded-lg p-4 text-center"><div class="text-2xl font-bold text-blue-600">${data.stages || 7}</div><div class="text-sm text-gray-600">Workflow Stages</div></div>`;
                html += `<div class="bg-green-50 rounded-lg p-4 text-center"><div class="text-2xl font-bold text-green-600">${data.files ? data.files.length : 0}</div><div class="text-sm text-gray-600">Files Generated</div></div>`;
                html += `<div class="bg-purple-50 rounded-lg p-4 text-center"><div class="text-2xl font-bold text-purple-600">2</div><div class="text-sm text-gray-600">AI Agents</div></div>`;
                html += `</div>`;
                
                // Generated content
                if (data.code) {
                    html += `<div class="mb-6">`;
                    html += `<h4 class="text-lg font-semibold text-gray-900 mb-3">⚡ Generated Code</h4>`;
                    html += `<div class="code-block">${escapeHtml(data.code)}</div>`;
                    html += `</div>`;
                }
                
                if (data.review) {
                    html += `<div class="mb-6">`;
                    html += `<h4 class="text-lg font-semibold text-gray-900 mb-3">🔍 AI Review</h4>`;
                    html += `<div class="prose max-w-none bg-gray-50 p-4 rounded-lg">${escapeHtml(data.review)}</div>`;
                    html += `</div>`;
                }
                
                html += `</div>`;
                
                // Download buttons
                html += `<div class="mt-6 flex space-x-4">`;
                html += `<button onclick="downloadCode()" class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">📥 Download main.py</button>`;
                html += `<button onclick="downloadReadme()" class="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">📄 Download README.md</button>`;
                html += `</div>`;
            } else {
                html = `<div class="text-center py-8 text-gray-500">`;
                html += `<p>Generation failed. Please try again.</p>`;
                html += `</div>`;
            }
            
            resultsContent.innerHTML = html;
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function downloadFile(filename, content) {
            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        let generatedData = {};

        function downloadCode() {
            if (generatedData.code) {
                downloadFile('main.py', generatedData.code);
            }
        }

        function downloadReadme() {
            const projectName = document.getElementById('project-name').value || 'AI Generated Project';
            const prompt = document.getElementById('prompt').value;
            
            const readme = `# ${projectName}

## Description
${prompt}

## Usage
```python
python main.py
```

## Generated on
${new Date().toLocaleString()}

---
*This project was generated using Multi-Agent AI Collaboration*
`;
            downloadFile('README.md', readme);
        }

        async function generateProject(event) {
            event.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            const projectName = document.getElementById('project-name').value.trim();
            const generateBtn = document.getElementById('generate-btn');
            
            if (!prompt || !projectName) {
                showError('Please fill in both project description and name.');
                return;
            }
            
            // Reset UI
            hideError();
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('status-messages').innerHTML = '';
            
            // Disable button
            generateBtn.disabled = true;
            generateBtn.textContent = '⏳ Generating...';
            
            try {
                addStatusMessage('🚀 Starting AI collaboration workflow...', 'info');
                
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        project_name: projectName
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.success) {
                    addStatusMessage('✅ Project generated successfully!', 'success');
                    generatedData = data;
                    showResults(data);
                } else {
                    throw new Error(data.error || 'Generation failed');
                }
                
            } catch (error) {
                console.error('Error:', error);
                showError(`Generation failed: ${error.message}`);
                addStatusMessage(`❌ Error: ${error.message}`, 'error');
            } finally {
                // Re-enable button
                generateBtn.disabled = false;
                generateBtn.textContent = '🚀 Generate Project';
            }
        }

        // Check API status on load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                console.log('API Status:', status);
            } catch (error) {
                console.error('Failed to check API status:', error);
            }
        });
    </script>
</body>
</html>'''

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status = {
                "claude": "✅ Available" if os.getenv("ANTHROPIC_API_KEY") else "❌ No API key",
                "chatgpt": "✅ Available" if os.getenv("OPENAI_API_KEY") else "❌ No API key",
                "server": "✅ Running"
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/generate':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                prompt = data.get('prompt', '').strip()
                project_name = data.get('project_name', '').strip()
                
                if not prompt or not project_name:
                    raise ValueError("Both prompt and project name are required")
                
                # Simple mock response for demonstration
                # In a real implementation, this would call the AI agents
                result = {
                    "success": True,
                    "stages": 7,
                    "code": generate_simple_code(prompt),
                    "review": f"This code for '{prompt}' looks good and follows Python best practices. The implementation is clean and well-structured.",
                    "files": [
                        {"filename": "main.py", "description": "Main implementation"},
                        {"filename": "README.md", "description": "Project documentation"},
                        {"filename": "requirements.txt", "description": "Dependencies"}
                    ]
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                logger.error(f"Error in POST /api/generate: {e}")
                error_response = {"success": False, "error": str(e)}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def generate_simple_code(prompt):
    """Generate a simple code template based on the prompt"""
    code_template = f'''#!/usr/bin/env python3
"""
{prompt}

This is a basic implementation template.
Customize this code according to your specific requirements.
"""

def main():
    """
    Main function for: {prompt}
    """
    print("Starting {prompt.lower()}...")
    
    # TODO: Implement your logic here
    # This is a template - customize based on your needs
    
    try:
        # Add your implementation here
        result = "Implementation completed successfully"
        print(f"Result: {{result}}")
        
    except Exception as e:
        print(f"Error: {{e}}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Program completed successfully!")
    else:
        print("❌ Program failed!")
'''
    return code_template

def run_server():
    """Run the HTTP server"""
    PORT = 8080
    
    try:
        with HTTPServer(('localhost', PORT), CustomHandler) as httpd:
            logger.info(f"🚀 Multi-Agent AI Code Collaboration server running at:")
            logger.info(f"   http://localhost:{PORT}")
            logger.info(f"   Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == '__main__':
    run_server()