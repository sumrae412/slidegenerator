#!/usr/bin/env python3

import http.server
import socketserver
import json
import urllib.parse
import threading
import os
import zipfile
import io
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import anthropic
import datetime

load_dotenv()

@dataclass
class ProjectFile:
    filename: str
    content: str
    description: str

class ClaudeAgent:
    def __init__(self):
        self.client = None
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                print("✅ Claude client initialized")
        except Exception as e:
            print(f"❌ Claude init error: {e}")
    
    def analyze_requirements(self, prompt: str) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[{"role": "user", "content": f"Analyze and create implementation plan for: {prompt}"}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude planning error: {e}")
            return None
    
    def generate_code(self, prompt: str, planning: str = None) -> Optional[str]:
        if not self.client:
            return None
        try:
            context = f"\nPlan: {planning}" if planning else ""
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2500,
                messages=[{"role": "user", "content": f"Generate Python code for: {prompt}{context}"}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude code gen error: {e}")
            return None

class ChatGPTAgent:
    def __init__(self):
        self.client = None
        self.model = "gpt-3.5-turbo"
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                # Try GPT-4 first
                for model in ["gpt-4", "gpt-3.5-turbo"]:
                    try:
                        test = self.client.chat.completions.create(
                            model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
                        )
                        self.model = model
                        print(f"✅ OpenAI using: {model}")
                        break
                    except:
                        continue
        except Exception as e:
            print(f"❌ OpenAI init error: {e}")
    
    def review_code(self, code: str, prompt: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": f"Review this code for: {prompt}\n\nCode:\n{code}"}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"ChatGPT review error: {e}")
            return None

class SimpleWorkflow:
    def __init__(self):
        self.claude = ClaudeAgent()
        self.chatgpt = ChatGPTAgent()
        self.last_result = {}
    
    def run(self, prompt: str, project_name: str) -> Dict:
        result = {"stages": [], "files": [], "error": None}
        
        try:
            # Stage 1: Planning
            print("🎯 Stage 1: Planning...")
            planning = self.claude.analyze_requirements(prompt)
            if planning:
                result["stages"].append({"name": "Planning", "content": planning, "agent": "Claude"})
            
            # Stage 2: Code Generation  
            print("⚡ Stage 2: Code Generation...")
            code = self.claude.generate_code(prompt, planning)
            if code:
                result["stages"].append({"name": "Code Generation", "content": code, "agent": "Claude"})
            
            # Stage 3: Code Review
            print("🔍 Stage 3: Code Review...")
            review = self.chatgpt.review_code(code, prompt) if code else None
            if review:
                result["stages"].append({"name": "Code Review", "content": review, "agent": "ChatGPT"})
            
            # Stage 4: Generate Project Files
            print("📦 Stage 4: Project Generation...")
            if code:
                result["files"] = self._generate_files(code, project_name, prompt)
            
            self.last_result = result
            print("✅ Workflow completed!")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Workflow error: {e}")
        
        return result
    
    def _generate_files(self, code: str, project_name: str, prompt: str) -> List[Dict]:
        files = []
        
        # Main code file
        files.append({
            "filename": "main.py",
            "content": code,
            "description": "Main implementation"
        })
        
        # README
        readme = f"""# {project_name.title()}

## Description
{prompt}

## Usage
```python
python main.py
```

## Generated
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        files.append({
            "filename": "README.md", 
            "content": readme,
            "description": "Documentation"
        })
        
        return files

# Global workflow instance
workflow = SimpleWorkflow()

class SimpleHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode())
        elif self.path.startswith('/download/'):
            self.handle_download()
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                prompt = data.get('prompt', '').strip()
                project_name = data.get('project_name', '').strip()
                
                if not prompt or not project_name:
                    self.send_error(400, "Missing prompt or project name")
                    return
                
                # Run workflow
                result = workflow.run(prompt, project_name)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)
    
    def handle_download(self):
        try:
            if not workflow.last_result.get("files"):
                self.send_error(404, "No files to download")
                return
            
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file in workflow.last_result["files"]:
                    zip_file.writestr(file["filename"], file["content"])
            
            zip_data = zip_buffer.getvalue()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/zip')
            self.send_header('Content-Disposition', 'attachment; filename="project.zip"')
            self.send_header('Content-Length', str(len(zip_data)))
            self.end_headers()
            self.wfile.write(zip_data)
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def get_html(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>🤖 Simple Multi-Agent Collaboration</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .input-section { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .results { margin-top: 20px; }
        .stage { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .code { background: #f8f8f8; padding: 15px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
        .button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .button:hover { background: #45a049; }
        .button:disabled { background: #cccccc; }
        input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        textarea { height: 100px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .loading { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Simple Multi-Agent Collaboration</h1>
        <p>Claude + ChatGPT → Reliable Code Generation</p>
    </div>

    <div class="input-section">
        <h3>Describe Your Project</h3>
        <textarea id="prompt" placeholder="Example: Create a simple calculator with basic operations"></textarea>
        <input type="text" id="projectName" placeholder="Project Name (e.g., calculator)">
        <br><br>
        <button id="generateBtn" class="button" onclick="generateProject()">🚀 Generate Project</button>
    </div>

    <div id="status"></div>
    <div id="results" class="results"></div>

    <script>
        async function generateProject() {
            const prompt = document.getElementById('prompt').value.trim();
            const projectName = document.getElementById('projectName').value.trim();
            
            if (!prompt || !projectName) {
                showStatus('Please enter both a project description and name.', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.textContent = '🔄 Generating...';
            
            showStatus('Starting workflow...', 'loading');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, project_name: projectName})
                });
                
                const result = await response.json();
                
                if (result.error) {
                    showStatus('Error: ' + result.error, 'error');
                } else {
                    showStatus('✅ Project generated successfully!', 'success');
                    displayResults(result);
                }
                
            } catch (error) {
                showStatus('Network error: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Generate Project';
            }
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function displayResults(result) {
            let html = '<h3>🎉 Results</h3>';
            
            // Show stages
            if (result.stages && result.stages.length > 0) {
                result.stages.forEach(stage => {
                    html += `
                        <div class="stage">
                            <h4>${stage.name} (${stage.agent})</h4>
                            <div class="code">${stage.content}</div>
                        </div>
                    `;
                });
            }
            
            // Show files
            if (result.files && result.files.length > 0) {
                html += '<h3>📁 Generated Files</h3>';
                result.files.forEach(file => {
                    html += `
                        <div class="stage">
                            <h4>📄 ${file.filename}</h4>
                            <p><em>${file.description}</em></p>
                            <div class="code">${file.content}</div>
                        </div>
                    `;
                });
                
                html += '<br><a href="/download/project.zip" class="button">📦 Download Project ZIP</a>';
            }
            
            document.getElementById('results').innerHTML = html;
        }
        
        // Check server status on load
        fetch('/status').then(r => r.json()).then(data => {
            showStatus('🟢 Server ready', 'success');
        }).catch(e => {
            showStatus('🔴 Server connection issue', 'error');
        });
    </script>
</body>
</html>'''

def main():
    PORT = 8080
    
    print("🚀 Starting Simple Multi-Agent Server...")
    print(f"📱 Open: http://localhost:{PORT}")
    print("🔧 Simple HTTP server - should work everywhere!")
    print("\n✅ Features:")
    print("  - No complex dependencies")
    print("  - No WebSocket issues") 
    print("  - No connection timeouts")
    print("  - Direct HTTP requests")
    print("  - Simple and reliable")
    print("\n" + "="*50)
    
    with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()