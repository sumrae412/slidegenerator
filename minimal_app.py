#!/usr/bin/env python3

import http.server
import socketserver
import json
import os
from typing import Optional
from dotenv import load_dotenv
import anthropic
import openai

load_dotenv()

class CodeGenerator:
    def __init__(self):
        # Initialize Claude
        self.claude = None
        try:
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.claude = anthropic.Anthropic(api_key=claude_key)
                print("✅ Claude ready")
        except Exception as e:
            print(f"❌ Claude error: {e}")
        
        # Initialize OpenAI
        self.openai = None
        self.openai_model = "gpt-3.5-turbo"
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai = openai.OpenAI(api_key=openai_key)
                # Try GPT-4 first
                for model in ["gpt-4", "gpt-3.5-turbo"]:
                    try:
                        test = self.openai.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=5
                        )
                        self.openai_model = model
                        print(f"✅ OpenAI using: {model}")
                        break
                    except:
                        continue
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
    
    def generate_project(self, prompt: str) -> dict:
        result = {"success": False, "code": "", "review": "", "error": ""}
        
        try:
            # Step 1: Generate code with Claude
            print("🎯 Generating code...")
            if self.claude:
                response = self.claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"Generate complete Python code for: {prompt}\n\nInclude proper error handling, docstrings, and make it production-ready."
                    }]
                )
                result["code"] = response.content[0].text
                print("✅ Code generated")
            else:
                result["error"] = "Claude not available"
                return result
            
            # Step 2: Review with OpenAI
            print("🔍 Reviewing code...")
            if self.openai and result["code"]:
                response = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{
                        "role": "user",
                        "content": f"Review this Python code and provide feedback:\n\n{result['code']}"
                    }],
                    max_tokens=1000
                )
                result["review"] = response.choices[0].message.content
                print("✅ Code reviewed")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Error: {e}")
        
        return result

# Global generator
generator = CodeGenerator()

class SimpleHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/generate':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                prompt = data.get('prompt', '').strip()
                if not prompt:
                    self.send_error(400, "No prompt provided")
                    return
                
                result = generator.generate_project(prompt)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_html(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>🤖 AI Code Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .input-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        textarea { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-family: Arial, sans-serif; resize: vertical; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .result { margin-top: 20px; padding: 15px; border-radius: 6px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .code-block { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 15px; margin: 10px 0; white-space: pre-wrap; font-family: 'Courier New', monospace; overflow-x: auto; }
        .loading { text-align: center; color: #666; }
        .tabs { display: flex; margin: 20px 0 0 0; border-bottom: 2px solid #ddd; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; background: #f8f9fa; margin-right: 5px; border-radius: 6px 6px 0 0; }
        .tab.active { background: white; border-bottom: 2px solid #4CAF50; }
        .tab-content { display: none; padding: 20px; background: white; border-radius: 0 6px 6px 6px; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Code Generator</h1>
        <p style="text-align: center; color: #666;">Powered by Claude + ChatGPT</p>
        
        <div class="input-group">
            <label for="prompt">What do you want to build?</label>
            <textarea id="prompt" rows="4" placeholder="Example: Create a simple calculator that can add, subtract, multiply, and divide two numbers with error handling"></textarea>
        </div>
        
        <button id="generateBtn" onclick="generateCode()">🚀 Generate Code</button>
        
        <div id="result"></div>
        
        <div id="tabs" style="display: none;">
            <div class="tabs">
                <div class="tab active" onclick="showTab('code')">📄 Generated Code</div>
                <div class="tab" onclick="showTab('review')">🔍 AI Review</div>
            </div>
            <div id="code-tab" class="tab-content active">
                <div id="generated-code"></div>
            </div>
            <div id="review-tab" class="tab-content">
                <div id="code-review"></div>
            </div>
        </div>
    </div>

    <script>
        async function generateCode() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                showResult('Please enter a description of what you want to build.', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.textContent = '🔄 Generating...';
            
            document.getElementById('tabs').style.display = 'none';
            showResult('🎯 AI is generating your code... This may take 30-60 seconds.', 'loading');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showResult('✅ Code generated successfully!', 'success');
                    
                    document.getElementById('generated-code').innerHTML = 
                        '<div class="code-block">' + escapeHtml(result.code) + '</div>';
                    
                    document.getElementById('code-review').innerHTML = 
                        '<div class="code-block">' + escapeHtml(result.review) + '</div>';
                    
                    document.getElementById('tabs').style.display = 'block';
                } else {
                    showResult('❌ Error: ' + result.error, 'error');
                }
                
            } catch (error) {
                showResult('❌ Network error: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🚀 Generate Code';
            }
        }
        
        function showResult(message, type) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `<div class="result ${type}">${message}</div>`;
        }
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Show selected tab
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Allow Enter to submit
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generateCode();
            }
        });
    </script>
</body>
</html>'''

def main():
    PORT = 8888
    
    print("🚀 Starting AI Code Generator...")
    print(f"📱 Open: http://localhost:{PORT}")
    print("🔧 Simple HTTP server - should work on any system!")
    print("\n" + "="*50)
    
    try:
        with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
            print(f"✅ Server running on port {PORT}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {PORT} is already in use. Trying port {PORT+1}...")
            PORT = PORT + 1
            with socketserver.TCPServer(("", PORT), SimpleHandler) as httpd:
                print(f"✅ Server running on port {PORT}")
                print(f"📱 Open: http://localhost:{PORT}")
                httpd.serve_forever()
        else:
            print(f"❌ Server error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()