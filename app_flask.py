from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import asyncio
import os
import zipfile
import io
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
import openai
import anthropic
import re
import datetime
import threading
import time

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class WorkflowStage(Enum):
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"
    INITIAL_CODE = "initial_code"
    REVIEW_ANALYSIS = "review_analysis"
    REFINEMENT = "refinement"
    FINAL_VALIDATION = "final_validation"
    PROJECT_GENERATION = "project_generation"
    COMPLETE = "complete"

@dataclass
class WorkflowResult:
    stage: str
    agent: str
    content: str
    feedback: Optional[str] = None
    suggestions: Optional[List[str]] = None

@dataclass
class ProjectFile:
    filename: str
    content: str
    description: str

class ClaudeAgent:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize Claude client: {str(e)}")
    
    def analyze_requirements(self, prompt: str) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this coding request and create a detailed implementation plan:

Request: {prompt}

Please provide:
1. Requirements analysis
2. Architecture approach
3. Key components needed
4. File structure recommendations
5. Potential challenges
6. Recommended implementation strategy

Be specific and technical."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude API error: {str(e)}")
            return None
    
    def generate_code(self, prompt: str, planning_context: str = None) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        
        try:
            context = f"\nImplementation Context:\n{planning_context}\n\n" if planning_context else ""
            full_prompt = f"""Generate high-quality Python code for: {prompt}{context}

Requirements:
- Follow best practices and PEP 8
- Include proper error handling
- Add comprehensive docstrings
- Make code modular and reusable
- Include type hints where appropriate

Please provide complete, working code."""
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude Code Generation error: {str(e)}")
            return None
    
    def refine_code(self, code: str, feedback: str) -> Optional[str]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": f"""Improve this code based on the feedback:

Original Code:
{code}

Feedback:
{feedback}

Please provide the refined code with improvements."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude refinement error: {str(e)}")
            return None

class ChatGPTAgent:
    def __init__(self):
        self.client = None
        self.current_model = "gpt-3.5-turbo"
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return
            
            self.client = openai.OpenAI(api_key=api_key)
            
            # Try GPT-4 first, fallback to GPT-3.5
            for model in ["gpt-4", "gpt-3.5-turbo"]:
                try:
                    test_response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5
                    )
                    self.current_model = model
                    print(f"Using OpenAI model: {model}")
                    break
                except:
                    continue
                    
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {str(e)}")
    
    def review_build_plan(self, plan: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not plan.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"""Review this implementation plan:

Original Request: {original_prompt}
Plan: {plan}

Provide:
1. Completeness analysis
2. Architecture feedback  
3. Key improvements needed

Keep response focused and actionable."""
                }],
                max_tokens=1200
            )
            
            review_text = response.choices[0].message.content
            
            # Extract suggestions
            suggestions = []
            lines = review_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    suggestions.append(line[1:].strip())
                    if len(suggestions) >= 3:
                        break
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": "Reviewed"
            }
        except Exception as e:
            print(f"ChatGPT plan review error: {str(e)}")
            return None
    
    def comprehensive_review(self, code: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{
                    "role": "user", 
                    "content": f"""Review this Python code:

Request: {original_prompt}
Code: {code}

Provide:
1. Correctness assessment
2. Best practices check
3. Specific improvements

Be concise and actionable."""
                }],
                max_tokens=1200
            )
            
            review_text = response.choices[0].message.content
            
            # Extract suggestions
            suggestions = []
            lines = review_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    suggestions.append(line[1:].strip())
                    if len(suggestions) >= 3:
                        break
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": "Reviewed"
            }
        except Exception as e:
            print(f"ChatGPT code review error: {str(e)}")
            return None
    
    def final_validation(self, code: str, original_prompt: str) -> Optional[str]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"""Final validation for:

Request: {original_prompt}
Code: {code}

Provide:
1. Quality assessment
2. Production readiness
3. Final recommendation

Keep it concise."""
                }],
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"ChatGPT validation error: {str(e)}")
            return None

class ProjectGenerator:
    def generate_project_structure(self, final_code: str, project_name: str, original_prompt: str) -> List[ProjectFile]:
        project_name = project_name.lower().replace(' ', '_')
        files = []
        
        # Main code file
        files.append(ProjectFile(
            filename="main.py",
            content=final_code,
            description="Main implementation file"
        ))
        
        # README
        readme_content = f"""# {project_name.replace('_', ' ').title()}

## Description
{original_prompt}

## Usage
```python
python main.py
```

## Generated
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*Generated using Multi-Agent AI Collaboration*
"""
        files.append(ProjectFile(
            filename="README.md",
            content=readme_content,
            description="Project documentation"
        ))
        
        # Requirements
        files.append(ProjectFile(
            filename="requirements.txt",
            content="# Add your dependencies here\n",
            description="Python dependencies"
        ))
        
        # Test file
        test_content = """import unittest
from main import *

class TestMain(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""
        files.append(ProjectFile(
            filename="test_main.py",
            content=test_content,
            description="Basic unit tests"
        ))
        
        return files

class WorkflowOrchestrator:
    def __init__(self):
        self.claude = ClaudeAgent()
        self.chatgpt = ChatGPTAgent()
        self.project_gen = ProjectGenerator()
        self.results = []
        self.project_files = []
    
    def run_workflow(self, prompt: str, project_name: str, session_id: str, max_iterations: int = 1):
        """Run workflow in separate thread to avoid blocking"""
        thread = threading.Thread(
            target=self._execute_workflow,
            args=(prompt, project_name, session_id, max_iterations)
        )
        thread.start()
    
    def _execute_workflow(self, prompt: str, project_name: str, session_id: str, max_iterations: int):
        self.results = []
        
        try:
            # Stage 1: Planning
            self._emit_progress(session_id, 1, 7, "Claude analyzing requirements...")
            planning = self.claude.analyze_requirements(prompt)
            if planning:
                self.results.append(WorkflowResult("planning", "Claude", planning))
                self._emit_result(session_id, "planning", planning)
            else:
                self._emit_error(session_id, "Planning failed")
                return
            
            # Stage 2: Plan Review
            self._emit_progress(session_id, 2, 7, "ChatGPT reviewing plan...")
            plan_review = self.chatgpt.review_build_plan(planning, prompt)
            enhanced_planning = planning
            if plan_review:
                self.results.append(WorkflowResult("plan_review", "ChatGPT", plan_review["review"]))
                self._emit_result(session_id, "plan_review", plan_review["review"])
                if plan_review["suggestions"]:
                    suggestions_text = "\n".join([f"- {s}" for s in plan_review["suggestions"]])
                    enhanced_planning = f"{planning}\n\nSuggestions:\n{suggestions_text}"
            
            # Stage 3: Code Generation
            self._emit_progress(session_id, 3, 7, "Claude generating code...")
            initial_code = self.claude.generate_code(prompt, enhanced_planning)
            if initial_code:
                self.results.append(WorkflowResult("initial_code", "Claude", initial_code))
                self._emit_result(session_id, "code", initial_code)
            else:
                self._emit_error(session_id, "Code generation failed")
                return
            
            # Stage 4: Code Review
            self._emit_progress(session_id, 4, 7, "ChatGPT reviewing code...")
            review_result = self.chatgpt.comprehensive_review(initial_code, prompt)
            current_code = initial_code
            if review_result:
                self.results.append(WorkflowResult("review_analysis", "ChatGPT", review_result["review"]))
                self._emit_result(session_id, "review", review_result["review"])
            
            # Stage 5: Refinement (if needed)
            if review_result and review_result["suggestions"] and max_iterations > 0:
                self._emit_progress(session_id, 5, 7, "Claude refining code...")
                feedback = "\n".join(review_result["suggestions"][:2])  # Limit feedback
                refined_code = self.claude.refine_code(current_code, feedback)
                if refined_code:
                    current_code = refined_code
                    self.results.append(WorkflowResult("refinement", "Claude", refined_code))
                    self._emit_result(session_id, "refined_code", refined_code)
            
            # Stage 6: Final Validation
            self._emit_progress(session_id, 6, 7, "ChatGPT final validation...")
            final_validation = self.chatgpt.final_validation(current_code, prompt)
            if final_validation:
                self.results.append(WorkflowResult("final_validation", "ChatGPT", final_validation))
                self._emit_result(session_id, "validation", final_validation)
            
            # Stage 7: Project Generation
            self._emit_progress(session_id, 7, 7, "Generating project files...")
            self.project_files = self.project_gen.generate_project_structure(current_code, project_name, prompt)
            
            # Send completion
            self._emit_complete(session_id, self.results, self.project_files)
            
        except Exception as e:
            self._emit_error(session_id, f"Workflow error: {str(e)}")
    
    def _emit_progress(self, session_id: str, current: int, total: int, message: str):
        socketio.emit('progress', {
            'current': current,
            'total': total,
            'message': message,
            'percentage': int((current / total) * 100)
        }, room=session_id)
    
    def _emit_result(self, session_id: str, stage: str, content: str):
        socketio.emit('stage_result', {
            'stage': stage,
            'content': content
        }, room=session_id)
    
    def _emit_error(self, session_id: str, message: str):
        socketio.emit('error', {'message': message}, room=session_id)
    
    def _emit_complete(self, session_id: str, results: List[WorkflowResult], files: List[ProjectFile]):
        socketio.emit('workflow_complete', {
            'results': [asdict(r) for r in results],
            'files': [asdict(f) for f in files]
        }, room=session_id)

# Global orchestrator instance
orchestrator = WorkflowOrchestrator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_workflow', methods=['POST'])
def start_workflow():
    data = request.json
    prompt = data.get('prompt', '').strip()
    project_name = data.get('project_name', '').strip()
    session_id = data.get('session_id', 'default')
    max_iterations = data.get('max_iterations', 1)
    
    if not prompt or not project_name:
        return jsonify({'error': 'Missing prompt or project name'}), 400
    
    # Start workflow in background
    orchestrator.run_workflow(prompt, project_name, session_id, max_iterations)
    
    return jsonify({'status': 'started'})

@app.route('/download_project/<project_name>')
def download_project(project_name):
    if not orchestrator.project_files:
        return "No project files available", 404
    
    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in orchestrator.project_files:
            file_path = f"{project_name}/{file.filename}"
            zip_file.writestr(file_path, file.content)
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f"{project_name}.zip",
        mimetype='application/zip'
    )

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('join')
def handle_join(data):
    session_id = data['session_id']
    socketio.join_room(session_id)
    emit('joined', {'session_id': session_id})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Code Collaboration</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .input-section { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .progress-section { margin-bottom: 20px; }
        .progress-bar { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s ease; }
        .results-section { display: none; }
        .stage-result { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .code-block { background: #f8f8f8; padding: 15px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
        .button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .button:hover { background: #45a049; }
        .button:disabled { background: #cccccc; cursor: not-allowed; }
        .input-field { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        .textarea-field { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; min-height: 100px; }
        .error { color: red; padding: 10px; background: #ffebee; border-radius: 4px; margin: 10px 0; }
        .success { color: green; padding: 10px; background: #e8f5e8; border-radius: 4px; margin: 10px 0; }
        .tabs { display: flex; border-bottom: 1px solid #ddd; margin: 20px 0; }
        .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }
        .tab.active { border-bottom: 2px solid #4CAF50; background: #f0f8f0; }
        .tab-content { display: none; padding: 20px; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Multi-Agent Code Collaboration</h1>
        <h2>Claude + ChatGPT → Reliable Project Generation</h2>
    </div>

    <div class="input-section">
        <h3>💭 Describe Your Project</h3>
        <textarea id="prompt" class="textarea-field" placeholder="Example: Create a simple web scraper for product prices"></textarea>
        <input type="text" id="projectName" class="input-field" placeholder="Project Name (e.g., my_project)">
        <label>Max Refinement Iterations: <input type="number" id="maxIterations" value="1" min="0" max="2" style="width: 60px;"></label>
        <br><br>
        <button id="generateBtn" class="button">🚀 Generate Project</button>
    </div>

    <div class="progress-section">
        <div id="progressStatus" style="margin-bottom: 10px;"></div>
        <div class="progress-bar">
            <div id="progressFill" class="progress-fill" style="width: 0%;"></div>
        </div>
    </div>

    <div id="errorSection"></div>

    <div id="resultsSection" class="results-section">
        <div class="tabs">
            <div class="tab active" data-tab="overview">📋 Overview</div>
            <div class="tab" data-tab="planning">🎯 Planning</div>
            <div class="tab" data-tab="code">⚡ Code</div>
            <div class="tab" data-tab="files">📦 Files</div>
        </div>

        <div id="overview" class="tab-content active">
            <h3>Project Summary</h3>
            <div id="summaryStats"></div>
        </div>

        <div id="planning" class="tab-content">
            <div id="planningContent"></div>
        </div>

        <div id="code" class="tab-content">
            <div id="codeContent"></div>
        </div>

        <div id="files" class="tab-content">
            <div id="filesContent"></div>
            <button id="downloadBtn" class="button" style="margin-top: 20px;">📦 Download Project ZIP</button>
        </div>
    </div>

    <script>
        const socket = io();
        const sessionId = Math.random().toString(36).substr(2, 9);
        let currentProjectName = '';
        
        // Join session
        socket.emit('join', {session_id: sessionId});
        
        // Tab functionality
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');
            });
        });
        
        // Generate button click
        document.getElementById('generateBtn').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value.trim();
            const projectName = document.getElementById('projectName').value.trim();
            const maxIterations = parseInt(document.getElementById('maxIterations').value);
            
            if (!prompt || !projectName) {
                showError('Please enter both a project description and name.');
                return;
            }
            
            currentProjectName = projectName;
            document.getElementById('generateBtn').disabled = true;
            document.getElementById('errorSection').innerHTML = '';
            document.getElementById('resultsSection').style.display = 'none';
            
            try {
                const response = await fetch('/start_workflow', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        project_name: projectName,
                        session_id: sessionId,
                        max_iterations: maxIterations
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to start workflow');
                }
            } catch (error) {
                showError('Failed to start workflow: ' + error.message);
                document.getElementById('generateBtn').disabled = false;
            }
        });
        
        // Download button click
        document.getElementById('downloadBtn').addEventListener('click', () => {
            if (currentProjectName) {
                window.open(`/download_project/${currentProjectName}`, '_blank');
            }
        });
        
        // Socket event handlers
        socket.on('progress', (data) => {
            document.getElementById('progressFill').style.width = data.percentage + '%';
            document.getElementById('progressStatus').textContent = data.message;
        });
        
        socket.on('stage_result', (data) => {
            handleStageResult(data.stage, data.content);
        });
        
        socket.on('workflow_complete', (data) => {
            document.getElementById('generateBtn').disabled = false;
            document.getElementById('progressStatus').innerHTML = '<span class="success">🎉 Project Generated Successfully!</span>';
            document.getElementById('resultsSection').style.display = 'block';
            
            // Update summary
            document.getElementById('summaryStats').innerHTML = `
                <p><strong>Stages Completed:</strong> 7/7</p>
                <p><strong>Project Files:</strong> ${data.files.length}</p>
                <p><strong>AI Agents Used:</strong> 2 (Claude + ChatGPT)</p>
            `;
            
            // Display files
            displayFiles(data.files);
        });
        
        socket.on('error', (data) => {
            showError(data.message);
            document.getElementById('generateBtn').disabled = false;
        });
        
        function handleStageResult(stage, content) {
            if (stage === 'planning') {
                document.getElementById('planningContent').innerHTML = `
                    <div class="stage-result">
                        <h4>🎯 Requirements Analysis</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            } else if (stage === 'plan_review') {
                document.getElementById('planningContent').innerHTML += `
                    <div class="stage-result">
                        <h4>🔍 Plan Review</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            } else if (stage === 'code') {
                document.getElementById('codeContent').innerHTML = `
                    <div class="stage-result">
                        <h4>⚡ Generated Code</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            } else if (stage === 'review') {
                document.getElementById('codeContent').innerHTML += `
                    <div class="stage-result">
                        <h4>🔍 Code Review</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            } else if (stage === 'refined_code') {
                document.getElementById('codeContent').innerHTML += `
                    <div class="stage-result">
                        <h4>🔄 Refined Code</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            } else if (stage === 'validation') {
                document.getElementById('codeContent').innerHTML += `
                    <div class="stage-result">
                        <h4>✅ Final Validation</h4>
                        <div class="code-block">${content}</div>
                    </div>
                `;
            }
        }
        
        function displayFiles(files) {
            let filesHtml = '<h3>📁 Project Files</h3>';
            files.forEach((file, index) => {
                filesHtml += `
                    <div class="stage-result">
                        <h4>📄 ${file.filename}</h4>
                        <p><em>${file.description}</em></p>
                        <div class="code-block">${file.content}</div>
                    </div>
                `;
            });
            document.getElementById('filesContent').innerHTML = filesHtml + document.getElementById('downloadBtn').outerHTML;
        }
        
        function showError(message) {
            document.getElementById('errorSection').innerHTML = `<div class="error">❌ ${message}</div>`;
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("🚀 Starting Flask app...")
    print("📱 Open: http://localhost:5001")
    print("🔧 More reliable than Streamlit for long-running processes!")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)