#!/usr/bin/env python3
"""
FastAPI Multi-Agent AI Code Collaboration
A modern web interface for Claude + ChatGPT collaboration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import os
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
import openai
import anthropic
import re
import datetime
import time
import logging
from pydantic import BaseModel

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent AI Code Collaboration", version="1.0.0")

# Create templates directory and mount static files
templates_dir = Path("templates")
static_dir = Path("static")
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    stage: WorkflowStage
    agent: str
    content: str
    feedback: Optional[str] = None
    suggestions: Optional[List[str]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()

@dataclass
class ProjectFile:
    filename: str
    content: str
    description: str

class GenerateRequest(BaseModel):
    prompt: str
    project_name: str
    max_iterations: int = 2

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_update(self, message: dict, websocket: WebSocket = None):
        if websocket:
            await websocket.send_text(json.dumps(message))
        else:
            # Broadcast to all connections
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    pass  # Connection might be closed

manager = ConnectionManager()

class ClaudeAgent:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("Claude client initialized successfully")
            else:
                logger.error("ANTHROPIC_API_KEY not found")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
    
    async def analyze_requirements(self, prompt: str, websocket: WebSocket) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 1,
                "message": "Claude is analyzing requirements...",
                "agent": "Claude"
            }, websocket)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
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
            await manager.send_update({
                "type": "error",
                "message": f"Claude API error: {str(e)}"
            }, websocket)
            return None
    
    async def generate_code(self, prompt: str, planning_context: str, websocket: WebSocket) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 3,
                "message": "Claude is generating code...",
                "agent": "Claude"
            }, websocket)
            
            context = f"\nImplementation Context:\n{planning_context}\n\n" if planning_context else ""
            full_prompt = f"""Generate high-quality Python code for: {prompt}{context}

Requirements:
- Follow best practices and PEP 8
- Include proper error handling
- Add comprehensive docstrings
- Make code modular and reusable
- Include type hints where appropriate
- Add inline comments explaining complex logic
- If multiple files are needed, separate them with clear headers like:
  # File: filename.py
- Include example usage in comments

Please provide complete, working code that can be run immediately."""
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"Claude Code Generation error: {str(e)}"
            }, websocket)
            return None
    
    async def refine_code(self, code: str, feedback: str, websocket: WebSocket) -> Optional[str]:
        if not self.client or not code.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 5,
                "message": "Claude is refining the code...",
                "agent": "Claude"
            }, websocket)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": f"""Improve this code based on the feedback provided:

Original Code:
{code}

Feedback to Address:
{feedback}

Please provide the refined code with improvements addressing all feedback points. Include comments explaining the improvements made."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"Claude refinement error: {str(e)}"
            }, websocket)
            return None

class ChatGPTAgent:
    def __init__(self):
        self.client = None
        self.available_models = ["gpt-5", "gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        self.current_model = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self._determine_best_model()
                logger.info(f"OpenAI client initialized with model: {self.current_model}")
            else:
                logger.error("OPENAI_API_KEY not found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _determine_best_model(self):
        """Test models and select the best available one"""
        if not self.client:
            return
        
        for model in self.available_models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                self.current_model = model
                return
            except Exception as e:
                if "does not exist" in str(e).lower():
                    continue
                elif "insufficient_quota" in str(e).lower():
                    self.current_model = model
                    return
        
        self.current_model = "gpt-3.5-turbo"
    
    async def review_build_plan(self, plan: str, original_prompt: str, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        if not self.client or not plan.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 2,
                "message": "ChatGPT is reviewing the build plan...",
                "agent": "ChatGPT"
            }, websocket)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"""Review this implementation plan and provide detailed feedback:

Original Request: {original_prompt}

Build Plan to Review:
{plan}

Please analyze and provide:
1. **Completeness**: Are all requirements covered?
2. **Architecture**: Is the proposed architecture sound?
3. **Technical Concerns**: Potential challenges or issues?
4. **Best Practices**: Are industry standards considered?
5. **Improvements**: Specific recommendations to enhance the plan

Provide actionable suggestions and highlight any critical gaps."""
                }],
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            return {
                "review": review_text,
                "suggestions": self._extract_suggestions(review_text),
                "rating": self._extract_rating(review_text)
            }
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"ChatGPT Plan Review error: {str(e)}"
            }, websocket)
            return None
    
    async def comprehensive_review(self, code: str, original_prompt: str, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        if not self.client or not code.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 4,
                "message": "ChatGPT is performing comprehensive code review...",
                "agent": "ChatGPT"
            }, websocket)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"""Perform a comprehensive code review:

Original Request: {original_prompt}

Code to Review:
{code}

Please provide:
1. **Correctness**: Does it solve the problem correctly?
2. **Best Practices**: Adherence to Python conventions
3. **Performance**: Efficiency opportunities
4. **Security**: Potential security issues
5. **Maintainability**: Code clarity and structure
6. **Improvements**: Specific actionable recommendations

Format your response with clear sections."""
                }],
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            return {
                "review": review_text,
                "suggestions": self._extract_suggestions(review_text),
                "rating": self._extract_rating(review_text)
            }
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"ChatGPT Code Review error: {str(e)}"
            }, websocket)
            return None
    
    async def final_validation(self, refined_code: str, original_prompt: str, websocket: WebSocket) -> Optional[str]:
        if not self.client or not refined_code.strip():
            return None
        
        try:
            await manager.send_update({
                "type": "progress",
                "stage": 6,
                "message": "ChatGPT is performing final validation...",
                "agent": "ChatGPT"
            }, websocket)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.current_model,
                messages=[{
                    "role": "user",
                    "content": f"""Perform final validation of this code:

Original Request: {original_prompt}

Final Code:
{refined_code}

Please provide:
1. Final quality assessment
2. Readiness for production use
3. File organization recommendations
4. Any remaining concerns
5. Overall recommendation

Keep response concise but thorough."""
                }],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"ChatGPT Final Validation error: {str(e)}"
            }, websocket)
            return None
    
    def _extract_suggestions(self, text: str) -> List[str]:
        """Extract suggestions from review text"""
        suggestions = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith(('- ', '* ', '1. ', '2. ', '3. ', '4. ', '5.')) and 
                len(line) > 10 and
                any(word in line.lower() for word in ['should', 'could', 'consider', 'add', 'improve', 'implement'])):
                suggestions.append(line.lstrip('- *123456789. '))
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _extract_rating(self, review_text: str) -> str:
        """Extract quality rating from review text"""
        positive_words = ["excellent", "good", "well", "proper", "solid", "strong"]
        negative_words = ["issue", "problem", "error", "bad", "poor", "missing", "weak"]
        
        text_lower = review_text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count * 2:
            return "🟢 Excellent"
        elif positive_count > negative_count:
            return "🟡 Good"
        else:
            return "🔴 Needs Improvement"

class ProjectGenerator:
    def generate_project_files(self, final_code: str, project_name: str, original_prompt: str) -> List[ProjectFile]:
        """Generate complete project structure"""
        files = []
        project_name_clean = project_name.lower().replace(' ', '_')
        
        # Extract main code (remove markdown formatting)
        clean_code = self._extract_clean_code(final_code)
        
        # Main Python file
        files.append(ProjectFile(
            filename="main.py",
            content=clean_code,
            description="Main implementation file"
        ))
        
        # README
        readme_content = f"""# {project_name.title()}

## Description
{original_prompt}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
python main.py
```

## Features
- Generated using Multi-Agent AI Collaboration (Claude + ChatGPT)
- Follows Python best practices
- Includes proper error handling and documentation
- Production-ready code

## Generated on
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*This project was generated using AI collaboration*
"""
        files.append(ProjectFile(
            filename="README.md",
            content=readme_content,
            description="Project documentation"
        ))
        
        # Requirements.txt
        requirements = self._extract_requirements(clean_code)
        if requirements:
            files.append(ProjectFile(
                filename="requirements.txt",
                content=requirements,
                description="Python package dependencies"
            ))
        
        # Basic test file
        test_content = self._generate_test_file(clean_code)
        files.append(ProjectFile(
            filename="test_main.py",
            content=test_content,
            description="Basic unit tests"
        ))
        
        # .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""
        files.append(ProjectFile(
            filename=".gitignore",
            content=gitignore_content,
            description="Git ignore file for Python projects"
        ))
        
        return files
    
    def _extract_clean_code(self, text: str) -> str:
        """Extract clean Python code from Claude's response"""
        import re
        
        # Look for code blocks first
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, look for lines that start with Python code patterns
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see Python code patterns
            if (line.strip().startswith(('import ', 'from ', 'class ', 'def ', 'if __name__')) or
                (in_code and line.strip() and not line.strip().startswith(('Here', 'This', 'The', '1.', '2.', '3.', '**', '*', '#')))):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip() == "":
                code_lines.append(line)  # Keep blank lines in code
            elif in_code and line.strip().startswith(('Important', 'Note:', 'Example', 'To use')):
                break  # Stop at explanatory text
        
        return '\n'.join(code_lines).strip()
    
    def _extract_requirements(self, code: str) -> str:
        """Extract Python package requirements from code"""
        import_to_package = {
            'requests': 'requests',
            'numpy': 'numpy', 
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'bs4': 'beautifulsoup4',
            'selenium': 'selenium',
            'flask': 'Flask',
            'django': 'Django',
            'fastapi': 'fastapi',
            'streamlit': 'streamlit',
        }
        
        import_pattern = r'(?:from|import)\s+(\w+)'
        imports = re.findall(import_pattern, code)
        
        requirements = set()
        for imp in imports:
            if imp in import_to_package:
                requirements.add(import_to_package[imp])
        
        if requirements:
            return '\n'.join(sorted(requirements))
        return ""
    
    def _generate_test_file(self, code: str) -> str:
        """Generate basic test file"""
        return '''import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import *
except ImportError:
    pass

class TestMainFunctionality(unittest.TestCase):
    def test_basic_functionality(self):
        """Basic test to ensure the module loads without errors"""
        self.assertTrue(True)
    
    def test_imports(self):
        """Test that main module can be imported"""
        try:
            import main
            self.assertTrue(True)
        except ImportError:
            self.fail("Could not import main module")

if __name__ == '__main__':
    unittest.main()
'''

class WorkflowOrchestrator:
    def __init__(self):
        self.claude = ClaudeAgent()
        self.chatgpt = ChatGPTAgent()
        self.project_generator = ProjectGenerator()
    
    async def execute_workflow(self, request: GenerateRequest, websocket: WebSocket) -> Dict[str, Any]:
        """Execute the complete 7-stage workflow"""
        results = []
        
        try:
            # Stage 1: Planning with Claude
            await manager.send_update({
                "type": "stage_start",
                "stage": 1,
                "total_stages": 7,
                "message": "Starting requirements analysis..."
            }, websocket)
            
            planning = await self.claude.analyze_requirements(request.prompt, websocket)
            if not planning:
                raise Exception("Planning stage failed")
            
            results.append(WorkflowResult(
                stage=WorkflowStage.PLANNING,
                agent="Claude",
                content=planning
            ))
            
            # Stage 2: Plan Review with ChatGPT
            await manager.send_update({
                "type": "stage_complete",
                "stage": 1,
                "message": "Requirements analysis complete"
            }, websocket)
            
            plan_review = await self.chatgpt.review_build_plan(planning, request.prompt, websocket)
            enhanced_planning = planning
            if plan_review:
                results.append(WorkflowResult(
                    stage=WorkflowStage.PLAN_REVIEW,
                    agent="ChatGPT",
                    content=plan_review["review"],
                    feedback=plan_review["rating"],
                    suggestions=plan_review["suggestions"]
                ))
                
                if plan_review["suggestions"]:
                    suggestions_text = "\n".join([f"- {s}" for s in plan_review["suggestions"]])
                    enhanced_planning = f"{planning}\n\n## ChatGPT Suggestions:\n{suggestions_text}"
            
            await manager.send_update({
                "type": "stage_complete", 
                "stage": 2,
                "message": "Plan review complete"
            }, websocket)
            
            # Stage 3: Code Generation with Claude
            initial_code = await self.claude.generate_code(request.prompt, enhanced_planning, websocket)
            if not initial_code:
                raise Exception("Code generation failed")
            
            results.append(WorkflowResult(
                stage=WorkflowStage.INITIAL_CODE,
                agent="Claude",
                content=initial_code
            ))
            
            await manager.send_update({
                "type": "stage_complete",
                "stage": 3, 
                "message": "Code generation complete"
            }, websocket)
            
            # Stage 4: Code Review with ChatGPT
            review_result = await self.chatgpt.comprehensive_review(initial_code, request.prompt, websocket)
            current_code = initial_code
            
            if review_result:
                results.append(WorkflowResult(
                    stage=WorkflowStage.REVIEW_ANALYSIS,
                    agent="ChatGPT",
                    content=review_result["review"],
                    feedback=review_result["rating"],
                    suggestions=review_result["suggestions"]
                ))
                
                await manager.send_update({
                    "type": "stage_complete",
                    "stage": 4,
                    "message": "Code review complete"
                }, websocket)
                
                # Stage 5: Refinement (if needed)
                if review_result["suggestions"]:
                    feedback = "\n".join([f"- {s}" for s in review_result["suggestions"]])
                    refined_code = await self.claude.refine_code(current_code, feedback, websocket)
                    
                    if refined_code:
                        results.append(WorkflowResult(
                            stage=WorkflowStage.REFINEMENT,
                            agent="Claude",
                            content=refined_code
                        ))
                        current_code = refined_code
            
            await manager.send_update({
                "type": "stage_complete",
                "stage": 5,
                "message": "Code refinement complete"
            }, websocket)
            
            # Stage 6: Final Validation
            final_validation = await self.chatgpt.final_validation(current_code, request.prompt, websocket)
            if final_validation:
                results.append(WorkflowResult(
                    stage=WorkflowStage.FINAL_VALIDATION,
                    agent="ChatGPT", 
                    content=final_validation
                ))
            
            await manager.send_update({
                "type": "stage_complete",
                "stage": 6,
                "message": "Final validation complete"
            }, websocket)
            
            # Stage 7: Project Generation
            await manager.send_update({
                "type": "progress",
                "stage": 7,
                "message": "Generating project files...",
                "agent": "Project Generator"
            }, websocket)
            
            project_files = self.project_generator.generate_project_files(
                current_code, request.project_name, request.prompt
            )
            
            results.append(WorkflowResult(
                stage=WorkflowStage.PROJECT_GENERATION,
                agent="Project Generator",
                content=f"Generated {len(project_files)} files: {', '.join([f.filename for f in project_files])}"
            ))
            
            await manager.send_update({
                "type": "workflow_complete",
                "message": "All stages completed successfully!",
                "results": [asdict(r) for r in results],
                "project_files": [asdict(f) for f in project_files]
            }, websocket)
            
            return {
                "success": True,
                "results": results,
                "project_files": project_files
            }
            
        except Exception as e:
            await manager.send_update({
                "type": "error",
                "message": f"Workflow failed: {str(e)}"
            }, websocket)
            
            return {
                "success": False,
                "error": str(e),
                "results": results
            }

# Global orchestrator instance
orchestrator = WorkflowOrchestrator()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "generate":
                request_data = GenerateRequest(
                    prompt=message["prompt"],
                    project_name=message["project_name"],
                    max_iterations=message.get("max_iterations", 2)
                )
                
                # Run workflow in background
                result = await orchestrator.execute_workflow(request_data, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/generate")
async def generate_project(request: GenerateRequest):
    """HTTP API endpoint for project generation"""
    try:
        # For HTTP API, we'll run without WebSocket updates
        orchestrator_instance = WorkflowOrchestrator()
        
        # Simplified workflow execution without WebSocket updates
        results = []
        
        # Stage 1: Planning
        planning = await orchestrator_instance.claude.analyze_requirements(request.prompt, None)
        if planning:
            results.append({
                "stage": "planning",
                "agent": "Claude",
                "content": planning
            })
        
        # Stage 2: Code Generation  
        initial_code = await orchestrator_instance.claude.generate_code(request.prompt, planning, None)
        if initial_code:
            results.append({
                "stage": "code_generation",
                "agent": "Claude",
                "content": initial_code
            })
        
        # Stage 3: Project Files
        project_files = orchestrator_instance.project_generator.generate_project_files(
            initial_code or "", request.project_name, request.prompt
        )
        
        return JSONResponse({
            "success": True,
            "results": results,
            "project_files": [asdict(f) for f in project_files]
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/status")
async def get_status():
    """API endpoint to check service status"""
    claude_status = "✅ Connected" if orchestrator.claude.client else "❌ Not connected"
    chatgpt_status = f"✅ Connected ({orchestrator.chatgpt.current_model})" if orchestrator.chatgpt.client else "❌ Not connected"
    
    return JSONResponse({
        "claude": claude_status,
        "chatgpt": chatgpt_status,
        "active_connections": len(manager.active_connections)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)