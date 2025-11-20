import streamlit as st
import asyncio
import os
import zipfile
import tempfile
import shutil
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai
import anthropic
import re
import datetime

load_dotenv()

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

@dataclass
class ProjectFile:
    filename: str
    content: str
    description: str

class ProjectGenerator:
    def __init__(self):
        self.files: List[ProjectFile] = []
        self.project_name = ""
    
    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Extract code blocks from markdown text"""
        # Pattern to match ```python or ``` code blocks
        pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for i, match in enumerate(matches):
            # Try to extract filename from comments or context
            lines = match.strip().split('\n')
            filename = f"script_{i+1}.py"
            
            # Look for filename hints in comments
            for line in lines[:5]:  # Check first 5 lines
                if '# File:' in line or '# Filename:' in line:
                    potential_name = line.split(':')[1].strip()
                    if potential_name.endswith('.py'):
                        filename = potential_name
                        break
                elif line.startswith('# ') and '.py' in line:
                    # Extract .py filename from comment
                    py_match = re.search(r'(\w+\.py)', line)
                    if py_match:
                        filename = py_match.group(1)
                        break
            
            code_blocks.append((filename, match.strip()))
        
        return code_blocks
    
    def generate_project_structure(self, final_code: str, project_name: str, original_prompt: str) -> List[ProjectFile]:
        """Generate a complete project structure"""
        self.project_name = project_name.lower().replace(' ', '_')
        self.files = []
        
        # Extract code blocks
        code_blocks = self.extract_code_blocks(final_code)
        
        if not code_blocks:
            # If no code blocks found, treat entire content as main script
            code_blocks = [("main.py", final_code)]
        
        # Add main code files
        for filename, code in code_blocks:
            self.files.append(ProjectFile(
                filename=filename,
                content=code,
                description=f"Main implementation: {filename}"
            ))
        
        # Generate README.md
        readme_content = self.generate_readme(original_prompt, code_blocks)
        self.files.append(ProjectFile(
            filename="README.md",
            content=readme_content,
            description="Project documentation and usage instructions"
        ))
        
        # Generate requirements.txt
        requirements = self.extract_requirements(final_code)
        if requirements:
            self.files.append(ProjectFile(
                filename="requirements.txt",
                content=requirements,
                description="Python package dependencies"
            ))
        
        # Generate setup.py (if it's a package)
        if len(code_blocks) > 1 or any('class' in code for _, code in code_blocks):
            setup_content = self.generate_setup_py()
            self.files.append(ProjectFile(
                filename="setup.py",
                content=setup_content,
                description="Package setup and installation script"
            ))
        
        # Generate test file
        test_content = self.generate_test_file(code_blocks[0][1] if code_blocks else "")
        self.files.append(ProjectFile(
            filename="test_main.py",
            content=test_content,
            description="Basic unit tests for the main functionality"
        ))
        
        # Generate .gitignore
        gitignore_content = self.generate_gitignore()
        self.files.append(ProjectFile(
            filename=".gitignore",
            content=gitignore_content,
            description="Git ignore file for Python projects"
        ))
        
        return self.files
    
    def generate_readme(self, original_prompt: str, code_blocks: List[Tuple[str, str]]) -> str:
        """Generate README.md content"""
        return f"""# {self.project_name.replace('_', ' ').title()}

## Description
{original_prompt}

## Generated Files
{chr(10).join([f"- **{filename}**: Main implementation" for filename, _ in code_blocks])}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
python {code_blocks[0][0] if code_blocks else 'main.py'}
```

## Features
- Generated using AI collaboration (Claude + ChatGPT)
- Follows Python best practices
- Includes error handling and documentation
- Ready for production use

## Testing
```bash
python test_main.py
```

## Generated on
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*This project was generated using Multi-Agent AI Collaboration*
"""
    
    def extract_requirements(self, code: str) -> str:
        """Extract Python package requirements from code"""
        # Common imports to requirements mapping
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
            'asyncio': '',  # Built-in
            'json': '',     # Built-in
            'os': '',       # Built-in
            'sys': '',      # Built-in
            'datetime': '', # Built-in
        }
        
        # Find import statements
        import_pattern = r'(?:from|import)\s+(\w+)'
        imports = re.findall(import_pattern, code)
        
        requirements = set()
        for imp in imports:
            if imp in import_to_package and import_to_package[imp]:
                requirements.add(import_to_package[imp])
        
        if requirements:
            return '\n'.join(sorted(requirements))
        return ""
    
    def generate_setup_py(self) -> str:
        """Generate setup.py for package installation"""
        return f'''from setuptools import setup, find_packages

setup(
    name="{self.project_name}",
    version="1.0.0",
    author="AI Generated",
    description="AI-generated Python project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().splitlines() if os.path.exists("requirements.txt") else [],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7+",
    ],
)
'''
    
    def generate_test_file(self, main_code: str) -> str:
        """Generate basic test file"""
        # Extract function/class names for testing
        functions = re.findall(r'def (\w+)\(', main_code)
        classes = re.findall(r'class (\w+)[\(:]', main_code)
        
        test_content = '''import unittest
import sys
import os

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

'''
        
        if functions or classes:
            # Import the main module
            test_content += "from main import *\n\n"
            
            test_content += "class TestMainFunctionality(unittest.TestCase):\n"
            
            # Generate test methods for functions
            for func in functions[:3]:  # Limit to first 3 functions
                test_content += f'''
    def test_{func}(self):
        """Test the {func} function"""
        # TODO: Add specific test cases for {func}
        # Example: result = {func}(test_input)
        # self.assertEqual(result, expected_output)
        pass
'''
            
            # Generate test methods for classes
            for cls in classes[:3]:  # Limit to first 3 classes
                test_content += f'''
    def test_{cls.lower()}_creation(self):
        """Test {cls} class instantiation"""
        # TODO: Add test for {cls} class
        # Example: instance = {cls}()
        # self.assertIsInstance(instance, {cls})
        pass
'''
        else:
            test_content += '''class TestMainFunctionality(unittest.TestCase):
    def test_basic_functionality(self):
        """Basic test to ensure the module loads without errors"""
        # TODO: Add specific test cases
        self.assertTrue(True)
'''
        
        test_content += '''

if __name__ == '__main__':
    unittest.main()
'''
        
        return test_content
    
    def generate_gitignore(self) -> str:
        """Generate .gitignore for Python projects"""
        return '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
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
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
'''

class ClaudeAgent:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("ANTHROPIC_API_KEY environment variable not set")
                return
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize Claude client: {str(e)}")
    
    async def analyze_requirements(self, prompt: str) -> Optional[str]:
        if not self.client or not prompt.strip():
            return None
        
        try:
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
            st.error(f"Claude API error: {str(e)}")
            return None
    
    async def generate_code(self, prompt: str, planning_context: str = None) -> Optional[str]:
        """Generate code using regular Claude API"""
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
            st.error(f"Claude Code Generation API error: {str(e)}")
            return None
    
    async def refine_code(self, code: str, feedback: str) -> Optional[str]:
        if not self.client or not code.strip():
            return None
        
        try:
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

Please provide the refined code with improvements addressing all feedback points. Include comments explaining the improvements made. If multiple files are suggested, separate them clearly."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"Claude API error: {str(e)}")
            return None

class ChatGPTAgent:
    def __init__(self):
        self.client = None
        self.available_models = ["gpt-5", "gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        self.current_model = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY environment variable not set")
                return
            self.client = openai.OpenAI(api_key=api_key)
            self._determine_best_model()
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _determine_best_model(self):
        """Test models in order of preference and select the first available one"""
        if not self.client:
            return
        
        for model in self.available_models:
            try:
                # Test with a simple query
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                self.current_model = model
                st.success(f"✅ Using OpenAI model: {model}")
                return
            except Exception as e:
                if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                    continue
                elif "insufficient_quota" in str(e).lower() or "rate_limit" in str(e).lower():
                    # Model exists but quota/rate limit issue
                    self.current_model = model
                    st.warning(f"⚠️ Using {model} (quota/rate limit detected)")
                    return
                else:
                    continue
        
        # If no model works, default to gpt-3.5-turbo
        self.current_model = "gpt-3.5-turbo"
        st.warning(f"⚠️ Defaulting to {self.current_model} - some models may not be available")
    
    def _get_optimal_max_tokens(self, model: str, requested_tokens: int) -> int:
        """Get optimal max tokens based on model capabilities"""
        model_limits = {
            "gpt-5": min(requested_tokens * 2, 4000),  # Enhanced capabilities for GPT-5
            "gpt-4o": min(requested_tokens * 1.5, 4000),  # Better efficiency
            "gpt-4": min(requested_tokens, 4000),
            "gpt-4-turbo": min(requested_tokens * 1.3, 4000),
            "gpt-3.5-turbo": min(requested_tokens, 4000),
            "gpt-3.5-turbo-16k": min(requested_tokens, 4000)
        }
        return model_limits.get(model, requested_tokens)
    
    def _make_api_call(self, messages: list, max_tokens: int):
        """Make API call with fallback to other models if current one fails"""
        if not self.client:
            return None
        
        # Try current model first
        models_to_try = [self.current_model] + [m for m in self.available_models if m != self.current_model]
        
        for model in models_to_try:
            try:
                optimal_tokens = self._get_optimal_max_tokens(model, max_tokens)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=optimal_tokens
                )
                # Update current model if different one worked
                if model != self.current_model:
                    self.current_model = model
                    st.info(f"🔄 Switched to model: {model}")
                return response
            except Exception as e:
                error_msg = str(e).lower()
                if "does not exist" in error_msg or "not found" in error_msg:
                    continue
                elif "insufficient_quota" in error_msg:
                    st.warning(f"⚠️ Quota exceeded for {model}, trying next model...")
                    continue
                elif "rate_limit" in error_msg:
                    st.warning(f"⚠️ Rate limit hit for {model}, trying next model...")
                    continue
                else:
                    # Other error, try next model
                    continue
        
        # If all models fail
        raise Exception("All OpenAI models failed or are unavailable")
    
    def comprehensive_review(self, code: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self._make_api_call(
                messages=[{
                    "role": "user", 
                    "content": f"""Perform a comprehensive code review for this Python code:

Original Request: {original_prompt}

Code to Review:
{code}

Please provide:
1. **Correctness**: Does it solve the problem correctly?
2. **Best Practices**: Adherence to Python conventions
3. **Performance**: Efficiency and optimization opportunities
4. **Security**: Potential security issues
5. **Maintainability**: Code clarity and structure
6. **Project Structure**: Should this be split into multiple files?
7. **Testing**: Suggestions for test cases
8. **Improvements**: Specific actionable recommendations

Format your response with clear sections and be specific about issues and solutions."""
                }],
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            
            # Extract improvement suggestions (simple parsing)
            suggestions = []
            if "improvements:" in review_text.lower():
                suggestions_section = review_text.lower().split("improvements:")[1]
                suggestions = [s.strip() for s in suggestions_section.split('\n') if s.strip() and not s.strip().startswith('#')][:5]
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": self._extract_rating(review_text)
            }
        except Exception as e:
            st.error(f"ChatGPT API error: {str(e)}")
            return None
    
    def _extract_rating(self, review_text: str) -> str:
        # Simple heuristic to determine code quality
        positive_indicators = ["good", "excellent", "well", "proper", "correct"]
        negative_indicators = ["issue", "problem", "error", "bad", "poor", "missing"]
        
        text_lower = review_text.lower()
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        
        if positive_count > negative_count * 2:
            return "🟢 Excellent"
        elif positive_count > negative_count:
            return "🟡 Good"
        else:
            return "🔴 Needs Improvement"
    
    def final_validation(self, refined_code: str, original_prompt: str) -> Optional[str]:
        if not self.client or not refined_code.strip():
            return None
        
        try:
            response = self._make_api_call(
                messages=[{
                    "role": "user",
                    "content": f"""Perform final validation of this refined code:

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
            st.error(f"ChatGPT API error: {str(e)}")
            return None
    
    def review_build_plan(self, plan: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        """Review the build plan and suggest improvements"""
        if not self.client or not plan.strip():
            return None
        
        try:
            response = self._make_api_call(
                messages=[{
                    "role": "user",
                    "content": f"""Review this implementation plan and provide detailed feedback:

Original Request: {original_prompt}

Build Plan to Review:
{plan}

Please analyze and provide:

1. **Completeness**: Are all requirements covered? What's missing?
2. **Architecture**: Is the proposed architecture sound? Any improvements?
3. **Technical Concerns**: Potential technical challenges or issues?
4. **Best Practices**: Are industry standards and best practices considered?
5. **Scalability**: Will this solution scale appropriately?
6. **Security**: Any security considerations missing?
7. **Dependencies**: Are all necessary dependencies identified?
8. **File Structure**: Is the proposed file organization optimal?
9. **Testing Strategy**: What testing approach should be included?
10. **Improvements**: Specific recommendations to enhance the plan

Provide actionable suggestions and highlight any critical gaps."""
                }],
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            
            # Extract key suggestions (simple parsing)
            suggestions = []
            if "improvements:" in review_text.lower():
                suggestions_section = review_text.lower().split("improvements:")[1]
                suggestions = [s.strip() for s in suggestions_section.split('\n') if s.strip() and not s.strip().startswith('#')][:7]
            elif "recommendations:" in review_text.lower():
                suggestions_section = review_text.lower().split("recommendations:")[1]
                suggestions = [s.strip() for s in suggestions_section.split('\n') if s.strip() and not s.strip().startswith('#')][:7]
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": self._extract_plan_rating(review_text)
            }
        except Exception as e:
            st.error(f"ChatGPT API error: {str(e)}")
            return None
    
    def _extract_plan_rating(self, review_text: str) -> str:
        """Extract plan quality rating from review text"""
        positive_indicators = ["comprehensive", "solid", "good", "well-planned", "thorough"]
        negative_indicators = ["missing", "incomplete", "lacks", "weak", "insufficient", "overlooked"]
        
        text_lower = review_text.lower()
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        
        if positive_count > negative_count * 1.5:
            return "🟢 Solid Plan"
        elif positive_count > negative_count:
            return "🟡 Good Plan"
        else:
            return "🔴 Needs Enhancement"

class WorkflowOrchestrator:
    def __init__(self):
        self.claude = ClaudeAgent()
        self.chatgpt = ChatGPTAgent()
        self.project_generator = ProjectGenerator()
        self.results: List[WorkflowResult] = []
    
    async def execute_collaborative_workflow(self, prompt: str, project_name: str, max_iterations: int = 2) -> Tuple[List[WorkflowResult], List[ProjectFile]]:
        self.results = []
        project_files = []
        
        try:
            # Stage 1: Planning with Claude
            st.info("🎯 Stage 1: Analyzing requirements and creating implementation plan...")
            planning = await self.claude.analyze_requirements(prompt)
            if planning:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.PLANNING,
                    agent="Claude",
                    content=planning
                ))
            else:
                st.error("❌ Planning stage failed")
                return self.results, project_files
            
            # Stage 2: Plan Review with ChatGPT
            st.info("🔍 Stage 2: Reviewing build plan and identifying improvements...")
            plan_review = self.chatgpt.review_build_plan(planning, prompt)
            if plan_review:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.PLAN_REVIEW,
                    agent="ChatGPT",
                    content=plan_review["review"],
                    feedback=plan_review["rating"],
                    suggestions=plan_review["suggestions"]
                ))
                
                # Update planning context with suggestions
                enhanced_planning = planning
                if plan_review["suggestions"]:
                    suggestions_text = "\n".join([f"- {s}" for s in plan_review["suggestions"]])
                    enhanced_planning = f"{planning}\n\n## ChatGPT Plan Review Suggestions:\n{suggestions_text}"
            else:
                st.warning("⚠️ Plan review failed, proceeding with original plan")
                enhanced_planning = planning
            
            # Stage 3: Code Generation with Claude
            st.info("⚡ Stage 3: Generating initial code implementation...")
            initial_code = await self.claude.generate_code(prompt, enhanced_planning)
            if initial_code:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.INITIAL_CODE,
                    agent="Claude (Code Generator)",
                    content=initial_code
                ))
            else:
                st.error("❌ Code generation stage failed")
                return self.results, project_files
            
            # Stage 4: Comprehensive Review with ChatGPT
            st.info("🔍 Stage 4: Performing comprehensive code review...")
            review_result = self.chatgpt.comprehensive_review(initial_code, prompt)
            if review_result:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.REVIEW_ANALYSIS,
                    agent="ChatGPT",
                    content=review_result["review"],
                    feedback=review_result["rating"],
                    suggestions=review_result["suggestions"]
                ))
                
                # Stage 5: Iterative Refinement
                current_code = initial_code
                for iteration in range(max_iterations):
                    st.info(f"🔄 Stage 5.{iteration + 1}: Refining code based on feedback...")
                    
                    if review_result["suggestions"]:
                        feedback = "\n".join([f"- {s}" for s in review_result["suggestions"]])
                        refined_code = await self.claude.refine_code(current_code, feedback)
                        
                        if refined_code:
                            self.results.append(WorkflowResult(
                                stage=WorkflowStage.REFINEMENT,
                                agent=f"Claude (Iteration {iteration + 1})",
                                content=refined_code
                            ))
                            current_code = refined_code
                            
                            # Re-review if this isn't the last iteration
                            if iteration < max_iterations - 1:
                                review_result = self.chatgpt.comprehensive_review(current_code, prompt)
                        else:
                            st.warning(f"⚠️ Refinement iteration {iteration + 1} failed")
                            break
                    else:
                        st.info("✅ No suggestions for improvement - code looks good!")
                        break
            else:
                st.error("❌ Code review stage failed")
                return self.results, project_files
            
            # Stage 6: Final Validation
            st.info("✅ Stage 6: Final validation and quality assessment...")
            final_validation = self.chatgpt.final_validation(current_code, prompt)
            if final_validation:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.FINAL_VALIDATION,
                    agent="ChatGPT",
                    content=final_validation
                ))
            
            # Stage 7: Project Generation
            st.info("📦 Stage 7: Generating project files and structure...")
            project_files = self.project_generator.generate_project_structure(
                current_code, project_name, prompt
            )
            
            self.results.append(WorkflowResult(
                stage=WorkflowStage.PROJECT_GENERATION,
                agent="Project Generator",
                content=f"Generated {len(project_files)} project files: {', '.join([f.filename for f in project_files])}"
            ))
            
            self.results.append(WorkflowResult(
                stage=WorkflowStage.COMPLETE,
                agent="System",
                content="Collaborative workflow completed successfully!"
            ))
            
        except Exception as e:
            st.error(f"❌ Workflow error: {str(e)}")
            self.results.append(WorkflowResult(
                stage=WorkflowStage.COMPLETE,
                agent="System",
                content=f"Workflow failed with error: {str(e)}"
            ))
        
        return self.results, project_files

def create_project_zip(project_files: List[ProjectFile], project_name: str) -> bytes:
    """Create a ZIP file containing all project files"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in project_files:
            # Create file path within the project directory
            file_path = f"{project_name}/{file.filename}"
            zip_file.writestr(file_path, file.content)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def display_project_files(project_files: List[ProjectFile]):
    """Display project files with preview and download options"""
    if not project_files:
        st.info("No project files generated")
        return
    
    st.subheader("📁 Generated Project Files")
    
    # Create tabs for each file
    file_tabs = st.tabs([f"📄 {file.filename}" for file in project_files])
    
    for i, (tab, file) in enumerate(zip(file_tabs, project_files)):
        with tab:
            st.write(f"**Description:** {file.description}")
            
            # Show file content
            if file.filename.endswith('.py'):
                st.code(file.content, language='python')
            elif file.filename.endswith('.md'):
                st.markdown(file.content)
            elif file.filename.endswith(('.txt', '.gitignore')):
                st.code(file.content, language='text')
            else:
                st.text(file.content)
            
            # Individual file download
            st.download_button(
                label=f"📥 Download {file.filename}",
                data=file.content,
                file_name=file.filename,
                mime="text/plain",
                key=f"download_{i}"
            )

def display_workflow_results(results: List[WorkflowResult], project_files: List[ProjectFile]):
    """Display workflow results in an organized manner"""
    
    # Create tabs for different stages
    tab_names = ["📋 Overview", "🎯 Planning", "🔍 Plan Review", "⚡ Code Gen", "🔍 Code Review", "🔄 Refinement", "✅ Final", "📦 Project"]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.subheader("Workflow Summary")
        
        # Progress indicator
        completed_stages = len([r for r in results if r.stage != WorkflowStage.COMPLETE])
        progress = min(completed_stages / 7.0, 1.0)
        st.progress(progress)
        st.write(f"Completed {completed_stages}/7 stages")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agents Used", len(set(r.agent for r in results if r.agent != "System")))
        with col2:
            refinement_count = len([r for r in results if r.stage == WorkflowStage.REFINEMENT])
            st.metric("Refinement Cycles", refinement_count)
        with col3:
            st.metric("Project Files", len(project_files))
    
    # Planning Tab
    with tabs[1]:
        planning_results = [r for r in results if r.stage == WorkflowStage.PLANNING]
        if planning_results:
            st.subheader("🎯 Requirements Analysis & Planning")
            st.markdown(planning_results[0].content)
        else:
            st.info("No planning results available")
    
    # Plan Review Tab
    with tabs[2]:
        plan_review_results = [r for r in results if r.stage == WorkflowStage.PLAN_REVIEW]
        if plan_review_results:
            st.subheader("🔍 Build Plan Review & Suggestions")
            result = plan_review_results[0]
            if result.feedback:
                st.write(f"**Plan Quality Rating:** {result.feedback}")
            st.markdown(result.content)
            if result.suggestions:
                st.subheader("Key Plan Improvements")
                for suggestion in result.suggestions:
                    st.write(f"• {suggestion}")
        else:
            st.info("No plan review results available")
    
    # Code Generation Tab
    with tabs[3]:
        code_results = [r for r in results if r.stage == WorkflowStage.INITIAL_CODE]
        if code_results:
            st.subheader("⚡ Initial Code Generation")
            st.code(code_results[0].content, language="python")
        else:
            st.info("No initial code available")
    
    # Code Review Tab
    with tabs[4]:
        review_results = [r for r in results if r.stage == WorkflowStage.REVIEW_ANALYSIS]
        if review_results:
            st.subheader("🔍 Comprehensive Code Review")
            result = review_results[0]
            if result.feedback:
                st.write(f"**Quality Rating:** {result.feedback}")
            st.markdown(result.content)
            if result.suggestions:
                st.subheader("Key Suggestions")
                for suggestion in result.suggestions:
                    st.write(f"• {suggestion}")
        else:
            st.info("No review results available")
    
    # Refinement Tab
    with tabs[5]:
        refinement_results = [r for r in results if r.stage == WorkflowStage.REFINEMENT]
        if refinement_results:
            st.subheader("🔄 Code Refinement")
            for i, result in enumerate(refinement_results):
                with st.expander(f"Refinement Iteration {i + 1} - {result.agent}"):
                    st.code(result.content, language="python")
        else:
            st.info("No refinement cycles performed")
    
    # Final Tab
    with tabs[6]:
        final_results = [r for r in results if r.stage == WorkflowStage.FINAL_VALIDATION]
        if final_results:
            st.subheader("✅ Final Validation")
            st.markdown(final_results[0].content)
        else:
            st.info("No final validation available")
    
    # Project Tab
    with tabs[7]:
        if project_files:
            display_project_files(project_files)
            
            # Download entire project as ZIP
            st.subheader("📦 Download Complete Project")
            project_name = project_files[0].filename.split('/')[0] if '/' in project_files[0].filename else "ai_generated_project"
            
            zip_data = create_project_zip(project_files, project_name)
            st.download_button(
                label="📦 Download Complete Project (ZIP)",
                data=zip_data,
                file_name=f"{project_name}.zip",
                mime="application/zip"
            )
        else:
            st.info("No project files generated")

def main():
    st.set_page_config(
        page_title="Multi-Agent Code Collaboration with Project Generation",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent Code Collaboration")
    st.subheader("Claude + GPT-5/ChatGPT → Complete Project Package")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Workflow Configuration")
        max_iterations = st.slider("Max Refinement Iterations", 1, 5, 2)
        
        # Model status
        st.header("🤖 AI Model Status")
        
        # Create a dummy ChatGPT agent to check model availability
        temp_agent = ChatGPTAgent()
        if temp_agent.current_model:
            st.success(f"OpenAI: {temp_agent.current_model}")
        else:
            st.error("OpenAI: Not available")
        
        # Show model preferences
        with st.expander("🔧 Model Preferences"):
            st.write("**OpenAI Model Priority:**")
            st.write("1. **GPT-5** 🚀 (latest & most advanced)")
            st.write("2. **GPT-4o** (multimodal capabilities)")
            st.write("3. GPT-4 (high quality)")
            st.write("4. GPT-4 Turbo (fast & efficient)")  
            st.write("5. GPT-3.5 Turbo (reliable)")
            st.write("6. GPT-3.5 Turbo 16K (extended context)")
            st.info("App automatically selects the best available model")
        
        st.header("📋 Required Environment Variables")
        st.code("""
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
        """)
        
        st.header("🔄 Workflow Stages")
        st.write("""
        1. **Planning** - Claude analyzes requirements
        2. **Plan Review** - ChatGPT reviews and suggests improvements
        3. **Code Generation** - Claude creates enhanced implementation
        4. **Code Review** - ChatGPT performs comprehensive analysis
        5. **Refinement** - Claude improves code based on feedback
        6. **Validation** - ChatGPT provides final assessment
        7. **Project Generation** - Creates complete project structure
        """)
        
        st.header("📦 Project Features")
        st.write("""
        - **Multiple Files** - Organized project structure
        - **README.md** - Documentation and usage
        - **requirements.txt** - Dependencies
        - **test_main.py** - Basic unit tests
        - **setup.py** - Package installation
        - **.gitignore** - Git ignore file
        - **ZIP Download** - Complete project package
        """)
    
    # Main interface
    with st.container():
        st.header("💭 Describe Your Coding Project")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            prompt = st.text_area(
                "Enter a detailed description of what you want to build:",
                placeholder="Example: Create a web scraper that extracts product information from e-commerce sites, handles rate limiting, and saves data to both CSV and JSON formats with error handling and logging.",
                height=120
            )
        with col2:
            project_name = st.text_input(
                "Project Name:",
                placeholder="my_awesome_project",
                help="Used for folder structure and package naming"
            )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            start_workflow = st.button("🚀 Generate Complete Project", type="primary")
        with col2:
            if start_workflow and not prompt.strip():
                st.warning("Please enter a project description first.")
            elif start_workflow and not project_name.strip():
                st.warning("Please enter a project name.")
    
    # Execute workflow
    if start_workflow and prompt.strip() and project_name.strip():
        orchestrator = WorkflowOrchestrator()
        
        with st.spinner("🔄 Running collaborative workflow and generating project..."):
            results, project_files = asyncio.run(
                orchestrator.execute_collaborative_workflow(prompt, project_name, max_iterations)
            )
        
        if results:
            st.success("🎉 Complete project generated successfully!")
            display_workflow_results(results, project_files)
        else:
            st.error("❌ Workflow failed to produce results.")

if __name__ == "__main__":
    main()