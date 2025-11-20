import streamlit as st
import asyncio
import os
import zipfile
import io
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai
import anthropic
import re
import datetime
from slide_generator_pkg.document_parser import DocumentParser

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
            st.error(f"Claude API error: {str(e)}")
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
- Add inline comments explaining complex logic

Please provide complete, working code that can be run immediately."""
            
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
            st.error(f"Claude Code Generation API error: {str(e)}")
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
                    "content": f"""Improve this code based on the feedback provided:

Original Code:
{code}

Feedback to Address:
{feedback}

Please provide the refined code with improvements addressing all feedback points."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"Claude API error: {str(e)}")
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
                st.error("OPENAI_API_KEY environment variable not set")
                return
            
            self.client = openai.OpenAI(api_key=api_key)
            
            # Try to use GPT-4 or fall back to GPT-3.5
            models_to_try = ["gpt-4", "gpt-3.5-turbo"]
            
            for model in models_to_try:
                try:
                    test_response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    self.current_model = model
                    st.success(f"✅ Using OpenAI model: {model}")
                    break
                except:
                    continue
            
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def review_build_plan(self, plan: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not plan.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
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
3. **Technical Concerns**: Potential challenges?
4. **Improvements**: Specific recommendations

Keep response concise but thorough."""
                }],
                max_tokens=1500
            )
            
            review_text = response.choices[0].message.content
            
            # Simple suggestion extraction
            suggestions = []
            lines = review_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    suggestions.append(line[1:].strip())
                    if len(suggestions) >= 5:
                        break
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": "🟡 Reviewed"
            }
        except Exception as e:
            st.error(f"ChatGPT Plan Review API error: {str(e)}")
            return None
    
    def comprehensive_review(self, code: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{
                    "role": "user", 
                    "content": f"""Perform a code review for this Python code:

Original Request: {original_prompt}

Code to Review:
{code}

Please provide:
1. **Correctness**: Does it solve the problem correctly?
2. **Best Practices**: Adherence to Python conventions
3. **Improvements**: Specific actionable recommendations

Format your response clearly and be specific."""
                }],
                max_tokens=1500
            )
            
            review_text = response.choices[0].message.content
            
            # Extract suggestions
            suggestions = []
            lines = review_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    suggestions.append(line[1:].strip())
                    if len(suggestions) >= 5:
                        break
            
            return {
                "review": review_text,
                "suggestions": suggestions,
                "rating": "🟡 Reviewed"
            }
        except Exception as e:
            st.error(f"ChatGPT Code Review API error: {str(e)}")
            return None
    
    def final_validation(self, refined_code: str, original_prompt: str) -> Optional[str]:
        if not self.client or not refined_code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
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
3. Overall recommendation

Keep response concise."""
                }],
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"ChatGPT Final Validation API error: {str(e)}")
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

## Generated on
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*Generated using Multi-Agent AI Collaboration*
"""
        files.append(ProjectFile(
            filename="README.md",
            content=readme_content,
            description="Project documentation"
        ))
        
        # Basic requirements.txt
        requirements = """# Add your required packages here
# Example:
# requests
# pandas
# numpy
"""
        files.append(ProjectFile(
            filename="requirements.txt",
            content=requirements,
            description="Python package dependencies"
        ))
        
        # Simple test file
        test_content = """import unittest
from main import *

class TestMain(unittest.TestCase):
    def test_basic_functionality(self):
        # Add your tests here
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

def create_project_zip(project_files: List[ProjectFile], project_name: str) -> bytes:
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in project_files:
            file_path = f"{project_name}/{file.filename}"
            zip_file.writestr(file_path, file.content)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def main():
    st.set_page_config(
        page_title="Simple Multi-Agent Code Collaboration",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Simple Multi-Agent Code Collaboration")
    st.subheader("Claude + ChatGPT → Reliable Project Generation")
    
    # Connection status
    connection_status = st.empty()
    connection_status.success("🟢 App Running")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_iterations = st.slider("Max Refinement Iterations", 1, 2, 1)
        
        st.header("📋 Environment Variables")
        st.code("""
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
        """)
        
        st.header("🔄 Simple 7-Stage Process")
        st.write("""
        1. Planning (Claude)
        2. Plan Review (ChatGPT)  
        3. Code Generation (Claude)
        4. Code Review (ChatGPT)
        5. Refinement (Claude)
        6. Final Validation (ChatGPT)
        7. Project Generation
        """)
    
    # Main interface
    st.header("💭 Describe Your Project")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area(
            "What do you want to build?",
            placeholder="Example: Create a simple web scraper for product prices",
            height=100
        )
    with col2:
        project_name = st.text_input("Project Name:", placeholder="my_project")
    
    if st.button("🚀 Generate Project", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a project description.")
            return
        if not project_name.strip():
            st.warning("Please enter a project name.")
            return
        
        # Initialize agents
        claude = ClaudeAgent()
        chatgpt = ChatGPTAgent()
        project_gen = ProjectGenerator()

        # Initialize DocumentParser with both API keys for intelligent routing
        claude_key = os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        doc_parser = DocumentParser(
            claude_api_key=claude_key,
            openai_api_key=openai_key,
            preferred_llm='auto'
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Stage 1: Planning
            status_text.info("🎯 Stage 1/7: Planning...")
            progress_bar.progress(1/7)
            
            planning = claude.analyze_requirements(prompt)
            if not planning:
                st.error("❌ Planning failed")
                return
            
            # Stage 2: Plan Review
            status_text.info("🔍 Stage 2/7: Plan Review...")
            progress_bar.progress(2/7)
            
            plan_review = chatgpt.review_build_plan(planning, prompt)
            enhanced_planning = planning
            if plan_review and plan_review["suggestions"]:
                suggestions_text = "\n".join([f"- {s}" for s in plan_review["suggestions"]])
                enhanced_planning = f"{planning}\n\nSuggestions:\n{suggestions_text}"
            
            # Stage 3: Code Generation
            status_text.info("⚡ Stage 3/7: Code Generation...")
            progress_bar.progress(3/7)
            
            initial_code = claude.generate_code(prompt, enhanced_planning)
            if not initial_code:
                st.error("❌ Code generation failed")
                return
            
            # Stage 4: Code Review
            status_text.info("🔍 Stage 4/7: Code Review...")
            progress_bar.progress(4/7)
            connection_status.info("🟡 Processing Stage 4...")
            
            review_result = chatgpt.comprehensive_review(initial_code, prompt)
            current_code = initial_code
            
            # Stage 5: Refinement (if needed)
            if review_result and review_result["suggestions"] and max_iterations > 0:
                status_text.info("🔄 Stage 5/7: Refinement...")
                progress_bar.progress(5/7)
                
                feedback = "\n".join([f"- {s}" for s in review_result["suggestions"][:3]])  # Limit feedback
                refined_code = claude.refine_code(current_code, feedback)
                if refined_code:
                    current_code = refined_code
            
            # Stage 6: Final Validation
            status_text.info("✅ Stage 6/7: Final Validation...")
            progress_bar.progress(6/7)
            
            final_validation = chatgpt.final_validation(current_code, prompt)
            
            # Stage 7: Project Generation
            status_text.info("📦 Stage 7/7: Project Generation...")
            progress_bar.progress(7/7)
            
            project_files = project_gen.generate_project_structure(current_code, project_name, prompt)
            
            # Complete
            progress_bar.progress(1.0)
            status_text.success("🎉 Project Generated Successfully!")
            connection_status.success("🟢 Workflow Completed")
            
            # Display results
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🎯 Planning", "⚡ Code", "📦 Files"])
            
            with tab1:
                st.subheader("Project Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stages Completed", "7/7")
                with col2:
                    st.metric("Project Files", len(project_files))
                with col3:
                    st.metric("AI Agents Used", "2")
            
            with tab2:
                if planning:
                    st.subheader("🎯 Requirements Analysis")
                    st.markdown(planning)
                if plan_review:
                    st.subheader("🔍 Plan Review")
                    st.markdown(plan_review["review"])
            
            with tab3:
                st.subheader("⚡ Generated Code")
                st.code(current_code, language="python")
                
                if review_result:
                    st.subheader("🔍 Code Review")
                    st.markdown(review_result["review"])
                
                if final_validation:
                    st.subheader("✅ Final Validation")
                    st.markdown(final_validation)
            
            with tab4:
                st.subheader("📁 Project Files")
                
                for i, file in enumerate(project_files):
                    with st.expander(f"📄 {file.filename}"):
                        st.write(f"**Description:** {file.description}")
                        if file.filename.endswith('.py'):
                            st.code(file.content, language='python')
                        elif file.filename.endswith('.md'):
                            st.markdown(file.content)
                        else:
                            st.text(file.content)
                        
                        st.download_button(
                            label=f"📥 Download {file.filename}",
                            data=file.content,
                            file_name=file.filename,
                            mime="text/plain",
                            key=f"download_{i}"
                        )
                
                # ZIP download
                st.subheader("📦 Download Complete Project")
                zip_data = create_project_zip(project_files, project_name)
                st.download_button(
                    label="📦 Download Project ZIP",
                    data=zip_data,
                    file_name=f"{project_name}.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            connection_status.error("🔴 Error occurred")

if __name__ == "__main__":
    main()