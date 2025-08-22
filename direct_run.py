#!/usr/bin/env python3
"""
Direct AI Code Generator - No Web Server Required
Run this script directly to generate code using Claude + ChatGPT
"""

import os
from dotenv import load_dotenv
import anthropic
import openai
import datetime

load_dotenv()

class AICodeGenerator:
    def __init__(self):
        print("🚀 Initializing AI Code Generator...")
        print("="*50)
        
        # Initialize Claude
        self.claude = None
        try:
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.claude = anthropic.Anthropic(api_key=claude_key)
                print("✅ Claude API connected")
            else:
                print("❌ ANTHROPIC_API_KEY not found in .env")
        except Exception as e:
            print(f"❌ Claude initialization error: {e}")
        
        # Initialize OpenAI
        self.openai = None
        self.openai_model = "gpt-3.5-turbo"
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai = openai.OpenAI(api_key=openai_key)
                
                # Try different models
                for model in ["gpt-4", "gpt-3.5-turbo"]:
                    try:
                        test_response = self.openai.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=5
                        )
                        self.openai_model = model
                        print(f"✅ OpenAI API connected using {model}")
                        break
                    except Exception as e:
                        if "does not exist" in str(e):
                            continue
                        else:
                            print(f"⚠️ {model} unavailable: {str(e)[:50]}...")
                            continue
            else:
                print("❌ OPENAI_API_KEY not found in .env")
        except Exception as e:
            print(f"❌ OpenAI initialization error: {e}")
        
        print("="*50)
    
    def generate_code(self, prompt: str) -> dict:
        """Generate code using Claude and get review from ChatGPT"""
        result = {
            "prompt": prompt,
            "planning": "",
            "code": "",
            "review": "",
            "final_code": "",
            "files": [],
            "success": False,
            "error": ""
        }
        
        try:
            # Stage 1: Planning with Claude
            print("🎯 Stage 1: Analyzing requirements...")
            if self.claude:
                planning_response = self.claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": f"""Analyze this coding request and create an implementation plan:

Request: {prompt}

Provide:
1. Requirements breakdown
2. Architecture approach
3. Key components needed
4. Implementation strategy

Be concise but thorough."""
                    }]
                )
                result["planning"] = planning_response.content[0].text
                print("✅ Planning completed")
            
            # Stage 2: Code Generation with Claude
            print("⚡ Stage 2: Generating code...")
            if self.claude:
                code_prompt = f"""Generate high-quality Python code for: {prompt}

{f"Implementation Plan: {result['planning']}" if result['planning'] else ""}

Requirements:
- Include proper error handling
- Add clear docstrings and comments
- Follow Python best practices
- Make it production-ready
- Include example usage if appropriate

Provide complete, working code."""

                code_response = self.claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2500,
                    messages=[{
                        "role": "user",
                        "content": code_prompt
                    }]
                )
                result["code"] = code_response.content[0].text
                result["final_code"] = result["code"]  # Start with initial code
                print("✅ Code generated")
            
            # Stage 3: Code Review with ChatGPT
            print("🔍 Stage 3: Reviewing code...")
            if self.openai and result["code"]:
                review_response = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[{
                        "role": "user",
                        "content": f"""Review this Python code and provide detailed feedback:

Original Request: {prompt}

Code to Review:
{result['code']}

Please provide:
1. Code correctness assessment
2. Best practices evaluation
3. Potential improvements
4. Security considerations
5. Performance notes

Be specific and actionable."""
                    }],
                    max_tokens=1500
                )
                result["review"] = review_response.choices[0].message.content
                print("✅ Code reviewed")
            
            # Stage 4: Generate Project Files
            print("📦 Stage 4: Creating project files...")
            result["files"] = self._create_project_files(result["final_code"], prompt)
            print("✅ Project files created")
            
            result["success"] = True
            print("🎉 Code generation completed successfully!")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Error during generation: {e}")
        
        return result
    
    def _create_project_files(self, code: str, prompt: str) -> list:
        """Create project files including main code, README, etc."""
        files = []
        
        # Main Python file
        files.append({
            "filename": "main.py",
            "content": code,
            "description": "Main implementation file"
        })
        
        # README file
        readme_content = f"""# AI Generated Project

## Description
{prompt}

## Usage
```python
python main.py
```

## Features
- Generated using AI collaboration (Claude + ChatGPT)
- Follows Python best practices
- Includes proper error handling
- Production-ready code

## Generated On
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*This project was generated using Multi-Agent AI Collaboration*
"""
        
        files.append({
            "filename": "README.md",
            "content": readme_content,
            "description": "Project documentation"
        })
        
        # Requirements file (basic template)
        requirements_content = """# Add your required packages here
# Common packages you might need:
# requests
# numpy
# pandas
# flask
# fastapi
"""
        
        files.append({
            "filename": "requirements.txt",
            "content": requirements_content,
            "description": "Python package dependencies"
        })
        
        # Basic test file
        test_content = """import unittest
from main import *

class TestMainFunctionality(unittest.TestCase):
    def test_basic_functionality(self):
        # Add your tests here
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""
        
        files.append({
            "filename": "test_main.py",
            "content": test_content,
            "description": "Basic unit tests"
        })
        
        return files
    
    def save_project(self, result: dict, project_name: str):
        """Save all project files to disk"""
        if not result["success"] or not result["files"]:
            print("❌ No files to save")
            return
        
        # Create project directory
        project_dir = f"{project_name}_project"
        os.makedirs(project_dir, exist_ok=True)
        
        print(f"💾 Saving project to: {project_dir}/")
        
        # Save each file
        for file in result["files"]:
            file_path = os.path.join(project_dir, file["filename"])
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file["content"])
            print(f"   ✅ {file['filename']} saved")
        
        print(f"🎉 Project saved successfully in '{project_dir}' directory!")
        return project_dir
    
    def display_result(self, result: dict):
        """Display the generation results"""
        print("\n" + "="*60)
        print("📋 GENERATION RESULTS")
        print("="*60)
        
        if not result["success"]:
            print(f"❌ Generation failed: {result['error']}")
            return
        
        if result["planning"]:
            print("\n🎯 PLANNING:")
            print("-" * 30)
            print(result["planning"])
        
        if result["code"]:
            print("\n⚡ GENERATED CODE:")
            print("-" * 30)
            print(result["code"])
        
        if result["review"]:
            print("\n🔍 AI REVIEW:")
            print("-" * 30)
            print(result["review"])
        
        if result["files"]:
            print(f"\n📦 PROJECT FILES ({len(result['files'])} files):")
            print("-" * 30)
            for file in result["files"]:
                print(f"   📄 {file['filename']} - {file['description']}")

def main():
    print("🤖 AI Code Generator - Direct Run")
    print("No web server required - runs directly in terminal!")
    print("="*60)
    
    generator = AICodeGenerator()
    
    # Check if APIs are available
    if not generator.claude and not generator.openai:
        print("\n❌ No AI APIs available. Please check your .env file:")
        print("   ANTHROPIC_API_KEY=your_anthropic_api_key")
        print("   OPENAI_API_KEY=your_openai_api_key")
        return
    
    print("\n💭 What do you want to build?")
    print("Examples:")
    print("  - Create a simple calculator with basic operations")
    print("  - Build a web scraper for product prices")
    print("  - Make a todo list manager with file storage")
    print("  - Create a password generator with customization")
    
    # Get user input
    while True:
        prompt = input("\n🎯 Enter your project description: ").strip()
        if prompt:
            break
        print("Please enter a description of what you want to build.")
    
    project_name = input("📁 Enter project name (or press Enter for 'my_project'): ").strip()
    if not project_name:
        project_name = "my_project"
    
    print(f"\n🚀 Starting AI collaboration to build: {prompt}")
    print("This may take 1-2 minutes...")
    
    # Generate the code
    result = generator.generate_code(prompt)
    
    # Display results
    generator.display_result(result)
    
    # Ask if user wants to save
    if result["success"]:
        save_choice = input(f"\n💾 Save project files to disk? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            project_dir = generator.save_project(result, project_name)
            print(f"\n🎉 All done! Your project is ready in: {project_dir}")
            print(f"   cd {project_dir}")
            print("   python main.py")

if __name__ == "__main__":
    main()