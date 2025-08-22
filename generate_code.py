#!/usr/bin/env python3
"""
Simple AI Code Generator - Pre-configured example
Edit the PROMPT variable below to generate code for your project
"""

import os
from dotenv import load_dotenv
import anthropic
import openai
import datetime

load_dotenv()

# 🎯 EDIT THIS TO GENERATE YOUR PROJECT:
PROMPT = "Create a simple web scraper that extracts product titles and prices from an e-commerce website, with error handling and CSV export functionality"
PROJECT_NAME = "web_scraper"

class SimpleAI:
    def __init__(self):
        print("🚀 Initializing AI Code Generator...")
        
        # Claude
        self.claude = None
        try:
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.claude = anthropic.Anthropic(api_key=claude_key)
                print("✅ Claude connected")
        except Exception as e:
            print(f"❌ Claude error: {e}")
        
        # OpenAI
        self.openai = None
        self.model = "gpt-3.5-turbo"
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai = openai.OpenAI(api_key=openai_key)
                # Try GPT-4
                try:
                    test = self.openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5
                    )
                    self.model = "gpt-4"
                    print("✅ OpenAI GPT-4 connected")
                except:
                    print("✅ OpenAI GPT-3.5 connected")
        except Exception as e:
            print(f"❌ OpenAI error: {e}")
    
    def generate(self, prompt: str):
        print("="*60)
        print(f"🎯 GENERATING: {prompt}")
        print("="*60)
        
        # Step 1: Generate code
        print("\n⚡ Step 1: Claude generating code...")
        code = ""
        if self.claude:
            try:
                response = self.claude.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": f"Generate complete Python code for: {prompt}\n\nInclude proper imports, error handling, docstrings, and example usage."
                    }]
                )
                code = response.content[0].text
                print("✅ Code generated!")
            except Exception as e:
                print(f"❌ Code generation failed: {e}")
                return
        
        # Step 2: Review code
        print("\n🔍 Step 2: ChatGPT reviewing code...")
        review = ""
        if self.openai and code:
            try:
                response = self.openai.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": f"Review this Python code for correctness, best practices, and improvements:\n\n{code}"
                    }],
                    max_tokens=1000
                )
                review = response.choices[0].message.content
                print("✅ Code reviewed!")
            except Exception as e:
                print(f"❌ Code review failed: {e}")
        
        # Step 3: Create files
        print("\n📦 Step 3: Creating project files...")
        self.save_project(code, review, prompt)
    
    def extract_python_code(self, text: str) -> str:
        """Extract Python code from Claude's response, removing markdown and explanatory text"""
        import re
        
        # Look for code blocks first
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, look for lines that start with import, class, def, etc.
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

    def save_project(self, code: str, review: str, prompt: str):
        project_dir = f"{PROJECT_NAME}_generated"
        os.makedirs(project_dir, exist_ok=True)
        
        # Extract clean Python code
        clean_code = self.extract_python_code(code)
        
        # Save main.py
        with open(f"{project_dir}/main.py", 'w') as f:
            f.write(clean_code)
        print(f"   ✅ main.py saved")
        
        # Save README.md
        readme = f"""# {PROJECT_NAME.replace('_', ' ').title()}

## Description
{prompt}

## Generated Code
The main implementation is in `main.py`.

## AI Review
{review}

## Usage
```bash
pip install -r requirements.txt
python main.py
```

## Generated on
{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*Generated using Claude + ChatGPT collaboration*
"""
        with open(f"{project_dir}/README.md", 'w') as f:
            f.write(readme)
        print(f"   ✅ README.md saved")
        
        # Save requirements.txt
        requirements = """# Add your dependencies here
# Common ones you might need:
requests
beautifulsoup4
pandas
csv
"""
        with open(f"{project_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
        print(f"   ✅ requirements.txt saved")
        
        print(f"\n🎉 Project saved to: {project_dir}/")
        print(f"   cd {project_dir}")
        print(f"   python main.py")
        
        # Also display the results
        print("\n" + "="*60)
        print("📋 GENERATED CODE:")
        print("="*60)
        print(code)
        
        print("\n" + "="*60)
        print("🔍 AI REVIEW:")
        print("="*60)
        print(review)

def main():
    print("🤖 Simple AI Code Generator")
    print("="*40)
    print(f"📝 Project: {PROMPT}")
    print(f"📁 Name: {PROJECT_NAME}")
    print("="*40)
    
    ai = SimpleAI()
    
    if not ai.claude and not ai.openai:
        print("❌ No AI APIs available. Check your .env file.")
        return
    
    print("\n🚀 Starting generation...")
    ai.generate(PROMPT)

if __name__ == "__main__":
    main()