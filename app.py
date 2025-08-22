import streamlit as st
import asyncio
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import openai
from claude_code_sdk import query, ClaudeCodeOptions

load_dotenv()

class ClaudeCodeAgent:
    def __init__(self):
        self.options = ClaudeCodeOptions()
    
    async def generate_code(self, prompt: str) -> Optional[str]:
        if not prompt.strip():
            return None
        
        try:
            full_prompt = f"Generate Python code for: {prompt}"
            result_parts = []
            
            async for message in query(prompt=full_prompt, options=self.options):
                if hasattr(message, 'content') and message.content:
                    if hasattr(message.content, 'text'):
                        result_parts.append(message.content.text)
                    elif isinstance(message.content, str):
                        result_parts.append(message.content)
                    elif hasattr(message.content, '__iter__'):
                        for block in message.content:
                            if hasattr(block, 'text'):
                                result_parts.append(block.text)
            
            return '\n'.join(result_parts) if result_parts else None
        except Exception as e:
            st.error(f"Claude Code API error: {str(e)}")
            return None

class ChatGPTAgent:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY environment variable not set")
                return
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def review_code(self, code: str) -> Optional[str]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": f"Review this Python code for correctness, best practices, and suggest improvements:\n\n{code}"
                }],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"ChatGPT API error: {str(e)}")
            return None

def create_feedback_summary(claude_code: str, cal_review: str) -> str:
    if not claude_code and not cal_review:
        return "No results to summarize."
    
    if not claude_code:
        return "Claude Code failed to generate code."
    
    if not cal_review:
        return "Code generated successfully, but ChatGPT review failed."
    
    return f"✅ Code Generation: Completed\n✅ Code Review: Completed\n\nRecommendation: Review Cal's analysis and implement suggested improvements."

async def process_dual_agent_request(prompt: str, claude_agent: ClaudeCodeAgent, chatgpt_agent: ChatGPTAgent) -> Dict[str, Any]:
    claude_code = await claude_agent.generate_code(prompt)
    
    cal_review = None
    if claude_code:
        cal_review = chatgpt_agent.review_code(claude_code)
    
    feedback = create_feedback_summary(claude_code, cal_review)
    
    return {
        "claude_code": claude_code,
        "cal_review": cal_review,
        "feedback": feedback
    }

def main():
    st.set_page_config(
        page_title="Dual-Agent Code Interface",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Dual-Agent Code Interface")
    st.subheader("Claude Code + ChatGPT (Cal) Collaboration")
    
    claude_agent = ClaudeCodeAgent()
    chatgpt_agent = ChatGPTAgent()
    
    with st.container():
        st.header("📝 Prompt Input")
        prompt = st.text_area(
            "Enter your coding task description:",
            placeholder="Example: Create a function to calculate the Fibonacci sequence",
            height=100
        )
        
        submit_button = st.button("🚀 Generate & Review Code", type="primary")
    
    if submit_button and prompt.strip():
        with st.spinner("Processing your request..."):
            results = asyncio.run(process_dual_agent_request(prompt, claude_agent, chatgpt_agent))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔧 Claude Code Output")
            if results["claude_code"]:
                st.code(results["claude_code"], language="python")
            else:
                st.error("Failed to generate code")
        
        with col2:
            st.header("🔍 Cal's Analysis")
            if results["cal_review"]:
                st.markdown(results["cal_review"])
            else:
                st.error("Failed to review code")
        
        st.header("📊 Final Feedback")
        st.info(results["feedback"])
    
    elif submit_button:
        st.warning("Please enter a prompt before submitting.")
    
    with st.expander("ℹ️ Setup Instructions"):
        st.markdown("""
        ### Required Environment Variables
        Create a `.env` file with:
        ```
        CLAUDE_CODE_API_KEY=your_claude_code_api_key
        OPENAI_API_KEY=your_openai_api_key
        ```
        
        ### Installation
        ```bash
        pip install streamlit python-dotenv openai claude-code-sdk
        ```
        
        ### Run
        ```bash
        streamlit run app.py
        ```
        """)

if __name__ == "__main__":
    main()