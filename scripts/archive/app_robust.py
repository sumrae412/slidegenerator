import streamlit as st
import asyncio
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai
import anthropic

load_dotenv()

class WorkflowStage(Enum):
    PLANNING = "planning"
    INITIAL_CODE = "initial_code"
    REVIEW_ANALYSIS = "review_analysis"
    REFINEMENT = "refinement"
    FINAL_VALIDATION = "final_validation"
    COMPLETE = "complete"

@dataclass
class WorkflowResult:
    stage: WorkflowStage
    agent: str
    content: str
    feedback: Optional[str] = None
    suggestions: Optional[List[str]] = None

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
4. Potential challenges
5. Recommended implementation strategy

Be specific and technical."""
                }]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"Claude API error: {str(e)}")
            return None
    
    async def generate_code(self, prompt: str, planning_context: str = None) -> Optional[str]:
        """Generate code using regular Claude API as fallback"""
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
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-5-sonnet-20241022",
                max_tokens=2500,
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
                max_tokens=2000,
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
            st.error(f"Claude API error: {str(e)}")
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
    
    def comprehensive_review(self, code: str, original_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.client or not code.strip():
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
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
6. **Testing**: Suggestions for test cases
7. **Improvements**: Specific actionable recommendations

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
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"""Perform final validation of this refined code:

Original Request: {original_prompt}

Final Code:
{refined_code}

Please provide:
1. Final quality assessment
2. Readiness for production use
3. Any remaining concerns
4. Overall recommendation

Keep response concise but thorough."""
                }],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"ChatGPT API error: {str(e)}")
            return None

class WorkflowOrchestrator:
    def __init__(self):
        self.claude = ClaudeAgent()
        self.chatgpt = ChatGPTAgent()
        self.results: List[WorkflowResult] = []
    
    async def execute_collaborative_workflow(self, prompt: str, max_iterations: int = 2) -> List[WorkflowResult]:
        self.results = []
        
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
                return self.results
            
            # Stage 2: Code Generation with Claude (fallback from Claude Code)
            st.info("⚡ Stage 2: Generating initial code implementation...")
            initial_code = await self.claude.generate_code(prompt, planning)
            if initial_code:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.INITIAL_CODE,
                    agent="Claude (Code Generator)",
                    content=initial_code
                ))
            else:
                st.error("❌ Code generation stage failed")
                return self.results
            
            # Stage 3: Comprehensive Review with ChatGPT
            st.info("🔍 Stage 3: Performing comprehensive code review...")
            review_result = self.chatgpt.comprehensive_review(initial_code, prompt)
            if review_result:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.REVIEW_ANALYSIS,
                    agent="ChatGPT",
                    content=review_result["review"],
                    feedback=review_result["rating"],
                    suggestions=review_result["suggestions"]
                ))
                
                # Stage 4: Iterative Refinement
                current_code = initial_code
                for iteration in range(max_iterations):
                    st.info(f"🔄 Stage 4.{iteration + 1}: Refining code based on feedback...")
                    
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
                return self.results
            
            # Stage 5: Final Validation
            st.info("✅ Stage 5: Final validation and quality assessment...")
            final_validation = self.chatgpt.final_validation(current_code, prompt)
            if final_validation:
                self.results.append(WorkflowResult(
                    stage=WorkflowStage.FINAL_VALIDATION,
                    agent="ChatGPT",
                    content=final_validation
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
        
        return self.results

def display_workflow_results(results: List[WorkflowResult]):
    """Display workflow results in an organized manner"""
    
    # Create tabs for different stages
    tab_names = ["📋 Overview", "🎯 Planning", "⚡ Code Gen", "🔍 Review", "🔄 Refinement", "✅ Final"]
    tabs = st.tabs(tab_names)
    
    # Overview Tab
    with tabs[0]:
        st.subheader("Workflow Summary")
        
        # Progress indicator
        completed_stages = len([r for r in results if r.stage != WorkflowStage.COMPLETE])
        progress = min(completed_stages / 5.0, 1.0)
        st.progress(progress)
        st.write(f"Completed {completed_stages}/5 stages")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agents Used", len(set(r.agent for r in results if r.agent != "System")))
        with col2:
            refinement_count = len([r for r in results if r.stage == WorkflowStage.REFINEMENT])
            st.metric("Refinement Cycles", refinement_count)
        with col3:
            final_code = next((r for r in reversed(results) if r.stage in [WorkflowStage.REFINEMENT, WorkflowStage.INITIAL_CODE]), None)
            if final_code:
                code_lines = len(final_code.content.split('\n'))
                st.metric("Final Code Lines", code_lines)
    
    # Planning Tab
    with tabs[1]:
        planning_results = [r for r in results if r.stage == WorkflowStage.PLANNING]
        if planning_results:
            st.subheader("🎯 Requirements Analysis & Planning")
            st.markdown(planning_results[0].content)
        else:
            st.info("No planning results available")
    
    # Code Generation Tab
    with tabs[2]:
        code_results = [r for r in results if r.stage == WorkflowStage.INITIAL_CODE]
        if code_results:
            st.subheader("⚡ Initial Code Generation")
            st.code(code_results[0].content, language="python")
        else:
            st.info("No initial code available")
    
    # Review Tab
    with tabs[3]:
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
    with tabs[4]:
        refinement_results = [r for r in results if r.stage == WorkflowStage.REFINEMENT]
        if refinement_results:
            st.subheader("🔄 Code Refinement")
            for i, result in enumerate(refinement_results):
                with st.expander(f"Refinement Iteration {i + 1} - {result.agent}"):
                    st.code(result.content, language="python")
        else:
            st.info("No refinement cycles performed")
    
    # Final Tab
    with tabs[5]:
        final_results = [r for r in results if r.stage == WorkflowStage.FINAL_VALIDATION]
        if final_results:
            st.subheader("✅ Final Validation")
            st.markdown(final_results[0].content)
        
        # Show final code
        final_code = next((r for r in reversed(results) if r.stage in [WorkflowStage.REFINEMENT, WorkflowStage.INITIAL_CODE]), None)
        if final_code:
            st.subheader("🎉 Final Code")
            st.code(final_code.content, language="python")
            
            # Download button
            st.download_button(
                label="📥 Download Final Code",
                data=final_code.content,
                file_name="generated_code.py",
                mime="text/python"
            )

def main():
    st.set_page_config(
        page_title="Robust Multi-Agent Code Collaboration",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Robust Multi-Agent Code Collaboration")
    st.subheader("Claude + ChatGPT Working Together (Robust Version)")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Workflow Configuration")
        max_iterations = st.slider("Max Refinement Iterations", 1, 5, 2)
        
        st.header("📋 Required Environment Variables")
        st.code("""
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
        """)
        
        st.header("🔄 Workflow Stages")
        st.write("""
        1. **Planning** - Claude analyzes requirements
        2. **Code Generation** - Claude creates implementation
        3. **Review** - ChatGPT performs comprehensive analysis
        4. **Refinement** - Claude improves code based on feedback
        5. **Validation** - ChatGPT provides final assessment
        """)
        
        st.header("🛡️ Robust Features")
        st.write("""
        - Better error handling
        - Fallback mechanisms
        - Progress tracking
        - Graceful failure recovery
        """)
    
    # Main interface
    with st.container():
        st.header("💭 Describe Your Coding Task")
        prompt = st.text_area(
            "Enter a detailed description of what you want to build:",
            placeholder="Example: Create a web scraper that extracts product information from e-commerce sites, handles rate limiting, and saves data to both CSV and JSON formats with error handling and logging.",
            height=120
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            start_workflow = st.button("🚀 Start Collaborative Workflow", type="primary")
        with col2:
            if start_workflow and not prompt.strip():
                st.warning("Please enter a coding task description first.")
    
    # Execute workflow
    if start_workflow and prompt.strip():
        orchestrator = WorkflowOrchestrator()
        
        with st.spinner("🔄 Running collaborative workflow..."):
            results = asyncio.run(orchestrator.execute_collaborative_workflow(prompt, max_iterations))
        
        if results:
            st.success("🎉 Workflow completed!")
            display_workflow_results(results)
        else:
            st.error("❌ Workflow failed to produce results.")

if __name__ == "__main__":
    main()