"""
Presentation Intelligence Module

Handles AI-powered presentation enhancements including:
1. Smart slide title generation
2. Speaker notes generation
3. Presentation quality review and analysis

Uses Claude API to create engaging slide titles, comprehensive speaker notes,
and analyze overall presentation quality.

Features:
- Smart slide title generation (action-oriented, contextual)
- Speaker notes generation with talking points
- Transition suggestions between slides
- Presentation quality analysis (flow, coherence, redundancy, completeness)
- Actionable recommendations for improvement
- Natural, conversational tone
- Cost tracking and optimization
- Fallback to basic notes when API unavailable
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Anthropic Claude for intelligent speaker notes
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic library not available - speaker notes generation disabled")

# Import data models and utilities
from .data_models import SlideContent
from .utils import CostTracker

logger = logging.getLogger(__name__)


class PresentationIntelligence:
    """
    Generates AI-powered speaker notes and smart slide titles.

    Uses Claude API to:
    1. Create engaging, contextual slide titles (3-7 words)
    2. Generate comprehensive speaker notes with talking points
    3. Suggest smooth transitions between slides

    Cost optimization:
    - Smart titles use Haiku model (faster, cheaper)
    - Speaker notes use Sonnet model (better quality)
    """

    # Claude pricing (as of 2025)
    CLAUDE_PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},  # per 1M tokens
        'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
        'claude-3-5-haiku-20241022': {'input': 1.00, 'output': 5.00},  # per 1M tokens
    }

    # Default models
    DEFAULT_TITLE_MODEL = "claude-3-5-haiku-20241022"  # Fast, cheap for titles
    DEFAULT_NOTES_MODEL = "claude-3-5-sonnet-20241022"  # Better quality for notes

    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, claude_api_key: Optional[str] = None,
                 cost_tracker: Optional[CostTracker] = None):
        """
        Initialize PresentationIntelligence with Claude API integration.

        Args:
            claude_api_key: Anthropic Claude API key (falls back to ANTHROPIC_API_KEY env var)
            cost_tracker: Optional CostTracker instance for cost tracking
        """
        self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')

        # Initialize Claude client
        self.client = None
        if self.api_key and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("✅ Claude API client initialized for speaker notes")
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                self.client = None
        elif self.api_key and not ANTHROPIC_AVAILABLE:
            logger.warning("Claude API key provided but anthropic library not installed - run: pip install anthropic")
        else:
            logger.info("No Claude API key - speaker notes will be basic summaries only")

        # Cost tracking
        self.cost_tracker = cost_tracker or CostTracker()
        self.total_cost = 0.0
        self.notes_generated = 0
        self.titles_generated = 0

    def generate_smart_title(
        self,
        bullets: List[str],
        context: str = "",
        original_title: str = "",
        model: str = None
    ) -> Dict[str, any]:
        """
        Generate engaging, professional slide title using Claude API.

        This method analyzes the slide's bullet points and surrounding context
        to create a concise, action-oriented title that captures the main message.

        Args:
            bullets: List of bullet points for the slide
            context: Surrounding context (previous/next slides, section info)
            original_title: Original heading from document (optional)
            model: Claude model to use (defaults to DEFAULT_TITLE_MODEL)

        Returns:
            Dictionary containing:
                - title (str): Generated title (3-7 words)
                - cost (float): API cost in USD
                - confidence (float): Confidence score 0-1
                - reasoning (str): Explanation of why this title was chosen
                - input_tokens (int): Number of input tokens
                - output_tokens (int): Number of output tokens
                - cached (bool): Whether response was cached

        Example:
            >>> result = intel.generate_smart_title(
            ...     bullets=["ML models need data", "Training improves accuracy"],
            ...     context="Section: Introduction to AI",
            ...     original_title="Machine Learning"
            ... )
            >>> print(result['title'])
            "Building Effective ML Models"
        """
        if not bullets or len(bullets) == 0:
            logger.warning("No bullets provided for title generation")
            return self._create_fallback_title_result(original_title)

        if not self.client:
            logger.warning("Claude API not available - using original title")
            return self._create_fallback_title_result(original_title)

        model = model or self.DEFAULT_TITLE_MODEL

        # Build prompt
        prompt = self._build_title_generation_prompt(bullets, context, original_title)

        # Call Claude API with retry logic
        import time
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"Calling Claude API for title generation (attempt {attempt + 1}/{self.MAX_RETRIES})")

                # Make API call
                response = self.client.messages.create(
                    model=model,
                    max_tokens=100,  # Short response - just a title
                    temperature=0.7,  # Some creativity for engaging titles
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Extract title from response
                title_text = response.content[0].text.strip()

                # Parse response to extract title and metadata
                result = self._parse_title_response(
                    title_text,
                    original_title,
                    response,
                    model
                )

                # Track cost if tracker is available
                if self.cost_tracker:
                    self.cost_tracker.track_api_call(
                        provider='claude',
                        model=model,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        cached=False,
                        slide_id=None,
                        call_type='smart_title',
                        success=True,
                        error=None
                    )
                    logger.debug(f"Tracked API cost: ${result['cost']:.4f}")

                self.total_cost += result['cost']
                self.titles_generated += 1

                logger.info(f"Generated smart title: '{result['title']}' (confidence: {result['confidence']:.2f})")
                return result

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying...")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Rate limit error after {self.MAX_RETRIES} attempts")
                    return self._create_error_title_result(original_title, str(e))

            except anthropic.APIError as e:
                logger.error(f"Claude API error on attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    return self._create_error_title_result(original_title, str(e))

            except Exception as e:
                logger.error(f"Unexpected error during title generation: {e}")
                return self._create_error_title_result(original_title, str(e))

        # Should not reach here, but fallback just in case
        return self._create_fallback_title_result(original_title)

    def _build_title_generation_prompt(
        self,
        bullets: List[str],
        context: str,
        original_title: str
    ) -> str:
        """
        Build the prompt for Claude API to generate slide title.

        Args:
            bullets: List of bullet points
            context: Surrounding context
            original_title: Original heading from document

        Returns:
            Formatted prompt string
        """
        bullets_text = "\n".join(f"• {bullet}" for bullet in bullets)

        prompt = f"""Generate a concise, engaging slide title (3-7 words) for a professional presentation.

Original title: {original_title if original_title else "(none)"}

Context: {context if context else "(none)"}

Slide content:
{bullets_text}

Requirements:
- Capture the main message or key takeaway
- Professional yet engaging tone
- Avoid generic titles like "Overview" or "Introduction"
- Action-oriented when appropriate (e.g., "Building", "Understanding", "Implementing")
- No punctuation at the end
- Must be 3-7 words
- Use title case (capitalize major words)

After the title, on a new line, briefly explain (one sentence) why you chose this title and rate your confidence (0-1).

Format your response like this:
TITLE: [Your title here]
REASONING: [Brief explanation]
CONFIDENCE: [0.0-1.0]"""

        return prompt

    def _parse_title_response(
        self,
        response_text: str,
        original_title: str,
        api_response: any,
        model: str
    ) -> Dict[str, any]:
        """
        Parse Claude API response to extract title and metadata.

        Args:
            response_text: Raw text from Claude
            original_title: Original title (fallback)
            api_response: Full API response object
            model: Model used

        Returns:
            Parsed result dictionary
        """
        try:
            # Try to parse structured response
            lines = response_text.strip().split('\n')
            title = None
            reasoning = ""
            confidence = 0.8  # Default confidence

            for line in lines:
                line = line.strip()
                if line.startswith('TITLE:'):
                    title = line.replace('TITLE:', '').strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                        # Clamp to 0-1 range
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        logger.warning("Could not parse confidence score, using default")

            # If structured parsing failed, use the whole response as title
            if not title:
                # Take first line or whole response if it's short
                title = lines[0].strip() if lines else response_text.strip()
                reasoning = "Generated from slide content"
                confidence = 0.7

            # Validate title length (3-7 words)
            word_count = len(title.split())
            if word_count < 3 or word_count > 7:
                logger.warning(f"Title has {word_count} words (expected 3-7): '{title}'")
                # Adjust confidence down if title is too long/short
                confidence *= 0.8

            # Calculate cost
            input_tokens = api_response.usage.input_tokens
            output_tokens = api_response.usage.output_tokens

            # Get pricing for this model
            pricing = self.CLAUDE_PRICING.get(model, self.CLAUDE_PRICING[self.DEFAULT_TITLE_MODEL])
            input_cost = (input_tokens / 1_000_000) * pricing['input']
            output_cost = (output_tokens / 1_000_000) * pricing['output']
            total_cost = input_cost + output_cost

            return {
                'title': title,
                'cost': total_cost,
                'confidence': confidence,
                'reasoning': reasoning,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cached': False,
                'model': model
            }

        except Exception as e:
            logger.error(f"Error parsing title response: {e}")
            return self._create_error_title_result(original_title, str(e))

    def _create_fallback_title_result(self, original_title: str) -> Dict[str, any]:
        """
        Create fallback result when AI generation is not available.

        Args:
            original_title: Original title to use as fallback

        Returns:
            Fallback result dictionary
        """
        return {
            'title': original_title or "Slide Title",
            'cost': 0.0,
            'confidence': 0.5,
            'reasoning': "Using original title (AI generation not available)",
            'input_tokens': 0,
            'output_tokens': 0,
            'cached': False,
            'model': 'fallback'
        }

    def _create_error_title_result(self, original_title: str, error: str) -> Dict[str, any]:
        """
        Create error result when API call fails.

        Args:
            original_title: Original title to use as fallback
            error: Error message

        Returns:
            Error result dictionary
        """
        logger.error(f"Title generation failed: {error}")
        return {
            'title': original_title or "Slide Title",
            'cost': 0.0,
            'confidence': 0.0,
            'reasoning': f"Error: {error}",
            'input_tokens': 0,
            'output_tokens': 0,
            'cached': False,
            'model': 'error'
        }

    def generate_speaker_notes(self, slide: SlideContent,
                              next_slide_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate helpful speaker notes for presenter.

        Args:
            slide: SlideContent object with title and bullets
            next_slide_title: Optional title of next slide for transition suggestion

        Returns:
            {
                'notes': str,  # Full speaker notes (2-3 sentences per bullet)
                'talking_points': List[str],  # Key talking points
                'transition': str,  # Suggested transition to next slide
                'cost': float  # API cost
            }
        """
        if not self.client:
            logger.warning("Claude API not available - returning basic speaker notes")
            return self._generate_basic_notes(slide, next_slide_title)

        try:
            # Build prompt for speaker notes generation
            prompt = self._build_speaker_notes_prompt(slide, next_slide_title)

            logger.info(f"🎤 Generating speaker notes for slide: '{slide.title}'")

            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.7,  # Slightly higher for more natural/conversational tone
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract response
            content = message.content[0].text

            # Calculate cost
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            self.total_cost += cost
            self.notes_generated += 1

            # Track cost
            self.cost_tracker.track_api_call(
                provider='claude',
                model='claude-3-5-sonnet-20241022',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                slide_id=slide.title,
                call_type='speaker_notes'
            )

            logger.info(f"✅ Speaker notes generated (cost: ${cost:.4f})")

            # Parse response into structured format
            result = self._parse_speaker_notes_response(content, cost)
            return result

        except Exception as e:
            logger.error(f"Speaker notes generation failed: {e}")
            return self._generate_basic_notes(slide, next_slide_title)

    def _build_speaker_notes_prompt(self, slide: SlideContent,
                                   next_slide_title: Optional[str] = None) -> str:
        """
        Build prompt for Claude to generate speaker notes.

        Args:
            slide: SlideContent object
            next_slide_title: Optional next slide title for transitions

        Returns:
            Prompt string
        """
        bullets = '\n'.join([f"- {bullet}" for bullet in slide.content]) if slide.content else "No bullets"

        prompt = f"""Generate concise speaker notes for this presentation slide.

Title: {slide.title}
Bullets:
{bullets}

Speaker notes should:
- Expand on each bullet point with 2-3 sentences of talking points
- Include relevant examples or analogies where appropriate
- Use a natural, conversational tone suitable for verbal delivery
- Help the presenter explain complex concepts clearly
- Be comprehensive but concise (avoid overly verbose explanations)

"""

        if next_slide_title:
            prompt += f"""- Suggest a smooth transition to the next slide: "{next_slide_title}"

"""

        prompt += """Format your response EXACTLY as follows:

NOTES:
[2-3 detailed paragraphs expanding on the bullet points with natural talking points]

KEY POINTS:
- [First key talking point]
- [Second key talking point]
- [Third key talking point]

"""

        if next_slide_title:
            prompt += """TRANSITION:
[1-2 sentences suggesting how to transition from this slide to the next]
"""

        return prompt

    def _parse_speaker_notes_response(self, content: str, cost: float) -> Dict[str, Any]:
        """
        Parse Claude's response into structured format.

        Args:
            content: Raw response from Claude
            cost: API call cost

        Returns:
            Structured dict with notes, talking_points, transition, cost
        """
        # Initialize result
        result = {
            'notes': '',
            'talking_points': [],
            'transition': '',
            'cost': cost
        }

        # Split response into sections
        sections = content.split('\n')
        current_section = None

        for line in sections:
            line = line.strip()

            if line.startswith('NOTES:'):
                current_section = 'notes'
                continue
            elif line.startswith('KEY POINTS:'):
                current_section = 'key_points'
                continue
            elif line.startswith('TRANSITION:'):
                current_section = 'transition'
                continue

            # Process content based on current section
            if current_section == 'notes' and line:
                result['notes'] += line + ' '
            elif current_section == 'key_points' and line.startswith('-'):
                # Remove leading dash and whitespace
                point = line[1:].strip()
                if point:
                    result['talking_points'].append(point)
            elif current_section == 'transition' and line:
                result['transition'] += line + ' '

        # Clean up whitespace
        result['notes'] = result['notes'].strip()
        result['transition'] = result['transition'].strip()

        # If parsing failed, use entire content as notes
        if not result['notes']:
            result['notes'] = content.strip()

        return result

    def _generate_basic_notes(self, slide: SlideContent,
                            next_slide_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate basic speaker notes without AI (fallback).

        Args:
            slide: SlideContent object
            next_slide_title: Optional next slide title

        Returns:
            Basic notes dict
        """
        # Create basic notes from slide content
        notes_parts = [f"This slide covers: {slide.title}."]

        if slide.content:
            notes_parts.append("\nKey points to discuss:")
            for i, bullet in enumerate(slide.content, 1):
                notes_parts.append(f"{i}. {bullet}")

        notes = ' '.join(notes_parts)

        # Create basic talking points
        talking_points = slide.content[:5] if slide.content else []

        # Create basic transition
        transition = ""
        if next_slide_title:
            transition = f"Moving on, we'll now discuss {next_slide_title}."

        return {
            'notes': notes,
            'talking_points': talking_points,
            'transition': transition,
            'cost': 0.0
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate API call cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        pricing = self.CLAUDE_PRICING['claude-3-5-sonnet-20241022']
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        return input_cost + output_cost

    def generate_batch_speaker_notes(self, slides: List[SlideContent],
                                    max_slides: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate speaker notes for multiple slides.

        Args:
            slides: List of SlideContent objects
            max_slides: Maximum number of slides to generate notes for

        Returns:
            Dict with 'notes' (dict mapping slide index to notes data) and 'summary' (cost summary)
        """
        if max_slides:
            slides = slides[:max_slides]

        logger.info(f"📊 Generating speaker notes for {len(slides)} slides")

        notes_dict = {}
        total_cost = 0.0

        for i, slide in enumerate(slides):
            # Get next slide title for transition suggestion
            next_slide_title = slides[i + 1].title if i + 1 < len(slides) else None

            # Generate notes
            result = self.generate_speaker_notes(slide, next_slide_title)
            notes_dict[i] = result
            total_cost += result.get('cost', 0.0)

        summary = {
            'total_cost': total_cost,
            'notes_generated': len(notes_dict),
            'slides_processed': len(slides)
        }

        logger.info(f"✅ Speaker notes generation complete:")
        logger.info(f"   Notes generated: {summary['notes_generated']}")
        logger.info(f"   Total cost: ${summary['total_cost']:.4f}")

        return {
            'notes': notes_dict,
            'summary': summary
        }

    def analyze_presentation_quality(self, slides: List[SlideContent]) -> Dict[str, Any]:
        """
        Analyze entire presentation for quality issues.

        This method evaluates the presentation across multiple dimensions:
        - Flow: Logical progression and smooth transitions
        - Coherence: Topic connectivity and narrative consistency
        - Redundancy: Duplicate or overly similar content
        - Completeness: Coverage gaps and missing transitions

        Args:
            slides: List of all SlideContent objects in presentation

        Returns:
            {
                'quality_score': float,  # 0-100 overall quality
                'scores': {
                    'flow': float,        # Logical progression (0-100)
                    'coherence': float,   # Topic connectivity (0-100)
                    'redundancy': float,  # Duplicate content (0-100, higher=less redundant)
                    'completeness': float # Coverage/gaps (0-100)
                },
                'issues': [
                    {
                        'type': str,        # 'redundancy', 'gap', 'flow', 'inconsistency'
                        'severity': str,    # 'low', 'medium', 'high'
                        'slides': List[int], # Affected slide indices
                        'description': str
                    }
                ],
                'recommendations': List[str],  # Actionable suggestions
                'strengths': List[str],        # What's working well
                'cost': float
            }

        Example:
            >>> result = intel.analyze_presentation_quality(slides)
            >>> print(f"Quality score: {result['quality_score']}/100")
            >>> print(f"Issues found: {len(result['issues'])}")
        """
        if not slides or len(slides) == 0:
            logger.warning("No slides provided for quality analysis")
            return self._create_empty_quality_result()

        if not self.client:
            logger.warning("Claude API not available - quality review disabled")
            return self._create_empty_quality_result()

        try:
            logger.info(f"📊 Analyzing presentation quality ({len(slides)} slides)...")

            # Build presentation outline for analysis
            outline = self._build_presentation_outline(slides)

            # Build comprehensive analysis prompt
            prompt = self._build_quality_analysis_prompt(outline, len(slides))

            # Call Claude API
            logger.debug("Calling Claude API for presentation quality analysis")
            response = self.client.messages.create(
                model=self.DEFAULT_NOTES_MODEL,  # Use Sonnet for better analysis quality
                max_tokens=2000,  # Need more tokens for comprehensive analysis
                temperature=0.3,  # Lower temperature for more analytical/objective output
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response
            analysis_text = response.content[0].text.strip()

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Track cost
            if self.cost_tracker:
                self.cost_tracker.track_api_call(
                    provider='claude',
                    model=self.DEFAULT_NOTES_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=False,
                    slide_id=None,
                    call_type='quality_review',
                    success=True,
                    error=None
                )

            self.total_cost += cost

            # Parse response into structured format
            result = self._parse_quality_analysis_response(analysis_text, cost)

            logger.info(f"✅ Quality analysis complete (score: {result['quality_score']:.1f}/100, cost: ${cost:.4f})")
            logger.info(f"   Issues found: {len(result['issues'])}")
            logger.info(f"   Recommendations: {len(result['recommendations'])}")

            return result

        except Exception as e:
            logger.error(f"Presentation quality analysis failed: {e}")
            return self._create_error_quality_result(str(e))

    def _build_presentation_outline(self, slides: List[SlideContent]) -> str:
        """
        Build a structured outline of the presentation for analysis.

        Args:
            slides: List of SlideContent objects

        Returns:
            Formatted presentation outline string
        """
        outline_parts = []

        for i, slide in enumerate(slides, 1):
            # Slide number and title
            outline_parts.append(f"\n{i}. {slide.title}")

            # Add bullets if available
            if slide.content and len(slide.content) > 0:
                for bullet in slide.content[:5]:  # Limit to first 5 bullets to save tokens
                    outline_parts.append(f"   • {bullet}")
                if len(slide.content) > 5:
                    outline_parts.append(f"   ... ({len(slide.content) - 5} more bullets)")
            else:
                outline_parts.append("   (no content)")

        return '\n'.join(outline_parts)

    def _build_quality_analysis_prompt(self, outline: str, slide_count: int) -> str:
        """
        Build the prompt for Claude to analyze presentation quality.

        Args:
            outline: Formatted presentation outline
            slide_count: Total number of slides

        Returns:
            Prompt string
        """
        prompt = f"""Analyze this presentation for quality issues and provide actionable recommendations.

PRESENTATION OUTLINE ({slide_count} slides):
{outline}

ANALYSIS REQUIREMENTS:

Evaluate the presentation across 4 dimensions (score each 0-100):

1. FLOW (0-100): Does the presentation progress logically? Are transitions smooth between slides?
   - Check for logical sequencing
   - Identify abrupt topic changes
   - Evaluate narrative progression

2. COHERENCE (0-100): Are topics well-connected? Is there a clear narrative thread?
   - Check for consistent themes
   - Identify disconnected sections
   - Evaluate overall story arc

3. REDUNDANCY (0-100): Any duplicate or overly similar content? (Higher score = less redundant)
   - Identify repeated concepts
   - Find slides covering same ground
   - Spot unnecessary overlap

4. COMPLETENESS (0-100): Missing information or transitions? Are topics adequately covered?
   - Identify missing context
   - Find knowledge gaps
   - Spot abrupt endings or incomplete coverage

IDENTIFY SPECIFIC ISSUES:
For each issue found, specify:
- Type: 'redundancy', 'gap', 'flow', or 'inconsistency'
- Severity: 'low', 'medium', or 'high'
- Affected slide numbers
- Clear description of the problem

PROVIDE RECOMMENDATIONS:
- 3-5 specific, actionable suggestions for improvement
- Prioritize high-impact changes
- Be concrete (e.g., "Merge slides 5 and 6" not "Reduce redundancy")

IDENTIFY STRENGTHS:
- 2-3 things the presentation does well
- What's working effectively

Return your analysis as valid JSON in this EXACT format (no markdown formatting, no code blocks):

{{
  "quality_score": 85,
  "scores": {{
    "flow": 90,
    "coherence": 85,
    "redundancy": 80,
    "completeness": 85
  }},
  "issues": [
    {{
      "type": "redundancy",
      "severity": "medium",
      "slides": [3, 7],
      "description": "Slides 3 and 7 cover similar concepts about X"
    }}
  ],
  "recommendations": [
    "Merge slides 3 and 7 to eliminate redundancy",
    "Add transition slide between sections at slide 12"
  ],
  "strengths": [
    "Clear logical progression in first section",
    "Good use of examples throughout"
  ]
}}

IMPORTANT: Return ONLY the JSON object. Do not include any markdown formatting, code blocks, or explanatory text."""

        return prompt

    def _parse_quality_analysis_response(self, response_text: str, cost: float) -> Dict[str, Any]:
        """
        Parse Claude's quality analysis response into structured format.

        Args:
            response_text: Raw response from Claude
            cost: API call cost

        Returns:
            Structured quality analysis dictionary
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                # Remove markdown code block formatting
                lines = cleaned_text.split('\n')
                # Find first line that starts with {
                start_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('{')), 0)
                # Find last line that ends with }
                end_idx = next((i for i in range(len(lines)-1, -1, -1) if lines[i].strip().endswith('}')), len(lines)-1)
                cleaned_text = '\n'.join(lines[start_idx:end_idx+1])

            # Parse JSON response
            analysis = json.loads(cleaned_text)

            # Validate and ensure all required fields are present
            result = {
                'quality_score': float(analysis.get('quality_score', 0)),
                'scores': {
                    'flow': float(analysis.get('scores', {}).get('flow', 0)),
                    'coherence': float(analysis.get('scores', {}).get('coherence', 0)),
                    'redundancy': float(analysis.get('scores', {}).get('redundancy', 0)),
                    'completeness': float(analysis.get('scores', {}).get('completeness', 0))
                },
                'issues': analysis.get('issues', []),
                'recommendations': analysis.get('recommendations', []),
                'strengths': analysis.get('strengths', []),
                'cost': cost
            }

            # Clamp scores to 0-100 range
            result['quality_score'] = max(0.0, min(100.0, result['quality_score']))
            for key in result['scores']:
                result['scores'][key] = max(0.0, min(100.0, result['scores'][key]))

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quality analysis JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            # Fallback: try to extract scores manually
            return self._parse_quality_fallback(response_text, cost)

        except Exception as e:
            logger.error(f"Error parsing quality analysis response: {e}")
            return self._create_error_quality_result(str(e))

    def _parse_quality_fallback(self, response_text: str, cost: float) -> Dict[str, Any]:
        """
        Fallback parser when JSON parsing fails - extract what we can.

        Args:
            response_text: Raw response text
            cost: API call cost

        Returns:
            Best-effort quality analysis dictionary
        """
        import re

        result = {
            'quality_score': 0.0,
            'scores': {
                'flow': 0.0,
                'coherence': 0.0,
                'redundancy': 0.0,
                'completeness': 0.0
            },
            'issues': [],
            'recommendations': [],
            'strengths': [],
            'cost': cost
        }

        # Try to extract quality_score
        quality_match = re.search(r'"quality_score":\s*(\d+(?:\.\d+)?)', response_text)
        if quality_match:
            result['quality_score'] = float(quality_match.group(1))

        # Try to extract individual scores
        for score_name in ['flow', 'coherence', 'redundancy', 'completeness']:
            score_match = re.search(rf'"{score_name}":\s*(\d+(?:\.\d+)?)', response_text)
            if score_match:
                result['scores'][score_name] = float(score_match.group(1))

        logger.warning("Used fallback parser for quality analysis - some data may be missing")
        return result

    def _create_empty_quality_result(self) -> Dict[str, Any]:
        """
        Create empty result when quality analysis cannot be performed.

        Returns:
            Empty quality analysis dictionary
        """
        return {
            'quality_score': 0.0,
            'scores': {
                'flow': 0.0,
                'coherence': 0.0,
                'redundancy': 0.0,
                'completeness': 0.0
            },
            'issues': [],
            'recommendations': ['Quality analysis unavailable - Claude API not configured'],
            'strengths': [],
            'cost': 0.0
        }

    def _create_error_quality_result(self, error: str) -> Dict[str, Any]:
        """
        Create error result when quality analysis fails.

        Args:
            error: Error message

        Returns:
            Error quality analysis dictionary
        """
        logger.error(f"Quality analysis error: {error}")
        return {
            'quality_score': 0.0,
            'scores': {
                'flow': 0.0,
                'coherence': 0.0,
                'redundancy': 0.0,
                'completeness': 0.0
            },
            'issues': [{
                'type': 'error',
                'severity': 'high',
                'slides': [],
                'description': f"Analysis failed: {error}"
            }],
            'recommendations': ['Unable to complete quality analysis due to error'],
            'strengths': [],
            'cost': 0.0
        }

    def generate_presentation_outline(
        self,
        topic: str,
        audience: str,
        duration_minutes: int,
        objectives: List[str],
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate complete presentation outline from topic and objectives.

        This method creates a comprehensive, well-structured presentation outline
        that can be used to generate slides without requiring a source document.

        Args:
            topic: Presentation topic/title
            audience: Target audience description (e.g., "C-level executives", "Software engineers")
            duration_minutes: Desired presentation length in minutes
            objectives: List of learning/business objectives to achieve
            additional_context: Any additional requirements or constraints

        Returns:
            {
                'presentation_title': str,
                'estimated_slides': int,
                'structure': List[dict],  # Detailed slide outlines
                'speaking_time': int,     # Estimated minutes
                'cost': float,            # API cost in USD
                'input_tokens': int,
                'output_tokens': int
            }

        Structure format for each slide:
        [
            {
                'slide_number': 1,
                'slide_type': 'title',  # title, section_header, content, conclusion, qa
                'title': str,
                'key_points': List[str],  # 3-5 points per slide
                'notes': str  # Suggested content/direction
            }
        ]

        Example:
            >>> outline = intel.generate_presentation_outline(
            ...     topic="Cloud Computing for Executives",
            ...     audience="C-level executives, non-technical",
            ...     duration_minutes=20,
            ...     objectives=["Explain cloud benefits", "Show ROI", "Address security"]
            ... )
            >>> print(f"Generated {outline['estimated_slides']} slides")
        """
        if not self.client:
            logger.error("Claude API not available - outline generation requires API access")
            return self._create_error_outline_result("Claude API not configured")

        if not topic or not audience or duration_minutes <= 0 or not objectives:
            logger.error("Invalid input parameters for outline generation")
            return self._create_error_outline_result("Missing required parameters")

        # Estimate slide count (approximately 2 minutes per slide)
        estimated_slides = max(5, duration_minutes // 2)

        logger.info(f"🎯 Generating presentation outline: '{topic}'")
        logger.info(f"   Audience: {audience}")
        logger.info(f"   Duration: {duration_minutes} minutes (~{estimated_slides} slides)")
        logger.info(f"   Objectives: {len(objectives)}")

        # Build comprehensive prompt
        prompt = self._build_outline_generation_prompt(
            topic, audience, duration_minutes, objectives, additional_context, estimated_slides
        )

        # Call Claude API with retry logic
        import time
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"Calling Claude API for outline generation (attempt {attempt + 1}/{self.MAX_RETRIES})")

                # Make API call - use Sonnet for better quality on complex task
                response = self.client.messages.create(
                    model=self.DEFAULT_NOTES_MODEL,
                    max_tokens=4000,  # Need substantial tokens for full outline
                    temperature=0.7,  # Balance between creativity and consistency
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Extract response
                outline_text = response.content[0].text.strip()

                # Parse response
                result = self._parse_outline_response(
                    outline_text,
                    topic,
                    duration_minutes,
                    response
                )

                # Track cost
                if self.cost_tracker:
                    self.cost_tracker.track_api_call(
                        provider='claude',
                        model=self.DEFAULT_NOTES_MODEL,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        cached=False,
                        slide_id=None,
                        call_type='outline_generation',
                        success=True,
                        error=None
                    )

                self.total_cost += result['cost']

                logger.info(f"✅ Outline generated: {len(result['structure'])} slides (cost: ${result['cost']:.4f})")
                return result

            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}, retrying...")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Rate limit error after {self.MAX_RETRIES} attempts")
                    return self._create_error_outline_result(str(e))

            except anthropic.APIError as e:
                logger.error(f"Claude API error on attempt {attempt + 1}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    return self._create_error_outline_result(str(e))

            except Exception as e:
                logger.error(f"Unexpected error during outline generation: {e}")
                return self._create_error_outline_result(str(e))

        # Should not reach here, but fallback just in case
        return self._create_error_outline_result("Unknown error")

    def _build_outline_generation_prompt(
        self,
        topic: str,
        audience: str,
        duration_minutes: int,
        objectives: List[str],
        additional_context: str,
        estimated_slides: int
    ) -> str:
        """
        Build comprehensive prompt for outline generation.

        Args:
            topic: Presentation topic
            audience: Target audience
            duration_minutes: Duration in minutes
            objectives: List of objectives
            additional_context: Additional requirements
            estimated_slides: Estimated number of slides

        Returns:
            Formatted prompt string
        """
        objectives_text = '\n'.join(f"- {obj}" for obj in objectives)

        # Build additional context section to avoid nested f-strings with backslashes
        context_section = ""
        if additional_context:
            context_section = f"ADDITIONAL CONTEXT:\n{additional_context}\n\n"

        prompt = f"""Create a comprehensive presentation outline for a professional presentation.

PRESENTATION DETAILS:
Topic: {topic}
Audience: {audience}
Duration: {duration_minutes} minutes
Target Slides: {estimated_slides} slides (approximately 2 minutes per slide)

OBJECTIVES:
{objectives_text}

{context_section}YOUR TASK:
Create a well-structured presentation outline with the following components:

1. OPENING (1-2 slides):
   - Title slide with compelling title
   - Optional: Agenda/overview (for presentations >15 minutes)

2. MAIN BODY ({estimated_slides - 4} to {estimated_slides - 3} slides):
   - Divide content into 3-5 logical sections
   - Each section should have a clear theme/topic
   - 2-4 content slides per section
   - Each slide should have 3-5 key points

3. CLOSING (2 slides):
   - Conclusion/summary slide with key takeaways
   - Q&A or "Next Steps" slide

GUIDELINES:
- Match complexity and terminology to the {audience} level
- Ensure logical flow from introduction to conclusion
- Balance depth vs. breadth based on {duration_minutes} minute timeframe
- Make content actionable and relevant to objectives
- Use clear, engaging slide titles (3-7 words each)
- Avoid jargon unless appropriate for technical audiences
- Include speaker notes with suggested talking points

SLIDE TYPES:
- "title": Opening/title slide
- "section_header": Section divider slides
- "content": Regular content slides with bullets
- "conclusion": Summary/conclusion slide
- "qa": Questions and answers or next steps

Return your response as a valid JSON array with this EXACT structure (no markdown formatting, no code blocks):

[
  {{
    "slide_number": 1,
    "slide_type": "title",
    "title": "{topic}",
    "key_points": [],
    "notes": "Opening slide with presentation title, presenter name, and date"
  }},
  {{
    "slide_number": 2,
    "slide_type": "content",
    "title": "What We'll Cover Today",
    "key_points": [
      "First main topic area",
      "Second main topic area",
      "Third main topic area",
      "Expected outcomes"
    ],
    "notes": "Set expectations and preview the presentation flow"
  }},
  {{
    "slide_number": 3,
    "slide_type": "section_header",
    "title": "Section 1: [Section Name]",
    "key_points": [],
    "notes": "Transition to first major section"
  }},
  {{
    "slide_number": 4,
    "slide_type": "content",
    "title": "Engaging Content Title",
    "key_points": [
      "First key point with specific details",
      "Second key point with actionable insights",
      "Third key point with relevant examples",
      "Fourth point connecting to objectives"
    ],
    "notes": "Explain each point with examples. Connect to [specific objective]. Use analogies if needed for {audience}."
  }}
]

IMPORTANT:
- Return ONLY the JSON array
- Do NOT include markdown code blocks (no triple backticks)
- Do NOT include any explanatory text before or after the JSON
- Ensure {estimated_slides - 2} to {estimated_slides + 2} total slides
- Make every slide meaningful and aligned with objectives"""

        return prompt

    def _parse_outline_response(
        self,
        response_text: str,
        topic: str,
        duration_minutes: int,
        api_response: any
    ) -> Dict[str, Any]:
        """
        Parse Claude's outline response into structured format.

        Args:
            response_text: Raw JSON response from Claude
            topic: Original topic
            duration_minutes: Duration in minutes
            api_response: Full API response object

        Returns:
            Parsed outline dictionary
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                # Remove markdown code block formatting
                lines = cleaned_text.split('\n')
                # Find first line that starts with [
                start_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('[')), 0)
                # Find last line that ends with ]
                end_idx = next((i for i in range(len(lines)-1, -1, -1) if lines[i].strip().endswith(']')), len(lines)-1)
                cleaned_text = '\n'.join(lines[start_idx:end_idx+1])

            # Parse JSON array
            structure = json.loads(cleaned_text)

            # Validate structure
            if not isinstance(structure, list):
                raise ValueError("Response is not a JSON array")

            # Validate each slide has required fields
            for i, slide in enumerate(structure):
                if not all(key in slide for key in ['slide_number', 'slide_type', 'title', 'key_points', 'notes']):
                    logger.warning(f"Slide {i+1} missing required fields, adding defaults")
                    slide.setdefault('slide_number', i + 1)
                    slide.setdefault('slide_type', 'content')
                    slide.setdefault('title', f'Slide {i+1}')
                    slide.setdefault('key_points', [])
                    slide.setdefault('notes', '')

            # Calculate cost
            input_tokens = api_response.usage.input_tokens
            output_tokens = api_response.usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            return {
                'presentation_title': topic,
                'estimated_slides': len(structure),
                'structure': structure,
                'speaking_time': duration_minutes,
                'cost': cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse outline JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return self._create_error_outline_result(f"Invalid JSON response: {e}")

        except Exception as e:
            logger.error(f"Error parsing outline response: {e}")
            return self._create_error_outline_result(str(e))

    def _create_error_outline_result(self, error: str) -> Dict[str, Any]:
        """
        Create error result when outline generation fails.

        Args:
            error: Error message

        Returns:
            Error result dictionary
        """
        logger.error(f"Outline generation failed: {error}")
        return {
            'presentation_title': '',
            'estimated_slides': 0,
            'structure': [],
            'speaking_time': 0,
            'cost': 0.0,
            'input_tokens': 0,
            'output_tokens': 0,
            'error': error
        }

    def create_slides_from_outline(self, outline: Dict[str, Any]) -> List[SlideContent]:
        """
        Convert outline structure to SlideContent objects.

        This method transforms the AI-generated outline into a list of SlideContent
        objects that can be used by the presentation generators (PowerPoint, Google Slides).

        Args:
            outline: Outline dictionary from generate_presentation_outline()

        Returns:
            List of SlideContent objects ready for presentation generation

        Example:
            >>> outline = intel.generate_presentation_outline(...)
            >>> slides = intel.create_slides_from_outline(outline)
            >>> print(f"Created {len(slides)} slides")
        """
        slides = []

        if 'error' in outline or not outline.get('structure'):
            logger.error("Cannot create slides from invalid outline")
            return slides

        logger.info(f"🎨 Converting outline to {len(outline['structure'])} slide objects")

        for slide_spec in outline['structure']:
            # Map slide type to heading level for consistent formatting
            heading_level = self._map_slide_type_to_heading_level(slide_spec['slide_type'])

            # Create SlideContent object
            slide = SlideContent(
                title=slide_spec['title'],
                content=slide_spec['key_points'],
                slide_type=slide_spec['slide_type'],
                heading_level=heading_level,
                speaker_notes=slide_spec.get('notes', '')  # Store AI-generated notes
            )

            slides.append(slide)

            logger.debug(f"   Slide {slide_spec['slide_number']}: {slide_spec['title']} ({slide_spec['slide_type']})")

        logger.info(f"✅ Created {len(slides)} SlideContent objects")
        return slides

    def _map_slide_type_to_heading_level(self, slide_type: str) -> int:
        """
        Map slide type to appropriate heading level for formatting.

        Args:
            slide_type: Slide type string

        Returns:
            Heading level (1-4)
        """
        # Map slide types to heading levels for consistent formatting
        type_to_level = {
            'title': 1,          # H1 for title slides
            'section_header': 2, # H2 for section dividers
            'content': 4,        # H4 for content slides
            'conclusion': 3,     # H3 for conclusions
            'qa': 4              # H4 for Q&A
        }

        return type_to_level.get(slide_type, 4)  # Default to H4

    def generate_qa_slides(
        self,
        slides: List[SlideContent],
        num_questions: int = 5,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate Q&A slides from presentation content.

        Analyzes the presentation and generates realistic questions that an audience
        might ask, along with concise, bullet-point answers. This helps presenters
        prepare for Q&A sessions.

        Args:
            slides: List of all presentation slides
            num_questions: Number of Q&A pairs to generate (default: 5)
            focus_areas: Optional list of areas to focus questions on

        Returns:
            {
                'qa_slides': List[SlideContent],  # Ready-to-add Q&A slides
                'questions': List[dict],           # Detailed Q&A info
                'coverage_areas': List[str],       # Topics covered
                'cost': float                      # API cost in USD
            }

        Question format:
        [
            {
                'question': str,                   # The question text
                'answer_bullets': List[str],       # 2-3 bullet point answers
                'source_slides': List[int],        # Which slides informed this
                'difficulty': str,                 # 'basic', 'intermediate', 'advanced'
                'category': str                    # 'clarification', 'implementation', 'concern', etc.
            }
        ]

        Example:
            >>> result = intel.generate_qa_slides(slides, num_questions=5)
            >>> print(f"Generated {len(result['qa_slides'])} Q&A slides")
            >>> for qa in result['questions']:
            ...     print(f"Q: {qa['question']}")
            ...     print(f"Category: {qa['category']}, Difficulty: {qa['difficulty']}")
        """
        if not slides or len(slides) == 0:
            logger.warning("No slides provided for Q&A generation")
            return self._create_empty_qa_result()

        if not self.client:
            logger.warning("Claude API not available - Q&A generation disabled")
            return self._create_empty_qa_result()

        try:
            logger.info(f"🤔 Generating {num_questions} Q&A slides from {len(slides)} presentation slides...")

            # Build presentation outline for Q&A generation
            presentation_outline = self._build_presentation_outline(slides)

            # Build Q&A generation prompt
            prompt = self._build_qa_generation_prompt(
                presentation_outline,
                num_questions,
                focus_areas
            )

            # Call Claude API
            logger.debug("Calling Claude API for Q&A generation")
            response = self.client.messages.create(
                model=self.DEFAULT_NOTES_MODEL,  # Use Sonnet for better quality
                max_tokens=3000,  # Need more tokens for multiple Q&As
                temperature=0.7,  # Some creativity for realistic questions
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response
            qa_text = response.content[0].text.strip()

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Track cost
            if self.cost_tracker:
                self.cost_tracker.track_api_call(
                    provider='claude',
                    model=self.DEFAULT_NOTES_MODEL,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=False,
                    slide_id=None,
                    call_type='qa_generation',
                    success=True,
                    error=None
                )

            self.total_cost += cost

            # Parse response into structured format
            result = self._parse_qa_response(qa_text, cost)

            # Create SlideContent objects from questions
            qa_slides = []
            for qa_item in result['questions']:
                slide = self._create_qa_slide_from_question(qa_item)
                qa_slides.append(slide)

            result['qa_slides'] = qa_slides

            logger.info(f"✅ Q&A generation complete: {len(qa_slides)} slides created (cost: ${cost:.4f})")
            logger.info(f"   Topics covered: {', '.join(result['coverage_areas'][:3])}")

            return result

        except Exception as e:
            logger.error(f"Q&A generation failed: {e}")
            return self._create_error_qa_result(str(e))

    def _build_qa_generation_prompt(
        self,
        presentation_outline: str,
        num_questions: int,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Build the prompt for Claude to generate Q&A content.

        Args:
            presentation_outline: Formatted presentation outline
            num_questions: Number of Q&As to generate
            focus_areas: Optional list of focus areas

        Returns:
            Prompt string
        """
        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFocus areas: {', '.join(focus_areas)}"

        prompt = f"""Generate realistic Q&A based on this presentation to help the presenter prepare for audience questions.

PRESENTATION SUMMARY:
{presentation_outline}{focus_text}

Generate {num_questions} common questions an audience might ask after seeing this presentation.

QUESTION CATEGORIES TO COVER:
1. Clarifications - "What exactly does X mean?" / "Can you explain Y in more detail?"
2. Implementation - "How do we actually do this?" / "What's the first step?"
3. Concerns - "What about Y risk/challenge?" / "How do we handle Z problem?"
4. Next steps - "What should we do next?" / "What's the timeline?"
5. Comparisons - "How does this compare to Z?" / "Why not use X instead?"

REQUIREMENTS FOR EACH Q&A:
- Question should be realistic and specific (not generic)
- Answer should be 2-3 concise bullets (8-15 words each)
- Cover different aspects of the presentation
- Mix of basic and advanced questions
- Anticipate concerns, objections, and clarifications
- Use natural, conversational language

Return as valid JSON (no markdown formatting, no code blocks):

[
  {{
    "question": "How long does a typical cloud migration take?",
    "answer_bullets": [
      "Planning phase: 2-3 months for assessment and strategy",
      "Migration execution: 3-6 months depending on complexity",
      "Optimization phase: Ongoing for first 6-12 months"
    ],
    "source_slides": [5, 8, 12],
    "difficulty": "intermediate",
    "category": "implementation"
  }},
  {{
    "question": "What are the main security risks we should prepare for?",
    "answer_bullets": [
      "Data breaches during transfer - use encryption and secure channels",
      "Misconfigured access controls - implement least-privilege principles",
      "Compliance gaps - audit cloud provider certifications before migration"
    ],
    "source_slides": [9, 14],
    "difficulty": "advanced",
    "category": "concern"
  }}
]

IMPORTANT:
- Return ONLY the JSON array
- No markdown code blocks (```json)
- No explanatory text before or after
- Ensure valid JSON syntax
- Include exactly {num_questions} Q&A pairs"""

        return prompt

    def _parse_qa_response(self, response_text: str, cost: float) -> Dict[str, Any]:
        """
        Parse Claude's Q&A response into structured format.

        Args:
            response_text: Raw response from Claude
            cost: API call cost

        Returns:
            Structured Q&A dictionary
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                # Remove markdown code block formatting
                lines = cleaned_text.split('\n')
                # Find first line that starts with [
                start_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('[')), 0)
                # Find last line that ends with ]
                end_idx = next((i for i in range(len(lines)-1, -1, -1) if lines[i].strip().endswith(']')), len(lines)-1)
                cleaned_text = '\n'.join(lines[start_idx:end_idx+1])

            # Parse JSON response
            questions = json.loads(cleaned_text)

            # Validate structure
            if not isinstance(questions, list):
                raise ValueError("Response is not a list of questions")

            # Extract unique coverage areas from categories
            coverage_areas = list(set(q.get('category', 'general') for q in questions))

            result = {
                'questions': questions,
                'coverage_areas': coverage_areas,
                'cost': cost
            }

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Q&A JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            # Fallback: try to extract Q&As manually
            return self._parse_qa_fallback(response_text, cost)

        except Exception as e:
            logger.error(f"Error parsing Q&A response: {e}")
            return self._create_error_qa_result(str(e))

    def _parse_qa_fallback(self, response_text: str, cost: float) -> Dict[str, Any]:
        """
        Fallback parser when JSON parsing fails - extract what we can.

        Args:
            response_text: Raw response text
            cost: API call cost

        Returns:
            Best-effort Q&A dictionary
        """
        logger.warning("Using fallback parser for Q&A - some data may be incomplete")

        # Try to extract questions and answers from text
        questions = []
        lines = response_text.split('\n')

        current_question = None
        current_bullets = []

        for line in lines:
            line = line.strip()

            # Check if line looks like a question
            if line.endswith('?'):
                # Save previous Q&A if exists
                if current_question and current_bullets:
                    questions.append({
                        'question': current_question,
                        'answer_bullets': current_bullets,
                        'source_slides': [],
                        'difficulty': 'intermediate',
                        'category': 'general'
                    })

                # Start new question
                current_question = line
                current_bullets = []

            # Check if line looks like a bullet point
            elif line.startswith(('-', '•', '*')) and current_question:
                bullet = line[1:].strip()
                if bullet:
                    current_bullets.append(bullet)

        # Save last Q&A
        if current_question and current_bullets:
            questions.append({
                'question': current_question,
                'answer_bullets': current_bullets,
                'source_slides': [],
                'difficulty': 'intermediate',
                'category': 'general'
            })

        return {
            'questions': questions,
            'coverage_areas': ['general'],
            'cost': cost
        }

    def _create_qa_slide_from_question(self, qa_item: Dict[str, Any]) -> SlideContent:
        """
        Convert Q&A item to SlideContent object.

        Args:
            qa_item: Q&A dictionary with question, answer_bullets, etc.

        Returns:
            SlideContent object formatted as Q&A slide
        """
        # Format title: "Q: [Question]"
        title = f"Q: {qa_item['question']}"

        # Answer bullets (already formatted)
        bullets = qa_item['answer_bullets']

        # Create Q&A metadata
        qa_info = {
            'difficulty': qa_item.get('difficulty', 'intermediate'),
            'category': qa_item.get('category', 'general'),
            'source_slides': qa_item.get('source_slides', [])
        }

        return SlideContent(
            title=title,
            content=bullets,
            slide_type='qa',
            heading_level=3,
            qa_info=qa_info
        )

    def _create_empty_qa_result(self) -> Dict[str, Any]:
        """
        Create empty result when Q&A generation cannot be performed.

        Returns:
            Empty Q&A result dictionary
        """
        return {
            'qa_slides': [],
            'questions': [],
            'coverage_areas': [],
            'cost': 0.0
        }

    def _create_error_qa_result(self, error: str) -> Dict[str, Any]:
        """
        Create error result when Q&A generation fails.

        Args:
            error: Error message

        Returns:
            Error Q&A result dictionary
        """
        logger.error(f"Q&A generation error: {error}")
        return {
            'qa_slides': [],
            'questions': [],
            'coverage_areas': [],
            'cost': 0.0,
            'error': error
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get presentation intelligence statistics (titles and speaker notes).

        Returns:
            Dict with cost and performance statistics
        """
        return {
            'total_cost': self.total_cost,
            'notes_generated': self.notes_generated,
            'titles_generated': self.titles_generated,
            'api_available': self.client is not None
        }
