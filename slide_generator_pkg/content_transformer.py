"""
Content Transformer Module

Transforms presentation content for different audience complexity levels.
Supports beginner, intermediate, expert, and executive audiences.
Also handles multilingual translation of presentations.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, replace

import anthropic

from .data_models import SlideContent
from .utils import CostTracker

logger = logging.getLogger(__name__)


# Supported languages mapping (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'ru': 'Russian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'sv': 'Swedish',
    'tr': 'Turkish',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'cs': 'Czech'
}

# Languages that use Right-to-Left text direction
RTL_LANGUAGES = {'ar', 'he', 'fa', 'ur'}

# CJK (Chinese, Japanese, Korean) languages for special handling
CJK_LANGUAGES = {'zh', 'ja', 'ko'}

# Technical terms to preserve across all languages (case-insensitive matching)
TECHNICAL_TERMS_TO_PRESERVE = {
    'API', 'REST', 'HTTP', 'HTTPS', 'JSON', 'XML', 'SQL', 'NoSQL',
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'K8s',
    'CI/CD', 'DevOps', 'MLOps', 'AI', 'ML', 'GPU', 'CPU',
    'SaaS', 'PaaS', 'IaaS', 'SDK', 'CLI', 'GUI', 'UI', 'UX',
    'OAuth', 'JWT', 'SSL', 'TLS', 'VPN', 'DNS', 'CDN',
    'GitHub', 'GitLab', 'Jira', 'Slack', 'Teams',
    'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go', 'Rust',
    'React', 'Vue', 'Angular', 'Node.js', 'Django', 'Flask',
    'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch'
}


class ContentTransformer:
    """
    Transforms presentation content for different audiences.

    Automatically adjusts slide content complexity, vocabulary, and focus
    based on target audience level (beginner, intermediate, expert, executive).

    Features:
    - Vocabulary adjustment (simplify or add technical depth)
    - Analogies for complex concepts (beginner)
    - Business impact focus (executive)
    - Technical terminology (expert)
    - Maintains core message and slide structure
    - Cost tracking for all transformations
    """

    # Complexity level definitions with transformation guidelines
    COMPLEXITY_LEVELS = {
        'beginner': {
            'name': 'Beginner',
            'description': 'Simple language, analogies, avoid jargon',
            'guidelines': [
                'Use simple, everyday language',
                'Avoid technical jargon or explain it clearly',
                'Use analogies and real-world examples',
                'Break down complex concepts into simple steps',
                'Focus on practical understanding over technical accuracy',
                'Use "you" and "we" to make it conversational'
            ],
            'vocabulary': 'elementary',
            'tone': 'friendly and educational'
        },
        'intermediate': {
            'name': 'Intermediate',
            'description': 'Some technical terms, balanced detail',
            'guidelines': [
                'Use some technical terminology with context',
                'Balance conceptual and practical information',
                'Assume basic background knowledge',
                'Provide moderate detail and explanation',
                'Include both "what" and "how"',
                'Professional but accessible tone'
            ],
            'vocabulary': 'professional with some technical terms',
            'tone': 'professional and informative'
        },
        'expert': {
            'name': 'Expert',
            'description': 'Technical terminology, detailed explanations',
            'guidelines': [
                'Use precise technical terminology freely',
                'Assume deep background knowledge',
                'Focus on implementation details and edge cases',
                'Include technical nuances and trade-offs',
                'Reference specific technologies, algorithms, or methodologies',
                'Emphasize technical accuracy and depth'
            ],
            'vocabulary': 'technical and precise',
            'tone': 'technical and authoritative'
        },
        'executive': {
            'name': 'Executive',
            'description': 'Business impact focus, high-level, actionable',
            'guidelines': [
                'Focus on business value and ROI',
                'Emphasize strategic implications',
                'Use business metrics and outcomes',
                'Minimize technical details',
                'Highlight risks, opportunities, and decisions',
                'Include timeline and resource implications',
                'Use action-oriented language'
            ],
            'vocabulary': 'business-focused and strategic',
            'tone': 'strategic and action-oriented'
        }
    }

    def __init__(self, client: anthropic.Anthropic, cost_tracker: Optional[CostTracker] = None):
        """
        Initialize ContentTransformer with Claude API client.

        Args:
            client: Initialized Anthropic client for API calls
            cost_tracker: Optional CostTracker instance for cost monitoring
        """
        self.client = client
        self.cost_tracker = cost_tracker or CostTracker()

        # Performance metrics
        self.transformations_performed = 0
        self.total_transform_time = 0.0

        logger.info("✅ ContentTransformer initialized")

    def adjust_complexity(self, slide: SlideContent, target_level: str) -> Dict[str, Any]:
        """
        Rewrite slide content for target audience level.

        Args:
            slide: SlideContent to transform
            target_level: 'beginner', 'intermediate', 'expert', 'executive'

        Returns:
            {
                'transformed_slide': SlideContent,
                'changes': List[str],  # What was changed
                'original_level': str,  # Detected original complexity
                'target_level': str,
                'cost': float,
                'tokens_used': int
            }

        Raises:
            ValueError: If target_level is invalid or slide has no content
        """
        # Validate target level
        if target_level not in self.COMPLEXITY_LEVELS:
            raise ValueError(
                f"Invalid target_level '{target_level}'. "
                f"Must be one of: {', '.join(self.COMPLEXITY_LEVELS.keys())}"
            )

        # Validate slide has content
        if not slide.content or len(slide.content) == 0:
            raise ValueError("Slide has no content to transform")

        logger.info(f"Transforming slide '{slide.title}' to {target_level} level...")

        # Build transformation prompt
        prompt = self._build_transformation_prompt(slide, target_level)

        # Call Claude API
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,  # Low temperature for consistent transformations
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response text
            response_text = response.content[0].text

            # Track cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            if self.cost_tracker:
                self.cost_tracker.track_api_call(
                    provider='claude',
                    model='claude-3-5-sonnet-20241022',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=False,
                    slide_id=f"transform_{slide.title}",
                    call_type='complexity_adjustment',
                    success=True
                )

            # Parse JSON response
            result = self._parse_transformation_response(response_text, slide)

            # Calculate cost
            cost = self.cost_tracker._calculate_cost(
                'claude-3-5-sonnet-20241022',
                input_tokens,
                output_tokens
            ) if self.cost_tracker else 0.0

            # Create transformed slide
            transformed_slide = SlideContent(
                title=result['title'],
                content=result['bullets'],
                slide_type=slide.slide_type,
                heading_level=slide.heading_level,
                subheader=result.get('subheader'),
                visual_cues=slide.visual_cues,
                visual_prompt=slide.visual_prompt,
                visual_image_url=slide.visual_image_url,
                visual_image_path=slide.visual_image_path,
                visual_type=slide.visual_type,
                bullet_markers=slide.bullet_markers,
                speaker_notes=slide.speaker_notes,
                speaker_talking_points=slide.speaker_talking_points,
                speaker_transition=slide.speaker_transition
            )

            self.transformations_performed += 1

            return {
                'transformed_slide': transformed_slide,
                'changes': result['changes'],
                'original_level': result.get('original_level', 'unknown'),
                'target_level': target_level,
                'cost': cost,
                'tokens_used': input_tokens + output_tokens
            }

        except anthropic.APIError as e:
            logger.error(f"Claude API error during transformation: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse transformation response as JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON response from Claude API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during transformation: {e}")
            raise

    def adjust_presentation_complexity(
        self,
        slides: List[SlideContent],
        target_level: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Batch adjust entire presentation to target complexity level.

        Args:
            slides: List of SlideContent objects to transform
            target_level: 'beginner', 'intermediate', 'expert', 'executive'
            progress_callback: Optional callback(current, total, slide_title) for progress updates

        Returns:
            {
                'transformed_slides': List[SlideContent],
                'individual_results': List[Dict],  # Per-slide transformation results
                'total_cost': float,
                'total_tokens': int,
                'slides_processed': int,
                'avg_cost_per_slide': float
            }
        """
        logger.info(f"Starting batch transformation of {len(slides)} slides to {target_level} level...")

        transformed_slides = []
        individual_results = []
        total_cost = 0.0
        total_tokens = 0

        for idx, slide in enumerate(slides):
            # Skip slides without content
            if not slide.content or len(slide.content) == 0:
                logger.debug(f"Skipping slide '{slide.title}' - no content")
                transformed_slides.append(slide)
                continue

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(slides), slide.title)

            try:
                # Transform slide
                result = self.adjust_complexity(slide, target_level)

                transformed_slides.append(result['transformed_slide'])
                individual_results.append({
                    'slide_title': slide.title,
                    'original_bullets': len(slide.content),
                    'transformed_bullets': len(result['transformed_slide'].content),
                    'changes': result['changes'],
                    'cost': result['cost'],
                    'tokens': result['tokens_used']
                })

                total_cost += result['cost']
                total_tokens += result['tokens_used']

                logger.info(
                    f"Transformed slide {idx+1}/{len(slides)}: '{slide.title}' "
                    f"(${result['cost']:.4f}, {result['tokens_used']} tokens)"
                )

            except Exception as e:
                logger.error(f"Failed to transform slide '{slide.title}': {e}")
                # Keep original slide on error
                transformed_slides.append(slide)
                individual_results.append({
                    'slide_title': slide.title,
                    'error': str(e),
                    'cost': 0.0,
                    'tokens': 0
                })

        avg_cost = total_cost / len(slides) if slides else 0.0

        logger.info(
            f"Batch transformation complete: {len(slides)} slides, "
            f"${total_cost:.4f} total cost, {total_tokens:,} tokens"
        )

        return {
            'transformed_slides': transformed_slides,
            'individual_results': individual_results,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'slides_processed': len(slides),
            'avg_cost_per_slide': avg_cost
        }

    def _build_transformation_prompt(self, slide: SlideContent, target_level: str) -> str:
        """
        Build Claude API prompt for transforming slide to target complexity level.

        Args:
            slide: Slide to transform
            target_level: Target complexity level

        Returns:
            Formatted prompt string
        """
        level_config = self.COMPLEXITY_LEVELS[target_level]

        # Format current bullets
        bullets_text = '\n'.join(f"• {bullet}" for bullet in slide.content)

        # Build guidelines text
        guidelines_text = '\n'.join(f"- {guideline}" for guideline in level_config['guidelines'])

        prompt = f"""You are a presentation content expert. Rewrite this slide for a {level_config['name']} audience.

CURRENT SLIDE:
Title: {slide.title}
{f'Subheader: {slide.subheader}' if slide.subheader else ''}

Bullets:
{bullets_text}

TARGET AUDIENCE: {level_config['name']} ({level_config['description']})

TRANSFORMATION GUIDELINES:
{guidelines_text}

TARGET VOCABULARY: {level_config['vocabulary']}
TARGET TONE: {level_config['tone']}

REQUIREMENTS:
1. Maintain the same core message and key points
2. Keep the same number of bullets ({len(slide.content)}) - do not add or remove bullets
3. Adjust vocabulary, examples, and focus to match {target_level} audience
4. Each bullet should be a complete sentence
5. Bullets should be concise (8-20 words each)
6. For beginner: Add analogies, simplify jargon
7. For intermediate: Balance technical and accessible language
8. For expert: Use precise technical terminology, add depth
9. For executive: Focus on business value, ROI, strategic implications

ALSO PROVIDE:
- First, detect the original complexity level of the slide (beginner/intermediate/expert/executive)
- List specific changes you made (e.g., "Simplified 'microservices' to 'small independent components'")

Return your response as JSON in this exact format:
{{
  "original_level": "intermediate",
  "title": "Rewritten slide title (if needed)",
  "subheader": "Optional subheader (null if not applicable)",
  "bullets": [
    "First rewritten bullet point",
    "Second rewritten bullet point",
    "Third rewritten bullet point"
  ],
  "changes": [
    "Changed 'distributed system' to 'system split across multiple computers'",
    "Added analogy: 'like organizing a company into small teams'",
    "Removed technical term 'circuit breaker pattern'"
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text before or after."""

        return prompt

    def _parse_transformation_response(self, response_text: str, original_slide: SlideContent) -> Dict[str, Any]:
        """
        Parse Claude API response for slide transformation.

        Args:
            response_text: Raw response from Claude
            original_slide: Original slide (for fallback)

        Returns:
            Parsed transformation result

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Try to extract JSON from response (in case Claude adds extra text)
        response_text = response_text.strip()

        # Find JSON block (between first { and last })
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise json.JSONDecodeError(
                "No JSON object found in response",
                response_text,
                0
            )

        json_text = response_text[start_idx:end_idx+1]

        # Parse JSON
        result = json.loads(json_text)

        # Validate required fields
        required_fields = ['bullets', 'changes']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field in response: {field}")

        # Set defaults for optional fields
        if 'title' not in result or not result['title']:
            result['title'] = original_slide.title

        if 'subheader' not in result:
            result['subheader'] = original_slide.subheader

        if 'original_level' not in result:
            result['original_level'] = 'unknown'

        # Validate bullet count
        if len(result['bullets']) != len(original_slide.content):
            logger.warning(
                f"Bullet count mismatch: original={len(original_slide.content)}, "
                f"transformed={len(result['bullets'])}. Using transformed count."
            )

        return result

    def get_transformation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about transformations performed.

        Returns:
            Dictionary with transformation statistics
        """
        return {
            'transformations_performed': self.transformations_performed,
            'total_cost': self.cost_tracker.get_total_cost() if self.cost_tracker else 0.0,
            'total_tokens': self.cost_tracker.get_total_tokens() if self.cost_tracker else {},
            'avg_cost_per_transformation': (
                self.cost_tracker.get_total_cost() / self.transformations_performed
                if self.transformations_performed > 0 and self.cost_tracker else 0.0
            )
        }

    @classmethod
    def get_available_levels(cls) -> Dict[str, str]:
        """
        Get all available complexity levels with descriptions.

        Returns:
            Dictionary mapping level names to descriptions
        """
        return {
            level: config['description']
            for level, config in cls.COMPLEXITY_LEVELS.items()
        }

    @classmethod
    def validate_level(cls, level: str) -> bool:
        """
        Check if a complexity level is valid.

        Args:
            level: Level to validate

        Returns:
            True if valid, False otherwise
        """
        return level in cls.COMPLEXITY_LEVELS

    # ==================== TRANSLATION METHODS ====================

    def translate_slide(self, slide: SlideContent, target_language: str,
                       preserve_technical_terms: bool = True) -> Dict[str, Any]:
        """
        Translate a single slide to the target language.

        Args:
            slide: SlideContent to translate
            target_language: ISO 639-1 language code (e.g., 'es', 'fr', 'de')
            preserve_technical_terms: If True, preserve common technical terms

        Returns:
            {
                'translated_slide': SlideContent with translated content,
                'target_language': str (ISO code),
                'language_name': str (full language name),
                'cost': float (API cost in USD),
                'text_direction': str ('ltr' or 'rtl'),
                'success': bool,
                'error': Optional[str]
            }
        """
        # Validate language code
        if target_language not in SUPPORTED_LANGUAGES:
            return {
                'translated_slide': slide,
                'target_language': target_language,
                'language_name': 'Unknown',
                'cost': 0.0,
                'text_direction': 'ltr',
                'success': False,
                'error': f"Unsupported language code: {target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            }

        # Check if client is available
        if not self.client:
            return {
                'translated_slide': slide,
                'target_language': target_language,
                'language_name': SUPPORTED_LANGUAGES[target_language],
                'cost': 0.0,
                'text_direction': 'ltr',
                'success': False,
                'error': "No Claude API client available for translation"
            }

        language_name = SUPPORTED_LANGUAGES[target_language]
        text_direction = 'rtl' if target_language in RTL_LANGUAGES else 'ltr'

        try:
            # Build translation prompt
            prompt = self._build_translation_prompt(
                slide=slide,
                target_language=target_language,
                language_name=language_name,
                preserve_technical_terms=preserve_technical_terms
            )

            # Call Claude API for translation
            logger.info(f"Translating slide '{slide.title}' to {language_name}...")

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent translations
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response text
            response_text = response.content[0].text.strip()

            # Parse JSON response
            translation_data = self._parse_translation_response(response_text)

            if not translation_data:
                raise ValueError("Failed to parse translation response")

            # Create translated slide
            translated_slide = replace(
                slide,
                title=translation_data.get('title', slide.title),
                content=translation_data.get('bullets', slide.content),
                subheader=translation_data.get('subheader', slide.subheader) if slide.subheader else None,
                original_language='en',
                translated_language=target_language,
                text_direction=text_direction
            )

            # Calculate cost (Sonnet 3.5: $3/MTok input, $15/MTok output)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Use cost tracker's method if available
            if self.cost_tracker:
                cost = self.cost_tracker._calculate_cost(
                    'claude-3-5-sonnet-20241022',
                    input_tokens,
                    output_tokens
                )
                self.cost_tracker.track_api_call(
                    provider='claude',
                    model='claude-3-5-sonnet-20241022',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=False,
                    slide_id=f"translate_{slide.title}",
                    call_type='translation',
                    success=True
                )
            else:
                cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)

            logger.info(f"Successfully translated slide to {language_name} (cost: ${cost:.4f})")

            return {
                'translated_slide': translated_slide,
                'target_language': target_language,
                'language_name': language_name,
                'cost': cost,
                'text_direction': text_direction,
                'success': True,
                'error': None
            }

        except Exception as e:
            logger.error(f"Translation failed for slide '{slide.title}': {e}")
            return {
                'translated_slide': slide,
                'target_language': target_language,
                'language_name': language_name,
                'cost': 0.0,
                'text_direction': text_direction,
                'success': False,
                'error': str(e)
            }

    def translate_presentation(self, slides: List[SlideContent], target_language: str,
                              preserve_technical_terms: bool = True,
                              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Translate an entire presentation to the target language.

        Args:
            slides: List of SlideContent objects to translate
            target_language: ISO 639-1 language code (e.g., 'es', 'fr', 'de')
            preserve_technical_terms: If True, preserve common technical terms
            progress_callback: Optional callback(current, total, slide_title) for progress updates

        Returns:
            {
                'translated_slides': List[SlideContent] with translated content,
                'target_language': str (ISO code),
                'language_name': str (full language name),
                'total_cost': float (total API cost in USD),
                'slide_count': int,
                'success_count': int,
                'failed_slides': List[int] (indices of failed slides),
                'text_direction': str ('ltr' or 'rtl'),
                'success': bool,
                'errors': List[str]
            }
        """
        if target_language not in SUPPORTED_LANGUAGES:
            return {
                'translated_slides': slides,
                'target_language': target_language,
                'language_name': 'Unknown',
                'total_cost': 0.0,
                'slide_count': len(slides),
                'success_count': 0,
                'failed_slides': list(range(len(slides))),
                'text_direction': 'ltr',
                'success': False,
                'errors': [f"Unsupported language code: {target_language}"]
            }

        language_name = SUPPORTED_LANGUAGES[target_language]
        text_direction = 'rtl' if target_language in RTL_LANGUAGES else 'ltr'

        logger.info(f"Starting translation of {len(slides)} slides to {language_name}...")

        translated_slides = []
        total_cost = 0.0
        success_count = 0
        failed_slides = []
        errors = []

        for idx, slide in enumerate(slides):
            # Skip slides without content
            if not slide.content or len(slide.content) == 0:
                logger.debug(f"Skipping slide '{slide.title}' - no content")
                translated_slides.append(slide)
                continue

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(slides), slide.title)

            result = self.translate_slide(
                slide=slide,
                target_language=target_language,
                preserve_technical_terms=preserve_technical_terms
            )

            translated_slides.append(result['translated_slide'])
            total_cost += result['cost']

            if result['success']:
                success_count += 1
            else:
                failed_slides.append(idx)
                if result['error']:
                    errors.append(f"Slide {idx} ('{slide.title}'): {result['error']}")

        success = success_count == len(slides)

        logger.info(f"Translation complete: {success_count}/{len(slides)} slides successful, total cost: ${total_cost:.4f}")

        return {
            'translated_slides': translated_slides,
            'target_language': target_language,
            'language_name': language_name,
            'total_cost': total_cost,
            'slide_count': len(slides),
            'success_count': success_count,
            'failed_slides': failed_slides,
            'text_direction': text_direction,
            'success': success,
            'errors': errors
        }

    def _build_translation_prompt(self, slide: SlideContent, target_language: str,
                                  language_name: str, preserve_technical_terms: bool) -> str:
        """
        Build the translation prompt for Claude API.

        Args:
            slide: SlideContent to translate
            target_language: ISO 639-1 language code
            language_name: Full language name
            preserve_technical_terms: Whether to preserve technical terms

        Returns:
            Formatted prompt string
        """
        # Format bullets for the prompt
        bullets_text = "\n".join([f"• {bullet}" for bullet in slide.content])

        # Build technical terms guidance
        tech_terms_guidance = ""
        if preserve_technical_terms:
            tech_terms_guidance = f"""

**Technical Terms Preservation:**
- Keep common technical terms in English when appropriate: {', '.join(list(TECHNICAL_TERMS_TO_PRESERVE)[:20])}...
- Preserve acronyms (API, REST, HTTP, JSON, etc.)
- Keep product names (GitHub, AWS, Docker, etc.)
- Preserve programming language names
- Only translate if there's a well-established translation in {language_name}"""

        # Build CJK-specific guidance
        cjk_guidance = ""
        if target_language in CJK_LANGUAGES:
            cjk_guidance = """

**CJK Language Guidance:**
- Use appropriate character set (Simplified Chinese for 'zh', Kanji/Hiragana for 'ja', Hangul for 'ko')
- Maintain professional/business tone
- Use proper spacing conventions for the language"""

        # Build RTL-specific guidance
        rtl_guidance = ""
        if target_language in RTL_LANGUAGES:
            rtl_guidance = """

**Right-to-Left Language Guidance:**
- Ensure proper RTL text flow
- Keep numbers and Latin text in LTR within RTL context
- Maintain professional formal register"""

        # Subheader handling
        subheader_section = ""
        if slide.subheader:
            subheader_section = f"""
Subheader: {slide.subheader}"""

        prompt = f"""Translate the following presentation slide content to {language_name}.

**Important Preservation Requirements:**
- Maintain professional/business tone
- Preserve the exact same bullet point structure (same number of bullets)
- Keep bullet points concise and similar in length to the original
- Use formal/professional language register
- Maintain the semantic meaning and intent
- Preserve emphasis and key concepts{tech_terms_guidance}{cjk_guidance}{rtl_guidance}

**Original Slide Content:**

Title: {slide.title}{subheader_section}

Bullets:
{bullets_text}

**Required Output Format:**
Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, no additional text):

{{
  "title": "Translated slide title",{f'''
  "subheader": "Translated subheader",''' if slide.subheader else ''}
  "bullets": [
    "Translated bullet point 1",
    "Translated bullet point 2",
    "Translated bullet point 3"
  ]
}}

Ensure:
1. The number of bullets in your translation EXACTLY matches the original ({len(slide.content)} bullets)
2. Each bullet is a complete, professional sentence or phrase
3. The JSON is valid and parseable
4. No text outside the JSON object

Translate now:"""

        return prompt

    def _parse_translation_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the JSON response from Claude API.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            # Try to extract JSON from response (handle cases where Claude adds extra text)
            response_text = response_text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Extract content between code blocks
                lines = response_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not line.strip().startswith('```')):
                        json_lines.append(line)
                response_text = '\n'.join(json_lines).strip()

            # Find JSON object bounds
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx + 1]
                translation_data = json.loads(json_text)

                # Validate required fields
                if 'title' not in translation_data or 'bullets' not in translation_data:
                    logger.error("Translation response missing required fields (title, bullets)")
                    return None

                if not isinstance(translation_data['bullets'], list):
                    logger.error("Translation response 'bullets' is not a list")
                    return None

                return translation_data
            else:
                logger.error("Could not find JSON object in translation response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse translation JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing translation response: {e}")
            return None

    def verify_translation_quality(self, original: SlideContent, translated: SlideContent) -> Dict[str, Any]:
        """
        Verify the quality of a translation.

        Args:
            original: Original SlideContent
            translated: Translated SlideContent

        Returns:
            {
                'valid': bool,
                'issues': List[str],
                'warnings': List[str]
            }
        """
        issues = []
        warnings = []

        # Check bullet count match
        if len(original.content) != len(translated.content):
            issues.append(
                f"Bullet count mismatch: original has {len(original.content)}, "
                f"translated has {len(translated.content)}"
            )

        # Check for empty content
        if not translated.title or not translated.title.strip():
            issues.append("Translated title is empty")

        for idx, bullet in enumerate(translated.content):
            if not bullet or not bullet.strip():
                issues.append(f"Translated bullet {idx} is empty")

        # Check if content appears to be actually translated
        if translated.title == original.title:
            warnings.append("Translated title is identical to original (may not be translated)")

        # Check for reasonable length differences
        original_avg_len = sum(len(b) for b in original.content) / max(len(original.content), 1)
        translated_avg_len = sum(len(b) for b in translated.content) / max(len(translated.content), 1)

        # Allow 50% variation in length (some languages are naturally more verbose/concise)
        if translated_avg_len < original_avg_len * 0.3 or translated_avg_len > original_avg_len * 2.5:
            warnings.append(
                f"Significant length difference: original avg {original_avg_len:.1f} chars, "
                f"translated avg {translated_avg_len:.1f} chars"
            )

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """
        Get all supported languages for translation.

        Returns:
            Dictionary mapping ISO 639-1 codes to language names
        """
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def validate_language_code(cls, language_code: str) -> bool:
        """
        Check if a language code is supported.

        Args:
            language_code: ISO 639-1 language code to validate

        Returns:
            True if supported, False otherwise
        """
        return language_code in SUPPORTED_LANGUAGES
