"""
Visual Enhancements for Slide Generator

Provides AI-powered visual markers (emojis and icons) for bullet points
to enhance visual scanning and comprehension.
"""

import logging
import json
from typing import List, Dict, Optional
import anthropic

logger = logging.getLogger(__name__)


class VisualEnhancements:
    """Handles AI-powered visual enhancements for slide content"""

    def __init__(self, client: anthropic.Anthropic):
        """
        Initialize VisualEnhancements with Anthropic client.

        Args:
            client: Configured Anthropic client for Claude API calls
        """
        self.client = client
        self.total_cost = 0.0

    def suggest_visual_markers(self, bullets: List[str], slide_title: str = "") -> dict:
        """
        Suggest icons/emojis for each bullet point using Claude API.

        Args:
            bullets: List of bullet point texts
            slide_title: Optional slide title for additional context

        Returns:
            {
                'markers': dict,  # {index: {'emoji': str, 'icon_name': str, 'category': str}}
                'cost': float,
                'confidence': float
            }
        """
        if not bullets:
            return {'markers': {}, 'cost': 0.0, 'confidence': 1.0}

        try:
            # Build the prompt
            prompt = self._build_marker_prompt(bullets, slide_title)

            # Call Claude API
            logger.info(f"Requesting visual markers for {len(bullets)} bullets...")
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent icon suggestions
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Calculate cost (Claude 3.5 Sonnet pricing)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)
            self.total_cost += cost

            # Extract and parse the response
            response_text = response.content[0].text.strip()
            markers = self._parse_marker_response(response_text, len(bullets))

            # Calculate confidence based on coverage
            confidence = len(markers) / len(bullets) if bullets else 1.0

            logger.info(f"✅ Generated {len(markers)} visual markers (cost: ${cost:.4f})")

            return {
                'markers': markers,
                'cost': cost,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Failed to generate visual markers: {e}")
            # Graceful degradation - return empty markers
            return {'markers': {}, 'cost': 0.0, 'confidence': 0.0}

    def _build_marker_prompt(self, bullets: List[str], slide_title: str = "") -> str:
        """
        Build the prompt for visual marker generation.

        Args:
            bullets: List of bullet point texts
            slide_title: Optional slide title for context

        Returns:
            Formatted prompt string
        """
        # Format bullets for the prompt
        bullet_list = "\n".join([f"{i}. {bullet}" for i, bullet in enumerate(bullets)])

        title_context = f"\n\nSlide Title: {slide_title}" if slide_title else ""

        prompt = f"""For each bullet point below, suggest a relevant emoji and icon description that enhances visual comprehension and quick scanning.{title_context}

Bullets:
{bullet_list}

For each bullet point, provide:
1. **emoji**: A single, highly relevant emoji character
2. **icon_name**: A professional icon name (e.g., "dollar-sign", "shield-check", "chart-line")
3. **category**: The semantic category (e.g., "finance", "security", "performance", "education", "technology")

Guidelines:
- Choose emojis that are universally recognizable and professional
- Icons should match common icon libraries (FontAwesome, Material Icons)
- Categories help maintain visual consistency across slides
- Avoid obscure or ambiguous emojis
- Ensure each suggestion genuinely enhances understanding

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "0": {{"emoji": "💰", "icon_name": "dollar-sign", "category": "finance"}},
  "1": {{"emoji": "⚡", "icon_name": "lightning", "category": "performance"}},
  "2": {{"emoji": "🔒", "icon_name": "shield-check", "category": "security"}}
}}

Return JSON now:"""

        return prompt

    def _parse_marker_response(self, response_text: str, expected_count: int) -> Dict[int, dict]:
        """
        Parse the JSON response from Claude into structured markers.

        Args:
            response_text: Raw response text from Claude
            expected_count: Expected number of markers

        Returns:
            Dictionary mapping bullet indices to marker data
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where Claude might add markdown code blocks
            response_text = response_text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]

            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            markers_dict = json.loads(response_text)

            # Convert string keys to integers and validate structure
            markers = {}
            for key, value in markers_dict.items():
                try:
                    idx = int(key)
                    # Validate required fields
                    if all(field in value for field in ['emoji', 'icon_name', 'category']):
                        markers[idx] = {
                            'emoji': value['emoji'].strip(),
                            'icon_name': value['icon_name'].strip(),
                            'category': value['category'].strip()
                        }
                    else:
                        logger.warning(f"Marker at index {idx} missing required fields: {value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid marker key or value: {key} -> {value}: {e}")

            logger.info(f"Parsed {len(markers)}/{expected_count} visual markers from response")
            return markers

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error parsing markers: {e}")
            return {}

    def get_total_cost(self) -> float:
        """Get total cost of all visual marker generation calls"""
        return self.total_cost

    def reset_cost(self):
        """Reset the cost tracker"""
        self.total_cost = 0.0
