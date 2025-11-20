"""
Visual Generator Module

Handles AI-powered visual generation using OpenAI DALL-E 3 API.
Generates visual prompts and images for presentation slides based on content analysis.

Features:
- DALL-E 3 integration for high-quality image generation
- Content-aware visual prompt creation
- Smart caching to avoid regeneration
- Cost tracking and optimization
- Fallback to text descriptions when API unavailable
- Batch processing for efficiency
"""

import os
import logging
import hashlib
import requests
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# OpenAI for DALL-E image generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available - visual generation disabled")

# Anthropic Claude for AI-enhanced prompt generation
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.info("Anthropic library not available - using template-based prompts")

# Import data models and utilities
from .data_models import SlideContent
from .utils import CostTracker

logger = logging.getLogger(__name__)


class VisualGenerator:
    """
    Generates AI-powered visual prompts and images for presentation slides.

    Uses OpenAI DALL-E 3 to create contextually relevant visuals based on slide content.
    Includes intelligent caching, cost optimization, and fallback strategies.
    """

    # DALL-E 3 pricing (as of 2025)
    DALLE_PRICING = {
        'dall-e-3': {
            'standard-1024x1024': 0.040,  # $0.04 per image
            'standard-1024x1792': 0.080,  # $0.08 per image (portrait)
            'standard-1792x1024': 0.080,  # $0.08 per image (landscape)
            'hd-1024x1024': 0.080,        # $0.08 per HD image
            'hd-1024x1792': 0.120,        # $0.12 per HD image (portrait)
            'hd-1792x1024': 0.120,        # $0.12 per HD image (landscape)
        }
    }

    # Visual generation strategies by slide type
    VISUAL_STRATEGIES = {
        'technical': 'Clean, minimalist technical diagram or architecture visualization. Professional, modern style with subtle colors.',
        'data': 'Abstract data visualization or infographic style. Modern, colorful, engaging.',
        'concept': 'Creative metaphorical imagery that represents the concept. Artistic, inspiring.',
        'process': 'Clean flowchart or step-by-step visual. Organized, professional.',
        'executive': 'Professional business imagery. Clean, corporate, modern.',
        'educational': 'Clear, engaging educational visual. Friendly, approachable.',
        'default': 'Professional, clean, modern imagery related to the topic.'
    }

    def __init__(self, openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 cache_dir: str = '.visual_cache',
                 cost_tracker: Optional[CostTracker] = None,
                 max_parallel_workers: int = 5,
                 enable_ai_prompts: bool = True):
        """
        Initialize VisualGenerator with OpenAI and Anthropic API integration.

        Args:
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key for AI-enhanced prompts
            cache_dir: Directory for caching generated images
            cost_tracker: Optional CostTracker instance for cost tracking
            max_parallel_workers: Maximum number of concurrent image generation workers
            enable_ai_prompts: Use AI to generate better DALL-E prompts
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_parallel_workers = max_parallel_workers
        self.enable_ai_prompts = enable_ai_prompts

        # Initialize OpenAI client
        self.client = None
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("✅ DALL-E 3 visual generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        elif self.api_key and not OPENAI_AVAILABLE:
            logger.warning("OpenAI API key provided but library not installed - run: pip install openai")
        else:
            logger.info("No OpenAI API key - visual generation will use text descriptions only")

        # Initialize Anthropic client for AI-enhanced prompts
        self.anthropic_client = None
        if self.anthropic_api_key and ANTHROPIC_AVAILABLE and self.enable_ai_prompts:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                logger.info("✅ AI-enhanced visual prompt generation enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                self.anthropic_client = None
        elif not self.anthropic_api_key and self.enable_ai_prompts:
            logger.info("No Anthropic API key - using template-based prompts")

        # Cost tracking
        self.cost_tracker = cost_tracker or CostTracker()
        self.total_cost = 0.0
        self.images_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Visual prompt cache (in-memory)
        self._prompt_cache = OrderedDict()
        self._cache_max_size = 500

        # Image download cache (disk-based)
        self._image_cache_index = self._load_image_cache_index()

    def _load_image_cache_index(self) -> Dict[str, str]:
        """Load image cache index from disk"""
        index_path = self.cache_dir / 'cache_index.json'
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load image cache index: {e}")
        return {}

    def _save_image_cache_index(self):
        """Save image cache index to disk"""
        index_path = self.cache_dir / 'cache_index.json'
        try:
            with open(index_path, 'w') as f:
                json.dump(self._image_cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save image cache index: {e}")

    def _generate_cache_key(self, slide_title: str, slide_content: List[str],
                           visual_style: str) -> str:
        """Generate cache key from slide content"""
        cache_input = f"{slide_title}|{','.join(slide_content)}|{visual_style}".encode('utf-8')
        return hashlib.sha256(cache_input).hexdigest()

    def estimate_cost(self, num_slides: int, quality: str = 'standard',
                     size: str = '1024x1024') -> float:
        """
        Estimate cost for generating visuals for slides.

        Args:
            num_slides: Number of slides to generate visuals for
            quality: 'standard' or 'hd'
            size: '1024x1024', '1024x1792', or '1792x1024'

        Returns:
            Estimated cost in USD
        """
        key = f"{quality}-{size}"
        price_per_image = self.DALLE_PRICING['dall-e-3'].get(key, 0.040)
        return num_slides * price_per_image

    def analyze_slide_type(self, slide: SlideContent) -> str:
        """
        Analyze slide content to determine visual type.

        Args:
            slide: SlideContent object

        Returns:
            Visual type: 'technical', 'data', 'concept', 'process', 'executive', or 'educational'
        """
        title = slide.title.lower()
        content = ' '.join(slide.content).lower() if slide.content else ''

        # Technical indicators
        technical_keywords = ['architecture', 'system', 'api', 'database', 'infrastructure',
                             'deployment', 'framework', 'component', 'integration']
        if any(kw in title or kw in content for kw in technical_keywords):
            return 'technical'

        # Data indicators
        data_keywords = ['data', 'metrics', 'analytics', 'statistics', 'results', 'performance',
                        'analysis', 'chart', 'graph', 'trend']
        if any(kw in title or kw in content for kw in data_keywords):
            return 'data'

        # Process indicators
        process_keywords = ['process', 'workflow', 'steps', 'procedure', 'methodology',
                           'approach', 'strategy', 'pipeline', 'flow']
        if any(kw in title or kw in content for kw in process_keywords):
            return 'process'

        # Executive indicators
        executive_keywords = ['business', 'market', 'revenue', 'growth', 'strategy',
                             'leadership', 'executive', 'board', 'investment']
        if any(kw in title or kw in content for kw in executive_keywords):
            return 'executive'

        # Educational indicators
        educational_keywords = ['learning', 'education', 'training', 'course', 'lesson',
                               'tutorial', 'introduction', 'basics', 'overview']
        if any(kw in title or kw in content for kw in educational_keywords):
            return 'educational'

        # Default to concept for abstract topics
        return 'concept'

    def analyze_slides_batch(self, slides: List[SlideContent]) -> Dict[int, str]:
        """
        Analyze multiple slides concurrently to determine their visual types.
        Uses parallel processing for better performance on large slide decks.

        Args:
            slides: List of SlideContent objects

        Returns:
            Dict mapping slide index to visual type
        """
        def analyze_one(idx_slide):
            idx, slide = idx_slide
            return idx, self.analyze_slide_type(slide)

        if len(slides) <= 3:
            # For small numbers, sequential is faster (no threading overhead)
            return {i: self.analyze_slide_type(slide) for i, slide in enumerate(slides)}

        # Parallel analysis for larger slide sets
        results = {}
        with ThreadPoolExecutor(max_workers=min(10, len(slides))) as executor:
            futures = {executor.submit(analyze_one, (i, slide)): i
                      for i, slide in enumerate(slides)}

            for future in as_completed(futures):
                idx, slide_type = future.result()
                results[idx] = slide_type

        return results

    def create_visual_prompt_ai(self, slide: SlideContent) -> str:
        """
        Use AI (Claude) to create an optimized DALL-E prompt based on slide content.

        Args:
            slide: SlideContent object

        Returns:
            AI-generated DALL-E prompt string
        """
        if not self.anthropic_client:
            # Fallback to template-based approach
            return self.create_visual_prompt(slide)

        # Build context for AI
        title = slide.title
        content_text = '\n'.join(f"• {pt}" for pt in (slide.content[:5] if slide.content else []))

        # Check prompt cache
        cache_key = f"ai_prompt_{hashlib.sha256(f'{title}{content_text}'.encode()).hexdigest()}"
        if cache_key in self._prompt_cache:
            logger.debug(f"AI prompt cache hit for '{title}'")
            return self._prompt_cache[cache_key]

        try:
            prompt = f"""You are an expert at creating DALL-E image prompts for presentation slides.

Slide Title: {title}

Slide Content:
{content_text if content_text else '(No bullet points)'}

Task: Create a concise, highly effective DALL-E 3 prompt (max 200 characters) that will generate a professional, visually striking image for this slide.

Requirements:
1. The image should visually represent the slide's core concept
2. Professional, modern style suitable for business presentations
3. No text, labels, or words in the image
4. Clear, uncluttered composition
5. Appropriate for diverse audiences
6. Metaphorical or conceptual representation (not literal)

Output ONLY the DALL-E prompt, nothing else."""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )

            ai_prompt = response.content[0].text.strip()

            # Track cost (approximate)
            input_tokens = len(prompt.split()) * 1.3  # rough estimate
            output_tokens = len(ai_prompt.split()) * 1.3
            self.cost_tracker.track_api_call(
                provider='anthropic',
                model='claude-3-5-sonnet-20241022',
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cached=False,
                slide_id=title,
                call_type='visual_prompt_generation'
            )

            # Cache the result
            self._prompt_cache[cache_key] = ai_prompt
            if len(self._prompt_cache) > self._cache_max_size:
                self._prompt_cache.popitem(last=False)  # Remove oldest

            logger.info(f"✨ AI-generated prompt for '{title}': {ai_prompt[:80]}...")
            return ai_prompt

        except Exception as e:
            logger.warning(f"AI prompt generation failed, using template: {e}")
            return self.create_visual_prompt(slide)

    def create_visual_prompt(self, slide: SlideContent, visual_type: Optional[str] = None) -> str:
        """
        Create a DALL-E prompt based on slide content (template-based).

        Args:
            slide: SlideContent object
            visual_type: Optional override for visual type

        Returns:
            DALL-E prompt string
        """
        # Determine visual type
        vtype = visual_type or self.analyze_slide_type(slide)
        strategy = self.VISUAL_STRATEGIES.get(vtype, self.VISUAL_STRATEGIES['default'])

        # Extract key concepts from title and content
        title = slide.title
        key_points = slide.content[:3] if slide.content else []  # Top 3 bullet points

        # Build prompt
        prompt_parts = [
            f"Create a professional presentation visual for a slide titled '{title}'.",
        ]

        if key_points:
            key_concepts = ', '.join([pt[:50] for pt in key_points])  # Truncate long points
            prompt_parts.append(f"Key concepts: {key_concepts}.")

        prompt_parts.append(strategy)
        prompt_parts.append("No text or labels in the image. Suitable for a presentation background or visual element.")

        return ' '.join(prompt_parts)

    def generate_visual_prompt_text(self, slide: SlideContent) -> str:
        """
        Generate text-based visual description (fallback when DALL-E unavailable).

        Args:
            slide: SlideContent object

        Returns:
            Text description of recommended visual
        """
        visual_type = self.analyze_slide_type(slide)

        descriptions = {
            'technical': f"📐 Suggested visual: Technical diagram or architecture schematic for '{slide.title}'",
            'data': f"📊 Suggested visual: Data visualization or infographic for '{slide.title}'",
            'concept': f"💡 Suggested visual: Conceptual illustration representing '{slide.title}'",
            'process': f"🔄 Suggested visual: Process flowchart or step diagram for '{slide.title}'",
            'executive': f"💼 Suggested visual: Professional business imagery for '{slide.title}'",
            'educational': f"📚 Suggested visual: Educational graphic or learning visual for '{slide.title}'",
            'default': f"🎨 Suggested visual: Professional imagery related to '{slide.title}'"
        }

        return descriptions.get(visual_type, descriptions['default'])

    def generate_image(self, slide: SlideContent,
                      quality: str = 'standard',
                      size: str = '1024x1024',
                      style: str = 'natural',
                      use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Generate image using DALL-E 3.

        Args:
            slide: SlideContent object
            quality: 'standard' or 'hd'
            size: '1024x1024', '1024x1792', or '1792x1024'
            style: 'natural' or 'vivid'
            use_cache: Whether to use cached images

        Returns:
            Dict with 'url', 'prompt', 'cost', 'cached' keys, or None if generation failed
        """
        if not self.client:
            logger.warning("OpenAI client not available - returning text description only")
            return {
                'url': None,
                'prompt': self.generate_visual_prompt_text(slide),
                'cost': 0.0,
                'cached': False,
                'text_only': True
            }

        # Analyze slide type for cache key
        visual_type = self.analyze_slide_type(slide)

        # Create visual prompt (use AI if enabled)
        if self.enable_ai_prompts and self.anthropic_client:
            prompt = self.create_visual_prompt_ai(slide)
        else:
            prompt = self.create_visual_prompt(slide, visual_type)

        # Check cache
        cache_key = self._generate_cache_key(slide.title, slide.content or [], visual_type)

        if use_cache and cache_key in self._image_cache_index:
            self.cache_hits += 1
            cached_path = self.cache_dir / self._image_cache_index[cache_key]
            if cached_path.exists():
                logger.info(f"✅ Cache hit for slide '{slide.title}' (saved ${self.DALLE_PRICING['dall-e-3'][f'{quality}-{size}']:.3f})")
                return {
                    'url': str(cached_path),
                    'prompt': prompt,
                    'cost': 0.0,
                    'cached': True,
                    'local_path': str(cached_path)
                }

        # Generate new image
        self.cache_misses += 1
        try:
            logger.info(f"🎨 Generating DALL-E 3 image for slide '{slide.title}'...")
            logger.info(f"   Prompt: {prompt[:100]}...")

            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )

            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt

            # Calculate cost
            cost = self.DALLE_PRICING['dall-e-3'][f'{quality}-{size}']
            self.total_cost += cost
            self.images_generated += 1

            # Track cost
            self.cost_tracker.track_api_call(
                provider='openai',
                model='dall-e-3',
                input_tokens=0,  # DALL-E doesn't use tokens
                output_tokens=0,
                cached=False,
                slide_id=slide.title,
                call_type='image_generation'
            )

            logger.info(f"✅ Image generated successfully (cost: ${cost:.3f})")

            # Download and cache image
            local_path = None
            if use_cache:
                try:
                    image_filename = f"{cache_key}.png"
                    local_path = self.cache_dir / image_filename

                    # Download image
                    response_img = requests.get(image_url, timeout=30)
                    response_img.raise_for_status()

                    with open(local_path, 'wb') as f:
                        f.write(response_img.content)

                    # Update cache index
                    self._image_cache_index[cache_key] = image_filename
                    self._save_image_cache_index()

                    logger.info(f"💾 Image cached to {local_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache image: {e}")

            return {
                'url': image_url,
                'prompt': revised_prompt or prompt,
                'cost': cost,
                'cached': False,
                'local_path': str(local_path) if local_path else None
            }

        except Exception as e:
            logger.error(f"DALL-E 3 image generation failed: {e}")
            return {
                'url': None,
                'prompt': self.generate_visual_prompt_text(slide),
                'cost': 0.0,
                'cached': False,
                'error': str(e),
                'text_only': True
            }

    def _select_slides_intelligently(self, slides: List[SlideContent],
                                    max_slides: Optional[int] = None) -> List[Tuple[int, SlideContent]]:
        """
        Use AI to intelligently select which slides would benefit most from visuals.

        Args:
            slides: List of all SlideContent objects
            max_slides: Maximum number of slides to select

        Returns:
            List of (index, slide) tuples for selected slides
        """
        if not self.anthropic_client:
            # Fallback to key_slides strategy
            logger.info("AI not available, using key_slides fallback")
            selected = []
            for i, slide in enumerate(slides):
                if (slide.slide_type in ['title', 'section_title', 'subsection_title'] or
                    slide.heading_level in [1, 2, 3]):
                    selected.append((i, slide))
            return selected[:max_slides] if max_slides else selected

        try:
            # Build concise slide summary for AI analysis
            slide_summaries = []
            for i, slide in enumerate(slides[:50]):  # Limit to first 50 to avoid token limits
                content_preview = ', '.join(slide.content[:2]) if slide.content else '(no content)'
                slide_summaries.append(f"{i}. {slide.title} | {content_preview[:80]}")

            slides_text = '\n'.join(slide_summaries)

            prompt = f"""Analyze these {len(slide_summaries)} presentation slides and identify which ones would benefit MOST from AI-generated visuals.

Slides:
{slides_text}

Task: Select the {max_slides or 'top 10'} slides where a visual image would add the most value.

Prioritize slides that:
1. Introduce key concepts or sections (title slides, section headers)
2. Present complex ideas that visuals can clarify
3. Discuss processes, architectures, or systems
4. Would benefit from metaphorical or conceptual imagery
5. Are important milestones in the presentation flow

Avoid selecting slides that:
- Are purely textual lists with no visual concept
- Contain specific numbers/data (better as charts, not images)
- Are transitions or meta-slides

Output ONLY a JSON array of slide numbers (integers), nothing else. Example: [0, 3, 7, 12, 18]"""

            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse AI response
            ai_response = response.content[0].text.strip()
            # Extract JSON array
            import re
            json_match = re.search(r'\[[\d,\s]+\]', ai_response)
            if json_match:
                selected_indices = json.loads(json_match.group(0))
                selected_slides = [(i, slides[i]) for i in selected_indices if i < len(slides)]

                logger.info(f"🤖 AI selected {len(selected_slides)} slides for visuals: {selected_indices}")

                # Track cost
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(ai_response.split()) * 1.3
                self.cost_tracker.track_api_call(
                    provider='anthropic',
                    model='claude-3-5-sonnet-20241022',
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    cached=False,
                    call_type='intelligent_slide_selection'
                )

                return selected_slides[:max_slides] if max_slides else selected_slides
            else:
                raise ValueError(f"Could not parse AI response: {ai_response}")

        except Exception as e:
            logger.warning(f"Intelligent slide selection failed: {e}, using fallback")
            # Fallback to key_slides
            selected = []
            for i, slide in enumerate(slides):
                if (slide.slide_type in ['title', 'section_title', 'subsection_title'] or
                    slide.heading_level in [1, 2, 3]):
                    selected.append((i, slide))
            return selected[:max_slides] if max_slides else selected

    def _generate_image_task(self, slide_index: int, slide: SlideContent,
                            quality: str, size: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Worker function for parallel image generation.

        Args:
            slide_index: Index of slide in original list
            slide: SlideContent object
            quality: Image quality
            size: Image size

        Returns:
            Tuple of (slide_index, result_dict)
        """
        try:
            result = self.generate_image(slide, quality=quality, size=size)
            return (slide_index, result)
        except Exception as e:
            logger.error(f"Error generating image for slide {slide_index} ('{slide.title}'): {e}")
            return (slide_index, None)

    def generate_visuals_batch_parallel(self, slides: List[SlideContent],
                                       filter_strategy: str = 'key_slides',
                                       quality: str = 'standard',
                                       size: str = '1024x1024',
                                       max_slides: Optional[int] = None,
                                       use_parallel: bool = True) -> Dict[str, Any]:
        """
        Generate visuals for multiple slides with parallel processing and smart filtering.

        Args:
            slides: List of SlideContent objects
            filter_strategy: 'all', 'key_slides', 'smart', or 'none'
                - 'all': Generate for all slides
                - 'key_slides': Only section titles and important content slides
                - 'smart': Use AI to determine which slides need visuals
                - 'none': No visual generation
            quality: 'standard' or 'hd'
            size: Image size
            max_slides: Maximum number of slides to generate visuals for
            use_parallel: Enable parallel processing (default: True)

        Returns:
            Dict with 'visuals' (dict mapping slide index to visual data) and 'summary' (cost summary)
        """
        if filter_strategy == 'none':
            return {'visuals': {}, 'summary': {'total_cost': 0.0, 'images_generated': 0}}

        # Filter slides based on strategy
        slides_to_process = []

        if filter_strategy == 'smart':
            # Use AI to select slides (implemented below)
            slides_to_process = self._select_slides_intelligently(slides, max_slides)
        elif filter_strategy == 'key_slides':
            # Only process title slides and section headers
            for i, slide in enumerate(slides):
                if (slide.slide_type in ['title', 'section_title', 'subsection_title'] or
                    slide.heading_level in [1, 2, 3]):
                    slides_to_process.append((i, slide))
        else:  # 'all'
            slides_to_process = list(enumerate(slides))

        # Limit to max_slides if specified
        if max_slides:
            slides_to_process = slides_to_process[:max_slides]

        # Estimate cost
        estimated_cost = self.estimate_cost(len(slides_to_process), quality, size)
        logger.info(f"📊 Generating visuals for {len(slides_to_process)}/{len(slides)} slides")
        logger.info(f"💰 Estimated cost: ${estimated_cost:.2f}")

        if use_parallel:
            logger.info(f"⚡ Using parallel processing with {self.max_parallel_workers} workers")

        # Generate visuals
        visuals = {}
        total_cost = 0.0
        start_time = time.time()

        if use_parallel and len(slides_to_process) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                # Submit all tasks
                future_to_slide = {
                    executor.submit(self._generate_image_task, i, slide, quality, size): (i, slide)
                    for i, slide in slides_to_process
                }

                # Process results as they complete
                completed = 0
                for future in as_completed(future_to_slide):
                    slide_index, result = future.result()
                    if result:
                        visuals[slide_index] = result
                        total_cost += result.get('cost', 0.0)

                    completed += 1
                    logger.info(f"   Progress: {completed}/{len(slides_to_process)} slides processed")
        else:
            # Sequential processing (fallback or single slide)
            for i, slide in slides_to_process:
                result = self.generate_image(slide, quality=quality, size=size)
                if result:
                    visuals[i] = result
                    total_cost += result.get('cost', 0.0)

        elapsed_time = time.time() - start_time

        summary = {
            'total_cost': total_cost,
            'estimated_cost': estimated_cost,
            'images_generated': len([v for v in visuals.values() if not v.get('cached', False)]),
            'cache_hits': len([v for v in visuals.values() if v.get('cached', False)]),
            'slides_processed': len(slides_to_process),
            'total_slides': len(slides),
            'processing_time': elapsed_time,
            'parallel_mode': use_parallel and len(slides_to_process) > 1
        }

        logger.info(f"✅ Visual generation complete in {elapsed_time:.1f}s:")
        logger.info(f"   Images generated: {summary['images_generated']}")
        logger.info(f"   Cache hits: {summary['cache_hits']}")
        logger.info(f"   Actual cost: ${summary['total_cost']:.2f}")
        logger.info(f"   Processing mode: {'Parallel' if summary['parallel_mode'] else 'Sequential'}")

        return {
            'visuals': visuals,
            'summary': summary
        }

    def generate_visuals_batch(self, slides: List[SlideContent],
                              filter_strategy: str = 'key_slides',
                              quality: str = 'standard',
                              size: str = '1024x1024',
                              max_slides: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate visuals for multiple slides with smart filtering.
        This method now uses parallel processing by default.

        Args:
            slides: List of SlideContent objects
            filter_strategy: 'all', 'key_slides', 'smart', or 'none'
                - 'all': Generate for all slides
                - 'key_slides': Only section titles and important content slides
                - 'smart': Use AI to determine which slides need visuals
                - 'none': No visual generation
            quality: 'standard' or 'hd'
            size: Image size
            max_slides: Maximum number of slides to generate visuals for

        Returns:
            Dict with 'visuals' (dict mapping slide index to visual data) and 'summary' (cost summary)
        """
        # Call the new parallel implementation
        return self.generate_visuals_batch_parallel(
            slides=slides,
            filter_strategy=filter_strategy,
            quality=quality,
            size=size,
            max_slides=max_slides,
            use_parallel=True
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get visual generation statistics.

        Returns:
            Dict with cost and performance statistics
        """
        cache_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0

        return {
            'total_cost': self.total_cost,
            'images_generated': self.images_generated,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{cache_rate:.1f}%",
            'api_available': self.client is not None,
            'cache_dir': str(self.cache_dir),
            'cached_images': len(self._image_cache_index)
        }

    def clear_cache(self):
        """Clear all cached images and index"""
        try:
            # Remove cached image files
            for file in self.cache_dir.glob('*.png'):
                file.unlink()

            # Clear cache index
            index_path = self.cache_dir / 'cache_index.json'
            if index_path.exists():
                index_path.unlink()

            self._image_cache_index = {}
            logger.info("🗑️  Visual cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
