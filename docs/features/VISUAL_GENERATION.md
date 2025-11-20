# AI Visual Generation with DALL-E 3

## Overview

The slide generator now supports **AI-powered visual generation** using OpenAI's DALL-E 3 API. This feature automatically creates contextually relevant, professional images for your presentation slides based on their content.

## Features

### Intelligent Visual Type Detection
The system automatically analyzes each slide's content and determines the most appropriate visual style:

- **Technical Slides** → Clean diagrams, architecture visualizations
- **Data Slides** → Abstract data visualizations, infographics
- **Concept Slides** → Metaphorical imagery representing abstract ideas
- **Process Slides** → Flowcharts, step-by-step visuals
- **Executive Slides** → Professional business imagery
- **Educational Slides** → Clear, engaging educational visuals

### Cost Optimization

#### Pricing (as of 2025)
- **Standard Quality (1024x1024)**: $0.04 per image
- **HD Quality (1024x1024)**: $0.08 per image
- **HD Quality (1024x1792 or 1792x1024)**: $0.12 per image

#### Smart Filtering Strategies
To keep costs manageable, you can choose which slides get visuals:

1. **Key Slides Only** (Recommended)
   - Only generates visuals for title slides and section headers
   - Typical cost: $0.12 - $0.40 per document (3-10 key slides)
   - Best balance of visual impact and cost efficiency

2. **All Slides**
   - Generates visuals for every slide
   - Typical cost: $0.40 - $2.00 per document (10-50 slides)
   - Maximum visual richness

3. **None**
   - Disables visual generation
   - Falls back to text-based visual prompts

### Intelligent Caching

The visual generator includes a robust caching system:

- **Disk-based cache**: Stores generated images locally
- **Cache hit detection**: Identical content reuses cached images
- **Cost savings**: Eliminates redundant API calls
- **Persistent storage**: Cache survives application restarts

Example savings:
- First generation: 10 images = $0.40
- Regeneration with cache: 0 new images = $0.00 (10 cache hits)

## Usage

### 1. Web Interface

Enable visual generation in the UI:

```
☑️ Enable AI Visual Generation

Which slides should get visuals?
[Key Slides Only ▼]

Image Quality:
◉ Standard ($0.04/image)
○ HD ($0.08/image)
```

**Requirements:**
- OpenAI API key must be provided
- Key is used for both DALL-E 3 (images) and GPT models (bullets)

### 2. Python API

```python
from slide_generator_pkg import DocumentParser, SlideGenerator

# Initialize parser with visual generation enabled
parser = DocumentParser(
    openai_api_key="sk-...",
    enable_visual_generation=True,
    visual_filter='key_slides'  # 'all', 'key_slides', or 'none'
)

# Parse document (visuals generated automatically)
doc_structure = parser.parse_file('document.txt', 'document.txt')

# Check visual generation results
if 'visual_generation' in doc_structure.metadata:
    stats = doc_structure.metadata['visual_generation']
    print(f"Images generated: {stats['images_generated']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Total cost: ${stats['total_cost']:.2f}")

# Generate PowerPoint with AI images
generator = SlideGenerator()
pptx_path = generator.create_powerpoint(doc_structure)

# Images are automatically inserted into slides
```

### 3. Standalone Visual Generator

For advanced use cases, you can use the VisualGenerator directly:

```python
from slide_generator_pkg import VisualGenerator, SlideContent

# Initialize visual generator
visual_gen = VisualGenerator(
    openai_api_key="sk-...",
    cache_dir='.visual_cache'
)

# Create a slide
slide = SlideContent(
    title="Machine Learning Architecture",
    content=[
        "Data ingestion pipeline",
        "Feature engineering layer",
        "Model training and evaluation",
        "Deployment and monitoring"
    ]
)

# Generate image for single slide
result = visual_gen.generate_image(
    slide=slide,
    quality='standard',  # or 'hd'
    size='1024x1024'
)

if result and result['url']:
    print(f"Image URL: {result['url']}")
    print(f"Local path: {result['local_path']}")
    print(f"Cost: ${result['cost']:.3f}")
    print(f"Cached: {result['cached']}")

# Batch generation for multiple slides
slides = [slide1, slide2, slide3]
batch_result = visual_gen.generate_visuals_batch(
    slides=slides,
    filter_strategy='key_slides',
    quality='standard',
    max_slides=10
)

print(f"Generated {batch_result['summary']['images_generated']} images")
print(f"Total cost: ${batch_result['summary']['total_cost']:.2f}")
```

## Architecture

### Visual Generation Pipeline

```
Document Parsing
    ↓
Content Analysis → Slide Type Detection
    ↓
Visual Prompt Generation
    ↓
Cache Check (SHA256 hash of content + style)
    ↓
DALL-E 3 API Call (if cache miss)
    ↓
Image Download & Local Storage
    ↓
PowerPoint Integration (Image Insertion)
```

### Visual Prompt Strategy

Each slide type has an optimized prompt strategy:

**Technical Slide Example:**
```
Input:
  Title: "Microservices Architecture"
  Content: ["Service discovery", "Load balancing", "API gateway"]

Generated Prompt:
  "Create a professional presentation visual for a slide titled
   'Microservices Architecture'. Key concepts: Service discovery,
   Load balancing, API gateway. Clean, minimalist technical diagram
   or architecture visualization. Professional, modern style with
   subtle colors. No text or labels in the image. Suitable for a
   presentation background or visual element."
```

### Data Models

Visual fields added to `SlideContent`:

```python
@dataclass
class SlideContent:
    title: str
    content: List[str]
    # ... existing fields ...

    # Visual generation fields
    visual_prompt: Optional[str] = None        # DALL-E prompt
    visual_image_url: Optional[str] = None     # Remote image URL
    visual_image_path: Optional[str] = None    # Local cached path
    visual_type: Optional[str] = None          # Visual type category
```

## Cost Management

### Typical Costs by Document Type

| Document Type | Slides | Visual Strategy | Cost Range |
|--------------|--------|----------------|------------|
| **Short presentation** | 10-15 | Key slides only | $0.12 - $0.24 |
| **Medium presentation** | 20-30 | Key slides only | $0.24 - $0.40 |
| **Long presentation** | 40-60 | Key slides only | $0.40 - $0.80 |
| **Technical documentation** | 30-50 | All slides | $1.20 - $2.00 |
| **Executive deck** | 15-20 | All slides (HD) | $1.20 - $1.60 |

### Cost Estimation

Before generation, the system provides accurate cost estimates:

```python
# Estimate cost for 20 slides
visual_gen = VisualGenerator(openai_api_key="sk-...")
estimated_cost = visual_gen.estimate_cost(
    num_slides=20,
    quality='standard',
    size='1024x1024'
)
print(f"Estimated: ${estimated_cost:.2f}")  # $0.80
```

### Budget Controls

```python
# Limit maximum slides to control costs
parser = DocumentParser(
    openai_api_key="sk-...",
    enable_visual_generation=True,
    visual_filter='key_slides'
)

# Or manually control in batch generation
visual_gen.generate_visuals_batch(
    slides=all_slides,
    filter_strategy='key_slides',
    max_slides=10  # Hard limit: maximum 10 images
)
```

## Cache Management

### Cache Directory Structure

```
.visual_cache/
├── cache_index.json          # Mapping of content hashes to files
├── a3f2b1c9d8e7f6a5.png      # Cached image 1
├── b4e3c2d1f0a9b8c7.png      # Cached image 2
└── ...
```

### Cache Operations

```python
visual_gen = VisualGenerator(openai_api_key="sk-...")

# Get cache statistics
stats = visual_gen.get_statistics()
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"Cached images: {stats['cached_images']}")

# Clear cache (forces regeneration)
visual_gen.clear_cache()
```

### Cache Key Generation

Cache keys are SHA256 hashes of:
- Slide title
- Slide content (all bullet points)
- Visual type/style

This ensures:
- Same content = same image (cost savings)
- Different content = new image
- No collision risk

## Error Handling

The system gracefully handles various failure scenarios:

### No OpenAI API Key
```python
# Result: Text-based visual descriptions instead of images
{
    'url': None,
    'prompt': '📐 Suggested visual: Technical diagram for "Architecture"',
    'text_only': True
}
```

### API Rate Limit
```python
# Automatic retry with exponential backoff
# Falls back to text description if retries exhausted
```

### Image Download Failure
```python
# Returns URL without local cache
# PowerPoint generator attempts to insert from URL
```

## Integration with PowerPoint

### Image Placement

Generated images are automatically inserted into slides:

```
┌─────────────────────────────────────┐
│  Slide Title                        │
├─────────────────┬───────────────────┤
│                 │                   │
│  • Bullet 1     │                   │
│  • Bullet 2     │   [AI Image]      │
│  • Bullet 3     │                   │
│                 │                   │
├─────────────────┴───────────────────┤
│  🎨 Visual prompt caption (small)   │
└─────────────────────────────────────┘
```

**Layout Details:**
- Left column (4.5"): Bullet points
- Right column (4.3"): AI-generated image
- Bottom: Small caption with visual prompt (optional)

### Fallback Behavior

If visual generation is disabled or fails:
- Text-based visual prompts are shown instead
- Users can copy prompts to external image generators
- No disruption to slide generation process

## Best Practices

### 1. Start with Key Slides Only
- Most cost-effective approach
- Provides visual impact on important slides
- Typically 20-30% of total slides

### 2. Use Standard Quality
- 1024x1024 standard quality is sufficient for most presentations
- HD quality only needed for high-resolution displays or print

### 3. Review Cache Regularly
- Cache can grow large over time
- Periodically clear unused cached images
- Monitor cache directory size

### 4. Test Before Bulk Generation
- Generate a small test document first
- Verify image quality meets your needs
- Adjust settings as needed

### 5. Combine with Text Prompts
- Even with AI images, text prompts are valuable
- Provide flexibility for manual image creation
- Useful for custom brand imagery

## Troubleshooting

### Issue: Images Not Generating

**Check:**
1. OpenAI API key is provided and valid
2. `enable_visual_generation=True` in parser initialization
3. `visual_filter` is not set to 'none'
4. Sufficient API quota/credits in OpenAI account

**Solution:**
```python
# Verify configuration
parser = DocumentParser(
    openai_api_key="sk-...",  # Valid key
    enable_visual_generation=True,  # Enabled
    visual_filter='key_slides'  # Not 'none'
)
```

### Issue: High Costs

**Check:**
1. `visual_filter` is set to 'all' instead of 'key_slides'
2. Many slides in document
3. HD quality instead of standard

**Solution:**
```python
# Cost-optimized settings
parser = DocumentParser(
    openai_api_key="sk-...",
    enable_visual_generation=True,
    visual_filter='key_slides',  # Not 'all'
)

# Use standard quality
visual_gen.generate_image(slide, quality='standard')
```

### Issue: Cache Not Working

**Check:**
1. Cache directory permissions
2. Disk space availability
3. Cache index file corruption

**Solution:**
```python
# Clear and rebuild cache
visual_gen.clear_cache()

# Verify cache directory
import os
cache_dir = visual_gen.cache_dir
print(f"Cache dir: {cache_dir}")
print(f"Exists: {cache_dir.exists()}")
print(f"Writable: {os.access(cache_dir, os.W_OK)}")
```

## Future Enhancements

Planned improvements:

1. **Multi-image support**: Generate multiple image options per slide
2. **Style customization**: Custom visual styles (minimalist, bold, corporate, etc.)
3. **Aspect ratio options**: Portrait/landscape for different slide layouts
4. **Image editing**: Basic cropping, filters, overlays
5. **Brand integration**: Logo placement, color scheme matching
6. **Stable Diffusion support**: Alternative to DALL-E 3 for cost savings
7. **Local models**: Offline image generation with open-source models

## Support

For issues or questions:
- Check existing documentation in `/docs`
- Review test cases in `/tests/test_visual_generation.py`
- Run demo script: `python demo_visual_generation.py`

## License

Visual generation feature is part of the slide generator package and follows the same license terms.
