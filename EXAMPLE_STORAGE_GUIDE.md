# Example Storage System - Few-Shot Learning Guide

## Overview

The Example Storage System enables the slide generator to **learn from past successful outputs** through few-shot learning. As users generate slides, they can save high-quality examples that the system will use to improve future generations.

## Key Features

### 1. **Dynamic Few-Shot Learning**
- Claude API prompts automatically include relevant examples from the database
- Examples are selected based on semantic similarity to the current content
- Filters by style (professional, educational, technical, executive) and content type

### 2. **Example Storage**
- JSON-based storage (easy to version control and deploy)
- Organized by style category for fast retrieval
- Includes quality scores and user ratings

### 3. **Admin Interface**
- Web UI at `/admin/examples` for managing examples
- View statistics, filter examples, rate quality
- Star rating system (1-5 stars) for subjective quality

### 4. **API Endpoints**
- `POST /api/examples/save` - Save new example
- `GET /api/examples/list` - List examples with filters
- `GET /api/examples/stats` - Get storage statistics
- `POST /api/examples/rate` - Rate example quality

## Quick Start

### 1. Seed Initial Examples

The database is pre-seeded with 21 high-quality examples from the golden test set:

```bash
python seed_examples.py
```

Output:
```
🌱 Seeding example database from golden test set...
Found 21 golden examples

✅ Seeding complete!
   Total examples: 21
   By style:
     - professional: 4
     - educational: 2
     - technical: 2
     - executive: 2
   Average quality score: 85.0
```

### 2. View Examples in Admin Interface

Navigate to: `http://localhost:5001/admin/examples`

You'll see:
- **Statistics Dashboard**: Total examples, avg quality score, counts by style
- **Filters**: Filter by style, content type, or limit results
- **Example Cards**: View input text, generated bullets, metadata
- **Star Ratings**: Rate examples 1-5 stars

### 3. Save Examples via API

When generating slides, save good examples:

```python
import requests

example_data = {
    "input_text": "Your original document content here...",
    "generated_bullets": [
        "First generated bullet point",
        "Second generated bullet point",
        "Third generated bullet point"
    ],
    "context_heading": "Slide Title/Heading",
    "style": "professional",  # or educational, technical, executive
    "content_type": "paragraph",  # or table, list, heading
    "user_rating": 5,  # Optional: 1-5 stars
    "category_tags": ["business", "strategy"]  # Optional
}

response = requests.post(
    "http://localhost:5001/api/examples/save",
    json=example_data
)

print(response.json())
# {'success': True, 'example_id': 'professional_abc123', 'message': 'Example saved successfully'}
```

## How It Works

### Example Selection Process

When generating bullets for new content:

1. **Semantic Similarity** (if sklearn available):
   - Uses TF-IDF vectorization to compute similarity scores
   - Returns top 2-3 most similar examples

2. **Keyword Matching** (fallback):
   - Computes word overlap between input and stored examples
   - Returns examples with highest keyword overlap

3. **Style Filtering**:
   - Filters to examples matching the detected style
   - Falls back to any style if no matches found

### Integration with Claude API

The `_build_structured_prompt()` function now:

```python
def _build_structured_prompt(self, text, content_info, context_heading=None, style='professional'):
    # ... existing code ...

    # NEW: Load dynamic few-shot examples from storage
    example_bullets = self._get_dynamic_examples(text, style, content_type)

    # Fallback to hardcoded examples if storage is empty
    if not example_bullets:
        example_bullets = [hardcoded_examples...]

    # Include examples in prompt
    examples_text = '\n'.join(f"  - {ex}" for ex in example_bullets[:2])

    prompt = f"""Extract 3-5 key facts...

Examples:
{examples_text}

Content:
{text}
..."""
```

## Storage Structure

```
examples/
├── examples_db.json           # Main database (all examples)
├── quality_ratings.json       # Quality scores and ratings
└── categories/                # Category-specific files
    ├── professional.json
    ├── educational.json
    ├── technical.json
    └── executive.json
```

### Example Record Schema

```json
{
  "example_id": "professional_abc123def456",
  "input_text": "Original document content...",
  "generated_bullets": [
    "First bullet point",
    "Second bullet point"
  ],
  "context_heading": "Slide Title",
  "style": "professional",
  "content_type": "paragraph",
  "quality_score": 85.5,
  "user_rating": 5,
  "category_tags": ["business", "strategy"],
  "timestamp": "2025-11-19T05:15:00.000Z"
}
```

## API Reference

### Save Example
```http
POST /api/examples/save
Content-Type: application/json

{
  "input_text": "...",
  "generated_bullets": ["...", "..."],
  "context_heading": "...",
  "style": "professional",
  "content_type": "paragraph",
  "user_rating": 5
}

Response:
{
  "success": true,
  "example_id": "professional_abc123",
  "message": "Example saved successfully"
}
```

### List Examples
```http
GET /api/examples/list?style=professional&limit=10

Response:
{
  "examples": [
    {
      "example_id": "...",
      "input_text": "...",
      "generated_bullets": [...],
      ...
    }
  ],
  "count": 10
}
```

### Get Statistics
```http
GET /api/examples/stats

Response:
{
  "total_examples": 21,
  "by_style": {
    "professional": 4,
    "educational": 2,
    "technical": 2,
    "executive": 2
  },
  "by_content_type": {
    "paragraph": 15,
    "table": 2,
    "list": 2,
    "heading": 2
  },
  "avg_quality_score": 85.0,
  "rated_examples": 21
}
```

### Rate Example
```http
POST /api/examples/rate
Content-Type: application/json

{
  "example_id": "professional_abc123",
  "quality_score": 90.5,
  "user_rating": 5
}

Response:
{
  "success": true,
  "message": "Rating saved"
}
```

## Testing

### Run Integration Test

```bash
python test_example_storage_integration.py
```

Expected output:
```
🧪 Testing Example Storage System
============================================================
✅ Storage manager initialized
📊 Initial Stats: 21 examples
🔍 Testing example retrieval...
🎯 Testing similarity search...
➕ Testing example creation...
⭐ Testing rating system...
✅ All tests passed!

🔄 Testing Dynamic Example Loading for Claude API
============================================================
   Input: Machine learning algorithms process large datasets...
   Style: educational
   Similar examples (2):
      - Students apply machine learning...
      - Module introduces Python fundamentals...
✅ Dynamic loading test complete!

🎉 Example Storage Integration: FULLY FUNCTIONAL
```

## Quality Improvement Strategy

### Phase 1: Bootstrap (Current)
- Seeded with 21 golden examples from test suite
- Provides baseline few-shot learning capability
- Quality score: 85/100 average

### Phase 2: User Contributions
- Users save examples when they're satisfied with output
- Rate examples 1-5 stars for subjective quality
- System learns from real-world use cases

### Phase 3: Automated Quality Scoring
- Run `quality_metrics.py` on stored examples
- Filter low-quality examples automatically
- Keep only examples scoring >75/100

### Phase 4: Continuous Improvement
- Export high-rated examples to golden test set
- Run regression benchmarks to verify improvement
- Measure impact: baseline vs. learned examples

## Deployment

### Heroku Deployment

The `examples/` directory will be deployed with the app and persist between deploys (as long as you commit it to git).

**Option 1: Commit seeded examples (recommended)**
```bash
git add examples/
git commit -m "Add seeded example database"
git push heroku main
```

**Option 2: Re-seed on each deploy**
Add to Procfile:
```
release: python seed_examples.py
web: gunicorn wsgi:app
```

### Production Considerations

1. **Storage Limits**: JSON files scale to ~1000 examples comfortably
   - For >1000 examples, migrate to PostgreSQL or MongoDB
   - See `example_storage.py` for database migration path

2. **Version Control**:
   - Commit `examples/` directory with seeded examples
   - User-added examples persist in deployed environment
   - For multi-server deployments, use shared database

3. **Backup Strategy**:
   - Export examples periodically: `storage.export_examples_for_testing()`
   - Store backups in cloud storage (S3, Google Cloud Storage)
   - Version control high-quality examples

## Advanced Usage

### Export Examples for Testing

```python
from example_storage import get_storage_manager

storage = get_storage_manager()
storage.export_examples_for_testing('exported_test_cases.json')
```

### Programmatic Example Management

```python
from example_storage import get_storage_manager, BulletExample

storage = get_storage_manager()

# Create new example
example = BulletExample(
    input_text="...",
    generated_bullets=["...", "..."],
    context_heading="...",
    style="professional",
    content_type="paragraph",
    quality_score=90.0
)

example_id = storage.add_example(example)

# Find similar examples
similar = storage.get_similar_examples(
    "Your input text here",
    style="professional",
    limit=5
)

# Get examples by category
professional_examples = storage.get_examples_by_style("professional", limit=10)

# Rate example
storage.rate_example(example_id, quality_score=95.0, user_rating=5)
```

## Troubleshooting

### Issue: No examples loaded for Claude prompts

**Cause**: Storage directory not initialized

**Solution**:
```bash
python seed_examples.py
```

### Issue: Similarity search not working

**Cause**: sklearn not installed

**Solution**: Install dependencies
```bash
pip install scikit-learn numpy
```

Falls back to keyword matching automatically if sklearn unavailable.

### Issue: Examples not persisting

**Cause**: Directory permissions or file path issues

**Solution**: Check permissions
```bash
ls -la examples/
chmod 755 examples/
```

## Future Enhancements

### Planned Features

1. **Automatic Quality Scoring**
   - Run quality_metrics.py on new examples
   - Auto-reject low-quality examples (<60/100)

2. **Smart Example Pruning**
   - Remove duplicate or near-duplicate examples
   - Keep only top-rated examples per category
   - Maintain diversity of examples

3. **Multi-Modal Examples**
   - Store visual prompt examples
   - Store table formatting examples
   - Store speaker notes examples

4. **Collaborative Learning**
   - Share anonymized examples across instances
   - Community-voted "best examples"
   - Category-specific example packs

5. **A/B Testing**
   - Compare quality with/without examples
   - Measure impact of different example counts
   - Optimize few-shot prompt structure

## Summary

The Example Storage System enables **continuous improvement** of bullet generation quality through:

✅ **Automated few-shot learning** from stored examples
✅ **Semantic similarity matching** for relevant examples
✅ **User rating system** for quality feedback
✅ **Admin interface** for easy management
✅ **API integration** for programmatic access
✅ **Production-ready** JSON storage with migration path

**Result**: The system learns from successful outputs and gets better over time! 🎉
