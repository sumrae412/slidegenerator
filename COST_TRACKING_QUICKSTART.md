# Cost Tracking Quick Start Guide

## Installation

No additional installation required! Cost tracking is built into the slide generator.

## 30-Second Quick Start

```python
from slide_generator_pkg.document_parser import DocumentParser

# Initialize parser (cost tracking is automatic)
parser = DocumentParser()

# Process your document
slides = parser.parse_text(your_document_content)

# View costs
parser.print_cost_summary()
```

That's it! Cost tracking happens automatically.

## Common Tasks

### Get Total Cost

```python
cost = parser.get_total_cost()
print(f"Total: ${cost:.4f}")
```

### Export Cost Report

```python
parser.export_cost_report('cost_report.json')
```

### View Cache Savings

```python
summary = parser.get_cost_summary()
cache = summary['cache_statistics']
print(f"Cache saved: ${cache['cost_savings_usd']:.4f} ({cache['hit_rate_percent']:.1f}% hit rate)")
```

### Per-Slide Costs

```python
breakdown = parser.get_cost_breakdown()
for slide_id, data in breakdown['by_slide'].items():
    print(f"{slide_id}: ${data['cost']:.4f}")
```

### Compare Providers

```python
# Test Claude
p1 = DocumentParser(preferred_llm='claude')
s1 = p1.parse_text(doc)
print(f"Claude: ${p1.get_total_cost():.4f}")

# Test OpenAI
p2 = DocumentParser(preferred_llm='openai')
s2 = p2.parse_text(doc)
print(f"OpenAI: ${p2.get_total_cost():.4f}")
```

### Reset Between Documents

```python
parser.reset_cost_tracking()
slides = parser.parse_text(new_document)
```

## Run the Demo

```bash
python3 demo_cost_tracking.py
```

This will:
- Process a sample document
- Display comprehensive cost statistics
- Export a detailed JSON report
- Show all available cost breakdowns

## What Gets Tracked

| Metric | Description |
|--------|-------------|
| **Total Cost** | Sum of all API costs (excluding cached) |
| **Token Usage** | Input, output, and total tokens |
| **Cache Performance** | Hit rate and cost savings |
| **Per-Provider** | Claude vs OpenAI breakdown |
| **Per-Model** | Specific model usage (GPT-4o, Sonnet, etc.) |
| **Per-Call-Type** | Chat, embeddings, refinement |
| **Per-Slide** | Individual slide processing costs |

## Output Example

```
Total Cost: $0.0234
Total Calls: 8 (8 successful, 0 failed)

Token Usage:
  Input Tokens:  4,250
  Output Tokens: 1,120
  Total Tokens:  5,370

Cache Performance:
  Hit Rate: 25.0% (2/8 requests)
  Cost Savings: $0.0058 (19.8%)

Cost by Provider:
  CLAUDE: $0.0156 (5 calls)
  OPENAI: $0.0078 (3 calls)

Slides Processed: 3
Avg Cost per Slide: $0.0078
```

## Integration Points

The cost tracker automatically hooks into:

1. **Claude API calls** (`_call_claude_with_retry`)
2. **OpenAI API calls** (`_call_openai_with_retry`)
3. **Embeddings calls** (`_deduplicate_bullets_with_embeddings`)
4. **Cache hits** (`_get_cached_response`)

All API calls are tracked without any code changes needed.

## Pricing (2025)

| Model | Input | Output |
|-------|-------|--------|
| Claude Sonnet | $3/1M | $15/1M |
| GPT-4o | $2.50/1M | $10/1M |
| GPT-3.5 | $0.50/1M | $1.50/1M |
| Embeddings | $0.02/1M | - |

## Files Created

- `slide_generator_pkg/utils.py` - CostTracker class (350+ lines)
- `slide_generator_pkg/document_parser.py` - Integration (modified)
- `demo_cost_tracking.py` - Full demonstration script
- `COST_TRACKING.md` - Comprehensive documentation
- `COST_TRACKING_QUICKSTART.md` - This quick reference

## Support

For detailed documentation, see `COST_TRACKING.md`.

For examples, run `python3 demo_cost_tracking.py`.

For issues, check the error message and consult the troubleshooting section in the full documentation.
