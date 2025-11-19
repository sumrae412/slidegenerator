# Cost Tracking System Documentation

## Overview

The slide generator now includes a comprehensive cost tracking system that monitors API usage and costs for both Claude (Anthropic) and OpenAI API calls. This system provides detailed insights into token usage, API call patterns, and associated costs.

## Features

### Core Capabilities

- **Real-time Cost Tracking**: Track costs as API calls are made
- **Multi-Provider Support**: Separate tracking for Claude and OpenAI
- **Per-Slide Cost Attribution**: See costs broken down by individual slides
- **Cache Performance Monitoring**: Track cost savings from cached responses
- **Detailed Breakdowns**: Analyze costs by provider, model, call type, and slide
- **JSON Export**: Export detailed cost reports for analysis
- **Pretty Printing**: Human-readable cost summaries in console

### Token Tracking

- Input tokens (prompt/context)
- Output tokens (generated content)
- Total token counts
- Cached tokens saved (estimate)

### Cost Breakdowns

1. **By Provider**: Claude vs OpenAI
2. **By Model**: Specific model versions (GPT-4o, Claude Sonnet, etc.)
3. **By Call Type**: Chat completion, embeddings, refinement passes
4. **By Slide**: Individual slide processing costs

## Pricing Configuration

The system includes built-in pricing for common models (as of 2025):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-3.5-turbo | $0.50 | $1.50 |
| text-embedding-3-small | $0.02 | $0.00 |
| text-embedding-3-large | $0.13 | $0.00 |

## Usage

### Basic Usage

```python
from slide_generator_pkg.document_parser import DocumentParser

# Initialize parser (cost tracking is automatic)
parser = DocumentParser()

# Process a document
slides = parser.parse_text(document_content)

# Get total cost
total_cost = parser.get_total_cost()
print(f"Total cost: ${total_cost:.4f}")

# Print comprehensive summary
parser.print_cost_summary()
```

### Getting Cost Summary

```python
# Get full cost summary as dictionary
summary = parser.get_cost_summary()

# Access specific metrics
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['tokens']['total_tokens']:,}")
print(f"Cache hit rate: {summary['cache_statistics']['hit_rate_percent']}%")
print(f"Slides processed: {summary['slides_processed']}")
print(f"Average cost per slide: ${summary['avg_cost_per_slide']:.4f}")
```

### Getting Detailed Breakdowns

```python
# Get detailed breakdowns
breakdown = parser.get_cost_breakdown()

# By provider
for provider, data in breakdown['by_provider'].items():
    print(f"{provider}: ${data['cost']:.4f} ({data['calls']} calls)")

# By model
for model, data in breakdown['by_model'].items():
    print(f"{model}: ${data['cost']:.4f}")

# By call type
for call_type, data in breakdown['by_call_type'].items():
    print(f"{call_type}: ${data['cost']:.4f} ({data['calls']} calls)")

# Per-slide costs
for slide_id, data in breakdown['by_slide'].items():
    print(f"Slide {slide_id}: ${data['cost']:.4f}")
    print(f"  Input tokens: {data['input_tokens']:,}")
    print(f"  Output tokens: {data['output_tokens']:,}")
    print(f"  API calls: {len(data['calls'])}")
```

### Exporting Cost Reports

```python
# Export detailed report (includes all individual API calls)
parser.export_cost_report('cost_report_detailed.json', detailed=True)

# Export summary only (smaller file, high-level stats)
parser.export_cost_report('cost_report_summary.json', detailed=False)
```

### Resetting Cost Tracking

```python
# Reset cost tracking between documents
parser.reset_cost_tracking()

# Process new document with fresh cost tracking
slides = parser.parse_text(new_document)
```

## Cost Summary Structure

The `get_cost_summary()` method returns a dictionary with the following structure:

```python
{
    "session_start": "2025-11-19T10:30:00",
    "session_duration_seconds": 45.3,
    "total_cost_usd": 0.1234,
    "total_calls": 25,
    "successful_calls": 24,
    "failed_calls": 1,

    "tokens": {
        "input_tokens": 12500,
        "output_tokens": 3750,
        "total_tokens": 16250,
        "cached_tokens_saved": 5000
    },

    "cache_statistics": {
        "cache_hits": 8,
        "cache_misses": 17,
        "total_requests": 25,
        "hit_rate_percent": 32.0,
        "estimated_cost_without_cache": 0.1534,
        "actual_cost_with_cache": 0.1234,
        "cost_savings_usd": 0.0300,
        "cost_savings_percent": 19.6
    },

    "cost_by_provider": {
        "claude": {
            "calls": 15,
            "input_tokens": 7500,
            "output_tokens": 2250,
            "cost": 0.0787
        },
        "openai": {
            "calls": 9,
            "input_tokens": 5000,
            "output_tokens": 1500,
            "cost": 0.0447
        }
    },

    "cost_by_model": {
        "claude-3-5-sonnet-20241022": {
            "calls": 15,
            "input_tokens": 7500,
            "output_tokens": 2250,
            "cost": 0.0787
        },
        "gpt-4o": {
            "calls": 8,
            "input_tokens": 4500,
            "output_tokens": 1400,
            "cost": 0.0413
        },
        "text-embedding-3-small": {
            "calls": 1,
            "input_tokens": 500,
            "output_tokens": 0,
            "cost": 0.0034
        }
    },

    "cost_by_call_type": {
        "chat": {
            "calls": 23,
            "input_tokens": 12000,
            "output_tokens": 3650,
            "cost": 0.1200
        },
        "embedding": {
            "calls": 1,
            "input_tokens": 500,
            "output_tokens": 0,
            "cost": 0.0034
        },
        "refinement": {
            "calls": 1,
            "input_tokens": 0,
            "output_tokens": 100,
            "cost": 0.0000
        }
    },

    "slides_processed": 5,
    "avg_cost_per_slide": 0.0247
}
```

## Detailed Report Structure

The `export_cost_report()` method with `detailed=True` includes all individual API calls:

```json
{
    "...": "(all summary fields from above)",

    "individual_calls": [
        {
            "timestamp": "2025-11-19T10:30:05.123456",
            "provider": "claude",
            "model": "claude-3-5-sonnet-20241022",
            "call_type": "chat",
            "input_tokens": 500,
            "output_tokens": 150,
            "total_tokens": 650,
            "cost": 0.0039,
            "cached": false,
            "success": true,
            "error": null,
            "slide_id": "slide_1"
        },
        {
            "timestamp": "2025-11-19T10:30:07.456789",
            "provider": "openai",
            "model": "gpt-4o",
            "call_type": "chat",
            "input_tokens": 450,
            "output_tokens": 140,
            "cost": 0.0025,
            "cached": false,
            "success": true,
            "error": null,
            "slide_id": "slide_2"
        }
    ],

    "slide_breakdown": {
        "slide_1": {
            "calls": [
                "(list of call objects for this slide)"
            ],
            "input_tokens": 500,
            "output_tokens": 150,
            "cost": 0.0039
        },
        "slide_2": {
            "calls": [
                "(list of call objects for this slide)"
            ],
            "input_tokens": 450,
            "output_tokens": 140,
            "cost": 0.0025
        }
    }
}
```

## Cache Hit Tracking

The cost tracker integrates with the DocumentParser's LRU cache to estimate cost savings:

- **Cache Hits**: When content is retrieved from cache, no API call is made
- **Estimated Savings**: The system estimates what the API call would have cost
- **Cached Calls Tracked**: Cache hits are tracked separately in statistics
- **Hit Rate**: Percentage of requests served from cache vs new API calls

```python
summary = parser.get_cost_summary()
cache_stats = summary['cache_statistics']

print(f"Cache hit rate: {cache_stats['hit_rate_percent']}%")
print(f"Cost without cache: ${cache_stats['estimated_cost_without_cache']:.4f}")
print(f"Actual cost with cache: ${cache_stats['actual_cost_with_cache']:.4f}")
print(f"Savings: ${cache_stats['cost_savings_usd']:.4f} ({cache_stats['cost_savings_percent']:.1f}%)")
```

## Advanced Usage

### Custom Slide IDs

When processing slides programmatically, you can provide custom slide identifiers for better tracking:

```python
# The cost tracker automatically receives slide IDs from parse_text()
# But you can also track custom processing:

parser.cost_tracker.track_api_call(
    provider='claude',
    model='claude-3-5-sonnet-20241022',
    input_tokens=500,
    output_tokens=150,
    slide_id='custom_slide_id',
    call_type='chat',
    success=True
)
```

### Tracking Failed Calls

Failed API calls are automatically tracked with error information:

```python
summary = parser.get_cost_summary()
print(f"Failed calls: {summary['failed_calls']}")

# Access detailed error information
detailed = parser.cost_tracker.get_detailed_report()
for call in detailed['individual_calls']:
    if not call['success']:
        print(f"Failed call: {call['error']}")
```

### Comparing Costs Between Providers

```python
# Test with Claude
parser_claude = DocumentParser(preferred_llm='claude')
slides_claude = parser_claude.parse_text(doc)
cost_claude = parser_claude.get_total_cost()

# Test with OpenAI
parser_openai = DocumentParser(preferred_llm='openai')
slides_openai = parser_openai.parse_text(doc)
cost_openai = parser_openai.get_total_cost()

# Compare
print(f"Claude: ${cost_claude:.4f}")
print(f"OpenAI: ${cost_openai:.4f}")
print(f"Difference: ${abs(cost_claude - cost_openai):.4f}")
```

## Integration with Existing Code

The cost tracking system is **automatically integrated** into the DocumentParser. No code changes are required to start tracking costs:

```python
# Before (existing code)
parser = DocumentParser()
slides = parser.parse_text(document)

# After (same code, but now with cost tracking)
parser = DocumentParser()
slides = parser.parse_text(document)
cost = parser.get_total_cost()  # New feature available
```

## Console Output

Using `parser.print_cost_summary()` produces formatted console output:

```
======================================================================
API COST TRACKING SUMMARY
======================================================================

Session Duration: 45.3 seconds
Total Cost: $0.1234
Total Calls: 25 (24 successful, 1 failed)

Token Usage:
  Input Tokens:  12,500
  Output Tokens: 3,750
  Total Tokens:  16,250
  Cached Tokens Saved: 5,000

Cache Performance:
  Hit Rate: 32.0% (8/25 requests)
  Cost Savings: $0.0300 (19.6%)

Cost by Provider:
  CLAUDE: $0.0787 (15 calls, 9,750 tokens)
  OPENAI: $0.0447 (10 calls, 6,500 tokens)

Cost by Model:
  claude-3-5-sonnet-20241022: $0.0787 (15 calls)
  gpt-4o: $0.0413 (8 calls)
  text-embedding-3-small: $0.0034 (1 calls)

Slide Processing:
  Slides Processed: 5
  Avg Cost per Slide: $0.0247

======================================================================
```

## Error Handling

The cost tracking system gracefully handles errors:

- **Failed API calls**: Tracked with `success=False` and error message
- **Missing pricing data**: Falls back to default (Claude Sonnet) pricing with warning
- **No API keys**: Cost tracking still works, reports $0.00 costs
- **Cache errors**: Continues operation, may have reduced savings estimates

## Performance Impact

The cost tracking system has minimal performance impact:

- **Overhead**: < 0.1ms per API call (negligible)
- **Memory**: ~1KB per tracked call
- **Storage**: JSON exports are typically 50-500KB depending on detail level

## Best Practices

1. **Export Reports Regularly**: Save cost reports after processing each document
   ```python
   parser.export_cost_report(f'costs_{doc_id}.json')
   ```

2. **Reset Between Documents**: Clear tracking when starting new documents
   ```python
   parser.reset_cost_tracking()
   ```

3. **Monitor Cache Hit Rates**: Aim for >30% cache hit rate for cost efficiency
   ```python
   if parser.get_cost_summary()['cache_statistics']['hit_rate_percent'] < 30:
       print("Consider enabling cache warming or increasing cache size")
   ```

4. **Track Per-Document Costs**: Keep separate reports for budgeting
   ```python
   parser.reset_cost_tracking()
   slides = parser.parse_text(doc)
   parser.export_cost_report(f'reports/doc_{doc_id}_cost.json')
   ```

5. **Compare Providers**: Test both Claude and OpenAI to find optimal costs
   ```python
   # Use cost-sensitive mode for automatic selection
   parser = DocumentParser(cost_sensitive=True)
   ```

## Troubleshooting

### "No pricing data for model" warning

If you see warnings about missing pricing data, the system will use Claude Sonnet pricing as a fallback. To fix:

1. Check the model name is correct
2. Add custom pricing in `slide_generator_pkg/utils.py`:
   ```python
   PRICING = {
       'your-custom-model': {'input': 1.00, 'output': 2.00},
       ...
   }
   ```

### Costs seem too high

1. Check cache hit rate - low hit rates mean more API calls
2. Review per-slide breakdown to identify expensive operations
3. Consider using `cost_sensitive=True` mode
4. Enable batch processing: `enable_batch_processing=True`

### Costs don't match actual billing

The cost tracker provides **estimates** based on current pricing. Actual costs may vary due to:

- Pricing changes by providers
- Special rate limits or discounts
- Rounding differences
- Cached prompt tokens (Claude prompt caching not yet tracked)

Always verify against actual provider billing.

## Future Enhancements

Planned improvements:

- [ ] Claude prompt caching token tracking
- [ ] Real-time cost alerts (threshold warnings)
- [ ] Cost optimization suggestions
- [ ] Historical cost trending
- [ ] Budget tracking and limits
- [ ] Multi-document cost aggregation
- [ ] Cost projection for large batches

## See Also

- `demo_cost_tracking.py`: Complete working demonstration
- `slide_generator_pkg/utils.py`: CostTracker implementation
- `slide_generator_pkg/document_parser.py`: Integration points
- `CLAUDE.md`: Project documentation and best practices
