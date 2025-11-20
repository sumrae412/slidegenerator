"""
Cost Tracking System - Code Examples

Copy-paste ready examples for common cost tracking tasks.
"""

from slide_generator_pkg.document_parser import DocumentParser


# ==============================================================================
# EXAMPLE 1: Basic Usage - Get Total Cost
# ==============================================================================

def example_basic_cost():
    """Simple example: process document and get total cost"""
    parser = DocumentParser()

    # Process document
    document = "# My Document\n\nSome content here..."
    slides = parser.parse_text(document)

    # Get total cost
    cost = parser.get_total_cost()
    print(f"Total cost: ${cost:.4f}")


# ==============================================================================
# EXAMPLE 2: Get Comprehensive Summary
# ==============================================================================

def example_cost_summary():
    """Get detailed cost statistics"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Get full summary
    summary = parser.get_cost_summary()

    # Access specific metrics
    print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    print(f"Total calls: {summary['total_calls']}")
    print(f"Total tokens: {summary['tokens']['total_tokens']:,}")
    print(f"Cache hit rate: {summary['cache_statistics']['hit_rate_percent']}%")
    print(f"Slides processed: {summary['slides_processed']}")
    print(f"Avg cost per slide: ${summary['avg_cost_per_slide']:.4f}")


# ==============================================================================
# EXAMPLE 3: Per-Provider Breakdown
# ==============================================================================

def example_provider_breakdown():
    """Compare costs between Claude and OpenAI"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Get provider breakdown
    breakdown = parser.get_cost_breakdown()

    print("Cost by Provider:")
    for provider, data in breakdown['by_provider'].items():
        print(f"\n{provider.upper()}:")
        print(f"  Calls: {data['calls']}")
        print(f"  Input tokens: {data['input_tokens']:,}")
        print(f"  Output tokens: {data['output_tokens']:,}")
        print(f"  Total cost: ${data['cost']:.4f}")


# ==============================================================================
# EXAMPLE 4: Per-Slide Costs
# ==============================================================================

def example_per_slide_costs():
    """Analyze costs for each individual slide"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Get slide breakdown
    breakdown = parser.get_cost_breakdown()

    print("Cost per Slide:")
    for slide_id, data in breakdown['by_slide'].items():
        print(f"\n{slide_id}:")
        print(f"  API calls: {len(data['calls'])}")
        print(f"  Input tokens: {data['input_tokens']:,}")
        print(f"  Output tokens: {data['output_tokens']:,}")
        print(f"  Cost: ${data['cost']:.4f}")


# ==============================================================================
# EXAMPLE 5: Export Cost Report
# ==============================================================================

def example_export_report():
    """Export detailed cost report to JSON"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Export detailed report (includes all individual calls)
    parser.export_cost_report('cost_report_detailed.json', detailed=True)
    print("✅ Detailed report saved to cost_report_detailed.json")

    # Export summary only (smaller file)
    parser.export_cost_report('cost_report_summary.json', detailed=False)
    print("✅ Summary report saved to cost_report_summary.json")


# ==============================================================================
# EXAMPLE 6: Cache Performance Analysis
# ==============================================================================

def example_cache_analysis():
    """Analyze cache performance and savings"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Get cache statistics
    summary = parser.get_cost_summary()
    cache = summary['cache_statistics']

    print("Cache Performance:")
    print(f"  Hit rate: {cache['hit_rate_percent']:.1f}%")
    print(f"  Hits: {cache['cache_hits']}, Misses: {cache['cache_misses']}")
    print(f"  Cost without cache: ${cache['estimated_cost_without_cache']:.4f}")
    print(f"  Actual cost with cache: ${cache['actual_cost_with_cache']:.4f}")
    print(f"  Savings: ${cache['cost_savings_usd']:.4f} ({cache['cost_savings_percent']:.1f}%)")


# ==============================================================================
# EXAMPLE 7: Compare Claude vs OpenAI
# ==============================================================================

def example_compare_providers():
    """Compare costs between providers for same document"""
    document = "# Test Document\n\nSome content to process..."

    # Test with Claude
    parser_claude = DocumentParser(preferred_llm='claude')
    slides_claude = parser_claude.parse_text(document)
    cost_claude = parser_claude.get_total_cost()
    summary_claude = parser_claude.get_cost_summary()

    # Test with OpenAI
    parser_openai = DocumentParser(preferred_llm='openai')
    slides_openai = parser_openai.parse_text(document)
    cost_openai = parser_openai.get_total_cost()
    summary_openai = parser_openai.get_cost_summary()

    # Compare
    print("Provider Comparison:")
    print(f"\nClaude:")
    print(f"  Cost: ${cost_claude:.4f}")
    print(f"  Tokens: {summary_claude['tokens']['total_tokens']:,}")
    print(f"  Calls: {summary_claude['total_calls']}")

    print(f"\nOpenAI:")
    print(f"  Cost: ${cost_openai:.4f}")
    print(f"  Tokens: {summary_openai['tokens']['total_tokens']:,}")
    print(f"  Calls: {summary_openai['total_calls']}")

    print(f"\nDifference: ${abs(cost_claude - cost_openai):.4f}")


# ==============================================================================
# EXAMPLE 8: Process Multiple Documents
# ==============================================================================

def example_multiple_documents():
    """Track costs across multiple documents"""
    parser = DocumentParser()
    documents = [
        "# Document 1\n\nContent...",
        "# Document 2\n\nContent...",
        "# Document 3\n\nContent..."
    ]

    total_cost = 0
    for i, doc in enumerate(documents, 1):
        # Reset tracking for each document
        parser.reset_cost_tracking()

        # Process document
        slides = parser.parse_text(doc)
        cost = parser.get_total_cost()

        # Save individual report
        parser.export_cost_report(f'doc_{i}_cost.json')

        print(f"Document {i}: ${cost:.4f}")
        total_cost += cost

    print(f"\nTotal cost (all documents): ${total_cost:.4f}")


# ==============================================================================
# EXAMPLE 9: Pretty Print Summary
# ==============================================================================

def example_pretty_print():
    """Use built-in pretty printer for human-readable output"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Print formatted summary to console
    parser.print_cost_summary()


# ==============================================================================
# EXAMPLE 10: Cost by Model
# ==============================================================================

def example_model_breakdown():
    """Breakdown costs by specific model"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    breakdown = parser.get_cost_breakdown()

    print("Cost by Model:")
    for model, data in breakdown['by_model'].items():
        print(f"\n{model}:")
        print(f"  Calls: {data['calls']}")
        print(f"  Tokens: {data['input_tokens'] + data['output_tokens']:,}")
        print(f"  Cost: ${data['cost']:.4f}")


# ==============================================================================
# EXAMPLE 11: Cost by Call Type
# ==============================================================================

def example_call_type_breakdown():
    """Breakdown costs by type of API call"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    breakdown = parser.get_cost_breakdown()

    print("Cost by Call Type:")
    for call_type, data in breakdown['by_call_type'].items():
        print(f"\n{call_type.upper()}:")
        print(f"  Calls: {data['calls']}")
        print(f"  Input tokens: {data['input_tokens']:,}")
        print(f"  Output tokens: {data['output_tokens']:,}")
        print(f"  Cost: ${data['cost']:.4f}")


# ==============================================================================
# EXAMPLE 12: Monitor Costs During Processing
# ==============================================================================

def example_monitor_costs():
    """Check costs at different stages of processing"""
    parser = DocumentParser()

    # Process first part
    slides_part1 = parser.parse_text("# Part 1\n\nContent...")
    cost_part1 = parser.get_total_cost()
    print(f"After part 1: ${cost_part1:.4f}")

    # Process second part (costs accumulate)
    slides_part2 = parser.parse_text("# Part 2\n\nContent...")
    cost_total = parser.get_total_cost()
    cost_part2 = cost_total - cost_part1
    print(f"After part 2: ${cost_total:.4f} (part 2: ${cost_part2:.4f})")


# ==============================================================================
# EXAMPLE 13: Custom Cost Analysis
# ==============================================================================

def example_custom_analysis():
    """Perform custom cost analysis"""
    parser = DocumentParser()
    slides = parser.parse_text(document_content)

    # Get detailed report
    report = parser.cost_tracker.get_detailed_report()

    # Analyze individual calls
    expensive_calls = []
    for call in report['individual_calls']:
        if call['cost'] > 0.01:  # Calls costing more than $0.01
            expensive_calls.append(call)

    print(f"Found {len(expensive_calls)} expensive calls (>$0.01):")
    for call in expensive_calls:
        print(f"  {call['model']}: ${call['cost']:.4f} "
              f"({call['input_tokens']}+{call['output_tokens']} tokens)")


# ==============================================================================
# EXAMPLE 14: Estimate Cost Before Processing
# ==============================================================================

def example_cost_estimation():
    """Estimate cost based on document size"""
    # Rough estimation formula
    document = "# Large Document\n\nLots of content..."
    char_count = len(document)

    # Estimate tokens (approximately 1 token = 4 characters)
    estimated_tokens = char_count // 4

    # Estimate slides (roughly 1 slide per 500 characters)
    estimated_slides = max(1, char_count // 500)

    # Estimate cost (average $0.01 per slide for Claude Sonnet)
    estimated_cost = estimated_slides * 0.01

    print(f"Document size: {char_count:,} characters")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Estimated slides: {estimated_slides}")
    print(f"Estimated cost: ${estimated_cost:.4f}")

    # Now process and compare
    parser = DocumentParser()
    slides = parser.parse_text(document)
    actual_cost = parser.get_total_cost()

    print(f"\nActual cost: ${actual_cost:.4f}")
    print(f"Difference: ${abs(actual_cost - estimated_cost):.4f}")


# ==============================================================================
# EXAMPLE 15: Failed Calls Analysis
# ==============================================================================

def example_failed_calls():
    """Analyze failed API calls"""
    parser = DocumentParser()

    try:
        slides = parser.parse_text(document_content)
    except Exception as e:
        print(f"Processing failed: {e}")

    # Get summary to see failed calls
    summary = parser.get_cost_summary()
    print(f"Successful calls: {summary['successful_calls']}")
    print(f"Failed calls: {summary['failed_calls']}")

    # Get detailed info on failures
    report = parser.cost_tracker.get_detailed_report()
    for call in report['individual_calls']:
        if not call['success']:
            print(f"\nFailed call:")
            print(f"  Provider: {call['provider']}")
            print(f"  Model: {call['model']}")
            print(f"  Error: {call['error']}")
            print(f"  Timestamp: {call['timestamp']}")


# ==============================================================================
# Main - Run Examples
# ==============================================================================

if __name__ == "__main__":
    # Sample document for examples
    document_content = """
# Machine Learning Basics

Machine learning enables computers to learn from data without explicit programming.

## Key Concepts

- Supervised learning uses labeled data
- Unsupervised learning finds patterns in unlabeled data
- Reinforcement learning learns through trial and error

## Applications

Machine learning powers modern applications like recommendation systems,
natural language processing, and computer vision.
"""

    print("Running Cost Tracking Examples...\n")

    # Run basic example
    print("=" * 70)
    print("EXAMPLE 1: Basic Cost Tracking")
    print("=" * 70)
    example_basic_cost()

    # Uncomment to run other examples:
    # example_cost_summary()
    # example_provider_breakdown()
    # example_per_slide_costs()
    # example_export_report()
    # example_cache_analysis()
    # example_compare_providers()
    # example_pretty_print()

    print("\n✅ Example completed!")
    print("\nSee cost_tracking_examples.py for 15 ready-to-use examples.")
