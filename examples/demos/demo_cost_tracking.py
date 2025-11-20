#!/usr/bin/env python3
"""
Demonstration of the Cost Tracking System

This script demonstrates the comprehensive cost tracking features
for both Claude and OpenAI API calls in the slide generator.
"""

import os
import sys
import json
from slide_generator_pkg.document_parser import DocumentParser


def demo_cost_tracking():
    """Demonstrate cost tracking with sample document processing"""

    print("="*70)
    print("COST TRACKING SYSTEM DEMONSTRATION")
    print("="*70)
    print()

    # Sample document content
    sample_doc = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing computer programs that can access
data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data.
The model is provided with input-output pairs and learns to map inputs
to outputs. Common applications include classification and regression tasks.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to
find patterns and structure in the input data without predefined labels.
Clustering and dimensionality reduction are common techniques.

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by
taking actions in an environment to maximize cumulative reward. It's
commonly used in robotics, game playing, and autonomous systems.

## Applications

Machine learning powers many modern applications:
- Recommendation systems (Netflix, Amazon)
- Natural language processing (ChatGPT, translation)
- Computer vision (facial recognition, self-driving cars)
- Fraud detection in financial services
- Medical diagnosis and drug discovery

## Key Algorithms

Popular machine learning algorithms include:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Neural Networks
- K-Means Clustering
- Principal Component Analysis
"""

    # Initialize parser with API keys from environment
    # Note: You can also pass API keys directly to DocumentParser()
    parser = DocumentParser()

    print("Initializing DocumentParser with cost tracking...")
    print(f"  Claude API available: {parser.client is not None}")
    print(f"  OpenAI API available: {parser.openai_client is not None}")
    print()

    if not parser.client and not parser.openai_client:
        print("⚠️  No API keys found. Cost tracking will still work but with zero costs.")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variables to track real costs.")
        print()

    # Parse the document
    print("Processing sample document...")
    print(f"Document length: {len(sample_doc)} characters")
    print()

    try:
        # Parse document to slides
        slides = parser.parse_text(sample_doc)

        print(f"✅ Successfully generated {len(slides)} slides")
        print()

        # Display cost summary
        print("="*70)
        print("COST SUMMARY")
        print("="*70)

        # Get comprehensive cost summary
        summary = parser.get_cost_summary()

        print(f"\nTotal Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"  ├─ Successful: {summary['successful_calls']}")
        print(f"  └─ Failed: {summary['failed_calls']}")

        # Token usage
        tokens = summary['tokens']
        print(f"\nToken Usage:")
        print(f"  ├─ Input Tokens:  {tokens['input_tokens']:,}")
        print(f"  ├─ Output Tokens: {tokens['output_tokens']:,}")
        print(f"  ├─ Total Tokens:  {tokens['total_tokens']:,}")
        print(f"  └─ Cached Tokens Saved: {tokens['cached_tokens_saved']:,}")

        # Cache performance
        cache = summary['cache_statistics']
        print(f"\nCache Performance:")
        print(f"  ├─ Hit Rate: {cache['hit_rate_percent']:.1f}%")
        print(f"  ├─ Hits: {cache['cache_hits']}, Misses: {cache['cache_misses']}")
        print(f"  ├─ Cost Savings: ${cache['cost_savings_usd']:.4f}")
        print(f"  └─ Savings Percent: {cache['cost_savings_percent']:.1f}%")

        # Cost by provider
        if summary['cost_by_provider']:
            print(f"\nCost by Provider:")
            for provider, data in summary['cost_by_provider'].items():
                print(f"  {provider.upper()}:")
                print(f"    ├─ Calls: {data['calls']}")
                print(f"    ├─ Tokens: {data['input_tokens'] + data['output_tokens']:,}")
                print(f"    └─ Cost: ${data['cost']:.4f}")

        # Cost by model
        if summary['cost_by_model']:
            print(f"\nCost by Model:")
            for model, data in summary['cost_by_model'].items():
                print(f"  {model}:")
                print(f"    ├─ Calls: {data['calls']}")
                print(f"    └─ Cost: ${data['cost']:.4f}")

        # Cost by call type
        if summary['cost_by_call_type']:
            print(f"\nCost by Call Type:")
            for call_type, data in summary['cost_by_call_type'].items():
                print(f"  {call_type}:")
                print(f"    ├─ Calls: {data['calls']}")
                print(f"    └─ Cost: ${data['cost']:.4f}")

        # Per-slide costs
        if summary['slides_processed'] > 0:
            print(f"\nSlide Processing:")
            print(f"  ├─ Slides Processed: {summary['slides_processed']}")
            print(f"  └─ Avg Cost per Slide: ${summary['avg_cost_per_slide']:.4f}")

        print("\n" + "="*70)

        # Export detailed cost report
        report_path = "/home/user/slidegenerator/cost_report.json"
        parser.export_cost_report(report_path, detailed=True)
        print(f"\n📊 Detailed cost report exported to: {report_path}")

        # Show breakdown structure
        print("\n" + "="*70)
        print("COST BREAKDOWN STRUCTURE")
        print("="*70)

        breakdown = parser.get_cost_breakdown()

        print("\nAvailable breakdowns:")
        print(f"  ├─ By Provider: {len(breakdown['by_provider'])} providers")
        print(f"  ├─ By Model: {len(breakdown['by_model'])} models")
        print(f"  ├─ By Call Type: {len(breakdown['by_call_type'])} types")
        print(f"  └─ By Slide: {len(breakdown['by_slide'])} slides")

        # Show sample slide breakdown
        if breakdown['by_slide']:
            slide_ids = list(breakdown['by_slide'].keys())[:3]  # Show first 3 slides
            print("\nSample per-slide costs (first 3 slides):")
            for slide_id in slide_ids:
                slide_data = breakdown['by_slide'][slide_id]
                print(f"\n  Slide: {slide_id}")
                print(f"    ├─ Calls: {len(slide_data['calls'])}")
                print(f"    ├─ Input Tokens: {slide_data['input_tokens']:,}")
                print(f"    ├─ Output Tokens: {slide_data['output_tokens']:,}")
                print(f"    └─ Cost: ${slide_data['cost']:.4f}")

        print("\n" + "="*70)

        # Alternative: Use the built-in pretty printer
        print("\nAlternative: Using built-in cost summary printer:")
        print()
        parser.print_cost_summary()

        return True

    except Exception as e:
        print(f"\n❌ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_cost_comparison():
    """Demonstrate cost comparison between Claude and OpenAI"""

    print("\n" + "="*70)
    print("COST COMPARISON: CLAUDE VS OPENAI")
    print("="*70)
    print()

    sample_text = """
Cloud computing provides on-demand access to computing resources over the internet.
It offers scalability, flexibility, and cost-efficiency for businesses of all sizes.
Key benefits include reduced infrastructure costs, automatic updates, and global accessibility.
"""

    # Test with Claude
    print("Testing with Claude API...")
    parser_claude = DocumentParser(preferred_llm='claude')
    if parser_claude.client:
        slides_claude = parser_claude.parse_text(f"# Cloud Computing\n\n{sample_text}")
        cost_claude = parser_claude.get_total_cost()
        summary_claude = parser_claude.get_cost_summary()
        print(f"  ✅ Cost: ${cost_claude:.4f}")
        print(f"  ├─ Tokens: {summary_claude['tokens']['total_tokens']:,}")
        print(f"  └─ Calls: {summary_claude['total_calls']}")
    else:
        print("  ⚠️  Claude API not available")

    print()

    # Test with OpenAI
    print("Testing with OpenAI API...")
    parser_openai = DocumentParser(preferred_llm='openai')
    if parser_openai.openai_client:
        slides_openai = parser_openai.parse_text(f"# Cloud Computing\n\n{sample_text}")
        cost_openai = parser_openai.get_total_cost()
        summary_openai = parser_openai.get_cost_summary()
        print(f"  ✅ Cost: ${cost_openai:.4f}")
        print(f"  ├─ Tokens: {summary_openai['tokens']['total_tokens']:,}")
        print(f"  └─ Calls: {summary_openai['total_calls']}")
    else:
        print("  ⚠️  OpenAI API not available")

    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n🚀 Starting Cost Tracking Demonstration...\n")

    # Run main demo
    success = demo_cost_tracking()

    # Run comparison demo if both APIs are available
    # demo_cost_comparison()

    if success:
        print("\n✅ Demonstration completed successfully!")
        print("\nNext steps:")
        print("  1. Review the cost_report.json file for detailed breakdown")
        print("  2. Use parser.get_cost_summary() to access cost data programmatically")
        print("  3. Use parser.export_cost_report() to save reports after processing")
        print("  4. Use parser.reset_cost_tracking() when processing new documents")
    else:
        print("\n❌ Demonstration encountered errors")
        sys.exit(1)
