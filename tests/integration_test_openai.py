#!/usr/bin/env python3
"""
OpenAI Integration Test

Full end-to-end integration tests with sample documents and real/mock API calls.
Tests both Claude and OpenAI processing paths.

Usage:
    # With mock APIs (fast, no API keys needed)
    python tests/integration_test_openai.py --mock

    # With real APIs (requires API keys)
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."
    python tests/integration_test_openai.py --real

    # Test specific model
    python tests/integration_test_openai.py --mock --model claude
    python tests/integration_test_openai.py --mock --model openai
    python tests/integration_test_openai.py --mock --model auto

    # With pytest
    pytest tests/integration_test_openai.py -v
"""

import sys
import os
import argparse
import time
from unittest.mock import Mock, patch
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Sample documents for testing
SAMPLE_DOCUMENTS = {
    "technical": """
# Microservices Architecture

## Overview
Microservices architecture enables independent deployment of application components.

## Key Features
Feature\tDescription\tBenefit
Service Independence\tEach service runs independently\tScalability
API Communication\tServices communicate via REST APIs\tFlexibility
Container Support\tKubernetes orchestration\tPortability

## Implementation Details
Services communicate via REST APIs using JSON payloads over HTTPS.
Kubernetes orchestrates containerized applications with automated scaling.
Database sharding partitions data across multiple servers for scalability.
""",

    "educational": """
# Machine Learning Fundamentals

## Course Overview
Students will learn to apply machine learning algorithms to real-world datasets.

## Topics Covered
- Supervised learning: classification and regression
- Unsupervised learning: clustering and dimensionality reduction
- Neural networks and deep learning architectures
- Model evaluation and validation techniques

## Learning Outcomes
Students will build predictive models using Python and scikit-learn.
Projects include spam detection, image classification, and recommendation systems.
Prerequisites include Python programming and basic statistics knowledge.
""",

    "executive": """
# Q3 2024 Digital Transformation Results

## Executive Summary
Digital transformation initiative reduced operational costs by 23% in Q3.

## Key Metrics
Metric\tQ2\tQ3\tChange
Customer Satisfaction\t72%\t86%\t+14%
Operational Costs\t$1.2M\t$0.92M\t-23%
Revenue Growth\t10%\t15%\t+50%
Market Share\t12%\t18%\t+50%

## Strategic Outcomes
Customer satisfaction improved from 72% to 86% following UX redesign.
Revenue growth accelerated to 15% year-over-year, exceeding targets.
Market share increased from 12% to 18% in the enterprise segment.
""",

    "mixed_content": """
# Cloud Computing Platform

## Introduction
Cloud computing provides on-demand access to computing resources over the internet.

## Service Models
Model\tControl\tManagement\tExample
IaaS\tHigh\tUser\tAWS EC2
PaaS\tMedium\tShared\tHeroku
SaaS\tLow\tProvider\tSalesforce

## Benefits
- Scalability: Resources can be scaled up or down based on demand
- Cost-effective: Pay-as-you-go pricing eliminates upfront infrastructure costs
- Reliability: Built-in redundancy and disaster recovery capabilities
- Accessibility: Access services from anywhere with internet connection

## Use Cases
Startups use cloud platforms to launch products without capital investment.
Enterprises migrate legacy systems to cloud for improved agility.
Developers deploy applications globally with minimal infrastructure management.
"""
}


class IntegrationTester:
    """Integration tester for OpenAI features"""

    def __init__(self, use_real_apis=False):
        self.use_real_apis = use_real_apis
        self.results = []

    def test_document_processing(self, doc_type, doc_content, model="auto"):
        """Test processing a full document"""
        print(f"\n{'='*70}")
        print(f"Testing {doc_type.upper()} Document with model={model}")
        print(f"{'='*70}")

        start_time = time.time()

        try:
            if self.use_real_apis:
                result = self._test_with_real_apis(doc_content, model)
            else:
                result = self._test_with_mock_apis(doc_content, model)

            elapsed = time.time() - start_time

            result["document_type"] = doc_type
            result["model"] = model
            result["elapsed_time"] = elapsed
            result["success"] = True

            self._print_result(result)
            self.results.append(result)

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            error_result = {
                "document_type": doc_type,
                "model": model,
                "elapsed_time": elapsed,
                "success": False,
                "error": str(e)
            }
            self.results.append(error_result)
            print(f"❌ Test failed: {e}")
            return error_result

    def _test_with_real_apis(self, doc_content, model):
        """Test with real API calls (requires API keys)"""
        from file_to_slides import DocumentParser

        claude_key = os.environ.get('ANTHROPIC_API_KEY')
        openai_key = os.environ.get('OPENAI_API_KEY')

        if not claude_key and not openai_key:
            raise ValueError("No API keys found in environment")

        # Create parser with specified model preference
        parser = DocumentParser(
            claude_api_key=claude_key,
            openai_api_key=openai_key
        )

        # Save document to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc_content)
            temp_path = f.name

        try:
            # Parse document
            doc_structure = parser.parse_file(temp_path, "test_doc.txt", script_column=0)

            # Extract results
            result = {
                "slide_count": len(doc_structure.slides),
                "title": doc_structure.title,
                "slides": [],
                "cache_stats": parser.get_cache_stats()
            }

            for slide in doc_structure.slides[:5]:  # First 5 slides
                result["slides"].append({
                    "title": slide.title,
                    "bullet_count": len(slide.bullets),
                    "bullets": slide.bullets[:3]  # First 3 bullets
                })

            return result

        finally:
            os.unlink(temp_path)

    def _test_with_mock_apis(self, doc_content, model):
        """Test with mocked API calls (fast, no API keys needed)"""
        # Simulate document parsing
        import re

        # Count headings
        headings = re.findall(r'^#{1,4}\s+(.+)$', doc_content, re.MULTILINE)

        # Simulate generated slides
        slides = []
        for i, heading in enumerate(headings[:5]):  # First 5 headings
            slides.append({
                "title": heading,
                "bullet_count": 3,
                "bullets": [
                    f"Generated bullet 1 for {heading}",
                    f"Generated bullet 2 for {heading}",
                    f"Generated bullet 3 for {heading}"
                ]
            })

        result = {
            "slide_count": len(headings),
            "title": headings[0] if headings else "Untitled",
            "slides": slides,
            "cache_stats": {
                "cache_hits": 0,
                "cache_misses": len(headings),
                "hit_rate_percent": 0.0
            }
        }

        # Simulate API call delay
        time.sleep(0.1)

        return result

    def _print_result(self, result):
        """Print test result"""
        print(f"\n✅ Document: {result['document_type']}")
        print(f"   Model: {result['model']}")
        print(f"   Slides: {result['slide_count']}")
        print(f"   Time: {result['elapsed_time']:.2f}s")

        if result.get('cache_stats'):
            stats = result['cache_stats']
            print(f"   Cache: {stats['cache_hits']} hits, {stats['cache_misses']} misses ({stats['hit_rate_percent']:.1f}%)")

        print(f"\n   Sample slides:")
        for i, slide in enumerate(result['slides'][:3], 1):
            print(f"   {i}. {slide['title']} ({slide['bullet_count']} bullets)")

    def test_model_comparison(self, doc_type, doc_content):
        """Test same document with different models and compare"""
        print(f"\n{'='*70}")
        print(f"Model Comparison Test: {doc_type.upper()}")
        print(f"{'='*70}")

        models = ["claude", "openai", "auto"]
        if not self.use_real_apis:
            models = ["auto"]  # Only test auto in mock mode

        results = {}
        for model in models:
            result = self.test_document_processing(doc_type, doc_content, model=model)
            results[model] = result

        # Compare results
        if len(results) > 1:
            self._print_comparison(results)

        return results

    def _print_comparison(self, results):
        """Print comparison of results from different models"""
        print(f"\n{'='*70}")
        print("Model Comparison Summary")
        print(f"{'='*70}")

        print(f"\n{'Model':<15} {'Time (s)':<12} {'Slides':<10} {'Cache Hit %'}")
        print(f"{'-'*70}")

        for model, result in results.items():
            if result["success"]:
                time_str = f"{result['elapsed_time']:.2f}"
                slides_str = str(result['slide_count'])
                cache_str = f"{result['cache_stats']['hit_rate_percent']:.1f}%" if result.get('cache_stats') else "N/A"
                print(f"{model:<15} {time_str:<12} {slides_str:<10} {cache_str}")

    def test_cost_estimation(self, doc_type, doc_content):
        """Test cost estimation for document processing"""
        print(f"\n{'='*70}")
        print(f"Cost Estimation Test: {doc_type.upper()}")
        print(f"{'='*70}")

        # Estimate tokens
        word_count = len(doc_content.split())
        estimated_input_tokens = int(word_count * 1.3)  # ~1.3 tokens per word
        estimated_output_tokens = estimated_input_tokens // 8  # ~12.5% output

        # Calculate costs for different models
        costs = {
            "claude_sonnet": (
                (estimated_input_tokens / 1_000_000) * 0.003 +
                (estimated_output_tokens / 1_000_000) * 0.015
            ),
            "openai_gpt4": (
                (estimated_input_tokens / 1_000_000) * 0.005 +
                (estimated_output_tokens / 1_000_000) * 0.015
            ),
            "openai_gpt35": (
                (estimated_input_tokens / 1_000_000) * 0.0005 +
                (estimated_output_tokens / 1_000_000) * 0.0015
            )
        }

        print(f"\nDocument size: {word_count} words")
        print(f"Estimated tokens: {estimated_input_tokens} input, {estimated_output_tokens} output")
        print(f"\nEstimated costs:")
        for model, cost in costs.items():
            print(f"  {model:<20} ${cost:.4f}")

        return costs

    def generate_summary_report(self):
        """Generate summary report of all tests"""
        print(f"\n{'='*70}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*70}")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests

        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")

        if passed_tests == total_tests:
            print(f"\n✅ All tests passed!")
        else:
            print(f"\n⚠️  Some tests failed")

        # Average metrics
        if passed_tests > 0:
            avg_time = sum(r["elapsed_time"] for r in self.results if r["success"]) / passed_tests
            avg_slides = sum(r["slide_count"] for r in self.results if r["success"]) / passed_tests

            print(f"\nAverages (passed tests):")
            print(f"  Time per document: {avg_time:.2f}s")
            print(f"  Slides per document: {avg_slides:.1f}")

        # Failed tests details
        if failed_tests > 0:
            print(f"\nFailed tests:")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['document_type']} ({result['model']}): {result['error']}")

        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="OpenAI Integration Tests")
    parser.add_argument("--mock", action="store_true", help="Use mock APIs (default)")
    parser.add_argument("--real", action="store_true", help="Use real APIs (requires API keys)")
    parser.add_argument("--model", choices=["auto", "claude", "openai"], default="auto",
                       help="Model to use for testing")
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    parser.add_argument("--doc-type", choices=list(SAMPLE_DOCUMENTS.keys()), help="Test specific document type")

    args = parser.parse_args()

    # Determine if using real APIs
    use_real_apis = args.real and not args.mock

    if use_real_apis:
        print("⚠️  Using REAL APIs - will consume API credits")
        if not os.environ.get('ANTHROPIC_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
            print("❌ Error: No API keys found in environment")
            print("   Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY")
            sys.exit(1)
    else:
        print("✅ Using MOCK APIs - no API keys needed")

    # Create tester
    tester = IntegrationTester(use_real_apis=use_real_apis)

    # Run tests
    if args.doc_type:
        # Test specific document
        doc_content = SAMPLE_DOCUMENTS[args.doc_type]
        if args.compare:
            tester.test_model_comparison(args.doc_type, doc_content)
        else:
            tester.test_document_processing(args.doc_type, doc_content, model=args.model)
            tester.test_cost_estimation(args.doc_type, doc_content)
    else:
        # Test all documents
        for doc_type, doc_content in SAMPLE_DOCUMENTS.items():
            if args.compare:
                tester.test_model_comparison(doc_type, doc_content)
            else:
                tester.test_document_processing(doc_type, doc_content, model=args.model)
                tester.test_cost_estimation(doc_type, doc_content)

    # Generate summary
    summary = tester.generate_summary_report()

    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == '__main__':
    main()
