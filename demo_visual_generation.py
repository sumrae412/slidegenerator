#!/usr/bin/env python3
"""
Demo Script: AI Visual Generation with DALL-E 3

This script demonstrates how to use the VisualGenerator to create
AI-powered images for presentation slides using DALL-E 3.

Usage:
    python demo_visual_generation.py

Requirements:
    - OpenAI API key (set via OPENAI_API_KEY environment variable or pass directly)
    - Internet connection for DALL-E 3 API
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from slide_generator_pkg import (
    VisualGenerator,
    SlideContent,
    DocumentParser,
    SlideGenerator
)


def demo_basic_usage():
    """Demonstrate basic visual generation for a single slide"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Visual Generation")
    print("=" * 70)

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        print("   Falling back to text-based descriptions...")

    # Initialize visual generator
    visual_gen = VisualGenerator(openai_api_key=api_key)

    # Create a sample slide
    slide = SlideContent(
        title="Cloud Architecture Overview",
        content=[
            "Multi-region deployment with automatic failover",
            "Microservices-based architecture for scalability",
            "Kubernetes orchestration for container management",
            "Load balancing across availability zones"
        ],
        slide_type='content'
    )

    print(f"\n📝 Slide: {slide.title}")
    print(f"   Content: {len(slide.content)} bullet points")

    # Analyze slide type
    visual_type = visual_gen.analyze_slide_type(slide)
    print(f"   Detected type: {visual_type}")

    # Generate visual
    print("\n🎨 Generating visual...")
    result = visual_gen.generate_image(
        slide=slide,
        quality='standard',
        size='1024x1024'
    )

    if result:
        print("\n✅ Generation complete!")
        print(f"   Prompt: {result['prompt'][:100]}...")
        print(f"   Cost: ${result['cost']:.3f}")
        print(f"   Cached: {result['cached']}")

        if result.get('local_path'):
            print(f"   Saved to: {result['local_path']}")
        elif result.get('url'):
            print(f"   URL: {result['url'][:60]}...")
        else:
            print(f"   Text only: {result['prompt']}")


def demo_batch_generation():
    """Demonstrate batch visual generation for multiple slides"""
    print("\n" + "=" * 70)
    print("DEMO 2: Batch Visual Generation")
    print("=" * 70)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Skipping batch demo (no API key)")
        return

    visual_gen = VisualGenerator(openai_api_key=api_key)

    # Create multiple slides
    slides = [
        SlideContent(
            title="Machine Learning Pipeline",
            content=[
                "Data collection and preprocessing",
                "Feature engineering and selection",
                "Model training and validation",
                "Deployment and monitoring"
            ],
            slide_type='content',
            heading_level=2
        ),
        SlideContent(
            title="Data Analytics Dashboard",
            content=[
                "Real-time metrics visualization",
                "Interactive filtering and drill-down",
                "Custom report generation",
                "Automated alerts and notifications"
            ],
            slide_type='content',
            heading_level=3
        ),
        SlideContent(
            title="Security Best Practices",
            content=[
                "End-to-end encryption for data at rest",
                "Multi-factor authentication for all users",
                "Regular security audits and penetration testing",
                "Compliance with industry standards (SOC 2, GDPR)"
            ],
            slide_type='content',
            heading_level=3
        )
    ]

    print(f"\n📊 Processing {len(slides)} slides...")

    # Estimate cost
    estimated = visual_gen.estimate_cost(
        num_slides=len(slides),
        quality='standard',
        size='1024x1024'
    )
    print(f"💰 Estimated cost: ${estimated:.2f}")

    # Generate visuals (key slides only to save cost)
    print("\n🎨 Generating visuals (key slides only)...")
    result = visual_gen.generate_visuals_batch(
        slides=slides,
        filter_strategy='key_slides',  # Only section headers
        quality='standard',
        size='1024x1024'
    )

    # Display results
    summary = result['summary']
    print("\n✅ Batch generation complete!")
    print(f"   Slides processed: {summary['slides_processed']}/{summary['total_slides']}")
    print(f"   Images generated: {summary['images_generated']}")
    print(f"   Cache hits: {summary['cache_hits']}")
    print(f"   Total cost: ${summary['total_cost']:.2f}")

    # Show which slides got visuals
    print("\n📋 Visual assignment:")
    for i, slide in enumerate(slides):
        if i in result['visuals']:
            visual_data = result['visuals'][i]
            cached = "✓ cached" if visual_data.get('cached') else "✗ new"
            print(f"   [{i+1}] {slide.title}: {cached}")
        else:
            print(f"   [{i+1}] {slide.title}: skipped")


def demo_cost_tracking():
    """Demonstrate cost tracking and statistics"""
    print("\n" + "=" * 70)
    print("DEMO 3: Cost Tracking and Statistics")
    print("=" * 70)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Skipping cost tracking demo (no API key)")
        return

    visual_gen = VisualGenerator(openai_api_key=api_key)

    # Get current statistics
    stats = visual_gen.get_statistics()

    print("\n📊 Visual Generator Statistics:")
    print(f"   Total cost: ${stats['total_cost']:.2f}")
    print(f"   Images generated: {stats['images_generated']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']}")
    print(f"   Cached images: {stats['cached_images']}")
    print(f"   Cache directory: {stats['cache_dir']}")
    print(f"   API available: {stats['api_available']}")


def demo_full_pipeline():
    """Demonstrate full document-to-presentation pipeline with visuals"""
    print("\n" + "=" * 70)
    print("DEMO 4: Full Pipeline (Document → PowerPoint with AI Visuals)")
    print("=" * 70)

    api_key = os.getenv('OPENAI_API_KEY')
    claude_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        print("⚠️  Skipping full pipeline demo (no OpenAI API key)")
        return

    # Create a sample document
    sample_doc = """# Modern Web Development

## Introduction

Web development has evolved significantly over the past decade with the introduction of modern frameworks and tools.

## Frontend Technologies

Modern frontend development relies on component-based architectures:
- React for building interactive user interfaces
- Vue.js for progressive web applications
- Angular for enterprise-scale applications
- TypeScript for type-safe JavaScript development

## Backend Architecture

Server-side development has embraced microservices:
- Node.js for JavaScript-based backends
- Python FastAPI for high-performance APIs
- Go for scalable microservices
- Rust for systems-level performance

## DevOps and Deployment

Continuous integration and deployment pipelines:
- Docker containers for consistent environments
- Kubernetes for orchestration at scale
- GitHub Actions for automated CI/CD
- Terraform for infrastructure as code
"""

    # Save to temporary file
    temp_file = Path('temp_demo_document.txt')
    temp_file.write_text(sample_doc)

    try:
        print("\n📄 Created sample document (4 sections)")

        # Initialize parser with visual generation
        parser = DocumentParser(
            openai_api_key=api_key,
            claude_api_key=claude_key,
            enable_visual_generation=True,
            visual_filter='key_slides'  # Only title and section headers
        )

        print("🔍 Parsing document and generating visuals...")

        # Parse document (visuals generated automatically)
        doc_structure = parser.parse_file(
            str(temp_file),
            'temp_demo_document.txt',
            script_column=0  # Paragraph mode
        )

        print(f"\n✅ Parsing complete!")
        print(f"   Title: {doc_structure.title}")
        print(f"   Slides: {len(doc_structure.slides)}")

        # Check visual generation results
        if 'visual_generation' in doc_structure.metadata:
            visual_stats = doc_structure.metadata['visual_generation']
            print(f"\n🎨 Visual Generation Results:")
            print(f"   Images generated: {visual_stats['images_generated']}")
            print(f"   Cache hits: {visual_stats['cache_hits']}")
            print(f"   Total cost: ${visual_stats['total_cost']:.2f}")

        # Count slides with visuals
        slides_with_visuals = sum(
            1 for slide in doc_structure.slides
            if slide.visual_image_path or slide.visual_image_url
        )
        print(f"   Slides with visuals: {slides_with_visuals}/{len(doc_structure.slides)}")

        # Generate PowerPoint
        print("\n📊 Generating PowerPoint presentation...")
        generator = SlideGenerator()
        pptx_path = generator.create_powerpoint(doc_structure)

        print(f"\n✅ PowerPoint created: {pptx_path}")
        print(f"   AI images automatically inserted into slides")
        print(f"\n💡 Open the presentation to see AI-generated visuals!")

    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()


def demo_visual_types():
    """Demonstrate different visual types for different content"""
    print("\n" + "=" * 70)
    print("DEMO 5: Visual Type Detection")
    print("=" * 70)

    api_key = os.getenv('OPENAI_API_KEY')
    visual_gen = VisualGenerator(openai_api_key=api_key)

    # Different slide types
    test_slides = [
        ("Technical", SlideContent(
            title="System Architecture",
            content=["Microservices", "API Gateway", "Service Discovery"],
            slide_type='content'
        )),
        ("Data", SlideContent(
            title="Performance Metrics",
            content=["95% uptime", "2ms latency", "1M requests/day"],
            slide_type='content'
        )),
        ("Process", SlideContent(
            title="Development Workflow",
            content=["Code review", "Testing", "Deployment", "Monitoring"],
            slide_type='content'
        )),
        ("Executive", SlideContent(
            title="Business Strategy",
            content=["Market expansion", "Revenue growth", "Customer acquisition"],
            slide_type='content'
        )),
        ("Educational", SlideContent(
            title="Introduction to Machine Learning",
            content=["Supervised learning", "Unsupervised learning", "Reinforcement learning"],
            slide_type='content'
        ))
    ]

    print("\n🎨 Analyzing slide types and visual strategies:\n")

    for category, slide in test_slides:
        visual_type = visual_gen.analyze_slide_type(slide)
        prompt = visual_gen.create_visual_prompt(slide, visual_type)

        print(f"[{category}] {slide.title}")
        print(f"   Detected: {visual_type}")
        print(f"   Prompt: {prompt[:80]}...")
        print()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("AI VISUAL GENERATION DEMO")
    print("DALL-E 3 Integration for Slide Generation")
    print("=" * 70)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  WARNING: No OpenAI API key found!")
        print("   Set OPENAI_API_KEY environment variable to enable full demos.")
        print("   Some demos will show text-based fallbacks instead.\n")

    # Run demos
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Batch Generation", demo_batch_generation),
        ("Cost Tracking", demo_cost_tracking),
        ("Visual Types", demo_visual_types),
        ("Full Pipeline", demo_full_pipeline),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("DEMOS COMPLETE")
    print("=" * 70)
    print("\nFor more information, see VISUAL_GENERATION.md")
    print()


if __name__ == "__main__":
    main()
