#!/usr/bin/env python3
"""
Demo: Q&A Slide Generator

Demonstrates how to automatically generate FAQ/Q&A slides based on presentation content.
Helps presenters prepare for audience questions.
"""

import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slide_generator_pkg.data_models import SlideContent
from slide_generator_pkg.presentation_intelligence import PresentationIntelligence
from slide_generator_pkg.utils import CostTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_presentation() -> list:
    """
    Create a sample presentation about Cloud Migration.

    Returns:
        List of SlideContent objects
    """
    slides = [
        # Title slide
        SlideContent(
            title="Enterprise Cloud Migration Strategy",
            content=[],
            slide_type='title',
            heading_level=1
        ),

        # Section: Why Cloud?
        SlideContent(
            title="Why Migrate to Cloud?",
            content=[
                "Reduce infrastructure costs by 30-40% through pay-per-use pricing",
                "Scale resources up or down based on actual demand",
                "Improve disaster recovery with built-in redundancy and backups",
                "Enable remote work with anywhere access to applications",
                "Faster deployment of new services and features"
            ],
            slide_type='content',
            heading_level=3
        ),

        # Section: Migration Challenges
        SlideContent(
            title="Common Migration Challenges",
            content=[
                "Legacy applications may need refactoring or replacement",
                "Data transfer can take weeks for large datasets",
                "Security and compliance requirements vary by cloud provider",
                "Team needs training on new cloud platforms and tools",
                "Potential downtime during cutover period"
            ],
            slide_type='content',
            heading_level=3
        ),

        # Section: Migration Process
        SlideContent(
            title="Our Migration Process",
            content=[
                "Phase 1: Assessment and Planning (2-3 months)",
                "Phase 2: Proof of Concept with pilot applications (1-2 months)",
                "Phase 3: Migration of non-critical workloads (3-4 months)",
                "Phase 4: Migration of critical systems (4-6 months)",
                "Phase 5: Optimization and training (ongoing)"
            ],
            slide_type='content',
            heading_level=3
        ),

        # Section: Security Approach
        SlideContent(
            title="Security and Compliance",
            content=[
                "Implement zero-trust architecture with identity-based access",
                "Encrypt all data in transit and at rest",
                "Regular security audits and penetration testing",
                "Compliance certifications: SOC 2, ISO 27001, HIPAA",
                "24/7 security monitoring and incident response"
            ],
            slide_type='content',
            heading_level=3
        ),

        # Section: Expected Outcomes
        SlideContent(
            title="Expected Outcomes",
            content=[
                "30% reduction in infrastructure costs within first year",
                "99.9% uptime SLA (vs. current 95%)",
                "50% faster deployment of new features",
                "Improved disaster recovery (4-hour RTO vs. current 24-hour)",
                "Better scalability for peak periods (holiday season, sales)"
            ],
            slide_type='content',
            heading_level=3
        ),

        # Section: Timeline
        SlideContent(
            title="Timeline and Budget",
            content=[
                "Total timeline: 12-18 months from planning to completion",
                "Budget: $2.5M initial investment",
                "Expected annual savings: $1.2M (ROI in ~2 years)",
                "Key milestones: Pilot (Month 3), First production migration (Month 6)",
                "Dedicated team: 8 engineers, 2 project managers, 1 architect"
            ],
            slide_type='content',
            heading_level=3
        )
    ]

    return slides


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def main():
    """Main demo function."""
    print_separator()
    print("Q&A SLIDE GENERATOR - DEMO")
    print_separator()

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        print("   Get a key at: https://console.anthropic.com/")
        return

    print(f"\n✅ API key found: {api_key[:8]}...{api_key[-4:]}")

    # Create sample presentation
    print("\n" + "=" * 80)
    print("STEP 1: Creating Sample Presentation")
    print("=" * 80)

    slides = create_sample_presentation()
    print(f"\n📊 Created presentation with {len(slides)} slides:")
    for i, slide in enumerate(slides, 1):
        bullet_count = len(slide.content) if slide.content else 0
        print(f"   {i}. {slide.title} ({bullet_count} bullets)")

    # Initialize PresentationIntelligence
    print("\n" + "=" * 80)
    print("STEP 2: Initialize Q&A Generator")
    print("=" * 80)

    cost_tracker = CostTracker()
    intel = PresentationIntelligence(
        claude_api_key=api_key,
        cost_tracker=cost_tracker
    )
    print("\n✅ PresentationIntelligence initialized")

    # Generate Q&A slides
    print("\n" + "=" * 80)
    print("STEP 3: Generate Q&A Slides")
    print("=" * 80)

    print("\n🤔 Generating 5 Q&A slides...")
    print("   This analyzes the presentation content and creates realistic questions")
    print("   that an audience might ask, along with concise answers.\n")

    result = intel.generate_qa_slides(
        slides=slides,
        num_questions=5
    )

    # Display results
    print("\n" + "=" * 80)
    print("STEP 4: Results")
    print("=" * 80)

    print(f"\n✅ Generated {len(result['qa_slides'])} Q&A slides")
    print(f"📊 API Cost: ${result['cost']:.4f}")
    print(f"📋 Topics Covered: {', '.join(result['coverage_areas'])}")

    # Display each Q&A
    print("\n" + "=" * 80)
    print("GENERATED Q&A SLIDES")
    print("=" * 80)

    for i, qa in enumerate(result['questions'], 1):
        print(f"\n{'─' * 80}")
        print(f"Q&A SLIDE {i}")
        print(f"{'─' * 80}")

        print(f"\n📌 TITLE: Q: {qa['question']}")
        print(f"\n📊 Metadata:")
        print(f"   • Category: {qa['category']}")
        print(f"   • Difficulty: {qa['difficulty']}")
        print(f"   • Source slides: {qa.get('source_slides', 'N/A')}")

        print(f"\n💡 ANSWER:")
        for j, bullet in enumerate(qa['answer_bullets'], 1):
            print(f"   {j}. {bullet}")

        # Display corresponding SlideContent
        slide = result['qa_slides'][i-1]
        print(f"\n📄 Slide Object:")
        print(f"   • Title: {slide.title}")
        print(f"   • Type: {slide.slide_type}")
        print(f"   • Heading Level: {slide.heading_level}")
        print(f"   • Q&A Info: {slide.qa_info}")

    # Test with focus areas
    print("\n\n" + "=" * 80)
    print("STEP 5: Generate Q&A with Focus Areas")
    print("=" * 80)

    print("\n🎯 Generating 3 Q&A slides focused on 'security' and 'costs'...")

    result_focused = intel.generate_qa_slides(
        slides=slides,
        num_questions=3,
        focus_areas=['security', 'costs']
    )

    print(f"\n✅ Generated {len(result_focused['qa_slides'])} focused Q&A slides")
    print(f"📊 API Cost: ${result_focused['cost']:.4f}")

    print("\n" + "=" * 80)
    print("FOCUSED Q&A SLIDES")
    print("=" * 80)

    for i, qa in enumerate(result_focused['questions'], 1):
        print(f"\nQ{i}: {qa['question']}")
        print(f"    Category: {qa['category']}, Difficulty: {qa['difficulty']}")
        for bullet in qa['answer_bullets']:
            print(f"    • {bullet}")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_cost = result['cost'] + result_focused['cost']
    total_slides = len(result['qa_slides']) + len(result_focused['qa_slides'])

    print(f"\n✅ Successfully generated {total_slides} Q&A slides")
    print(f"📊 Total API Cost: ${total_cost:.4f}")
    print(f"💰 Average cost per Q&A: ${total_cost/total_slides:.4f}")

    print("\n" + "=" * 80)
    print("HOW TO USE IN YOUR APPLICATION")
    print("=" * 80)

    print("""
1. Import the PresentationIntelligence class:
   from slide_generator_pkg.presentation_intelligence import PresentationIntelligence

2. Initialize with your API key:
   intel = PresentationIntelligence(claude_api_key='your-key')

3. Generate Q&A slides:
   result = intel.generate_qa_slides(slides, num_questions=5)

4. Add Q&A slides to presentation:
   all_slides = slides + result['qa_slides']

5. The Q&A slides are ready to be added to PowerPoint or Google Slides!

RESULT STRUCTURE:
{
    'qa_slides': [SlideContent, ...],      # Ready-to-add slides
    'questions': [                          # Detailed Q&A info
        {
            'question': str,
            'answer_bullets': [str, ...],
            'source_slides': [int, ...],
            'difficulty': 'basic|intermediate|advanced',
            'category': 'clarification|implementation|concern|...'
        }
    ],
    'coverage_areas': [str, ...],          # Topics covered
    'cost': float                           # API cost in USD
}
    """)

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📊 Cost Tracking:")
    print(f"   • Total API calls: {cost_tracker.total_calls}")
    print(f"   • Total cost: ${cost_tracker.total_cost:.4f}")
    print(f"   • Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
