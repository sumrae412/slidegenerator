#!/usr/bin/env python3
"""
Interactive Demo for Presentation Outline Generator

Demonstrates the AI-powered outline generation feature with various examples.
Allows users to try different scenarios interactively.
"""

import os
import sys
import json
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slide_generator_pkg.presentation_intelligence import PresentationIntelligence
from slide_generator_pkg.document_parser import DocumentParser
from slide_generator_pkg.powerpoint_generator import PowerPointGenerator


# Pre-defined example scenarios
EXAMPLE_SCENARIOS = {
    "1": {
        "name": "Executive Briefing - Cloud Migration",
        "topic": "Enterprise Cloud Migration Strategy",
        "audience": "C-level executives, non-technical",
        "duration_minutes": 15,
        "objectives": [
            "Explain business benefits of cloud migration",
            "Present ROI and cost analysis",
            "Address security and compliance concerns",
            "Outline migration roadmap and timeline"
        ],
        "additional_context": "Focus on financial impact and risk mitigation"
    },
    "2": {
        "name": "Technical Deep Dive - Kubernetes",
        "topic": "Kubernetes Architecture and Best Practices",
        "audience": "DevOps engineers with Docker experience",
        "duration_minutes": 45,
        "objectives": [
            "Explain Kubernetes core components and architecture",
            "Demonstrate pod deployment and management",
            "Show resource allocation and scaling strategies",
            "Cover monitoring and troubleshooting techniques"
        ],
        "additional_context": "Include hands-on deployment examples"
    },
    "3": {
        "name": "Educational - Machine Learning Basics",
        "topic": "Introduction to Machine Learning",
        "audience": "University students, beginner level",
        "duration_minutes": 30,
        "objectives": [
            "Define machine learning and its applications",
            "Explain supervised vs unsupervised learning",
            "Demonstrate simple algorithms with examples",
            "Show real-world use cases in various industries"
        ],
        "additional_context": "Use accessible language and visual examples"
    },
    "4": {
        "name": "Sales Enablement - Product Launch",
        "topic": "Q1 2025 Product Launch: SmartAnalytics Pro",
        "audience": "Sales team and channel partners",
        "duration_minutes": 20,
        "objectives": [
            "Introduce new product features and capabilities",
            "Explain competitive positioning and differentiation",
            "Provide pricing structure and deal guidance",
            "Share customer success stories and testimonials"
        ],
        "additional_context": "Emphasize ROI and competitive advantages"
    },
    "5": {
        "name": "Workshop - API Design",
        "topic": "RESTful API Design Best Practices",
        "audience": "Software developers and architects",
        "duration_minutes": 60,
        "objectives": [
            "Explain REST principles and constraints",
            "Demonstrate authentication and authorization patterns",
            "Show versioning and backward compatibility strategies",
            "Cover error handling and documentation approaches"
        ],
        "additional_context": "Include code examples and design patterns"
    }
}


def print_banner(text: str):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_outline_summary(outline: dict):
    """Print outline summary"""
    print(f"\n{'='*80}")
    print("OUTLINE GENERATED SUCCESSFULLY".center(80))
    print(f"{'='*80}\n")

    print(f"Title: {outline['presentation_title']}")
    print(f"Estimated Slides: {outline['estimated_slides']}")
    print(f"Speaking Time: {outline['speaking_time']} minutes")
    print(f"API Cost: ${outline['cost']:.4f}")
    print(f"Tokens: {outline['input_tokens']} input, {outline['output_tokens']} output")


def print_outline_details(outline: dict):
    """Print detailed slide-by-slide breakdown"""
    print(f"\n{'='*80}")
    print("SLIDE-BY-SLIDE BREAKDOWN".center(80))
    print(f"{'='*80}\n")

    for slide in outline['structure']:
        # Print slide header
        slide_type_emoji = {
            'title': '🎬',
            'section_header': '📑',
            'content': '📊',
            'conclusion': '🎯',
            'qa': '❓'
        }
        emoji = slide_type_emoji.get(slide['slide_type'], '📄')

        print(f"\n{emoji} Slide {slide['slide_number']}: {slide['title']}")
        print(f"   Type: {slide['slide_type']}")

        # Print key points
        if slide['key_points']:
            print(f"   Key Points:")
            for point in slide['key_points']:
                print(f"      • {point}")

        # Print notes preview
        if slide['notes']:
            notes_preview = slide['notes'][:150] + "..." if len(slide['notes']) > 150 else slide['notes']
            print(f"   Speaker Notes: {notes_preview}")

        print(f"   {'-'*76}")


def run_example_scenario(scenario_key: str, intelligence: PresentationIntelligence):
    """Run a pre-defined example scenario"""
    scenario = EXAMPLE_SCENARIOS[scenario_key]

    print_banner(f"SCENARIO: {scenario['name']}")

    print("Configuration:")
    print(f"  Topic: {scenario['topic']}")
    print(f"  Audience: {scenario['audience']}")
    print(f"  Duration: {scenario['duration_minutes']} minutes")
    print(f"  Objectives:")
    for obj in scenario['objectives']:
        print(f"    • {obj}")
    if scenario['additional_context']:
        print(f"  Additional Context: {scenario['additional_context']}")

    print("\n🎯 Generating outline...\n")

    # Generate outline
    outline = intelligence.generate_presentation_outline(
        topic=scenario['topic'],
        audience=scenario['audience'],
        duration_minutes=scenario['duration_minutes'],
        objectives=scenario['objectives'],
        additional_context=scenario['additional_context']
    )

    if 'error' in outline:
        print(f"❌ Error: {outline['error']}")
        return None

    # Print results
    print_outline_summary(outline)
    print_outline_details(outline)

    return outline


def run_custom_scenario(intelligence: PresentationIntelligence):
    """Run a custom user-defined scenario"""
    print_banner("CUSTOM PRESENTATION")

    print("Enter your presentation details:\n")

    # Get user input
    topic = input("Topic: ").strip()
    if not topic:
        print("❌ Topic is required")
        return None

    audience = input("Target Audience: ").strip()
    if not audience:
        print("❌ Audience is required")
        return None

    try:
        duration_minutes = int(input("Duration (minutes): ").strip())
        if duration_minutes <= 0:
            print("❌ Duration must be positive")
            return None
    except ValueError:
        print("❌ Invalid duration")
        return None

    print("\nEnter objectives (one per line, empty line to finish):")
    objectives = []
    while True:
        obj = input(f"  Objective {len(objectives)+1}: ").strip()
        if not obj:
            break
        objectives.append(obj)

    if not objectives:
        print("❌ At least one objective is required")
        return None

    additional_context = input("\nAdditional context (optional): ").strip()

    print("\n🎯 Generating outline...\n")

    # Generate outline
    outline = intelligence.generate_presentation_outline(
        topic=topic,
        audience=audience,
        duration_minutes=duration_minutes,
        objectives=objectives,
        additional_context=additional_context
    )

    if 'error' in outline:
        print(f"❌ Error: {outline['error']}")
        return None

    # Print results
    print_outline_summary(outline)
    print_outline_details(outline)

    return outline


def export_outline_to_json(outline: dict, filename: str = None):
    """Export outline to JSON file"""
    if filename is None:
        filename = f"outline_{outline['presentation_title'].replace(' ', '_')}.json"

    with open(filename, 'w') as f:
        json.dump(outline, f, indent=2)

    print(f"\n✅ Outline exported to: {filename}")


def create_presentation_from_outline(outline: dict, parser: DocumentParser, filename: str = None):
    """Create PowerPoint presentation from outline"""
    print("\n🎨 Creating PowerPoint presentation from outline...")

    try:
        # Convert outline to DocumentStructure
        slides = parser.presentation_intelligence.create_slides_from_outline(outline)

        from slide_generator_pkg.data_models import DocumentStructure

        doc = DocumentStructure(
            title=outline['presentation_title'],
            slides=slides,
            metadata={
                'source': 'AI-generated outline',
                'duration_minutes': outline['speaking_time'],
                'slide_count': len(slides),
                'generation_cost': outline['cost']
            }
        )

        # Generate PowerPoint
        generator = PowerPointGenerator()

        if filename is None:
            filename = f"{outline['presentation_title'].replace(' ', '_')}.pptx"

        pptx_path = generator.create_presentation(
            doc=doc,
            output_path=filename
        )

        print(f"\n✅ PowerPoint presentation created: {pptx_path}")
        return pptx_path

    except Exception as e:
        print(f"\n❌ Error creating presentation: {e}")
        return None


def interactive_menu():
    """Main interactive menu"""
    print_banner("PRESENTATION OUTLINE GENERATOR - INTERACTIVE DEMO")

    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("\nPlease set your Claude API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nOr set it in your .env file")
        return

    # Initialize components
    intelligence = PresentationIntelligence(claude_api_key=api_key)
    parser = DocumentParser(claude_api_key=api_key)

    current_outline = None

    while True:
        print("\n" + "="*80)
        print("MAIN MENU".center(80))
        print("="*80)
        print("\nExample Scenarios:")
        for key, scenario in EXAMPLE_SCENARIOS.items():
            print(f"  {key}. {scenario['name']} ({scenario['duration_minutes']} min)")

        print("\nOptions:")
        print("  C. Create custom presentation")
        print("  E. Export current outline to JSON")
        print("  P. Create PowerPoint from current outline")
        print("  Q. Quit")

        choice = input("\nSelect option: ").strip().upper()

        if choice == 'Q':
            print("\n👋 Goodbye!")
            break

        elif choice == 'C':
            current_outline = run_custom_scenario(intelligence)

        elif choice in EXAMPLE_SCENARIOS:
            current_outline = run_example_scenario(choice, intelligence)

        elif choice == 'E':
            if current_outline:
                export_outline_to_json(current_outline)
            else:
                print("\n❌ No outline to export. Generate one first.")

        elif choice == 'P':
            if current_outline:
                create_presentation_from_outline(current_outline, parser)
            else:
                print("\n❌ No outline to convert. Generate one first.")

        else:
            print("\n❌ Invalid choice. Please try again.")

        # Show cost summary
        if current_outline and 'cost' in current_outline:
            print(f"\nSession cost so far: ${intelligence.total_cost:.4f}")


def quick_demo():
    """Quick non-interactive demo of all scenarios"""
    print_banner("QUICK DEMO - ALL SCENARIOS")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not set")
        return

    intelligence = PresentationIntelligence(claude_api_key=api_key)

    total_cost = 0.0
    results = []

    for key, scenario in EXAMPLE_SCENARIOS.items():
        print(f"\n{'='*80}")
        print(f"Running: {scenario['name']}")
        print(f"{'='*80}")

        outline = intelligence.generate_presentation_outline(
            topic=scenario['topic'],
            audience=scenario['audience'],
            duration_minutes=scenario['duration_minutes'],
            objectives=scenario['objectives'],
            additional_context=scenario['additional_context']
        )

        if 'error' not in outline:
            results.append({
                'name': scenario['name'],
                'duration': scenario['duration_minutes'],
                'slides': outline['estimated_slides'],
                'cost': outline['cost']
            })
            total_cost += outline['cost']

            print(f"\n✅ Success: {outline['estimated_slides']} slides, ${outline['cost']:.4f}")
        else:
            print(f"\n❌ Error: {outline['error']}")

    # Print summary
    print(f"\n{'='*80}")
    print("DEMO SUMMARY".center(80))
    print(f"{'='*80}\n")

    print(f"{'Scenario':<45} {'Duration':<10} {'Slides':<10} {'Cost'}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<45} {r['duration']} min {r['slides']:<10} ${r['cost']:.4f}")

    print("-"*80)
    print(f"{'TOTAL':<45} {'':<10} {'':<10} ${total_cost:.4f}")
    print(f"\n✅ Completed {len(results)} scenarios")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Presentation Outline Generator Demo')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo of all scenarios (non-interactive)')
    parser.add_argument('--scenario', type=str, choices=list(EXAMPLE_SCENARIOS.keys()),
                       help='Run a specific scenario')

    args = parser.parse_args()

    if args.quick:
        quick_demo()
    elif args.scenario:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ERROR: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        intelligence = PresentationIntelligence(claude_api_key=api_key)
        run_example_scenario(args.scenario, intelligence)
    else:
        interactive_menu()
