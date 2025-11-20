#!/usr/bin/env python
"""
Demo: Presentation Quality Review Feature

Demonstrates the AI-powered presentation quality analysis feature.
Shows how to use the feature and displays example output.
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slide_generator_pkg.presentation_intelligence import PresentationIntelligence
from slide_generator_pkg.document_parser import DocumentParser
from slide_generator_pkg.data_models import SlideContent


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_quality_report(result):
    """Print a formatted quality report"""
    print(f"\n📊 PRESENTATION QUALITY REPORT")
    print(f"{'─' * 80}")

    # Overall score
    print(f"\n🎯 Overall Quality Score: {result['quality_score']:.1f}/100")

    # Detailed scores
    print(f"\n📈 Detailed Scores:")
    print(f"   Flow (Logical Progression):     {result['scores']['flow']:.1f}/100")
    print(f"   Coherence (Topic Connectivity): {result['scores']['coherence']:.1f}/100")
    print(f"   Redundancy (Less is Better):    {result['scores']['redundancy']:.1f}/100")
    print(f"   Completeness (Coverage):        {result['scores']['completeness']:.1f}/100")

    # Issues
    if result['issues']:
        print(f"\n⚠️  Issues Found ({len(result['issues'])}):")
        for i, issue in enumerate(result['issues'], 1):
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            emoji = severity_emoji.get(issue['severity'], "⚪")
            print(f"   {i}. {emoji} [{issue['severity'].upper()}] {issue['type'].upper()}")
            print(f"      Slides: {issue['slides']}")
            print(f"      {issue['description']}")
    else:
        print(f"\n✅ No major issues detected")

    # Recommendations
    if result['recommendations']:
        print(f"\n💡 Recommendations ({len(result['recommendations'])}):")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")

    # Strengths
    if result['strengths']:
        print(f"\n✨ Strengths ({len(result['strengths'])}):")
        for i, strength in enumerate(result['strengths'], 1):
            print(f"   {i}. {strength}")

    # Cost
    print(f"\n💰 Analysis Cost: ${result['cost']:.4f}")
    print(f"{'─' * 80}")


def demo_standalone_usage():
    """Demo: Using PresentationIntelligence directly"""
    print_header("DEMO 1: Standalone Usage (Direct API)")

    print("\nThis demo shows how to use PresentationIntelligence directly.")
    print("\nCode:")
    print("""
    from slide_generator_pkg.presentation_intelligence import PresentationIntelligence
    from slide_generator_pkg.data_models import SlideContent

    # Initialize
    intel = PresentationIntelligence(claude_api_key="your-key")

    # Create slides
    slides = [
        SlideContent(title="Intro", content=["Point 1", "Point 2"]),
        SlideContent(title="Main", content=["Point A", "Point B"]),
    ]

    # Analyze
    result = intel.analyze_presentation_quality(slides)
    """)

    # Create sample slides
    slides = [
        SlideContent(
            title="Introduction to Machine Learning",
            content=[
                "Machine learning enables computers to learn from data",
                "Improves performance through experience",
                "Used in various applications from email filtering to self-driving cars"
            ],
            slide_type='content'
        ),
        SlideContent(
            title="Types of Machine Learning",
            content=[
                "Supervised learning: learns from labeled data",
                "Unsupervised learning: finds patterns in unlabeled data",
                "Reinforcement learning: learns through trial and error"
            ],
            slide_type='content'
        ),
        SlideContent(
            title="Applications and Use Cases",
            content=[
                "Natural language processing for chatbots",
                "Computer vision for image recognition",
                "Recommendation systems for personalization",
                "Fraud detection in financial transactions"
            ],
            slide_type='content'
        )
    ]

    print(f"\n📝 Sample Presentation ({len(slides)} slides):")
    for i, slide in enumerate(slides, 1):
        print(f"\n   Slide {i}: {slide.title}")
        for bullet in slide.content:
            print(f"      • {bullet}")

    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"\n✅ API key found - running analysis...")
        intel = PresentationIntelligence(claude_api_key=api_key)
        result = intel.analyze_presentation_quality(slides)
        print_quality_report(result)
    else:
        print(f"\n⚠️  ANTHROPIC_API_KEY not set - showing example output structure")
        print("\nExpected output structure:")
        example_output = {
            "quality_score": 85.0,
            "scores": {
                "flow": 90.0,
                "coherence": 88.0,
                "redundancy": 82.0,
                "completeness": 80.0
            },
            "issues": [
                {
                    "type": "gap",
                    "severity": "medium",
                    "slides": [2, 3],
                    "description": "Missing transition between Types and Applications"
                }
            ],
            "recommendations": [
                "Add a transition slide between 'Types' and 'Applications'",
                "Expand on how each ML type applies to the use cases shown",
                "Consider adding examples for each learning type"
            ],
            "strengths": [
                "Clear logical progression from intro to specifics",
                "Good coverage of fundamental ML concepts"
            ],
            "cost": 0.0234
        }
        print(json.dumps(example_output, indent=2))


def demo_document_parser_integration():
    """Demo: Using DocumentParser with quality review enabled"""
    print_header("DEMO 2: DocumentParser Integration")

    print("\nThis demo shows how to enable quality review in DocumentParser.")
    print("\nCode:")
    print("""
    from slide_generator_pkg.document_parser import DocumentParser

    # Initialize with quality review enabled
    parser = DocumentParser(
        claude_api_key="your-key",
        enable_quality_review=True  # Enable quality review
    )

    # Parse document
    doc_structure = parser.parse_file("document.docx", "document.docx")

    # Access quality review results
    quality_review = doc_structure.metadata.get('quality_review')
    if quality_review:
        print(f"Quality Score: {quality_review['quality_score']}")
    """)

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"\n✅ API key found - you can now use:")
        print(f"   parser = DocumentParser(enable_quality_review=True)")
        print(f"   Results will be stored in metadata['quality_review']")
    else:
        print(f"\n⚠️  To use this feature, set ANTHROPIC_API_KEY environment variable")


def demo_use_cases():
    """Demo: Real-world use cases"""
    print_header("DEMO 3: Real-World Use Cases")

    print("\n🎯 Use Case 1: Pre-Presentation Review")
    print("   Before delivering a presentation, run quality analysis to:")
    print("   • Identify redundant slides")
    print("   • Find missing transitions")
    print("   • Ensure logical flow")
    print("   • Get actionable improvement suggestions")

    print("\n🎯 Use Case 2: Automated Quality Gates")
    print("   Integrate into CI/CD pipeline:")
    print("   • Enforce minimum quality scores")
    print("   • Block presentations with critical issues")
    print("   • Track quality metrics over time")

    print("\n🎯 Use Case 3: Presentation Coaching")
    print("   Help presenters improve their slides:")
    print("   • Identify weak areas (flow, coherence)")
    print("   • Provide specific recommendations")
    print("   • Highlight what's working well")

    print("\n🎯 Use Case 4: Content Audit")
    print("   Review existing presentation library:")
    print("   • Find presentations that need updates")
    print("   • Identify common quality issues")
    print("   • Prioritize improvement efforts")


def demo_integration_example():
    """Demo: Full integration example"""
    print_header("DEMO 4: Full Integration Example")

    print("\n📋 Complete workflow example:")
    print("""
    # 1. Initialize parser with quality review
    parser = DocumentParser(
        claude_api_key=os.getenv('ANTHROPIC_API_KEY'),
        enable_quality_review=True,
        enable_speaker_notes=True  # Can combine with other features
    )

    # 2. Parse document
    doc_structure = parser.parse_file('presentation.docx', 'presentation.docx')

    # 3. Access quality review
    quality_review = doc_structure.metadata.get('quality_review')

    # 4. Check quality score
    if quality_review and quality_review['quality_score'] < 70:
        print("⚠️  Quality score below threshold!")
        print(f"Score: {quality_review['quality_score']:.1f}/100")

        # 5. Display issues
        for issue in quality_review['issues']:
            if issue['severity'] == 'high':
                print(f"❌ CRITICAL: {issue['description']}")

        # 6. Show recommendations
        print("\\n💡 Recommendations:")
        for rec in quality_review['recommendations']:
            print(f"  • {rec}")
    else:
        print("✅ Presentation quality looks good!")
    """)


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  AI PRESENTATION QUALITY REVIEW - FEATURE DEMO")
    print("=" * 80)
    print("\nThis feature analyzes presentations for:")
    print("  • Flow: Logical progression and smooth transitions")
    print("  • Coherence: Topic connectivity and narrative consistency")
    print("  • Redundancy: Duplicate or overly similar content")
    print("  • Completeness: Coverage gaps and missing information")

    # Run demos
    demo_standalone_usage()
    demo_document_parser_integration()
    demo_use_cases()
    demo_integration_example()

    # Final notes
    print_header("GETTING STARTED")
    print("\n1. Set your API key:")
    print("   export ANTHROPIC_API_KEY='your-key-here'")
    print("\n2. Enable in DocumentParser:")
    print("   parser = DocumentParser(enable_quality_review=True)")
    print("\n3. Or use PresentationIntelligence directly:")
    print("   intel = PresentationIntelligence(claude_api_key='your-key')")
    print("   result = intel.analyze_presentation_quality(slides)")
    print("\n4. Run tests:")
    print("   python tests/test_quality_review_simple.py")

    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
