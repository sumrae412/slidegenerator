"""
Document Structure Validation Tests

Tests the parser's ability to correctly identify slide boundaries and
create the expected number of slides from various document formats.

This is separate from bullet quality tests - it validates document parsing
and slide structure recognition.
"""

import sys
import os
from typing import List, Dict
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_to_slides import DocumentParser, SlideContent


@dataclass
class StructureTest:
    """Test case for document structure validation"""
    id: str
    description: str
    file_path: str
    script_column: int
    expected_slide_count: int
    expected_title_slides: int  # Slides with headings (no bullets)
    expected_content_slides: int  # Slides with bullet content
    expected_heading_patterns: List[str]  # Headers that should be recognized
    skip_patterns: List[str] = None  # Rows that should be skipped (e.g., stage directions)


# Structural test cases
STRUCTURE_TEST_SET = [
    StructureTest(
        id="video_script_with_lesson_headers",
        description="Video script with plain text lesson headers + script table",
        file_path="ryans_doc.txt",
        script_column=1,
        expected_slide_count=15,  # 12 lesson headers + 3 content groups
        expected_title_slides=12,  # Lesson headers should be title slides
        expected_content_slides=3,  # Narration paragraphs grouped by headings
        expected_heading_patterns=[
            "Lesson 1 - Specialization & Course Introduction",
            "C1W1L1_1 - Welcome to AI for Good",
            "C1W1L1_2 - What is \u201cAI for Good\u201d?",  # Unicode curly quotes
            "C1W1L1_3 - Microsoft AI for Good Lab",
            "C1W1L1_4 - The Courses in this Specialization",
            "C1W1L1_5 - Project spotlight: Charles Onu",
            "Lesson 2 - Introduction to Artificial Intelligence and Machine Learning",
            "C1W1L2_1 - What is Artificial Intelligence?",
            "C1W1L2_2 - How Supervised Learning Works",
            "C1W1L2_3 - Project spotlight: Felipe Oviedo",
            "C1W1L2_4 - Considering the Impact of Your AI for Good Project",
            "C1W1L2_5 - Summary Week 1"
        ],
        skip_patterns=[
            "[VISUAL:",  # Stage directions should be skipped
            "[CUT TO",
            "[ANIMATION"
        ]
    ),

    StructureTest(
        id="markdown_headings",
        description="Document with markdown-style headings",
        file_path=None,  # Will use inline content
        script_column=0,
        expected_slide_count=5,
        expected_title_slides=2,  # H1 and H2 headings
        expected_content_slides=3,  # Paragraph content
        expected_heading_patterns=[
            "# Main Title",
            "## Section 1"
        ]
    ),

    StructureTest(
        id="script_table_narration_column",
        description="Script table with narration in column 1",
        file_path="ryans_doc.txt",
        script_column=1,
        expected_slide_count=15,  # Same as video_script test
        expected_title_slides=12,
        expected_content_slides=3,  # Narration paragraphs grouped by headings
        expected_heading_patterns=[
            "Lesson 1 - Specialization & Course Introduction",
            "C1W1L1_1 - Welcome to AI for Good"
        ]
    ),

    StructureTest(
        id="script_table_stage_directions_column",
        description="Script table with stage directions in column 2 (extracts visual cues)",
        file_path="ryans_doc.txt",
        script_column=2,
        expected_slide_count=13,  # 12 lesson headers + 1 content slide from stage directions
        expected_title_slides=12,
        expected_content_slides=1,  # Stage direction lines grouped into content
        expected_heading_patterns=[
            "Lesson 1 - Specialization & Course Introduction"
        ]
    )
]


class StructureValidator:
    """Validates document structure and slide generation"""

    def __init__(self, api_key: str = None):
        self.parser = DocumentParser(claude_api_key=api_key)

    def validate_structure(self, test: StructureTest) -> Dict:
        """
        Run a structure validation test

        Returns:
            Dict with validation results and detailed findings
        """
        print(f"\n{'=' * 70}")
        print(f"Testing: {test.id}")
        print(f"Description: {test.description}")
        print(f"{'=' * 70}")

        # Parse the document
        if test.file_path:
            doc_structure = self.parser.parse_file(
                test.file_path,
                test.file_path,
                script_column=test.script_column,
                fast_mode=False
            )
        else:
            # For inline tests, create test content
            # (Not implemented yet - placeholder)
            return {"error": "Inline content tests not yet implemented"}

        # Count slides
        total_slides = len(doc_structure.slides)

        # Categorize slides
        title_slides = []
        content_slides = []

        for slide in doc_structure.slides:
            if not slide.content or len(slide.content) == 0:
                # No bullets = title/heading slide
                title_slides.append(slide)
            else:
                # Has bullets = content slide
                content_slides.append(slide)

        # Check for expected heading patterns
        found_headings = []
        missing_headings = []

        all_slide_titles = [slide.title for slide in doc_structure.slides]

        for expected_heading in test.expected_heading_patterns:
            # Check if this heading appears in any slide title
            found = False
            for slide_title in all_slide_titles:
                if expected_heading.strip() in slide_title.strip():
                    found = True
                    found_headings.append(expected_heading)
                    break

            if not found:
                missing_headings.append(expected_heading)

        # Calculate results
        slide_count_match = total_slides == test.expected_slide_count
        title_count_match = len(title_slides) == test.expected_title_slides
        content_count_match = len(content_slides) == test.expected_content_slides
        all_headings_found = len(missing_headings) == 0

        passed = (
            slide_count_match and
            title_count_match and
            content_count_match and
            all_headings_found
        )

        # Print results
        print(f"\n📊 SLIDE COUNT:")
        print(f"  Expected: {test.expected_slide_count}")
        print(f"  Actual:   {total_slides}")
        print(f"  Match:    {'✅' if slide_count_match else '❌'}")

        print(f"\n📋 SLIDE TYPES:")
        print(f"  Title slides - Expected: {test.expected_title_slides}, Actual: {len(title_slides)} {'✅' if title_count_match else '❌'}")
        print(f"  Content slides - Expected: {test.expected_content_slides}, Actual: {len(content_slides)} {'✅' if content_count_match else '❌'}")

        print(f"\n🎯 HEADING RECOGNITION:")
        print(f"  Expected headings: {len(test.expected_heading_patterns)}")
        print(f"  Found: {len(found_headings)} {'✅' if all_headings_found else '❌'}")

        if missing_headings:
            print(f"\n  ❌ MISSING HEADINGS:")
            for heading in missing_headings[:5]:  # Show first 5
                print(f"     - {heading}")

        print(f"\n{'=' * 70}")
        print(f"RESULT: {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"{'=' * 70}")

        # Show first 3 slides for debugging
        if not passed:
            print(f"\n🔍 FIRST 3 SLIDES (for debugging):")
            for i, slide in enumerate(doc_structure.slides[:3], 1):
                print(f"\n  Slide {i}: {slide.title}")
                print(f"    Type: {slide.slide_type}")
                print(f"    Bullets: {len(slide.content) if slide.content else 0}")
                if slide.content and len(slide.content) > 0:
                    print(f"    First bullet: {slide.content[0][:60]}...")

        return {
            'test_id': test.id,
            'passed': passed,
            'total_slides': total_slides,
            'expected_slides': test.expected_slide_count,
            'title_slides': len(title_slides),
            'expected_title_slides': test.expected_title_slides,
            'content_slides': len(content_slides),
            'expected_content_slides': test.expected_content_slides,
            'found_headings': len(found_headings),
            'missing_headings': missing_headings,
            'all_slide_titles': all_slide_titles
        }

    def run_all_tests(self) -> List[Dict]:
        """Run all structure validation tests"""
        results = []

        print(f"\n{'=' * 70}")
        print(f"STRUCTURE VALIDATION TEST SUITE")
        print(f"Testing {len(STRUCTURE_TEST_SET)} test cases")
        print(f"{'=' * 70}")

        for i, test in enumerate(STRUCTURE_TEST_SET, 1):
            print(f"\n[{i}/{len(STRUCTURE_TEST_SET)}]")
            result = self.validate_structure(test)
            results.append(result)

        # Summary
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)

        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        print(f"{'=' * 70}")

        failed_tests = [r for r in results if not r.get('passed', False)]
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_tests:
                if 'error' in result:
                    print(f"  - {result.get('test_id', 'unknown')}: {result['error']}")
                else:
                    test_id = result.get('test_id', 'unknown')
                    total = result.get('total_slides', 0)
                    expected = result.get('expected_slides', 0)
                    print(f"  - {test_id}: Generated {total} slides, expected {expected}")

        return results


def main():
    """Run structure validation tests from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate document structure and slide generation')
    parser.add_argument('--test', help='Run specific test by ID', default=None)
    parser.add_argument('--api-key', help='Claude API key (optional)', default=None)

    args = parser.parse_args()

    validator = StructureValidator(api_key=args.api_key)

    if args.test:
        # Run specific test
        test = next((t for t in STRUCTURE_TEST_SET if t.id == args.test), None)
        if not test:
            print(f"❌ Test '{args.test}' not found")
            print(f"Available tests: {[t.id for t in STRUCTURE_TEST_SET]}")
            sys.exit(1)

        result = validator.validate_structure(test)
        sys.exit(0 if result['passed'] else 1)
    else:
        # Run all tests
        results = validator.run_all_tests()
        all_passed = all(r.get('passed', False) for r in results)
        sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
