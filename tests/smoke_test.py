"""
Smoke Test - Quick validation before deployment

Run this before every deployment to catch obvious regressions.
Takes ~30 seconds vs full benchmark which takes ~5 minutes.

Usage:
    python tests/smoke_test.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_to_slides import DocumentParser

# Minimal test set for smoke testing
SMOKE_TESTS = [
    {
        "name": "Educational paragraph",
        "text": """Students in this course will learn to apply machine learning algorithms
                  to real-world datasets using Python and scikit-learn.""",
        "heading": "Machine Learning Basics",
        "expected_style": "educational",
        "min_bullets": 2,
        "must_contain": ["students", "machine learning"]
    },
    {
        "name": "Technical content",
        "text": """The microservices architecture enables independent deployment of application
                  components. Services communicate via REST APIs using JSON payloads.""",
        "heading": "Architecture",
        "expected_style": "technical",
        "min_bullets": 2,
        "must_contain": ["microservices", "services"]
    },
    {
        "name": "Table structure",
        "text": """Feature\tBasic\tPremium
Storage\t10GB\t100GB
Users\t5\t25""",
        "heading": "Pricing",
        "expected_style": "professional",
        "min_bullets": 2,
        "must_contain": ["storage", "users"]
    },
    {
        "name": "Executive metrics",
        "text": """Digital transformation reduced costs by 23% in Q3. Customer satisfaction
                  improved from 72% to 86% following the UX redesign.""",
        "heading": "Results",
        "expected_style": "executive",
        "min_bullets": 2,
        "must_contain": ["costs", "satisfaction"]
    }
]


def run_smoke_test():
    """
    Run quick smoke tests
    """
    print("\n" + "=" * 70)
    print("SMOKE TEST - Quick Validation")
    print("=" * 70 + "\n")

    # Test without API key (NLP fallback)
    parser = DocumentParser(claude_api_key=None)

    passed = 0
    failed = 0

    for i, test in enumerate(SMOKE_TESTS, 1):
        print(f"[{i}/{len(SMOKE_TESTS)}] {test['name']}...", end=' ')

        try:
            # Generate bullets
            bullets = parser._create_unified_bullets(
                test['text'],
                context_heading=test['heading']
            )

            # Validate
            errors = []

            # Check bullet count
            if len(bullets) < test['min_bullets']:
                errors.append(f"Too few bullets: {len(bullets)} < {test['min_bullets']}")

            # Check bullet length
            for bullet in bullets:
                word_count = len(bullet.split())
                if word_count < 5:
                    errors.append(f"Bullet too short: {word_count} words")
                if word_count > 20:
                    errors.append(f"Bullet too long: {word_count} words")

            # Check keywords
            bullets_text = ' '.join(bullets).lower()
            for keyword in test['must_contain']:
                if keyword not in bullets_text:
                    errors.append(f"Missing keyword: '{keyword}'")

            if errors:
                print("❌ FAIL")
                for error in errors:
                    print(f"  └─ {error}")
                print(f"  Generated bullets:")
                for j, bullet in enumerate(bullets, 1):
                    print(f"     {j}. {bullet}")
                failed += 1
            else:
                print("✅ PASS")
                passed += 1

        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"SMOKE TEST RESULTS: {passed}/{len(SMOKE_TESTS)} passed")
    print("=" * 70 + "\n")

    if failed > 0:
        print(f"❌ {failed} test(s) failed - DO NOT DEPLOY")
        sys.exit(1)
    else:
        print("✅ All smoke tests passed - Safe to deploy")
        sys.exit(0)


if __name__ == '__main__':
    run_smoke_test()
