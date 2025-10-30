#!/usr/bin/env python3
"""
Integration Tests for Slide Generator
Tests end-to-end functionality and deployment-specific issues
"""

import sys
import os
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_to_slides import DocumentParser

def test_class_methods_available():
    """
    Test that all critical methods are available on DocumentParser.
    This catches deployment issues where methods exist in code but aren't loaded.
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST 1: Class Method Availability")
    print("=" * 70)

    parser = DocumentParser(claude_api_key=None)

    # Critical methods that must be available
    required_methods = [
        '_create_unified_bullets',
        'parse_file',
        '_content_to_slides',
        'get_cache_stats',
        '_get_cached_response',
        '_cache_response',
        '_generate_cache_key',
        '_detect_table_structure',
        '_summarize_table',
        '_handle_minimal_input',
    ]

    # Optional methods that should be available if not disabled
    optional_methods = [
        '_validate_heading_hierarchy',
        '_optimize_slide_density',
        '_insert_section_dividers',
    ]

    missing_required = []
    missing_optional = []

    for method_name in required_methods:
        if not hasattr(parser, method_name):
            missing_required.append(method_name)
        else:
            method = getattr(parser, method_name)
            if not callable(method):
                missing_required.append(f"{method_name} (not callable)")
            else:
                print(f"  ✅ {method_name}")

    for method_name in optional_methods:
        if not hasattr(parser, method_name):
            missing_optional.append(method_name)
        else:
            method = getattr(parser, method_name)
            if callable(method):
                print(f"  ⚠️  {method_name} (optional)")
            else:
                missing_optional.append(f"{method_name} (not callable)")

    if missing_required:
        print("\n❌ FAIL: Missing required methods:")
        for method in missing_required:
            print(f"  - {method}")
        return False

    if missing_optional:
        print(f"\n⚠️  WARNING: Missing optional methods (may be disabled):")
        for method in missing_optional:
            print(f"  - {method}")

    print("\n✅ PASS: All required methods available")
    return True


def test_end_to_end_bullet_generation():
    """
    Test complete bullet generation workflow.
    This catches integration issues between components.
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST 2: End-to-End Bullet Generation")
    print("=" * 70)

    parser = DocumentParser(claude_api_key=None)

    test_cases = [
        {
            "name": "Educational paragraph",
            "text": "Machine learning algorithms learn patterns from data to make predictions.",
            "heading": "Introduction to ML",
            "min_bullets": 1,
            "max_bullets": 5,
        },
        {
            "name": "Table data",
            "text": "Feature\tBasic\tPremium\nStorage\t10GB\t100GB",
            "heading": "Plans",
            "min_bullets": 1,
            "max_bullets": 5,
        },
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\nTesting: {tc['name']}")
        try:
            bullets = parser._create_unified_bullets(tc['text'], context_heading=tc['heading'])

            if not bullets:
                print(f"  ❌ FAIL: No bullets generated")
                all_passed = False
                continue

            if len(bullets) < tc['min_bullets']:
                print(f"  ❌ FAIL: Too few bullets ({len(bullets)} < {tc['min_bullets']})")
                all_passed = False
                continue

            if len(bullets) > tc['max_bullets']:
                print(f"  ⚠️  WARNING: Many bullets ({len(bullets)} > {tc['max_bullets']})")

            print(f"  ✅ Generated {len(bullets)} bullets")
            for i, bullet in enumerate(bullets, 1):
                print(f"     {i}. {bullet[:60]}{'...' if len(bullet) > 60 else ''}")

        except Exception as e:
            print(f"  ❌ FAIL: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n✅ PASS: All end-to-end tests passed")
    else:
        print("\n❌ FAIL: Some end-to-end tests failed")

    return all_passed


def test_cache_functionality():
    """
    Test that caching system works correctly.
    This catches deployment issues with cache state management.
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST 3: Cache Functionality")
    print("=" * 70)

    parser = DocumentParser(claude_api_key=None)

    # Reset cache stats
    initial_stats = parser.get_cache_stats()
    print(f"Initial cache stats: {initial_stats}")

    text = "Machine learning uses data to train predictive models."
    heading = "ML Overview"

    # First call - should be cache miss
    print("\nFirst call (expecting cache miss)...")
    bullets1 = parser._create_unified_bullets(text, context_heading=heading)
    stats_after_first = parser.get_cache_stats()

    # Second call - should be cache hit
    print("Second call (expecting cache hit)...")
    bullets2 = parser._create_unified_bullets(text, context_heading=heading)
    stats_after_second = parser.get_cache_stats()

    # Verify cache worked
    if bullets1 != bullets2:
        print("❌ FAIL: Cache returned different results")
        return False

    if stats_after_second['cache_hits'] <= stats_after_first['cache_hits']:
        print(f"❌ FAIL: Cache hit count didn't increase")
        print(f"  After first:  {stats_after_first}")
        print(f"  After second: {stats_after_second}")
        return False

    hit_rate = (stats_after_second['cache_hits'] / max(stats_after_second['total_requests'], 1)) * 100

    print(f"\n✅ PASS: Cache working correctly")
    print(f"  Cache hits: {stats_after_second['cache_hits']}/{stats_after_second['total_requests']} ({hit_rate:.1f}%)")
    print(f"  Results identical: {bullets1 == bullets2}")

    return True


def test_error_handling():
    """
    Test that error handling works correctly.
    This catches issues with exception handling and fallbacks.
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST 4: Error Handling")
    print("=" * 70)

    parser = DocumentParser(claude_api_key=None)

    # Test with empty text
    print("\nTesting with empty text...")
    try:
        bullets = parser._create_unified_bullets("", context_heading="Empty")
        if bullets:
            print(f"  ⚠️  WARNING: Generated bullets from empty text: {bullets}")
        else:
            print(f"  ✅ Correctly returned empty result")
    except Exception as e:
        print(f"  ❌ FAIL: Exception on empty text: {e}")
        return False

    # Test with very short text
    print("\nTesting with very short text...")
    try:
        bullets = parser._create_unified_bullets("Data.", context_heading="Short")
        print(f"  ✅ Handled short text: {len(bullets)} bullets")
    except Exception as e:
        print(f"  ❌ FAIL: Exception on short text: {e}")
        return False

    # Test with very long text
    print("\nTesting with very long text...")
    try:
        long_text = " ".join(["Machine learning analyzes data."] * 100)
        bullets = parser._create_unified_bullets(long_text, context_heading="Long")
        print(f"  ✅ Handled long text: {len(bullets)} bullets")
    except Exception as e:
        print(f"  ❌ FAIL: Exception on long text: {e}")
        return False

    print("\n✅ PASS: Error handling working correctly")
    return True


def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUITE")
    print("=" * 70)
    print("\nThese tests catch deployment-specific issues like:")
    print("  - Missing methods (AttributeError)")
    print("  - Import failures")
    print("  - Cache state issues")
    print("  - Exception handling problems")
    print("=" * 70)

    tests = [
        ("Class Method Availability", test_class_methods_available),
        ("End-to-End Bullet Generation", test_end_to_end_bullet_generation),
        ("Cache Functionality", test_cache_functionality),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 70)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("=" * 70)

    if passed_count == total_count:
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} INTEGRATION TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
