#!/usr/bin/env python3
"""
Seed Example Database with Golden Test Set

This script initializes the example storage with high-quality examples
from the golden test set.
"""

import sys
sys.path.insert(0, '/home/user/slidegenerator')

from example_storage import ExampleStorageManager, BulletExample
from tests.golden_test_set import GOLDEN_TEST_SET


def seed_examples():
    """Seed the example database with golden test examples."""
    manager = ExampleStorageManager()

    print("🌱 Seeding example database from golden test set...")
    print(f"Found {len(GOLDEN_TEST_SET)} golden examples")

    added_count = 0
    for test_case in GOLDEN_TEST_SET:
        example = BulletExample(
            input_text=test_case['input_text'],
            generated_bullets=test_case['expected_bullets'],
            context_heading=test_case.get('context_heading', 'Example'),
            style=test_case['category'],
            content_type='paragraph',  # Default, can be refined
            quality_score=85.0,  # High quality since these are hand-crafted
            category_tags=[test_case['id'], test_case['category']]
        )

        manager.add_example(example)
        added_count += 1

    # Get stats
    stats = manager.get_stats()

    print(f"\n✅ Seeding complete!")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   By style:")
    for style, count in stats['by_style'].items():
        print(f"     - {style}: {count}")
    print(f"   Average quality score: {stats['avg_quality_score']:.1f}")

    return stats


if __name__ == "__main__":
    seed_examples()
