#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple translation demo - shows structure without special characters"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slide_generator_pkg.content_transformer import ContentTransformer, SUPPORTED_LANGUAGES
from slide_generator_pkg.data_models import SlideContent
from slide_generator_pkg.utils import CostTracker

def main():
    print("\n" + "="*80)
    print("MULTILINGUAL TRANSLATION FEATURE DEMONSTRATION")
    print("="*80)

    # Show supported languages
    print(f"\nSupported Languages ({len(SUPPORTED_LANGUAGES)}):")
    for code, name in sorted(list(SUPPORTED_LANGUAGES.items())[:10]):
        print(f"  {code}: {name}")
    print("  ... and 10 more languages")

    # Original slide
    original_slide = SlideContent(
        title="Cloud Cost Optimization",
        content=[
            "Reduce infrastructure costs by 40-60%",
            "Pay-per-use model eliminates upfront investment",
            "Automatic scaling prevents over-provisioning"
        ]
    )

    print("\nOriginal Slide (English):")
    print(f"  Title: {original_slide.title}")
    for i, bullet in enumerate(original_slide.content, 1):
        print(f"  {i}. {bullet}")

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n[INFO] No API key found - showing demo structure only")
        print("\nWith an API key, this would translate to:")
        print("  - Spanish (es)")
        print("  - French (fr)")
        print("  - German (de)")
        print("  - Chinese (zh)")
        print("  - Japanese (ja)")
        print("\nTo run live translations:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  python tests/demo_translation_simple.py")
        return

    # Run actual translations
    try:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=api_key)
        cost_tracker = CostTracker()
        transformer = ContentTransformer(client=claude_client, cost_tracker=cost_tracker)

        print("\n[OK] API key found - running live translations...")

        for lang_code in ['es', 'de', 'fr']:
            lang_name = SUPPORTED_LANGUAGES[lang_code]
            print(f"\nTranslating to {lang_name}...")

            result = transformer.translate_slide(original_slide, lang_code)

            if result['success']:
                translated = result['translated_slide']
                print(f"  Title: {translated.title}")
                print(f"  Bullets: {len(translated.content)} translated")
                print(f"  Cost: ${result['cost']:.4f}")
            else:
                print(f"  [FAIL] {result['error']}")

        print("\n[OK] Demo complete")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")

if __name__ == '__main__':
    main()
