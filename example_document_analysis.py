#!/usr/bin/env python3
"""
Example usage of document structure analysis.

This demonstrates how to use the analyze_document_structure() method
to automatically detect whether a document is table-based or text-based,
and get a suggested parsing mode.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slide_generator_pkg.document_parser import DocumentParser


def analyze_and_suggest(file_path):
    """
    Analyze a document and print suggestions for parsing mode.

    Args:
        file_path: Path to the document file
    """
    # Initialize parser
    parser = DocumentParser()

    # Detect file extension
    file_ext = file_path.split('.')[-1].lower()

    print(f"\nAnalyzing: {file_path}")
    print(f"Format: {file_ext.upper()}")
    print("-" * 60)

    # Analyze document structure
    analysis = parser.analyze_document_structure(file_path, file_ext)

    # Print results
    print(f"📊 Structure Analysis:")
    print(f"  • Tables detected:     {analysis['tables']}")
    print(f"  • Paragraphs detected: {analysis['paragraphs']}")
    print(f"  • Table cells:         {analysis['table_cells']}")
    print(f"\n🎯 Classification:")
    print(f"  • Primary type:        {analysis['primary_type'].upper()}")
    print(f"  • Confidence:          {analysis['confidence'].upper()}")

    # Provide parsing recommendations
    print(f"\n💡 Recommendation:")
    if analysis['suggested_mode'] == 0:
        print(f"  • Use PARAGRAPH MODE (script_column=0)")
        print(f"  • This document is primarily text-based")
        print(f"  • All paragraphs will be extracted and converted to slides")
    else:
        print(f"  • Use COLUMN MODE (script_column={analysis['suggested_mode']})")
        print(f"  • This document is primarily table-based")
        print(f"  • Extract content from column {analysis['suggested_mode']}")

    # Show sample code
    print(f"\n📝 Sample Code:")
    print(f"  parser = DocumentParser()")
    print(f"  slides = parser.parse_file(")
    print(f"      file_path='{file_path}',")
    print(f"      filename='{os.path.basename(file_path)}',")
    print(f"      script_column={analysis['suggested_mode']}")
    print(f"  )")

    return analysis


def main():
    """
    Example usage with command line arguments.

    Usage:
        python example_document_analysis.py <path-to-document>
    """
    if len(sys.argv) < 2:
        print("Usage: python example_document_analysis.py <path-to-document>")
        print("\nSupported formats: DOCX, TXT, PDF")
        print("\nExample:")
        print("  python example_document_analysis.py my_document.docx")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Analyze the document
    analysis = analyze_and_suggest(file_path)

    # Additional guidance based on confidence
    print(f"\n{'='*60}")
    if analysis['confidence'] == 'high':
        print("✅ High confidence recommendation - safe to use suggested mode")
    else:
        print("⚠️  Low confidence - document has mixed content")
        print("   Consider manually inspecting the document or trying both modes")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
