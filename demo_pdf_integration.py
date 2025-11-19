#!/usr/bin/env python3
"""
Demonstration of PDF Integration in DocumentParser

This script shows how the PDF parsing integration works.
"""
import logging
from slide_generator_pkg.document_parser import DocumentParser, PDF_AVAILABLE

# Set up logging to see the detailed process
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

def demonstrate_pdf_integration():
    """Demonstrate PDF parsing integration"""

    print("=" * 70)
    print("PDF INTEGRATION DEMONSTRATION")
    print("=" * 70)

    print("\n1. PDF Support Status")
    print(f"   PDF_AVAILABLE: {PDF_AVAILABLE}")

    if PDF_AVAILABLE:
        print("   ✅ PDF parsing libraries are installed")
        print("   Libraries available: pdfplumber and/or PyPDF2")
    else:
        print("   ⚠️ PDF parsing libraries not installed")
        print("   Install with: pip install pdfplumber")

    print("\n2. Creating DocumentParser Instance")
    parser = DocumentParser()
    print("   ✅ DocumentParser created")

    print("\n3. Available Parsing Methods")
    methods = []
    if hasattr(parser, '_parse_docx'):
        methods.append('DOCX')
    if hasattr(parser, '_parse_txt'):
        methods.append('TXT')
    if hasattr(parser, '_parse_pdf'):
        methods.append('PDF')

    print(f"   Supported formats: {', '.join(methods)}")

    print("\n4. PDF Parsing Workflow")
    print("   When parse_file() receives a PDF:")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │ 1. Check PDF_AVAILABLE flag                │")
    print("   │ 2. Create PDFParser instance                │")
    print("   │ 3. Detect if PDF is scanned (warn if so)   │")
    print("   │ 4. Extract text + tables (tab-delimited)   │")
    print("   │ 5. Write to temporary .txt file             │")
    print("   │ 6. Call _parse_txt() on temp file          │")
    print("   │ 7. Clean up temp file (in finally block)   │")
    print("   │ 8. Return processed content                 │")
    print("   └─────────────────────────────────────────────┘")

    print("\n5. Key Features")
    print("   ✅ Scanned PDF detection with warning")
    print("   ✅ Table extraction (via pdfplumber)")
    print("   ✅ Column filtering (script_column parameter)")
    print("   ✅ Reuses existing TXT parser logic")
    print("   ✅ Automatic temp file cleanup")
    print("   ✅ Helpful error messages")
    print("   ✅ Comprehensive logging")

    print("\n6. Error Handling")
    if not PDF_AVAILABLE:
        print("   Testing error message when libraries not installed:")
        print("   " + "-" * 60)
        try:
            parser._parse_pdf("test.pdf", script_column=2)
        except ValueError as e:
            for line in str(e).split('\n'):
                print(f"   {line}")
        print("   " + "-" * 60)

    print("\n7. Usage Example")
    print("   ```python")
    print("   from slide_generator_pkg.document_parser import DocumentParser")
    print()
    print("   parser = DocumentParser(claude_api_key='sk-...')")
    print()
    print("   # Parse PDF file")
    print("   doc = parser.parse_file(")
    print("       file_path='/path/to/presentation.pdf',")
    print("       filename='presentation.pdf',")
    print("       script_column=2  # Extract from column 2")
    print("   )")
    print()
    print("   # Generated slides")
    print("   for slide in doc.slides:")
    print("       print(f'{slide.title}: {len(slide.content)} bullets')")
    print("   ```")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  • Test with a real PDF file")
    print("  • Try with PDF containing tables")
    print("  • Test with scanned PDF to see warning")
    print()

if __name__ == '__main__':
    demonstrate_pdf_integration()
