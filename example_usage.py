"""
Example Usage of the Slide Generator Package

This script demonstrates how to use the modular slide_generator_pkg
to convert documents into presentations.
"""

import os
from slide_generator_pkg import (
    DocumentParser,
    SlideGenerator,
    GoogleSlidesGenerator,
    extract_google_doc_id,
    fetch_google_doc_content
)

def example_1_basic_document_to_powerpoint():
    """Example 1: Convert a text file to PowerPoint"""
    print("\n=== Example 1: Basic Document to PowerPoint ===")

    # Initialize parser with Claude API key (optional)
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    parser = DocumentParser(claude_api_key=api_key)

    # Parse a document (assumes you have a document file)
    doc_structure = parser.parse_file(
        file_path="path/to/your/document.txt",
        filename="document.txt",
        fast_mode=False  # Set True to skip AI processing
    )

    # Generate PowerPoint
    generator = SlideGenerator()
    pptx_path = generator.create_powerpoint(
        doc_structure=doc_structure,
        skip_visuals=False  # Set True to skip visual prompts
    )

    print(f"✅ PowerPoint created: {pptx_path}")


def example_2_google_doc_to_powerpoint():
    """Example 2: Convert Google Doc to PowerPoint"""
    print("\n=== Example 2: Google Doc to PowerPoint ===")

    # Extract document ID from Google Docs URL
    google_url = "https://docs.google.com/document/d/YOUR_DOC_ID/edit"
    doc_id = extract_google_doc_id(google_url)

    if doc_id:
        print(f"📄 Extracted Doc ID: {doc_id}")

        # Fetch content (works if doc is public or you provide credentials)
        content, error = fetch_google_doc_content(doc_id, credentials=None)

        if content:
            # Save to temp file
            temp_file = "/tmp/google_doc.txt"
            with open(temp_file, 'w') as f:
                f.write(content)

            # Parse and generate presentation
            parser = DocumentParser(claude_api_key=os.environ.get('ANTHROPIC_API_KEY'))
            doc_structure = parser.parse_file(temp_file, "Google Doc")

            generator = SlideGenerator()
            pptx_path = generator.create_powerpoint(doc_structure)

            print(f"✅ PowerPoint created: {pptx_path}")
        else:
            print(f"❌ Error fetching doc: {error}")


def example_3_create_google_slides():
    """Example 3: Create Google Slides instead of PowerPoint"""
    print("\n=== Example 3: Create Google Slides ===")

    # Parse document
    parser = DocumentParser(claude_api_key=os.environ.get('ANTHROPIC_API_KEY'))
    doc_structure = parser.parse_file("path/to/document.txt", "document.txt")

    # Create Google Slides (requires Google OAuth credentials)
    # You need to provide credentials dict from OAuth flow
    credentials = {
        'token': 'your-access-token',
        'refresh_token': 'your-refresh-token',
        'token_uri': 'https://oauth2.googleapis.com/token',
        'client_id': 'your-client-id',
        'client_secret': 'your-client-secret',
        'scopes': ['https://www.googleapis.com/auth/presentations']
    }

    generator = GoogleSlidesGenerator(credentials=credentials)
    presentation_url = generator.create_presentation(doc_structure)

    print(f"✅ Google Slides created: {presentation_url}")


def example_4_fast_mode_no_ai():
    """Example 4: Fast mode without AI processing"""
    print("\n=== Example 4: Fast Mode (No AI) ===")

    # Initialize parser without API key
    parser = DocumentParser(claude_api_key=None)

    # Parse in fast mode (uses basic text processing)
    doc_structure = parser.parse_file(
        file_path="path/to/document.txt",
        filename="document.txt",
        fast_mode=True  # Skip AI processing
    )

    # Generate PowerPoint
    generator = SlideGenerator()
    pptx_path = generator.create_powerpoint(doc_structure, skip_visuals=True)

    print(f"✅ PowerPoint created (fast mode): {pptx_path}")


def example_5_check_cache_stats():
    """Example 5: Check API cache statistics"""
    print("\n=== Example 5: API Cache Statistics ===")

    parser = DocumentParser(claude_api_key=os.environ.get('ANTHROPIC_API_KEY'))

    # Parse multiple documents to see caching in action
    # (assuming you have multiple files)

    # Get cache statistics
    stats = parser.get_cache_stats()
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Hit rate: {stats['hit_rate_percent']}%")
    print(f"Estimated savings: {stats['estimated_cost_savings']}")


if __name__ == "__main__":
    print("Slide Generator Package - Example Usage")
    print("=" * 50)

    # Uncomment the examples you want to run

    # example_1_basic_document_to_powerpoint()
    # example_2_google_doc_to_powerpoint()
    # example_3_create_google_slides()
    # example_4_fast_mode_no_ai()
    # example_5_check_cache_stats()

    print("\n✅ Examples complete!")
    print("\nTo use these examples:")
    print("1. Uncomment the example you want to run")
    print("2. Update file paths and credentials as needed")
    print("3. Set ANTHROPIC_API_KEY environment variable if using AI features")
