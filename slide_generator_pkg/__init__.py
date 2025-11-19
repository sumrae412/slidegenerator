"""
Slide Generator Package

A modular package for converting documents into presentation slides with AI-generated content.

Main Components:
- DocumentParser: Parse documents and generate bullet points
- SlideGenerator: Create PowerPoint presentations
- GoogleSlidesGenerator: Create Google Slides presentations
- Utils: Google Docs integration and helper functions

Quick Start:
    from slide_generator_pkg import DocumentParser, SlideGenerator

    # Parse document
    parser = DocumentParser(claude_api_key="your-api-key")
    doc_structure = parser.parse_file("document.txt", "document.txt")

    # Generate PowerPoint
    generator = SlideGenerator()
    pptx_path = generator.create_powerpoint(doc_structure)
"""

__version__ = "1.0.0"

# Import data models
from .data_models import SlideContent, DocumentStructure, SemanticChunk

# Import core components
from .document_parser import DocumentParser
from .powerpoint_generator import SlideGenerator
from .google_slides_generator import GoogleSlidesGenerator
from .semantic_analyzer import SemanticAnalyzer
from .visual_generator import VisualGenerator

# Import utilities
from .utils import (
    get_google_client_config,
    extract_google_doc_id,
    fetch_google_doc_content,
    CostTracker
)

# Public API
__all__ = [
    # Data Models
    'SlideContent',
    'DocumentStructure',
    'SemanticChunk',

    # Core Components
    'DocumentParser',
    'SlideGenerator',
    'GoogleSlidesGenerator',
    'SemanticAnalyzer',
    'VisualGenerator',

    # Utilities
    'get_google_client_config',
    'extract_google_doc_id',
    'fetch_google_doc_content',
    'CostTracker'
]
