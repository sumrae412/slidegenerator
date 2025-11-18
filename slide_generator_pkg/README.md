# Slide Generator Package

A modular Python package for converting documents into presentation slides with AI-generated bullet points and content.

## 📦 Package Structure

```
slide_generator_pkg/
├── __init__.py                    # Package initialization and public API
├── data_models.py                 # Core data classes (SlideContent, DocumentStructure, etc.)
├── document_parser.py             # Document parsing and bullet generation (313KB, 7123 lines)
├── powerpoint_generator.py        # PowerPoint presentation generation (152KB, 3236 lines)
├── google_slides_generator.py     # Google Slides presentation generation (11KB, 264 lines)
├── semantic_analyzer.py           # Semantic analysis using NLP/ML (12KB, 326 lines)
└── utils.py                       # Helper functions for Google Docs integration (7KB, 177 lines)
```

## 🚀 Quick Start

```python
from slide_generator_pkg import DocumentParser, SlideGenerator

# 1. Parse a document
parser = DocumentParser(claude_api_key="your-api-key")
doc_structure = parser.parse_file("document.txt", "document.txt")

# 2. Generate PowerPoint
generator = SlideGenerator()
pptx_path = generator.create_powerpoint(doc_structure)

print(f"✅ PowerPoint created: {pptx_path}")
```

## 📚 Components

### DocumentParser
Parses various document formats and generates bullet points using AI or NLP fallback.

**Features:**
- Supports TXT and DOCX formats
- AI-powered bullet generation using Claude API
- Lightweight NLP fallback using TF-IDF + TextRank + spaCy
- API response caching (saves 40-60% on API costs)
- Adaptive bullet count based on content density
- Table detection and summarization

**Key Methods:**
- `parse_file(file_path, filename, fast_mode=False)` - Main entry point
- `get_cache_stats()` - Returns API cache statistics

### SlideGenerator (PowerPoint)
Creates PowerPoint presentations with educational visual prompts.

**Features:**
- Heading hierarchy support (H1-H4)
- Section dividers and title slides
- Visual prompt generation
- AI-generated images (DALL-E integration)
- 12+ diagram types (flowcharts, architecture, timelines, etc.)
- HTML slide export

**Key Methods:**
- `create_powerpoint(doc_structure, skip_visuals=False)` - Generate PPTX file
- `create_html_slides(doc_structure)` - Generate HTML slides

### GoogleSlidesGenerator
Creates Google Slides presentations via API.

**Features:**
- Direct Google Slides creation
- OAuth 2.0 authentication
- Batch API requests for performance
- Supports all slide types (title, section, content, divider)

**Key Methods:**
- `create_presentation(doc_structure)` - Generate Google Slides and return URL

### Utilities
Helper functions for Google Docs integration.

**Functions:**
- `extract_google_doc_id(url)` - Extract document ID from Google Docs URL
- `fetch_google_doc_content(doc_id, credentials=None)` - Fetch doc content
- `get_google_client_config()` - Get OAuth client configuration

## 🔧 Dependencies

### Required
```bash
pip install python-pptx python-docx anthropic requests
```

### Optional (for enhanced features)
```bash
# NLP features
pip install nltk textstat spacy scikit-learn networkx

# Heavy semantic analysis
pip install sentence-transformers

# Google API integration
pip install google-api-python-client google-auth-oauthlib

# Image generation
pip install Pillow
```

## 💡 Usage Examples

### Example 1: Basic Usage
```python
from slide_generator_pkg import DocumentParser, SlideGenerator

parser = DocumentParser(claude_api_key="sk-...")
doc = parser.parse_file("notes.txt", "notes.txt")

generator = SlideGenerator()
pptx_path = generator.create_powerpoint(doc)
```

### Example 2: Google Docs to PowerPoint
```python
from slide_generator_pkg import extract_google_doc_id, fetch_google_doc_content, DocumentParser, SlideGenerator

# Extract doc ID and fetch content
doc_id = extract_google_doc_id("https://docs.google.com/document/d/ABC123/edit")
content, error = fetch_google_doc_content(doc_id)

# Save and parse
with open("/tmp/doc.txt", "w") as f:
    f.write(content)

parser = DocumentParser(claude_api_key="sk-...")
doc = parser.parse_file("/tmp/doc.txt", "Google Doc")

# Generate presentation
generator = SlideGenerator()
pptx_path = generator.create_powerpoint(doc)
```

### Example 3: Fast Mode (No AI)
```python
from slide_generator_pkg import DocumentParser, SlideGenerator

# Parse without AI processing
parser = DocumentParser()  # No API key
doc = parser.parse_file("notes.txt", "notes.txt", fast_mode=True)

# Generate simple presentation
generator = SlideGenerator()
pptx_path = generator.create_powerpoint(doc, skip_visuals=True)
```

### Example 4: Google Slides Output
```python
from slide_generator_pkg import DocumentParser, GoogleSlidesGenerator

parser = DocumentParser(claude_api_key="sk-...")
doc = parser.parse_file("notes.txt", "notes.txt")

# Requires OAuth credentials
credentials = {
    'token': 'access-token',
    'refresh_token': 'refresh-token',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'client_id': 'client-id',
    'client_secret': 'client-secret',
    'scopes': ['https://www.googleapis.com/auth/presentations']
}

generator = GoogleSlidesGenerator(credentials=credentials)
slides_url = generator.create_presentation(doc)
print(f"Google Slides: {slides_url}")
```

### Example 5: Check Cache Statistics
```python
parser = DocumentParser(claude_api_key="sk-...")

# Parse multiple documents...
doc1 = parser.parse_file("doc1.txt", "doc1.txt")
doc2 = parser.parse_file("doc2.txt", "doc2.txt")

# Check cache performance
stats = parser.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")
print(f"Savings: {stats['estimated_cost_savings']}")
```

## 🎯 Key Features

### AI-Powered Bullet Generation
- **Primary**: Claude API for high-quality, contextual bullets
- **Fallback**: TF-IDF + TextRank + spaCy ensemble for offline/free usage
- **Adaptive**: Automatically adjusts bullet count based on content density
- **Cached**: LRU cache saves 40-60% on API costs

### Smart Content Processing
- **Table Detection**: Automatically detects and summarizes tabular data
- **Heading Hierarchy**: Supports H1-H4 with proper slide organization
- **Semantic Analysis**: Groups related content intelligently
- **Quality Validation**: Filters out vague or low-quality bullets

### Flexible Output
- **PowerPoint**: Full-featured PPTX with visuals and formatting
- **Google Slides**: Direct API creation in user's Drive
- **HTML**: Web-based slide export

## 📝 API Reference

See `example_usage.py` for comprehensive examples.

## 🔑 Environment Variables

```bash
# Optional: Server-side Claude API key
export ANTHROPIC_API_KEY="sk-..."

# Optional: Google OAuth credentials
export GOOGLE_CREDENTIALS_JSON='{"web": {...}}'
```

## 📄 License

This package is part of the Slide Generator application.

## 🤝 Contributing

This package was automatically extracted from the monolithic `file_to_slides.py` application for better modularity and reusability.
