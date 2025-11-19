# Document Structure Analysis Feature

## Overview

The `analyze_document_structure()` method provides automatic detection of document structure to help determine the best parsing mode for slide generation. This feature analyzes documents to determine if they are table-dominant or text-dominant, and provides an intelligent recommendation for the `script_column` parameter.

## Location

**File:** `/home/user/slidegenerator/slide_generator_pkg/document_parser.py`

**Class:** `DocumentParser`

**Method:** `analyze_document_structure(file_path: str, file_ext: str) -> dict`

## Supported Formats

- **DOCX** - Microsoft Word documents
- **TXT** - Plain text files (with optional tab-delimited tables)
- **PDF** - PDF documents (requires pdfplumber or PyPDF2)

## Return Format

```python
{
    'tables': int,              # Number of tables found
    'paragraphs': int,          # Number of text paragraphs
    'table_cells': int,         # Total cells in tables
    'primary_type': str,        # 'table', 'text', or 'unknown'
    'suggested_mode': int,      # 0 for paragraph mode, 2+ for column mode
    'confidence': str           # 'high' or 'low'
}
```

## Decision Logic

### DOCX Files

1. **Count tables:** `len(doc.tables)`
2. **Count table cells:** Sum of all cells across all tables
3. **Count paragraphs:** Non-empty paragraphs only
4. **Decision:**
   - If `table_cells > paragraphs * 2` → **table mode** (suggested_mode=2)
   - Otherwise → **paragraph mode** (suggested_mode=0)
5. **Confidence:**
   - `high` if difference > 10
   - `low` otherwise

### TXT Files

1. **Count lines with tabs** vs **lines without tabs**
2. **Calculate tab percentage:** `(lines_with_tabs / total_lines) * 100`
3. **Decision:**
   - If tab percentage > 50% → **table mode** (suggested_mode=2)
   - Otherwise → **paragraph mode** (suggested_mode=0)
4. **Confidence:**
   - `high` if tab percentage > 80% or < 20%
   - `low` otherwise

### PDF Files

1. **Use PDFParser** to extract content and metadata
2. **Count tables** from metadata
3. **Estimate table cells:** `table_count * 15` (average cells per table)
4. **Count paragraphs** from extracted text
5. **Decision logic:** Same as DOCX

## Usage Examples

### Basic Usage

```python
from slide_generator_pkg.document_parser import DocumentParser

# Initialize parser
parser = DocumentParser()

# Analyze document
analysis = parser.analyze_document_structure(
    file_path='/path/to/document.docx',
    file_ext='docx'
)

# Check results
print(f"Primary type: {analysis['primary_type']}")
print(f"Suggested mode: {analysis['suggested_mode']}")
print(f"Confidence: {analysis['confidence']}")
```

### Using Analysis Results for Parsing

```python
from slide_generator_pkg.document_parser import DocumentParser

parser = DocumentParser(claude_api_key='your-api-key')

# Step 1: Analyze document structure
file_path = '/path/to/document.docx'
file_ext = 'docx'

analysis = parser.analyze_document_structure(file_path, file_ext)

# Step 2: Use suggested mode for parsing
if analysis['confidence'] == 'high':
    # High confidence - use suggested mode
    script_column = analysis['suggested_mode']
else:
    # Low confidence - you might want to try both modes
    print("Warning: Low confidence. Consider manual inspection.")
    script_column = analysis['suggested_mode']

# Step 3: Parse with suggested mode
slides = parser.parse_file(
    file_path=file_path,
    filename='document.docx',
    script_column=script_column
)
```

### Batch Analysis

```python
import os
from slide_generator_pkg.document_parser import DocumentParser

parser = DocumentParser()

# Analyze all documents in a directory
for filename in os.listdir('/path/to/documents'):
    if filename.endswith(('.docx', '.txt', '.pdf')):
        file_path = os.path.join('/path/to/documents', filename)
        file_ext = filename.split('.')[-1]

        analysis = parser.analyze_document_structure(file_path, file_ext)

        print(f"{filename}:")
        print(f"  Type: {analysis['primary_type']}")
        print(f"  Mode: {analysis['suggested_mode']}")
        print(f"  Confidence: {analysis['confidence']}")
        print()
```

## Test Cases

### Test Suite

Run the test suite to verify functionality:

```bash
python test_document_analysis.py
```

This tests:
- Table-heavy DOCX (should detect table mode)
- Text-heavy DOCX (should detect paragraph mode)
- Tab-delimited TXT (should detect table mode)
- Plain text TXT (should detect paragraph mode)
- Unsupported formats (should return 'unknown')

### Example Analysis Tool

Run the example tool to analyze any document:

```bash
python example_document_analysis.py /path/to/document.docx
```

This provides:
- Structure analysis (tables, paragraphs, cells)
- Classification (primary type, confidence)
- Parsing recommendation
- Sample code

## Error Handling

The method includes comprehensive error handling:

1. **File not found:** Returns `primary_type='unknown'` with all counts at 0
2. **Unsupported format:** Returns `primary_type='unknown'` with warning logged
3. **Parsing errors:** Catches exceptions and returns safe defaults
4. **Missing dependencies:** For PDFs without pdfplumber/PyPDF2, logs error and returns `unknown`

## Integration with Existing Code

This feature integrates seamlessly with the existing `parse_file()` workflow:

```python
# Before: Manual mode selection
slides = parser.parse_file(
    file_path='/path/to/document.docx',
    filename='document.docx',
    script_column=2  # Manual guess
)

# After: Automatic mode detection
analysis = parser.analyze_document_structure('/path/to/document.docx', 'docx')
slides = parser.parse_file(
    file_path='/path/to/document.docx',
    filename='document.docx',
    script_column=analysis['suggested_mode']  # Intelligent suggestion
)
```

## Performance

- **DOCX analysis:** Very fast (< 100ms for typical documents)
- **TXT analysis:** Instant (< 10ms)
- **PDF analysis:** Fast (< 500ms, depends on document size)

No LLM API calls are made during analysis - this is pure structural analysis.

## Logging

The method logs analysis results for debugging:

```
INFO: DOCX Analysis: 3 tables, 60 cells, 2 paragraphs → table (confidence: high)
INFO: TXT Analysis: 5/6 lines with tabs (83.3%) → table (confidence: high)
INFO: PDF Analysis: 2 tables, ~30 cells, 15 paragraphs → text (confidence: high)
```

## Confidence Levels

### High Confidence
- Clear distinction between table and text content
- Safe to use suggested mode automatically
- Difference between table_cells and paragraphs > 10

### Low Confidence
- Mixed content (roughly equal tables and text)
- Consider manual inspection
- May want to try both modes and compare results
- Difference between table_cells and paragraphs ≤ 10

## Future Enhancements

Potential improvements:
1. ML-based classification using document embeddings
2. Support for more formats (PPTX, HTML, Markdown)
3. Heading structure analysis (H1/H2/H3 ratio)
4. Content complexity scoring
5. Auto-detection of script column number (beyond 2)
6. Multi-column table detection

## Related Files

- **Implementation:** `/home/user/slidegenerator/slide_generator_pkg/document_parser.py`
- **PDF Parser:** `/home/user/slidegenerator/slide_generator_pkg/pdf_parser.py`
- **Test Suite:** `/home/user/slidegenerator/test_document_analysis.py`
- **Example Tool:** `/home/user/slidegenerator/example_document_analysis.py`

## API Reference

### `analyze_document_structure(file_path: str, file_ext: str) -> dict`

**Parameters:**
- `file_path` (str): Absolute path to the document file
- `file_ext` (str): File extension ('docx', 'txt', or 'pdf')

**Returns:**
- `dict`: Analysis results with structure information

**Raises:**
- No exceptions raised - returns safe defaults on error

**Example:**
```python
analysis = parser.analyze_document_structure('/docs/report.docx', 'docx')
# {
#     'tables': 3,
#     'paragraphs': 5,
#     'table_cells': 45,
#     'primary_type': 'table',
#     'suggested_mode': 2,
#     'confidence': 'high'
# }
```

## Contributing

To add support for new file formats:

1. Create a new helper method: `_analyze_<format>_structure(file_path: str) -> dict`
2. Add format detection in `analyze_document_structure()`
3. Follow the same return format
4. Add test cases
5. Update documentation

Example:

```python
def _analyze_markdown_structure(self, file_path: str) -> dict:
    """Analyze Markdown file structure"""
    # Your implementation here
    return {
        'tables': table_count,
        'paragraphs': paragraph_count,
        'table_cells': cell_count,
        'primary_type': 'text' or 'table',
        'suggested_mode': 0 or 2,
        'confidence': 'high' or 'low'
    }
```
