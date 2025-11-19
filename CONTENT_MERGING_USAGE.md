# Content Merging Feature Documentation

## Overview

The DocumentParser now includes intelligent content merging capabilities for handling mixed table/text documents. This feature improves bullet generation by providing contextual information from surrounding paragraphs when processing tables.

## New Methods

### 1. `_extract_content_blocks_from_docx(doc) -> List[dict]`

Extracts ordered list of content blocks from a DOCX document.

**Parameters:**
- `doc`: A python-docx Document object

**Returns:**
List of content blocks, where each block is:
- `{'type': 'heading', 'level': int, 'text': str}`
- `{'type': 'paragraph', 'text': str}`
- `{'type': 'table', 'data': [[str]], 'headers': [[str]]}`

**Example:**
```python
from docx import Document
from slide_generator_pkg.document_parser import DocumentParser

parser = DocumentParser()
doc = Document('my_document.docx')
content_blocks = parser._extract_content_blocks_from_docx(doc)

for block in content_blocks:
    print(f"Block type: {block['type']}")
```

### 2. `_merge_table_and_text_context(content_blocks) -> List[dict]`

Merges tables with surrounding text context for context-aware processing.

**Parameters:**
- `content_blocks`: List of content blocks from `_extract_content_blocks_from_docx()`

**Returns:**
List of merged blocks:
- `{'type': 'table_with_context', 'intro': str|None, 'table': dict, 'explanation': str|None}`
- `{'type': 'paragraph', 'text': str}` (standalone paragraphs)
- `{'type': 'heading', 'level': int, 'text': str}`

**Example:**
```python
# Extract blocks
content_blocks = parser._extract_content_blocks_from_docx(doc)

# Merge tables with context
merged_blocks = parser._merge_table_and_text_context(content_blocks)

for block in merged_blocks:
    if block['type'] == 'table_with_context':
        if block['intro']:
            print(f"Table intro: {block['intro']}")
        print(f"Table has {len(block['table']['data'])} rows")
        if block['explanation']:
            print(f"Table explanation: {block['explanation']}")
```

## Merging Rules

The merging logic follows these rules:

1. **Intro Detection**: A paragraph directly BEFORE a table (within 1 block) is treated as introductory context
2. **Explanation Detection**: A paragraph directly AFTER a table (within 1 block) is treated as explanatory context
3. **Minimum Length**: Paragraphs must be >20 characters to be considered context
4. **No Duplicate Merging**: A paragraph cannot be merged with multiple tables
5. **Standalone Preservation**: Paragraphs not adjacent to tables remain independent
6. **Heading Preservation**: All headings are kept in their original positions

## Example Document Structure

**Input Document:**
```
Heading 1: Performance Metrics
Paragraph: "The following table shows system performance..."  [INTRO]
Table: [Metric | Value | Status]
Paragraph: "As shown above, all metrics are acceptable."  [EXPLANATION]

Heading 2: Summary
Paragraph: "The system is performing well overall."  [STANDALONE]

Heading 3: User Statistics
Table: [User | Count]  [NO CONTEXT]
```

**After Merging:**
```
Heading: "Performance Metrics"
Table with Context:
  - Intro: "The following table shows system performance..."
  - Table: [Metric | Value | Status]
  - Explanation: "As shown above, all metrics are acceptable."

Heading: "Summary"
Paragraph: "The system is performing well overall."

Heading: "User Statistics"
Table with Context:
  - Intro: None
  - Table: [User | Count]
  - Explanation: None
```

## Integration Points

These methods can be integrated into the document parsing pipeline:

```python
def parse_docx_with_context(self, file_path: str) -> List[dict]:
    """Parse DOCX with context-aware table handling"""
    doc = Document(file_path)

    # Extract content blocks
    content_blocks = self._extract_content_blocks_from_docx(doc)

    # Merge tables with context
    merged_blocks = self._merge_table_and_text_context(content_blocks)

    # Process merged blocks for bullet generation
    for block in merged_blocks:
        if block['type'] == 'table_with_context':
            # Use intro and explanation for better bullet context
            context = []
            if block['intro']:
                context.append(block['intro'])

            # Process table...

            if block['explanation']:
                context.append(block['explanation'])

            # Generate bullets with full context
            bullets = self._create_bullet_points(
                text=' '.join(context),
                context_heading=current_heading
            )

    return merged_blocks
```

## Use Cases

1. **Financial Reports**: Tables with explanatory paragraphs
2. **Technical Documentation**: Code/configuration tables with descriptions
3. **Research Papers**: Data tables with interpretations
4. **Business Presentations**: Metrics tables with executive summaries
5. **Educational Materials**: Example tables with explanations

## Performance Notes

- **Two-pass algorithm**: First identifies merge candidates, then builds merged structure
- **Time complexity**: O(n) where n is the number of content blocks
- **Memory**: Creates new block list; original blocks unchanged
- **Typical overhead**: <100ms for documents with <100 blocks

## Testing

Run the test suite to verify functionality:

```bash
python test_content_merging.py
```

Expected output:
- ✓ All tables converted to table_with_context
- ✓ All headings preserved
- ✓ Paragraphs correctly merged or kept standalone
- ✓ Context detection (intro/explanation) working

## Future Enhancements

Potential improvements:
1. Configurable merge distance (currently fixed at 1 block)
2. Semantic similarity scoring for context relevance
3. Support for multiple context paragraphs
4. Context quality scoring
5. Enable/disable via configuration flag
