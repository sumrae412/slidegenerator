# File Format & Table Handling Enhancement Plan

**Goal:** Support multiple input formats (Google Docs, Word, PDF) and output formats (Google Slides, PowerPoint), with robust handling of mixed table/text content.

---

## ✅ Current State Assessment

### **Input Formats:**
- ✅ **Google Docs** - Via Google Docs API, exported as plain text
- ✅ **Word (.docx)** - Full support with python-docx library
- ✅ **Plain Text (.txt)** - Basic support
- ❌ **PDF** - NOT SUPPORTED

### **Output Formats:**
- ✅ **PowerPoint (.pptx)** - Full support via python-pptx
- ✅ **Google Slides** - Full support via Google Slides API
- ✅ Format selection in UI (`output_format` parameter)

### **Table & Text Handling:**
- ✅ **Table extraction from .docx** - Reads table cells
- ✅ **Table extraction from Google Docs (plain text)** - Tab-delimited parsing
- ✅ **Column selection** - Can extract specific column (e.g., column 2 for scripts)
- ✅ **Paragraph mode** - Can handle non-tabled text (`script_column=0`)
- ✅ **Mixed table + text** - Handles both in same document
- ⚠️ **Complex tables** - May struggle with merged cells, nested tables

---

## ❌ Current Gaps

### **Input Format Gaps:**

1. **PDF Support Missing**
   - Cannot process PDF files at all
   - Users must manually convert PDF → Word/Docs

2. **Limited Table Detection in PDFs**
   - Would need specialized PDF parsing
   - Table structure often lost in PDF extraction

3. **No Rich Formatting Preservation**
   - Bold, italic, colors not preserved from Word/Docs
   - Could enhance visual hierarchy

### **Table Handling Gaps:**

1. **Merged Cells**
   - Not properly handled in Word tables
   - May cause column misalignment

2. **Nested Tables**
   - Tables within tables not supported
   - Could cause parsing errors

3. **Complex Layouts**
   - Side-by-side tables
   - Tables with varying column counts

4. **Table Headers**
   - Not distinguished from data rows
   - Could use headers as context

---

## 🚀 Enhancement Plan

### **PHASE 1: PDF Input Support** ⭐⭐⭐ (HIGH PRIORITY)

#### **Approach: Multi-Strategy PDF Parsing**

**Libraries to Add:**
```python
# requirements.txt
PyPDF2>=3.0.1          # Basic PDF text extraction
pdfplumber>=0.10.0     # Advanced PDF parsing with table detection
pdf2image>=1.16.3      # Convert PDF to images (fallback)
pytesseract>=0.3.10    # OCR for scanned PDFs
```

#### **Implementation:**

**File:** `slide_generator_pkg/pdf_parser.py` (NEW)

```python
"""
PDF Parser Module
Handles PDF files with multiple extraction strategies
"""

import logging
from typing import Tuple, List
import os

logger = logging.getLogger(__name__)

# Try importing PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - PDF parsing will be limited")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF parsing will be limited")


class PDFParser:
    """Parse PDF files with intelligent table and text extraction"""

    def __init__(self):
        self.available_methods = []
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append('pdfplumber')
        if PYPDF2_AVAILABLE:
            self.available_methods.append('pypdf2')

    def parse_pdf(self, file_path: str) -> Tuple[str, List[dict]]:
        """
        Parse PDF file and extract text + tables

        Returns:
            (text_content, table_metadata)
        """
        if not self.available_methods:
            raise Exception("No PDF parsing libraries available. Install pdfplumber or PyPDF2")

        # Try pdfplumber first (best for tables)
        if PDFPLUMBER_AVAILABLE:
            return self._parse_with_pdfplumber(file_path)

        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            return self._parse_with_pypdf2(file_path)

        raise Exception("PDF parsing failed")

    def _parse_with_pdfplumber(self, file_path: str) -> Tuple[str, List[dict]]:
        """Parse PDF using pdfplumber (best for tables)"""
        import pdfplumber

        all_text = []
        all_tables = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"Processing PDF page {page_num}/{len(pdf.pages)}")

                # Extract tables first
                tables = page.extract_tables()

                if tables:
                    logger.info(f"Found {len(tables)} tables on page {page_num}")

                    for table_num, table in enumerate(tables):
                        # Convert table to text format
                        table_text = self._table_to_text(table, page_num, table_num)
                        all_text.append(table_text)

                        # Store table metadata
                        all_tables.append({
                            'page': page_num,
                            'table_num': table_num,
                            'rows': len(table),
                            'columns': len(table[0]) if table else 0,
                            'data': table
                        })

                # Extract remaining text (non-table)
                page_text = page.extract_text()
                if page_text:
                    # Remove table content (already extracted)
                    # This is approximate - pdfplumber doesn't perfectly separate
                    all_text.append(f"\n\n--- Page {page_num} ---\n{page_text}")

        full_text = "\n".join(all_text)
        logger.info(f"PDF parsing complete: {len(full_text)} chars, {len(all_tables)} tables")

        return full_text, all_tables

    def _parse_with_pypdf2(self, file_path: str) -> Tuple[str, List[dict]]:
        """Parse PDF using PyPDF2 (basic text extraction, no tables)"""
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        all_text = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(f"\n\n--- Page {page_num} ---\n{text}")

        full_text = "\n".join(all_text)

        logger.warning("PyPDF2 used - table structure may be lost. Consider using pdfplumber")
        return full_text, []  # No table metadata

    def _table_to_text(self, table: List[List[str]], page_num: int, table_num: int) -> str:
        """Convert table data to tab-delimited text (compatible with existing parser)"""

        if not table:
            return ""

        # Format as tab-delimited text
        lines = []
        lines.append(f"\n# Table {table_num + 1} (Page {page_num})\n")

        for row in table:
            # Clean cells
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            # Tab-delimited (compatible with existing txt parser)
            lines.append("\t".join(cleaned_row))

        return "\n".join(lines)

    def detect_scanned_pdf(self, file_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) vs text-based
        Returns True if PDF appears to be scanned
        """
        if not PYPDF2_AVAILABLE:
            return False

        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)

        # Sample first 3 pages
        pages_to_check = min(3, len(reader.pages))
        total_text_length = 0

        for i in range(pages_to_check):
            text = reader.pages[i].extract_text()
            total_text_length += len(text.strip())

        # If very little text extracted, likely scanned
        avg_text_per_page = total_text_length / pages_to_check

        is_scanned = avg_text_per_page < 50  # Less than 50 chars/page = likely scanned

        if is_scanned:
            logger.warning(f"PDF appears to be scanned (avg {avg_text_per_page:.0f} chars/page)")

        return is_scanned
```

#### **Integration into DocumentParser:**

**File:** `slide_generator_pkg/document_parser.py`

```python
# Add import at top
try:
    from .pdf_parser import PDFParser
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF parser not available")

# Update parse_file method
def parse_file(self, file_path: str, filename: str, script_column: int = 2, fast_mode: bool = False) -> DocumentStructure:
    """Parse DOCX, TXT, or PDF file and convert to slide structure"""
    file_ext = filename.lower().split('.')[-1]

    try:
        if file_ext == 'docx':
            content = self._parse_docx(file_path, script_column)
        elif file_ext == 'txt':
            content = self._parse_txt(file_path, script_column)
        elif file_ext == 'pdf':  # NEW
            if not PDF_AVAILABLE:
                raise Exception("PDF support not installed. Run: pip install pdfplumber PyPDF2")

            content = self._parse_pdf(file_path, script_column)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: .docx, .txt, .pdf")

        # ... rest of existing code

def _parse_pdf(self, file_path: str, script_column: int = 2) -> str:
    """Parse PDF file using PDFParser"""

    pdf_parser = PDFParser()

    # Check if scanned
    if pdf_parser.detect_scanned_pdf(file_path):
        logger.warning("⚠️ Scanned PDF detected - OCR not yet implemented. Text extraction may be poor.")
        # TODO: Add OCR support for scanned PDFs

    # Parse PDF
    text_content, tables = pdf_parser.parse_pdf(file_path)

    logger.info(f"PDF parsed: {len(text_content)} chars, {len(tables)} tables found")

    # If script_column mode, the tab-delimited table text will be parsed correctly
    # by existing _parse_txt logic

    # Save as temp .txt file and parse with existing logic
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(text_content)
        tmp_path = tmp.name

    try:
        # Reuse existing txt parser (handles tables and text)
        content = self._parse_txt(tmp_path, script_column)
        return content
    finally:
        # Clean up temp file
        os.remove(tmp_path)
```

#### **UI Updates:**

**File:** `templates/file_to_slides.html`

```html
<!-- Update file input to accept PDF -->
<div class="mb-4">
    <label class="block text-sm font-medium mb-2">Document Source</label>

    <div class="file-input-wrapper">
        <input type="file"
               name="file"
               accept=".docx,.txt,.pdf"  <!-- ADD .pdf -->
               class="file-input">
        <span class="file-label">Choose file (.docx, .txt, .pdf)</span>
    </div>

    <p class="text-xs text-gray-500 mt-1">
        Supported: Word documents (.docx), Text files (.txt), PDF files (.pdf)
    </p>
</div>

<!-- Add PDF warning -->
<div id="pdf-warning" class="hidden mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded">
    <p class="text-xs text-yellow-800">
        ⚠️ <strong>PDF Note:</strong> Scanned PDFs may have poor text extraction.
        For best results, use text-based PDFs or convert to Word first.
    </p>
</div>

<script>
// Show warning for PDF files
document.querySelector('input[type="file"]').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const warning = document.getElementById('pdf-warning');

    if (file && file.name.toLowerCase().endsWith('.pdf')) {
        warning.classList.remove('hidden');
    } else {
        warning.classList.add('hidden');
    }
});
</script>
```

#### **Backend Route Update:**

**File:** `file_to_slides.py`

```python
ALLOWED_EXTENSIONS = {'docx', 'txt', 'pdf'}  # Add 'pdf'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

**Testing Checklist:**
- [ ] Test with text-based PDF (articles, reports)
- [ ] Test with PDF containing tables
- [ ] Test with scanned PDF (should show warning)
- [ ] Test with multi-page PDF
- [ ] Test with PDF + table column selection

**Cost:** None (PDF parsing is local, no API calls)

**Effort:** 6-8 hours

---

### **PHASE 2: Enhanced Table Handling** ⭐⭐⭐

#### **2.1 Robust Merged Cell Support**

**Problem:** Merged cells in Word tables cause column misalignment

**Solution:** Detect merged cells and handle them intelligently

**File:** `slide_generator_pkg/document_parser.py`

```python
def _parse_docx(self, file_path: str, script_column: int = 2) -> str:
    """Enhanced DOCX parsing with merged cell support"""
    doc = Document(file_path)
    content = []

    for table in doc.tables:
        # Process each row
        for row_idx, row in enumerate(table.rows):
            cells = []

            for cell_idx, cell in enumerate(row.cells):
                # Check if this is a merged cell
                if self._is_merged_cell(cell):
                    # Get the original cell this merged from
                    original_cell = self._get_merged_cell_origin(cell, table, row_idx, cell_idx)
                    if original_cell and original_cell != cell:
                        continue  # Skip duplicate merged cells

                cell_text = cell.text.strip()
                cells.append(cell_text)

            # Handle column selection
            if script_column > 0 and len(cells) >= script_column:
                content.append(cells[script_column - 1])
            elif script_column == 0:
                content.extend(cells)

    return "\n".join(content)

def _is_merged_cell(self, cell) -> bool:
    """Check if cell is part of a merged cell"""
    try:
        # Check cell's gridSpan attribute
        tc = cell._element
        tcPr = tc.get_or_add_tcPr()
        gridSpan = tcPr.gridSpan

        if gridSpan is not None:
            span = int(gridSpan.val)
            return span > 1

        # Check vMerge (vertical merge)
        vMerge = tcPr.vMerge
        if vMerge is not None:
            return True

        return False
    except:
        return False
```

**Benefit:** Correctly handles complex Word table layouts

**Effort:** 3-4 hours

---

#### **2.2 Table Header Detection**

**Problem:** Table headers not distinguished from data

**Solution:** Detect header rows and use as context

```python
def _extract_table_with_headers(self, table) -> dict:
    """Extract table with header detection"""

    if not table.rows:
        return {'headers': [], 'data': []}

    # First row is usually header
    first_row = table.rows[0]

    # Detect if first row is header by checking:
    # 1. Bold formatting
    # 2. Different background color
    # 3. All caps text
    # 4. Shorter text than data rows

    is_header = self._is_header_row(first_row)

    if is_header:
        headers = [cell.text.strip() for cell in first_row.cells]
        data_rows = table.rows[1:]
    else:
        headers = []
        data_rows = table.rows

    data = []
    for row in data_rows:
        data.append([cell.text.strip() for cell in row.cells])

    return {
        'headers': headers,
        'data': data
    }

def _is_header_row(self, row) -> bool:
    """Detect if row is a header row"""

    # Check first cell for bold
    if row.cells:
        first_cell = row.cells[0]

        # Check if text is bold
        for paragraph in first_cell.paragraphs:
            for run in paragraph.runs:
                if run.bold:
                    return True

        # Check if all caps
        text = first_cell.text.strip()
        if text and text.isupper() and len(text.split()) <= 3:
            return True

    return False
```

**Benefit:** Use headers as context for better bullet generation

**Effort:** 2-3 hours

---

#### **2.3 Smart Table vs. Text Detection**

**Problem:** App doesn't auto-detect if document has tables

**Solution:** Analyze document structure and suggest best mode

```python
def analyze_document_structure(self, file_path: str, file_ext: str) -> dict:
    """
    Analyze document to detect tables and suggest best parsing mode
    """

    if file_ext == 'docx':
        doc = Document(file_path)

        # Count tables and paragraphs
        table_count = len(doc.tables)
        paragraph_count = len([p for p in doc.paragraphs if p.text.strip()])

        # Count total cells vs total text
        table_cells = sum(len(table.rows) * len(table.columns) for table in doc.tables)

        # Determine primary content type
        if table_count > 0 and table_cells > paragraph_count:
            primary_type = 'table'
            suggested_column = 2  # Default to column 2 for scripts
        else:
            primary_type = 'text'
            suggested_column = 0  # Paragraph mode

        return {
            'tables': table_count,
            'paragraphs': paragraph_count,
            'table_cells': table_cells,
            'primary_type': primary_type,
            'suggested_mode': suggested_column,
            'confidence': 'high' if abs(table_cells - paragraph_count) > 10 else 'low'
        }

    # Similar logic for TXT and PDF
    # ...

    return {'primary_type': 'unknown', 'suggested_mode': 0}
```

**UI Integration:**

```html
<!-- Auto-detect and suggest mode -->
<div id="document-analysis" class="hidden mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
    <p class="text-sm text-blue-900">
        📊 <strong>Document Analysis:</strong>
    </p>
    <p class="text-xs text-blue-800 mt-1" id="analysis-result">
        <!-- Filled by JavaScript -->
    </p>
    <button onclick="acceptSuggestion()" class="mt-2 text-xs bg-blue-600 text-white px-3 py-1 rounded">
        Use Suggested Mode
    </button>
</div>

<script>
async function analyzeDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/analyze-document', {
        method: 'POST',
        body: formData
    });

    const analysis = await response.json();

    if (analysis.primary_type === 'table') {
        document.getElementById('analysis-result').innerHTML =
            `Found ${analysis.tables} tables. Suggested mode: <strong>Table Column ${analysis.suggested_mode}</strong>`;
    } else {
        document.getElementById('analysis-result').innerHTML =
            `Found ${analysis.paragraphs} paragraphs. Suggested mode: <strong>Paragraph Mode</strong>`;
    }

    document.getElementById('document-analysis').classList.remove('hidden');
}

// Trigger analysis on file selection
document.querySelector('input[type="file"]').addEventListener('change', function(e) {
    if (e.target.files[0]) {
        analyzeDocument(e.target.files[0]);
    }
});
</script>
```

**Backend Endpoint:**

```python
@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    """Analyze uploaded document structure"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename
    file_ext = filename.rsplit('.', 1)[1].lower()

    # Save temporarily
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        parser = DocumentParser()
        analysis = parser.analyze_document_structure(filepath, file_ext)
        return jsonify(analysis)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
```

**Benefit:** Users don't need to guess the right mode

**Effort:** 4-5 hours

---

### **PHASE 3: Mixed Content Optimization** ⭐⭐

#### **3.1 Intelligent Content Merging**

**Problem:** Tables and paragraphs treated separately, losing context

**Solution:** Merge related table + text into same slides

```python
def _merge_table_and_text_context(self, content_blocks: List[dict]) -> List[dict]:
    """
    Merge tables with surrounding text context

    content_blocks = [
        {'type': 'paragraph', 'text': '...'},
        {'type': 'table', 'data': [[...]], 'headers': [...]},
        {'type': 'paragraph', 'text': '...'}
    ]
    """

    merged = []

    for i, block in enumerate(content_blocks):
        if block['type'] == 'table':
            # Look for paragraph before table (intro)
            intro = None
            if i > 0 and content_blocks[i-1]['type'] == 'paragraph':
                intro = content_blocks[i-1]['text']

            # Look for paragraph after table (explanation)
            explanation = None
            if i < len(content_blocks) - 1 and content_blocks[i+1]['type'] == 'paragraph':
                explanation = content_blocks[i+1]['text']

            merged.append({
                'type': 'table_with_context',
                'intro': intro,
                'table': block,
                'explanation': explanation
            })
        elif block['type'] == 'paragraph':
            # Only add if not already merged as table context
            if (i == 0 or content_blocks[i-1]['type'] != 'table') and \
               (i == len(content_blocks)-1 or content_blocks[i+1]['type'] != 'table'):
                merged.append(block)

    return merged
```

**Benefit:** Better context for bullet generation from tables

**Effort:** 3-4 hours

---

## 📊 Feature Matrix

| Feature | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|---------|---------|---------------|---------------|---------------|
| **Input: Google Docs** | ✅ | ✅ | ✅ | ✅ |
| **Input: Word (.docx)** | ✅ | ✅ | ✅ | ✅ |
| **Input: PDF** | ❌ | ✅ | ✅ | ✅ |
| **Input: Scanned PDF (OCR)** | ❌ | ⚠️ (warning) | ✅ | ✅ |
| **Output: PowerPoint** | ✅ | ✅ | ✅ | ✅ |
| **Output: Google Slides** | ✅ | ✅ | ✅ | ✅ |
| **Tables: Basic** | ✅ | ✅ | ✅ | ✅ |
| **Tables: Merged Cells** | ❌ | ❌ | ✅ | ✅ |
| **Tables: Header Detection** | ❌ | ❌ | ✅ | ✅ |
| **Mixed Table + Text** | ⚠️ | ⚠️ | ⚠️ | ✅ |
| **Auto-detect Mode** | ❌ | ❌ | ✅ | ✅ |

---

## 🎯 Implementation Roadmap

### **Week 1-2: PDF Support (Phase 1)**
- Install pdfplumber, PyPDF2
- Implement PDFParser class
- Integrate into DocumentParser
- Update UI for PDF upload
- Test with various PDFs

### **Week 3: Enhanced Tables (Phase 2)**
- Merged cell handling
- Header detection
- Auto-mode detection
- UI improvements

### **Week 4: Content Optimization (Phase 3)**
- Table + text merging
- Context-aware bullet generation
- Final testing

---

## 💰 Cost Analysis

**No additional API costs** - All file parsing is local

**Dependencies to Add:**
```bash
pip install pdfplumber PyPDF2  # ~$0 (open source)
```

**Development Time:**
- Phase 1 (PDF): 6-8 hours
- Phase 2 (Tables): 9-12 hours
- Phase 3 (Mixed): 7-9 hours
- **Total: 22-29 hours** (~3-4 weeks at part-time pace)

---

## ✅ Testing Strategy

### **Test Documents Needed:**

1. **PDF Tests:**
   - Text-based PDF (article)
   - PDF with tables
   - Multi-page PDF
   - Scanned PDF (to verify warning)

2. **Table Tests:**
   - Word doc with merged cells
   - Word doc with header row
   - Complex multi-column table
   - Mixed tables + paragraphs

3. **Mixed Content:**
   - Table followed by explanation paragraph
   - Paragraph introducing table
   - Alternating table/text sections

### **Validation Checklist:**

- [ ] PDF text extracted correctly
- [ ] PDF tables detected and parsed
- [ ] Merged cells handled without duplication
- [ ] Table headers used as context
- [ ] Mode auto-detection works
- [ ] Google Slides output works with all inputs
- [ ] PowerPoint output works with all inputs
- [ ] Mixed content maintains context

---

## 🚀 Quick Start (Phase 1 Only)

If you want just PDF support quickly:

```bash
# 1. Install dependencies
pip install pdfplumber PyPDF2

# 2. Create pdf_parser.py (code provided above)

# 3. Update document_parser.py (integration code provided)

# 4. Update file_to_slides.html (UI changes provided)

# 5. Test with sample PDF
python file_to_slides.py
# Upload a PDF and test
```

**Effort:** ~6 hours for PDF support only

---

## 📋 Acceptance Criteria

**Phase 1 Complete When:**
- ✅ Can upload and process PDF files
- ✅ PDF tables extracted as tab-delimited text
- ✅ Warning shown for scanned PDFs
- ✅ PDF content generates slides correctly

**Phase 2 Complete When:**
- ✅ Merged cells don't cause duplication
- ✅ Table headers detected and used
- ✅ Auto-detection suggests correct mode
- ✅ UI shows analysis results

**Phase 3 Complete When:**
- ✅ Table context includes surrounding text
- ✅ Bullets reference intro/explanation
- ✅ Mixed documents feel cohesive

---

## 🎯 Recommended Approach

**Start with Phase 1 (PDF support)** because:

1. **High user value** - Many documents exist as PDFs
2. **Independent feature** - Doesn't require other phases
3. **Quick to implement** - 6-8 hours
4. **Immediate benefit** - Expands supported formats by 33%

**Then add Phase 2 (Table enhancements)** for:
- More robust table handling
- Better user experience
- Professional quality

**Finally Phase 3 (Mixed content)** for:
- Advanced use cases
- Maximum quality
- Complete solution

---

**Last Updated:** 2025-11-19
**Status:** Ready for implementation
**Recommended Start:** Phase 1 (PDF Support)
