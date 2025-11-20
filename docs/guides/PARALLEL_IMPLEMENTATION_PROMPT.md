# Parallel Implementation Plan - Top 5 Improvements

**Goal:** Implement speaker notes, PDF support, enhanced tables, topic separation, and Google Docs heading preservation using multiple agents working in parallel.

**Strategy:** 5 agents working concurrently with careful coordination to avoid merge conflicts

**Estimated Time:** 15-20 hours sequential → **6-8 hours parallel** (60% faster)

---

## 🎯 **FOR CLAUDE CODE WEB:**

Please implement the following improvements using **5 parallel agents** with the coordination strategy below. All agents should work concurrently and use proper git branch management to avoid conflicts.

---

## 📋 **AGENT ASSIGNMENTS**

### **Agent 1: Speaker Notes Support** 🎤
**Priority:** CRITICAL (User's primary use case)
**Branch:** `feature/speaker-notes`
**Files to Modify:**
- `slide_generator_pkg/data_models.py`
- `slide_generator_pkg/document_parser.py`
- (PowerPoint/Google Slides generators - file paths TBD)

**Dependencies:** None (independent implementation)

**Tasks:**
1. Add `speaker_notes: Optional[str] = None` field to `SlideContent` dataclass
2. Fix bracketed text removal regex to handle multi-line brackets
3. Extract full script text and preserve as speaker notes in `_content_to_slides_no_table_mode()`
4. Add speaker notes output to PowerPoint generation (find `pptx_generator.py` or similar)
5. Add speaker notes output to Google Slides generation (find google slides code)
6. Test with `ryans_doc.txt`

**Detailed Instructions:**

#### Task 1.1: Update Data Model
**File:** `slide_generator_pkg/data_models.py`

Add speaker notes field to `SlideContent` dataclass:
```python
@dataclass
class SlideContent:
    """Represents content for a single slide"""
    title: str
    content: List[str]
    slide_type: str = 'content'
    heading_level: Optional[int] = None
    subheader: Optional[str] = None
    visual_cues: Optional[List[str]] = None
    speaker_notes: Optional[str] = None  # NEW: Full script text for presenter
    # ... rest of existing fields
```

#### Task 1.2: Fix Bracket Removal
**File:** `slide_generator_pkg/document_parser.py`

Find line ~856 with:
```python
raw_content = re.sub(r'\[.*?\]', '', raw_content)
```

Replace with:
```python
# Remove bracketed production notes (including multi-line)
raw_content = re.sub(r'\[.*?\]', '', raw_content, flags=re.DOTALL)
```

#### Task 1.3: Extract Speaker Notes
**File:** `slide_generator_pkg/document_parser.py`

In `_content_to_slides_no_table_mode()` method (around line 2000-2010):

Before creating slides, preserve original script:
```python
# BEFORE removing brackets, preserve original for speaker notes
if content_buffer:
    original_script = "\n\n".join(content_buffer)
    # Remove brackets for clean speaker notes
    cleaned_script = re.sub(r'\[.*?\]', '', original_script, flags=re.DOTALL).strip()
else:
    cleaned_script = None

# Generate bullets from cleaned text
bullet_points = self._create_bullet_points(
    combined_text,
    fast_mode,
    context_heading=slide_title,
    heading_ancestry=heading_ancestry
)

# Create slide with BOTH bullets and speaker notes
slides.append(SlideContent(
    title=slide_title,
    content=bullet_points,
    speaker_notes=cleaned_script,  # ADD THIS
    slide_type='script',
    heading_level=slide_heading_level,
    subheader=topic_sentence,
    visual_cues=slide_visual_cues
))
```

#### Task 1.4: PowerPoint Speaker Notes
**File:** Find the PowerPoint generation code (likely `pptx_generator.py` or in `file_to_slides.py`)

When creating content slides, add:
```python
# After adding title and bullets to slide
if slide_content.speaker_notes:
    notes_slide = slide.notes_slide
    notes_text_frame = notes_slide.notes_text_frame
    notes_text_frame.text = slide_content.speaker_notes
```

#### Task 1.5: Google Slides Speaker Notes
**File:** Find Google Slides generation code

Add to batch requests:
```python
if slide_content.speaker_notes:
    requests.append({
        'createShape': {
            'objectId': f'speaker_notes_{slide_id}',
            'shapeType': 'TEXT_BOX',
            'elementProperties': {
                'pageObjectId': slide_id,
                'size': {'width': {'magnitude': 1, 'unit': 'PT'}, 'height': {'magnitude': 1, 'unit': 'PT'}},
                'transform': {'scaleX': 0, 'scaleY': 0, 'translateX': 0, 'translateY': 0}
            }
        }
    })
    requests.append({
        'insertText': {
            'objectId': f'speaker_notes_{slide_id}',
            'text': slide_content.speaker_notes,
            'insertionIndex': 0
        }
    })
```

**Note:** Google Slides API doesn't directly support speaker notes. May need to use Notes page or alternative approach.

#### Task 1.6: Testing
Test with: `python file_to_slides.py` and upload `ryans_doc.txt`

Expected:
- Slide 12 (Considering the Impact...) has 3-5 bullets
- Speaker notes contain full script (lines 74-134) with brackets removed
- PowerPoint presenter view shows speaker notes
- No bracketed text `[Note to reviewer:]` or `[>>>click]` in any output

**Success Criteria:**
- ✅ SlideContent has speaker_notes field
- ✅ Multi-line brackets removed correctly
- ✅ Full script preserved in speaker notes
- ✅ PowerPoint shows notes in presenter view
- ✅ No brackets in final output
- ✅ Test passes with ryans_doc.txt

**Estimated Time:** 3-4 hours

---

### **Agent 2: PDF Input Support** 📄
**Priority:** CRITICAL (User requested)
**Branch:** `feature/pdf-input`
**Files to Modify:**
- `requirements.txt`
- `slide_generator_pkg/pdf_parser.py` (NEW FILE)
- `slide_generator_pkg/document_parser.py`
- `templates/file_to_slides.html`
- `file_to_slides.py`

**Dependencies:** None (independent implementation)

**Tasks:**
1. Add PDF parsing libraries to requirements.txt
2. Create PDFParser class for PDF text/table extraction
3. Integrate PDF support into DocumentParser
4. Update UI to accept PDF uploads
5. Update backend to handle PDF files
6. Test with sample PDFs

**Detailed Instructions:**

#### Task 2.1: Add Dependencies
**File:** `requirements.txt`

Add at end:
```
pdfplumber>=0.10.0
PyPDF2>=3.0.1
```

#### Task 2.2: Create PDFParser Class
**File:** `slide_generator_pkg/pdf_parser.py` (NEW FILE)

See full implementation in `FILE_FORMAT_ENHANCEMENT_PLAN.md` lines 84-250.

Key class structure:
```python
"""PDF Parser Module"""
import logging
from typing import Tuple, List

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

logger = logging.getLogger(__name__)

class PDFParser:
    """Parse PDF files with intelligent table and text extraction"""

    def __init__(self):
        self.available_methods = []
        if PDFPLUMBER_AVAILABLE:
            self.available_methods.append('pdfplumber')
        if PYPDF2_AVAILABLE:
            self.available_methods.append('pypdf2')

    def parse_pdf(self, file_path: str) -> Tuple[str, List[dict]]:
        """Parse PDF and extract text + tables"""
        if PDFPLUMBER_AVAILABLE:
            return self._parse_with_pdfplumber(file_path)
        if PYPDF2_AVAILABLE:
            return self._parse_with_pypdf2(file_path)
        raise Exception("No PDF parsing libraries available")

    def _parse_with_pdfplumber(self, file_path: str) -> Tuple[str, List[dict]]:
        """Use pdfplumber (best for tables)"""
        # See FILE_FORMAT_ENHANCEMENT_PLAN.md lines 143-184
        pass

    def _parse_with_pypdf2(self, file_path: str) -> Tuple[str, List[dict]]:
        """Use PyPDF2 (basic text extraction)"""
        # See FILE_FORMAT_ENHANCEMENT_PLAN.md lines 186-201
        pass

    def _table_to_text(self, table: List[List[str]], page_num: int, table_num: int) -> str:
        """Convert table to tab-delimited text"""
        # See FILE_FORMAT_ENHANCEMENT_PLAN.md lines 203-219
        pass

    def detect_scanned_pdf(self, file_path: str) -> bool:
        """Detect if PDF is scanned (image-based)"""
        # See FILE_FORMAT_ENHANCEMENT_PLAN.md lines 221-249
        pass
```

Full implementation available in `FILE_FORMAT_ENHANCEMENT_PLAN.md`.

#### Task 2.3: Integrate into DocumentParser
**File:** `slide_generator_pkg/document_parser.py`

Add import at top:
```python
try:
    from .pdf_parser import PDFParser
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF parser not available")
```

Update `parse_file()` method:
```python
def parse_file(self, file_path: str, filename: str, script_column: int = 2, fast_mode: bool = False):
    file_ext = filename.lower().split('.')[-1]

    if file_ext == 'docx':
        content = self._parse_docx(file_path, script_column)
    elif file_ext == 'txt':
        content = self._parse_txt(file_path, script_column)
    elif file_ext == 'pdf':  # NEW
        if not PDF_AVAILABLE:
            raise Exception("PDF support not installed. Run: pip install pdfplumber PyPDF2")
        content = self._parse_pdf(file_path, script_column)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    # ... rest of method
```

Add new `_parse_pdf()` method:
```python
def _parse_pdf(self, file_path: str, script_column: int = 2) -> str:
    """Parse PDF file using PDFParser"""
    pdf_parser = PDFParser()

    if pdf_parser.detect_scanned_pdf(file_path):
        logger.warning("⚠️ Scanned PDF detected - text extraction may be poor")

    text_content, tables = pdf_parser.parse_pdf(file_path)
    logger.info(f"PDF parsed: {len(text_content)} chars, {len(tables)} tables")

    # Save as temp .txt and parse with existing logic
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(text_content)
        tmp_path = tmp.name

    try:
        content = self._parse_txt(tmp_path, script_column)
        return content
    finally:
        os.remove(tmp_path)
```

#### Task 2.4: Update UI
**File:** `templates/file_to_slides.html`

Find file input (search for `accept="`):
```html
<input type="file"
       name="file"
       accept=".docx,.txt,.pdf"  <!-- ADD .pdf -->
       class="file-input">
```

Update help text:
```html
<p class="text-xs text-gray-500 mt-1">
    Supported: Word documents (.docx), Text files (.txt), PDF files (.pdf)
</p>
```

Add PDF warning:
```html
<div id="pdf-warning" class="hidden mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded">
    <p class="text-xs text-yellow-800">
        ⚠️ <strong>PDF Note:</strong> Scanned PDFs may have poor text extraction.
        For best results, use text-based PDFs.
    </p>
</div>

<script>
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

#### Task 2.5: Update Backend
**File:** `file_to_slides.py`

Update allowed extensions:
```python
ALLOWED_EXTENSIONS = {'docx', 'txt', 'pdf'}  # Add 'pdf'
```

#### Task 2.6: Testing
Test with:
- Text-based PDF (article, report)
- PDF with tables
- Multi-page PDF
- Scanned PDF (should show warning)

**Success Criteria:**
- ✅ PDF files upload successfully
- ✅ Tables extracted as tab-delimited text
- ✅ Warning shown for scanned PDFs
- ✅ Slides generated from PDF content
- ✅ No errors in parsing

**Estimated Time:** 6-8 hours

---

### **Agent 3: Enhanced Table Handling** 📊
**Priority:** HIGH (User reported issues)
**Branch:** `feature/enhanced-tables`
**Files to Modify:**
- `slide_generator_pkg/document_parser.py`

**Dependencies:** None (independent implementation)

**Tasks:**
1. Implement merged cell detection for Word tables
2. Implement table header detection
3. Implement auto-mode detection (table vs paragraph)
4. Add table + text context merging
5. Test with complex table documents

**Detailed Instructions:**

#### Task 3.1: Merged Cell Support
**File:** `slide_generator_pkg/document_parser.py`

Update `_parse_docx()` method (around line 1250):

Add helper methods:
```python
def _is_merged_cell(self, cell) -> bool:
    """Check if cell is part of a merged cell"""
    try:
        tc = cell._element
        tcPr = tc.get_or_add_tcPr()

        # Check horizontal merge (gridSpan)
        gridSpan = tcPr.gridSpan
        if gridSpan is not None and int(gridSpan.val) > 1:
            return True

        # Check vertical merge (vMerge)
        vMerge = tcPr.vMerge
        return vMerge is not None
    except:
        return False

def _get_merged_cell_origin(self, cell, table, row_idx, cell_idx):
    """Get the origin cell of a merged cell"""
    # For vertically merged cells, find the first cell in the merge
    for prev_row_idx in range(row_idx - 1, -1, -1):
        prev_cell = table.rows[prev_row_idx].cells[cell_idx]
        if not self._is_merged_cell(prev_cell):
            return prev_cell
        # Check if this is the start of the merge
        tc = prev_cell._element
        tcPr = tc.get_or_add_tcPr()
        vMerge = tcPr.vMerge
        if vMerge is not None and vMerge.val != 'continue':
            return prev_cell
    return cell
```

Update table parsing loop:
```python
for row_idx, row in enumerate(table.rows):
    cells = []
    processed_indices = set()

    for cell_idx, cell in enumerate(row.cells):
        if cell_idx in processed_indices:
            continue

        # Check if this is a merged cell
        if self._is_merged_cell(cell):
            origin = self._get_merged_cell_origin(cell, table, row_idx, cell_idx)
            if origin != cell:
                # Skip duplicate merged cells
                continue
            # For horizontally merged, mark subsequent cells as processed
            tc = cell._element
            tcPr = tc.get_or_add_tcPr()
            gridSpan = tcPr.gridSpan
            if gridSpan is not None:
                span = int(gridSpan.val)
                for i in range(1, span):
                    processed_indices.add(cell_idx + i)

        cell_text = cell.text.strip()
        cells.append(cell_text)

    # Process cells based on script_column...
```

#### Task 3.2: Table Header Detection
**File:** `slide_generator_pkg/document_parser.py`

Add helper methods:
```python
def _is_header_row(self, row) -> bool:
    """Detect if row is a header row"""
    if not row.cells:
        return False

    first_cell = row.cells[0]

    # Check for bold formatting
    for paragraph in first_cell.paragraphs:
        for run in paragraph.runs:
            if run.bold:
                return True

    # Check for all caps (common header pattern)
    text = first_cell.text.strip()
    if text and text.isupper() and len(text.split()) <= 3:
        return True

    return False

def _extract_table_with_headers(self, table) -> dict:
    """Extract table with header detection"""
    if not table.rows:
        return {'headers': [], 'data': []}

    first_row = table.rows[0]
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

    return {'headers': headers, 'data': data}
```

Use in table processing:
```python
for table in doc.tables:
    table_info = self._extract_table_with_headers(table)
    headers = table_info['headers']

    # Use headers as context for bullet generation
    if headers and script_column > 0:
        header_context = f"Table columns: {', '.join(headers)}"
        # Pass to bullet generation...
```

#### Task 3.3: Auto-Mode Detection
**File:** `slide_generator_pkg/document_parser.py`

Add method:
```python
def analyze_document_structure(self, file_path: str, file_ext: str) -> dict:
    """Analyze document to detect tables and suggest best parsing mode"""

    if file_ext == 'docx':
        doc = Document(file_path)

        table_count = len(doc.tables)
        paragraph_count = len([p for p in doc.paragraphs if p.text.strip()])

        # Count total cells
        table_cells = sum(len(table.rows) * len(table.rows[0].cells) if table.rows else 0
                         for table in doc.tables)

        if table_count > 0 and table_cells > paragraph_count:
            primary_type = 'table'
            suggested_column = 2
        else:
            primary_type = 'text'
            suggested_column = 0

        return {
            'tables': table_count,
            'paragraphs': paragraph_count,
            'table_cells': table_cells,
            'primary_type': primary_type,
            'suggested_mode': suggested_column,
            'confidence': 'high' if abs(table_cells - paragraph_count) > 10 else 'low'
        }

    return {'primary_type': 'unknown', 'suggested_mode': 0}
```

Add backend endpoint in `file_to_slides.py`:
```python
@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    """Analyze uploaded document structure"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = file.filename
    file_ext = filename.rsplit('.', 1)[1].lower()

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

#### Task 3.4: Table + Text Context Merging
**File:** `slide_generator_pkg/document_parser.py`

See `FILE_FORMAT_ENHANCEMENT_PLAN.md` lines 657-696 for full implementation.

#### Task 3.5: Testing
Test with:
- Word doc with merged cells
- Word doc with header row
- Complex multi-column table
- Mixed tables + paragraphs

**Success Criteria:**
- ✅ Merged cells handled without duplication
- ✅ Table headers detected and used as context
- ✅ Auto-detection suggests correct mode
- ✅ No column misalignment errors

**Estimated Time:** 9-12 hours

---

### **Agent 4: Topic Separation (Phase 2b)** 🎯
**Priority:** CRITICAL (User's #1 priority)
**Branch:** `feature/topic-separation`
**Files to Modify:**
- `slide_generator_pkg/document_parser.py`
- `slide_generator_pkg/semantic_analyzer.py`

**Dependencies:** Tests already exist in `tests/test_topic_separation.py`

**Tasks:**
1. Implement topic boundary detection (2.1)
2. Implement semantic clustering (2.2)
3. Implement smart splitting (2.3)
4. Integration testing with existing tests
5. Validate with real documents

**Detailed Instructions:**

#### Task 4.1: Topic Boundary Detection
**File:** `slide_generator_pkg/semantic_analyzer.py`

Add method:
```python
def detect_topic_boundaries(self, text: str, min_topic_length: int = 150) -> List[int]:
    """
    Detect where topics change in text using semantic similarity

    Returns list of sentence indices where topic boundaries occur
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available - using simple boundary detection")
        return self._simple_boundary_detection(text, min_topic_length)

    # Split into sentences
    sentences = self._split_into_sentences(text)

    if len(sentences) < 3:
        return []

    # Compute TF-IDF for each sentence
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except:
        return []

    # Compute cosine similarity between adjacent sentences
    boundaries = []
    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]

        # Low similarity = topic boundary
        if sim < 0.3:  # Threshold for topic change
            # Check minimum topic length
            if i > 0:
                prev_boundary = boundaries[-1] if boundaries else 0
                topic_text = " ".join(sentences[prev_boundary:i])
                if len(topic_text) >= min_topic_length:
                    boundaries.append(i)

    return boundaries

def _simple_boundary_detection(self, text: str, min_topic_length: int) -> List[int]:
    """Fallback: detect boundaries by paragraph breaks"""
    paragraphs = text.split('\n\n')
    boundaries = []
    char_count = 0

    for i, para in enumerate(paragraphs[:-1]):
        char_count += len(para)
        if char_count >= min_topic_length:
            boundaries.append(i)
            char_count = 0

    return boundaries
```

#### Task 4.2: Semantic Clustering
**File:** `slide_generator_pkg/semantic_analyzer.py`

Add method:
```python
def cluster_content_by_topic(self, content_blocks: List[str], num_topics: int = None) -> List[List[int]]:
    """
    Group content blocks by semantic similarity

    Returns list of clusters (each cluster is list of block indices)
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available - no clustering")
        return [[i] for i in range(len(content_blocks))]

    if len(content_blocks) < 2:
        return [[0]] if content_blocks else []

    # Auto-detect optimal number of topics
    if num_topics is None:
        num_topics = min(5, max(2, len(content_blocks) // 3))

    # Vectorize content
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(content_blocks)
    except:
        return [[i] for i in range(len(content_blocks))]

    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)

    # Group by cluster
    clusters = [[] for _ in range(num_topics)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    # Sort clusters by first occurrence
    clusters = [c for c in clusters if c]  # Remove empty
    clusters.sort(key=lambda c: c[0])

    return clusters
```

#### Task 4.3: Smart Splitting
**File:** `slide_generator_pkg/document_parser.py`

Add method to split long content:
```python
def _smart_split_content(self, text: str, max_bullets_per_slide: int = 5) -> List[str]:
    """
    Split long content into multiple slides intelligently

    Uses topic boundaries to determine split points
    """
    # Detect topic boundaries
    boundaries = self.semantic_analyzer.detect_topic_boundaries(text)

    if not boundaries:
        # No clear boundaries, split by length
        return self._split_by_length(text, max_bullets_per_slide)

    # Split at boundaries
    sentences = self._split_into_sentences(text)
    chunks = []
    prev_boundary = 0

    for boundary in boundaries:
        chunk = " ".join(sentences[prev_boundary:boundary])
        if chunk.strip():
            chunks.append(chunk)
        prev_boundary = boundary

    # Add final chunk
    final_chunk = " ".join(sentences[prev_boundary:])
    if final_chunk.strip():
        chunks.append(final_chunk)

    return chunks

def _split_by_length(self, text: str, max_bullets: int) -> List[str]:
    """Fallback: split by estimated bullet count"""
    sentences = self._split_into_sentences(text)

    # Assume ~1-2 sentences per bullet
    sentences_per_slide = max_bullets * 2

    chunks = []
    for i in range(0, len(sentences), sentences_per_slide):
        chunk = " ".join(sentences[i:i+sentences_per_slide])
        if chunk.strip():
            chunks.append(chunk)

    return chunks
```

Integrate into slide creation:
```python
# In _content_to_slides_no_table_mode()
if len(combined_text) > 500:  # Long content
    chunks = self._smart_split_content(combined_text)

    for chunk_idx, chunk in enumerate(chunks):
        topic_sentence, bullet_points = self._create_bullet_points(chunk, fast_mode)

        chunk_title = f"{slide_title} (Part {chunk_idx + 1})" if len(chunks) > 1 else slide_title

        slides.append(SlideContent(
            title=chunk_title,
            content=bullet_points,
            slide_type='script',
            heading_level=slide_heading_level
        ))
else:
    # Normal single slide
    # ... existing code
```

#### Task 4.4: Integration Testing
Run existing tests:
```bash
python3 -m pytest tests/test_topic_separation.py -v
```

All tests should pass.

#### Task 4.5: Validation
Test with:
- Long documents (>1000 words)
- Documents with clear topic shifts
- Mixed content types

**Success Criteria:**
- ✅ Topic boundaries detected accurately (>80% precision)
- ✅ Related content clustered together
- ✅ Long sections split into multiple slides
- ✅ All tests in `tests/test_topic_separation.py` pass
- ✅ No regression in bullet quality

**Estimated Time:** 8-10 hours

---

### **Agent 5: Google Docs Heading Preservation** 📑
**Priority:** HIGH (User pain point)
**Branch:** `feature/gdocs-headings`
**Files to Modify:**
- `file_to_slides.py` (Google Docs fetching code)
- `slide_generator_pkg/document_parser.py`

**Dependencies:** Google Docs API already integrated

**Tasks:**
1. Update Google Docs fetching to use API with styles
2. Parse paragraph styles (HEADING_1, HEADING_2, etc.)
3. Convert to markdown with proper heading levels
4. Test with complex Google Docs hierarchies
5. Maintain backward compatibility with .txt export

**Detailed Instructions:**

#### Task 5.1: Fetch Google Docs with Styles
**File:** `file_to_slides.py`

Find the Google Docs fetching code (search for `docs.documents().get`):

Update to preserve styles:
```python
def fetch_google_doc_with_styles(doc_id: str, credentials):
    """Fetch Google Doc with paragraph styles preserved"""
    from googleapiclient.discovery import build

    service = build('docs', 'v1', credentials=credentials)
    doc = service.documents().get(documentId=doc_id).execute()

    structured_content = []

    for element in doc.get('body', {}).get('content', []):
        if 'paragraph' not in element:
            continue

        para = element['paragraph']
        style = para.get('paragraphStyle', {})
        named_style = style.get('namedStyleType', 'NORMAL_TEXT')

        # Extract text
        text_runs = []
        for elem in para.get('elements', []):
            if 'textRun' in elem:
                text_runs.append(elem['textRun'].get('content', ''))

        text = ''.join(text_runs).strip()

        if not text:
            continue

        # Map named styles to heading levels
        heading_map = {
            'HEADING_1': 1,
            'HEADING_2': 2,
            'HEADING_3': 3,
            'HEADING_4': 4,
            'HEADING_5': 5,
            'HEADING_6': 6,
            'NORMAL_TEXT': None,
            'SUBTITLE': None,
            'TITLE': 1  # Treat title as H1
        }

        heading_level = heading_map.get(named_style)

        structured_content.append({
            'text': text,
            'heading_level': heading_level,
            'named_style': named_style
        })

    return structured_content
```

#### Task 5.2: Convert to Markdown
**File:** `file_to_slides.py`

Add conversion function:
```python
def structured_content_to_markdown(structured_content: List[dict]) -> str:
    """Convert structured content with styles to markdown"""
    lines = []

    for block in structured_content:
        text = block['text']
        heading_level = block['heading_level']

        if heading_level:
            # Convert to markdown heading
            markdown_line = '#' * heading_level + ' ' + text
        else:
            # Regular text
            markdown_line = text

        lines.append(markdown_line)

    return '\n\n'.join(lines)
```

#### Task 5.3: Integrate into Document Flow
**File:** `file_to_slides.py`

Update the Google Docs handling in the `/upload` route:

```python
# When processing Google Docs URL
if is_google_docs_url:
    doc_id = extract_doc_id(url)

    # Try to fetch with styles first
    try:
        structured_content = fetch_google_doc_with_styles(doc_id, credentials)
        markdown_content = structured_content_to_markdown(structured_content)

        # Save as temporary markdown file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(markdown_content)
        temp_file.close()

        filename = f"google_doc_{doc_id}.txt"
        file_path = temp_file.name

    except Exception as e:
        logger.warning(f"Could not fetch with styles: {e}, falling back to plain text")
        # Fallback to existing plain text export
        # ... existing code
```

#### Task 5.4: Update DocumentParser
**File:** `slide_generator_pkg/document_parser.py`

Ensure `_parse_txt()` handles markdown headings (should already work):

```python
# In _parse_txt(), markdown heading detection (should already exist around line 930)
if line.startswith('#'):
    heading_level = len(line) - len(line.lstrip('#'))
    heading_level = min(heading_level, 6)
    is_heading = True
    # ... process heading
```

#### Task 5.5: Testing
Test with:
- Google Doc with H1, H2, H3 headings
- Nested heading hierarchy
- Mixed headings and paragraphs
- Compare: API vs .txt export accuracy

**Success Criteria:**
- ✅ Google Docs H1/H2/H3 detected at 100% accuracy
- ✅ Heading hierarchy preserved correctly
- ✅ No regression on .txt file parsing
- ✅ Backward compatible with existing flow
- ✅ Error handling for API failures (fallback to .txt)

**Estimated Time:** 4-6 hours

---

## 🔄 **COORDINATION STRATEGY**

### **Git Branch Management**

Each agent creates their own feature branch from `main`:
```bash
# Agent 1
git checkout main
git pull origin main
git checkout -b feature/speaker-notes

# Agent 2
git checkout main
git pull origin main
git checkout -b feature/pdf-input

# Agent 3
git checkout main
git pull origin main
git checkout -b feature/enhanced-tables

# Agent 4
git checkout main
git pull origin main
git checkout -b feature/topic-separation

# Agent 5
git checkout main
git pull origin main
git checkout -b feature/gdocs-headings
```

### **File Conflict Prevention**

**Potential Conflicts:**
- `slide_generator_pkg/document_parser.py` - Agents 1, 2, 3, 4 all modify
- `file_to_slides.py` - Agents 2, 5 modify
- `requirements.txt` - Agent 2 modifies

**Resolution Strategy:**

#### For `document_parser.py`:
Each agent works on different methods:
- **Agent 1:** `_content_to_slides_no_table_mode()`, `_parse_txt()` (bracket fix)
- **Agent 2:** `parse_file()`, add `_parse_pdf()` (new method)
- **Agent 3:** `_parse_docx()`, add helper methods (new)
- **Agent 4:** `_smart_split_content()` (new method), modify slide creation

**Merge order:** Agent 2 → Agent 3 → Agent 4 → Agent 1
- Agent 2 adds PDF support (minimal changes)
- Agent 3 adds table helpers (new methods)
- Agent 4 adds splitting logic (integrates with Agent 1)
- Agent 1 merges last (integrates speaker notes with all above)

#### For `file_to_slides.py`:
- **Agent 2:** Modifies `ALLOWED_EXTENSIONS`, adds `/api/analyze-document`
- **Agent 5:** Modifies Google Docs fetching

**Merge order:** Agent 5 → Agent 2
- Agent 5 changes Google Docs flow
- Agent 2 adds PDF endpoints (no conflict)

#### For `requirements.txt`:
- **Agent 2:** Adds pdfplumber, PyPDF2

**No conflict:** Only one agent modifies

### **Merge Order (Critical Path)**

```
main
├── Agent 5 (gdocs-headings) ← Merge first (file_to_slides.py)
│   └── Agent 2 (pdf-input) ← Merge second (file_to_slides.py + requirements.txt)
│       └── Agent 3 (enhanced-tables) ← Merge third (document_parser.py helpers)
│           └── Agent 4 (topic-separation) ← Merge fourth (document_parser.py splitting)
│               └── Agent 1 (speaker-notes) ← Merge last (integrates all)
```

**Why this order:**
1. **Agent 5** - Independent Google Docs changes
2. **Agent 2** - Adds PDF support (builds on Agent 5's doc fetching)
3. **Agent 3** - Adds table helpers (doesn't conflict with PDF)
4. **Agent 4** - Adds topic separation (uses table parsing)
5. **Agent 1** - Speaker notes integrate with all above features

### **Testing Between Merges**

After each merge, run CI:
```bash
./scripts/quick_ci.sh
python3 -m pytest tests/ -v
```

Only proceed to next merge if all tests pass.

### **Communication Protocol**

Each agent should:
1. **Start:** Comment on their branch creation
2. **Progress:** Push commits frequently
3. **Complete:** Run local tests before marking ready
4. **Ready:** Signal when ready to merge
5. **Blocked:** Immediately notify if blocked by another agent

---

## ✅ **ACCEPTANCE CRITERIA**

### **After All Merges Complete:**

#### Speaker Notes (Agent 1):
- [ ] `ryans_doc.txt` generates slides with speaker notes
- [ ] PowerPoint shows notes in presenter view
- [ ] Google Slides has speaker notes (if supported)
- [ ] Bracketed text removed from all output
- [ ] Full script preserved in notes

#### PDF Input (Agent 2):
- [ ] PDF files upload successfully
- [ ] PDF tables extracted correctly
- [ ] Scanned PDF warning displays
- [ ] Slides generated from PDF content
- [ ] UI shows PDF in supported formats

#### Enhanced Tables (Agent 3):
- [ ] Merged cells handled correctly
- [ ] Table headers detected
- [ ] Auto-mode detection working
- [ ] No column misalignment errors
- [ ] Complex tables parse successfully

#### Topic Separation (Agent 4):
- [ ] Long content split into multiple slides
- [ ] Topic boundaries detected accurately
- [ ] Related content clustered together
- [ ] All `test_topic_separation.py` tests pass
- [ ] No regression in bullet quality

#### Google Docs Headings (Agent 5):
- [ ] H1/H2/H3 detected at 100% accuracy from Google Docs
- [ ] Heading hierarchy preserved
- [ ] Fallback to .txt export works
- [ ] No regression on .txt files
- [ ] Error handling for API failures

### **Overall Integration:**
- [ ] All CI tests pass (`./scripts/quick_ci.sh`)
- [ ] All pytest tests pass (`pytest tests/ -v`)
- [ ] No syntax errors
- [ ] No security regressions
- [ ] Speaker notes work with PDF input
- [ ] Topic separation works with enhanced tables
- [ ] All 5 features work together seamlessly

---

## 📊 **ESTIMATED TIMELINE**

### **Parallel Execution:**
- **Agent 1:** 3-4 hours
- **Agent 2:** 6-8 hours
- **Agent 3:** 9-12 hours (longest)
- **Agent 4:** 8-10 hours
- **Agent 5:** 4-6 hours

**Wall Clock Time (Parallel):** **9-12 hours** (limited by Agent 3)

**Sequential Merges:** ~2 hours (testing between each)

**Total:** **11-14 hours** vs. **30-40 hours sequential** = **60% time savings**

### **Merge Schedule:**

```
Hour 0:   All agents start in parallel
Hour 4-6: Agent 5 completes → Merge → Test
Hour 6-8: Agent 2 completes → Merge → Test
Hour 8-10: Agent 1, 4 complete
Hour 9-12: Agent 3 completes → Merge → Test
Hour 12-13: Agent 4 merges → Test
Hour 13-14: Agent 1 merges → Final test
Hour 14: ✅ ALL COMPLETE
```

---

## 🚀 **EXECUTION COMMAND**

**For Claude Code Web, execute:**

```
Implement the top 5 improvements for the slide generator app using 5 parallel agents:

1. Agent 1 (feature/speaker-notes): Add speaker notes support - preserve full script text in PowerPoint/Google Slides speaker notes
2. Agent 2 (feature/pdf-input): Add PDF input support with pdfplumber and PyPDF2
3. Agent 3 (feature/enhanced-tables): Enhance table handling - merged cells, headers, auto-detection
4. Agent 4 (feature/topic-separation): Complete topic separation Phase 2b - semantic clustering and smart splitting
5. Agent 5 (feature/gdocs-headings): Google Docs heading preservation using API instead of .txt export

Follow the coordination strategy in PARALLEL_IMPLEMENTATION_PROMPT.md:
- Each agent creates their own feature branch from main
- Merge order: Agent 5 → 2 → 3 → 4 → 1
- Run ./scripts/quick_ci.sh after each merge
- All agents work concurrently, merge sequentially

See PARALLEL_IMPLEMENTATION_PROMPT.md for full implementation details, file paths, code snippets, and acceptance criteria.
```

---

## 📚 **REFERENCE DOCUMENTS**

Agents should reference these files for implementation details:

- **Speaker Notes:** `VIDEO_SCRIPT_REQUIREMENTS.md` (Phases 1-3)
- **PDF Support:** `FILE_FORMAT_ENHANCEMENT_PLAN.md` (Phase 1, lines 68-385)
- **Table Handling:** `FILE_FORMAT_ENHANCEMENT_PLAN.md` (Phase 2, lines 388-646)
- **Topic Separation:** `IMPROVEMENT_RECOMMENDATIONS.md` (Section 2.1-2.3)
- **Google Docs API:** `VIDEO_SCRIPT_REQUIREMENTS.md` (Phase 4, lines 744-end)
- **Overall Roadmap:** `COMPREHENSIVE_IMPROVEMENT_ROADMAP.md`

**Test Files:**
- `ryans_doc.txt` - Video script with bracketed notes (speaker notes test)
- `tests/test_topic_separation.py` - Topic separation tests
- `tests/test_bullet_quality.py` - Bullet quality tests

---

**Status:** Ready for parallel execution
**Estimated Completion:** 11-14 hours (parallel) vs 30-40 hours (sequential)
**Risk:** Low (clear separation of concerns, detailed merge strategy)
**Blocker:** None - all dependencies installed, comprehensive plans available

