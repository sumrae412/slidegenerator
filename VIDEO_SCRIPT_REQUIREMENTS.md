# Video Script Document Requirements

**Use Case:** Creating engaging video teaching lessons (e.g., AI courses, online education)

**Test File:** `ryans_doc.txt` (typical video script with production notes)

---

## User Requirements (Confirmed)

Based on walkthrough with real-world scenario (video teaching lessons about AI):

1. ✅ **Skip sections with no content entirely** - Don't create slides for empty sections
2. ✅ **Headings with no content → Title-only slides** (empty bullet list)
3. ✅ **Ignore `[bracketed]` text** - Production notes, director cues, edit suggestions
4. ✅ **Bullets = main teaching points** extracted from script (AI summarization)
5. ✅ **Full script text → Speaker notes** (preserved verbatim after bracket removal)
6. ✅ **Respect H1/H2/H3 hierarchy** using heuristic detection from plain text

---

## Current Behavior vs. Expected

### ✅ **Working Correctly**

- **Empty slide creation:** Sections with headings but no content → title-only slides
- **Heading detection:** Video script patterns like `C1W1L1_1 - Welcome to AI` detected as H2
- **Hierarchy tracking:** H1/H2/H3 levels preserved

### ❌ **Broken/Missing**

1. **Bracketed text removal:**
   - **Current:** Removed at line 856 with `re.sub(r'\[.*?\]', '', raw_content)`
   - **Issue:** Non-greedy `.*?` stops at first `]`, breaking multi-line brackets
   - **Fix:** Use `re.sub(r'\[.*?\]', '', raw_content, flags=re.DOTALL)`

2. **Speaker notes field:**
   - **Current:** `SlideContent` has no `speaker_notes` attribute
   - **Missing:** Full script text preservation
   - **Fix:** Add `speaker_notes: Optional[str] = None` to `SlideContent` dataclass

3. **Script → Bullets extraction:**
   - **Current:** Fails to extract bullets from narrative script (lines 74-134 in ryans_doc.txt)
   - **Issue:** Script is conversational prose, not pre-formatted bullets
   - **Fix:** Use LLM to extract 3-5 key teaching points from script

4. **Google Docs heading preservation:**
   - **Current:** Heuristic detection from plain text (works ~70% of time)
   - **Issue:** H1/H2/H3 formatting lost when exporting to `.txt`
   - **Challenge:** Google Docs API doesn't expose paragraph styles via API
   - **Workaround:** Current heuristics are best we can do for `.txt` exports

---

## Implementation Plan

### **Phase 1: Core Infrastructure** (30 min)

#### 1.1 Add Speaker Notes Support

**File:** `slide_generator_pkg/data_models.py`

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
    # ... rest of fields
```

**Benefit:** Store full script alongside summarized bullets

---

#### 1.2 Fix Bracket Removal (Multi-line)

**File:** `slide_generator_pkg/document_parser.py:856`

**Current:**
```python
raw_content = re.sub(r'\[.*?\]', '', raw_content)
```

**Fixed:**
```python
# Remove bracketed production notes (including multi-line)
raw_content = re.sub(r'\[.*?\]', '', raw_content, flags=re.DOTALL)
```

**Benefit:** Correctly removes multi-line production notes like:
```
[Note to reviewer: the purpose of this video is to:
* Emphasize the AI does not always add value
* Data privacy considerations
* Word count ~950]
```

---

### **Phase 2: Script → Bullets + Speaker Notes** (2-3 hours)

#### 2.1 Extract Speaker Notes from Script

When processing script content (paragraphs between headings):

**File:** `slide_generator_pkg/document_parser.py`

**Add to `_content_to_slides_no_table_mode()` method:**

```python
# Around line 2000-2010 where slides are created

# BEFORE bracket removal, preserve original script for speaker notes
original_script_text = "\n\n".join(content_buffer)

# AFTER bracket removal, clean script text
cleaned_script = re.sub(r'\[.*?\]', '', original_script_text, flags=re.DOTALL)

# Extract bullets from script using LLM/NLP
bullet_points = self._create_bullet_points(
    cleaned_script,
    context_heading=slide_title,
    heading_ancestry=heading_stack
)

# Create slide with BOTH bullets and speaker notes
slides.append(SlideContent(
    title=slide_title,
    content=bullet_points,  # 3-5 key teaching points
    speaker_notes=cleaned_script.strip(),  # Full script text
    slide_type='script',
    heading_level=slide_heading_level,
    subheader=topic_sentence,
    visual_cues=slide_visual_cues
))
```

**Benefit:**
- Bullets = concise student-facing points
- Speaker notes = full script for instructor

---

#### 2.2 Enhance LLM Prompt for Script Extraction

**File:** `slide_generator_pkg/document_parser.py`

**Update `_create_llm_only_bullets()` prompt:**

```python
def _create_llm_only_bullets(self, text, heading="", max_bullets=5):
    """Generate bullets using Claude/OpenAI with context"""

    # Detect if this is a teaching script (conversational tone)
    is_script = self._is_teaching_script(text)

    if is_script:
        # Script-specific prompt
        prompt = f"""You are converting a video teaching script into student-facing slide bullets.

SCRIPT (Instructor narration):
{text}

CONTEXT: This is for a slide titled "{heading}"

Extract 3-5 KEY TEACHING POINTS that students should remember:
- Focus on core concepts, not narrative flow
- Use clear, concise language (8-15 words per bullet)
- Ensure bullets are parallel in structure
- Start with strong action verbs when possible
- Ignore production notes, visual cues, or meta-commentary

Return ONLY the bullet points, one per line, no numbering."""

    else:
        # Standard content prompt (existing)
        prompt = f"""Extract 3-5 key points from this content for a presentation slide...
        # ... existing prompt
        """

    # Send to LLM...
```

**New helper method:**

```python
def _is_teaching_script(self, text: str) -> bool:
    """Detect if text is a teaching script vs. structured content"""

    # Script indicators:
    # - First/second person ("you", "I", "we")
    # - Conversational transitions ("Now let's", "So", "And")
    # - Long paragraphs (>200 chars)
    # - Questions directed at audience

    script_indicators = [
        r'\b(you|I|we|let\'s|now)\b',  # First/second person
        r'\b(So|And|But|Now)\b',  # Conversational transitions
        r'[.!?]\s+[.!?]',  # Multiple sentences
        text.count('\n\n') < 3,  # Few paragraph breaks (flowing text)
        len(text) > 200  # Longer content
    ]

    matches = sum(1 for pattern in script_indicators[:4]
                  if re.search(pattern, text))
    matches += 1 if script_indicators[4] else 0

    # If 3+ indicators, it's a script
    return matches >= 3
```

**Benefit:** Tailored bullet extraction for teaching scripts vs. structured documents

---

### **Phase 3: PowerPoint/Google Slides Output** (1 hour)

#### 3.1 Add Speaker Notes to PowerPoint Generation

**File:** `pptx_generator.py` (or wherever PowerPoint is created)

```python
from pptx import Presentation
from pptx.util import Inches, Pt

# When creating content slides:
for slide_content in slides:
    slide = prs.slides.add_slide(content_layout)

    # Add title
    slide.shapes.title.text = slide_content.title

    # Add bullets
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    for bullet in slide_content.content:
        p = tf.add_paragraph()
        p.text = bullet
        p.level = 0

    # NEW: Add speaker notes if present
    if slide_content.speaker_notes:
        notes_slide = slide.notes_slide
        notes_text_frame = notes_slide.notes_text_frame
        notes_text_frame.text = slide_content.speaker_notes
```

**Benefit:** Full script preserved in PowerPoint presenter view

---

#### 3.2 Add Speaker Notes to Google Slides Output

**File:** `google_slides_generator.py`

```python
# When creating slides via Google Slides API:
requests = []

for slide_content in slides:
    # Create slide
    slide_id = f'slide_{index}'
    requests.append({
        'createSlide': {
            'objectId': slide_id,
            'slideLayoutReference': {'predefinedLayout': 'TITLE_AND_BODY'}
        }
    })

    # Add title and bullets
    # ... existing code ...

    # NEW: Add speaker notes
    if slide_content.speaker_notes:
        requests.append({
            'createSpeakerNotes': {
                'slideId': slide_id,
                'speakerNotesText': slide_content.speaker_notes
            }
        })

# Execute batch update
slides_service.presentations().batchUpdate(
    presentationId=presentation_id,
    body={'requests': requests}
).execute()
```

**Benefit:** Speaker notes sync to Google Slides

---

### **Phase 4: Google Docs Heading Preservation** (Research - Future)

**Challenge:** When Google Docs exports to `.txt`, heading styles are lost.

**Current Workaround:** Heuristic detection (works ~70% of time):
- Video script IDs: `C1W1L1_1 - Title` → H2
- ALL CAPS short lines → H1
- Title Case short lines with keywords → H2
- Markdown syntax: `### Heading` → H3

**Better Solution (Requires Google Docs API):**

```python
def _fetch_google_doc_with_styles(self, doc_id: str) -> List[dict]:
    """Fetch Google Doc with paragraph styles preserved"""

    from googleapiclient.discovery import build

    service = build('docs', 'v1', credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()

    structured_content = []

    for element in doc.get('body').get('content'):
        if 'paragraph' in element:
            para = element['paragraph']
            style = para.get('paragraphStyle', {})
            named_style = style.get('namedStyleType', 'NORMAL_TEXT')

            text = ''.join(
                elem.get('textRun', {}).get('content', '')
                for elem in para.get('elements', [])
            )

            # Map named styles to heading levels
            if named_style == 'HEADING_1':
                heading_level = 1
            elif named_style == 'HEADING_2':
                heading_level = 2
            elif named_style == 'HEADING_3':
                heading_level = 3
            else:
                heading_level = None

            structured_content.append({
                'text': text.strip(),
                'heading_level': heading_level,
                'named_style': named_style
            })

    return structured_content
```

**Implementation Path:**
1. Check if input is Google Docs URL (already supported)
2. Use Google Docs API to fetch document structure (instead of exporting to `.txt`)
3. Preserve heading levels directly from API
4. Apply bracket removal and other processing

**Effort:** 4-6 hours (requires OAuth, API integration, testing)

**Benefit:** 100% accurate heading detection for Google Docs

---

## Testing Strategy

### Test Cases for `ryans_doc.txt`

**Expected Output:**

| Slide # | Title | Bullets | Speaker Notes | Type |
|---------|-------|---------|---------------|------|
| 1 | C1 W1 Introduction to AI for Good | 0 | None | heading |
| 2 | Lesson 1 - Specialization & Course Introduction | 0 | None | heading |
| 3-6 | (Empty sections) | *SKIP* | *SKIP* | *SKIP* |
| 7 | Lesson 2 - Introduction to Artificial Intelligence | 0 | None | heading |
| 8-11 | (Empty sections) | *SKIP* | *SKIP* | *SKIP* |
| 12 | Considering the Impact of Your AI for Good Project | 3-5 | Full script (lines 74-134, brackets removed) | script |

**Slide 12 Expected Bullets (AI-extracted):**
- AI does not necessarily add value to every project
- Data privacy and personally identifiable information require extreme care
- Consider false positives vs. false negatives in model deployment
- Apply the "do no harm" principle with stakeholder input
- Evaluate both positive and negative potential impacts

**Slide 12 Expected Speaker Notes:**
```
As you've seen from the previous videos, the possible applications of AI are wide ranging...

[Full script from lines 74-134, with all [bracketed] content removed]

...After this video there's a short quiz for you to practice the subjects that you have learned so far.
```

---

### Validation Checklist

- [ ] Empty sections (C1W1L1_1 through L1_5) are skipped entirely
- [ ] Bracketed content `[Note to reviewer: ...]` removed from all slides
- [ ] Bracketed content `[>>>click]`, `[full screen TH]` removed from speaker notes
- [ ] Multi-line brackets `[a]actually this one goes...` removed
- [ ] Script content (lines 74-134) extracted as bullets (3-5 key points)
- [ ] Full script preserved in speaker notes field
- [ ] Heading hierarchy maintained (H1 → H2 → script slides)
- [ ] PowerPoint output includes speaker notes
- [ ] Google Slides output includes speaker notes

---

## Implementation Priority

**Phase 1 (Core Infrastructure)** - **MUST DO FIRST**
- Add `speaker_notes` field to `SlideContent`
- Fix bracket removal regex (multi-line support)
- **Time:** 30 minutes
- **Blockers:** None

**Phase 2 (Script Processing)** - **HIGH PRIORITY**
- Extract bullets from teaching scripts
- Preserve full script as speaker notes
- Detect teaching script vs. structured content
- **Time:** 2-3 hours
- **Blockers:** Requires Phase 1

**Phase 3 (Output Generation)** - **HIGH PRIORITY**
- Add speaker notes to PowerPoint output
- Add speaker notes to Google Slides output
- **Time:** 1 hour
- **Blockers:** Requires Phase 1 & 2

**Phase 4 (Google Docs API)** - **FUTURE ENHANCEMENT**
- Preserve heading styles from Google Docs directly
- **Time:** 4-6 hours
- **Blockers:** OAuth setup, API integration
- **Benefit:** 100% accurate heading detection (vs. 70% heuristic)

---

## Edge Cases to Consider

1. **Nested brackets:** `[Note: [edit this]]`
   - Current regex would only remove inner `[edit this]`
   - Fix: Use non-greedy + DOTALL flag

2. **Brackets spanning multiple paragraphs:**
   ```
   [Note to reviewer:

   This is a multi-paragraph
   production note]
   ```
   - Current regex stops at first newline
   - Fix: `re.DOTALL` flag

3. **Square brackets in actual content:** `The array [1, 2, 3] is sorted`
   - Risk: Legitimate brackets removed
   - Mitigation: Check for production note keywords (`Note to reviewer`, `>>>click`, `b-roll`)
   - Better fix: More specific pattern: `\[(?:Note|>>>|b-roll|full screen).*?\]`

4. **Empty headings with whitespace:**
   ```
   C1W1L1_1 - Welcome to AI for Good




   C1W1L1_2 - What is AI?
   ```
   - Should detect as empty and skip
   - Fix: `len(content.strip()) == 0`

5. **Script with no clear teaching points:**
   - Very short scripts (<100 words)
   - Purely transitional content ("Next, we'll move to...")
   - Fix: Minimum threshold for bullet extraction (>150 words)

---

## Success Metrics

**After implementation, `ryans_doc.txt` should produce:**

- ✅ 3 heading slides (not 12+)
- ✅ 1 content slide with 3-5 bullets (from lines 74-134)
- ✅ No bracketed text in any slide content
- ✅ Full script (500+ words) in speaker notes
- ✅ Bullets are concise teaching points, not full sentences from script
- ✅ PowerPoint has speaker notes visible in presenter view
- ✅ Google Slides has speaker notes accessible

**Quality Benchmarks:**
- Processing time: <10 seconds for typical video script (5-10 pages)
- Bullet relevance score: >80% (measured by keyword overlap with script)
- Speaker notes completeness: 100% of script content (minus brackets)

---

**Status:** Ready for implementation
**Next Step:** Implement Phase 1 (Core Infrastructure)
**Test File:** `ryans_doc.txt`
**Target Completion:** Phase 1-3 in ~4 hours

