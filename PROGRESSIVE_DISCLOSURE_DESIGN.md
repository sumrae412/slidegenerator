# Progressive Disclosure for Script Column Selector

## Overview
This document describes the implementation of progressive disclosure for the script column selector in the File to Slides application. The selector is hidden by default and appears intelligently based on document analysis, with auto-detection of tables and smart suggestions.

## Key Requirements Met

### 1. Hide by Default
- Script column selector container is initially `display: none`
- Hidden from UI flow until document is selected

### 2. Auto-Detection & Smart Showing
When document is uploaded/selected:
- Call existing `/api/analyze-document` endpoint
- Analyze for tables vs paragraphs
- Show selector only if tables detected OR allow no-table selection

### 3. Visual Indicators
- **Suggestion Badge**: "Suggested: Column 2" displayed when tables found
- **Auto-selection**: "No table - Use paragraphs" pre-selected when no tables detected
- **Disabled State**: Column options disabled when no tables found

### 4. Smooth Transitions
- Fade in/out with opacity transitions (300ms)
- Slide down animation when appearing
- All Tailwind CSS classes for consistency

### 5. One-Click Action
- "Use Suggestion" button for immediate column selection
- Updates select value and dismisses suggestion

## Implementation Strategy

### DOM Structure Changes

**Original Location (HIDE):**
```html
<!-- Lines 223-239: Script Column in grid layout -->
<!-- This will be hidden but kept for backward compatibility -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <div>
        <label for="script-column">...</label>
        <select id="script-column">...</select>
    </div>
    <!-- API Keys Section -->
</div>
```

**New Location (INSERT AFTER document info, line 557):**
```html
<!-- Progressive Disclosure: Script Column Selector -->
<div id="script-column-disclosure" class="hidden script-column-hidden mb-6">
    <!-- Compact card with suggestion -->
</div>
```

### JavaScript Integration Points

1. **analyzeDocument()** - Already calls `/api/analyze-document`
   - Enhanced to trigger `updateScriptColumnDisclosure()`

2. **displayAnalysis()** - Already shows analysis
   - Enhanced to populate script column selector state

3. **acceptSuggestion()** - Already exists
   - Enhanced to handle progressive disclosure close

4. **Document selection events** - File upload, URL input, Drive picker
   - All trigger document analysis

## Visual Design

### States

#### State 1: Hidden (Default)
- Container display: none
- No space taken in layout
- Minimal DOM footprint

#### State 2: No Tables Found
- Container visible
- Select shows "No table - Use paragraphs" (pre-selected)
- All column options disabled
- Informational text: "This document uses paragraph mode"
- Optional close button to hide

#### State 3: Tables Detected
- Container visible
- Select shows all column options enabled
- Suggestion badge: "Suggested: Column X"
- Info text: "Found X table(s) with script content"
- "Use Suggestion" button (primary action)
- Compact, inline presentation

### Color Scheme (Tailwind)
- Container: `bg-blue-50 border border-blue-200`
- Suggestion badge: `bg-blue-600 text-white text-xs px-2 py-1 rounded-full`
- "Use Suggestion" button: `bg-blue-600 hover:bg-blue-700 text-white`
- Disabled state: `bg-slate-100 text-slate-500`

### Transitions
- Fade in: `opacity-0` → `opacity-100` (300ms)
- Slide down: `transform -translate-y-4` → `translate-y-0` (300ms)
- Select focus: Ring effect with `focus:ring-blue-500`

## Integration with Existing Code

### Modified: analyzeDocument() function (line 920)
```javascript
async function analyzeDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/analyze-document', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.error('Analysis failed:', response.statusText);
            return;
        }

        const analysis = await response.json();
        displayAnalysis(analysis);
        updateScriptColumnDisclosure(analysis);  // NEW
    } catch (error) {
        console.error('Analysis failed:', error);
    }
}
```

### Modified: displayAnalysis() function (line 942)
```javascript
function displayAnalysis(analysis) {
    const resultDiv = document.getElementById('analysis-result');
    const panel = document.getElementById('document-analysis');

    if (analysis.primary_type === 'table') {
        resultDiv.innerHTML = `Found ${analysis.tables} table(s). Suggested mode: <strong>Table Column ${analysis.suggested_mode}</strong>`;
    } else {
        resultDiv.innerHTML = `Found ${analysis.paragraphs} paragraph(s). Suggested mode: <strong>Paragraph Mode</strong>`;
    }

    documentSuggestion = analysis.suggested_mode;
    panel.classList.remove('hidden');
    // Note: Progressive disclosure is now in updateScriptColumnDisclosure()
}
```

### New: updateScriptColumnDisclosure() function
```javascript
function updateScriptColumnDisclosure(analysis) {
    const container = document.getElementById('script-column-disclosure');
    const select = document.getElementById('script-column');
    const suggestedMode = analysis.suggested_mode;
    const hasMultipleTables = (analysis.tables || 0) > 1;

    // Update the hidden select
    select.value = suggestedMode;

    if (analysis.primary_type === 'table') {
        // Tables found - show selector with suggestion
        showTableModeDisclosure(analysis, suggestedMode, hasMultipleTables);
    } else {
        // No tables - show compact info or hide
        showNoTableModeDisclosure(analysis);
    }
}
```

### New: showTableModeDisclosure() function
```javascript
function showTableModeDisclosure(analysis, suggestedMode, hasMultipleTables) {
    const container = document.getElementById('script-column-disclosure');
    const select = document.getElementById('script-column');

    // Populate dropdown content
    const columnOptions = document.getElementById('disclosure-column-select');
    columnOptions.value = suggestedMode;

    // Update suggestion badge
    document.getElementById('suggested-column-badge').textContent =
        `Suggested: Column ${suggestedMode}`;

    // Update info text
    document.getElementById('disclosure-table-info').textContent =
        `Found ${analysis.tables} table(s). ${hasMultipleTables ? 'Choose which column contains your script.' : 'Column auto-detected.'}`;

    // Show container with animation
    container.classList.remove('hidden', 'script-column-hidden');
    container.classList.add('script-column-visible');

    // Trigger animation
    setTimeout(() => {
        container.style.opacity = '1';
        container.style.transform = 'translateY(0)';
    }, 10);
}
```

### New: showNoTableModeDisclosure() function
```javascript
function showNoTableModeDisclosure(analysis) {
    const container = document.getElementById('script-column-disclosure');
    const select = document.getElementById('script-column');

    // Auto-select paragraph mode
    select.value = '0';

    // Show compact info
    const infoDiv = document.getElementById('disclosure-no-table-info');
    infoDiv.textContent = `${analysis.paragraphs || 0} paragraph(s) detected. Using paragraph mode.`;

    // Show but with different styling (less prominent)
    container.classList.remove('hidden', 'script-column-hidden');
    container.classList.add('script-column-visible', 'script-column-no-table');
}
```

### Integration with Document Removal (line 1186)
```javascript
document.getElementById('remove-doc').addEventListener('click', () => {
    selectedDocUrl = null;
    selectedDocName = null;
    googleDocsUrlInput.value = '';
    fileUploadInput.value = '';
    docInfo.classList.add('hidden');
    document.getElementById('pdf-warning').classList.add('hidden');
    document.getElementById('document-analysis').classList.add('hidden');

    // NEW: Hide script column disclosure
    hideScriptColumnDisclosure();

    convertBtn.disabled = true;
    hideError();
    hideResults();
});
```

### New: hideScriptColumnDisclosure() function
```javascript
function hideScriptColumnDisclosure() {
    const container = document.getElementById('script-column-disclosure');
    container.classList.add('hidden', 'script-column-hidden');
    container.classList.remove('script-column-visible');
    container.style.opacity = '0';
}
```

## CSS Classes

Add to `<style>` section (around line 158):

```css
/* Progressive Disclosure for Script Column Selector */
.script-column-hidden {
    opacity: 0;
    transform: translateY(-16px);
}

.script-column-visible {
    opacity: 1;
    transform: translateY(0);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.script-column-no-table {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-color: #86efac;
}

.script-column-no-table .disclosure-label {
    color: #059669;
    font-weight: 600;
}

.suggestion-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.75rem;
    animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-8px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.use-suggestion-btn {
    white-space: nowrap;
    transition: all 0.2s ease;
}

.use-suggestion-btn:hover:not(:disabled) {
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    transform: translateY(-1px);
}
```

## HTML Markup to Insert (After line 557)

Insert this after the `doc-info` div:

```html
<!-- Progressive Disclosure: Script Column Selector -->
<div id="script-column-disclosure" class="hidden script-column-hidden mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg transition-all duration-300">
    <!-- Table Mode: With suggestion -->
    <div id="disclosure-table-mode">
        <div class="mb-3">
            <div class="flex items-center justify-between mb-2">
                <label for="disclosure-column-select" class="text-sm font-semibold text-slate-800">
                    Script Column
                </label>
                <span id="suggested-column-badge" class="suggestion-badge">
                    Suggested: Column 2
                </span>
            </div>

            <select id="disclosure-column-select"
                    class="w-full px-4 py-2 border border-blue-300 rounded-lg bg-white text-slate-800 text-sm transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-200">
                <option value="0" disabled>Select a column...</option>
                <option value="1">Column 1</option>
                <option value="2">Column 2</option>
                <option value="3">Column 3</option>
                <option value="4">Column 4</option>
                <option value="5">Column 5</option>
                <option value="6">Column 6</option>
            </select>

            <p id="disclosure-table-info" class="text-xs text-blue-700 mt-2">
                Found 1 table(s). Choose which column contains your script.
            </p>
        </div>

        <div class="flex gap-2">
            <button type="button"
                    onclick="applySuggestionFromDisclosure()"
                    class="use-suggestion-btn flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-all">
                Use Suggestion
            </button>
            <button type="button"
                    onclick="hideScriptColumnDisclosure()"
                    class="px-3 py-2 bg-white border border-slate-300 text-slate-700 text-sm font-medium rounded-lg hover:bg-slate-50 transition-all">
                Later
            </button>
        </div>
    </div>

    <!-- No Table Mode: Compact info -->
    <div id="disclosure-no-table-mode" class="hidden">
        <div class="flex items-center justify-between">
            <div>
                <p class="text-sm font-semibold text-emerald-900">Paragraph Mode Selected</p>
                <p id="disclosure-no-table-info" class="text-xs text-emerald-700 mt-1">
                    45 paragraphs detected. Using paragraph mode.
                </p>
            </div>
            <button type="button"
                    onclick="hideScriptColumnDisclosure()"
                    class="text-emerald-600 hover:text-emerald-800 transition-colors">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
    </div>
</div>
```

## Implementation Checklist

- [x] Design progressive disclosure states (hidden, no-table, tables found)
- [x] Create HTML markup with both table and no-table modes
- [x] Add CSS for smooth transitions and animations
- [x] Create JavaScript functions for show/hide logic
- [x] Integrate with existing analyzeDocument() flow
- [x] Update displayAnalysis() to trigger disclosure
- [x] Handle document removal to hide disclosure
- [x] Add "Use Suggestion" button with onclick handler
- [x] Position near document info display
- [x] Ensure backward compatibility with original select

## Testing Checklist

1. Upload document with tables
   - [ ] Script column selector appears with suggestion
   - [ ] Correct table count shown
   - [ ] Suggested column highlighted
   - [ ] "Use Suggestion" button works

2. Upload document without tables
   - [ ] Compact info shown instead
   - [ ] "No table - Use paragraphs" pre-selected
   - [ ] Less prominent styling
   - [ ] Can be dismissed

3. Remove/change document
   - [ ] Disclosure hides smoothly
   - [ ] No flash or jumping
   - [ ] UI is clean

4. Browser compatibility
   - [ ] Chrome/Firefox/Safari
   - [ ] Mobile responsiveness
   - [ ] Transitions work smoothly

5. Integration with existing flow
   - [ ] Form submission includes correct script_column
   - [ ] History save includes column selection
   - [ ] API keys section still visible
   - [ ] No layout shifts

## Files to Modify

1. **templates/file_to_slides.html**
   - Add CSS (lines 158-250)
   - Add HTML markup (after line 557)
   - Modify analyzeDocument() (line 920)
   - Modify displayAnalysis() (line 942)
   - Modify document removal handler (line 1186)
   - Add new JavaScript functions (around line 1822)

## No Backend Changes Required

The progressive disclosure is purely frontend logic that:
- Uses existing `/api/analyze-document` endpoint
- Maintains existing select element ID: `script-column`
- Integrates with existing form submission
- No new API endpoints needed
