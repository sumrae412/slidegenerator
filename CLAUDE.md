# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Flask web application that converts Google Docs into PowerPoint presentations or Google Slides with AI-generated bullet points and visual prompts. The app integrates with Google Drive for document selection, uses Google OAuth for authentication, and leverages Claude API for intelligent content processing.

## ⚠️ CRITICAL: NLP Approach Policy

**LOCKED IN: Smart NLP Approach (TF-IDF + spaCy)**

As of October 27, 2025, the fallback bullet generation uses **intelligent NLP** (TF-IDF sentence ranking + spaCy structure validation). This is the **permanent standard**.

### DO NOT:
- ❌ Revert to manual keyword-based filtering
- ❌ Replace with hardcoded word lists
- ❌ Remove spaCy or scikit-learn dependencies

### Quality Results:
- ✅ 80-85% success rate (up from 57% with manual approach)
- ✅ Automatic adaptation to any content type
- ✅ No maintenance required

**See**: `NLP_APPROACH_DECISION.md` for full rationale and test results.

**Reversion**: Only if explicitly requested by project owner.

## 🧪 CRITICAL: Quality Testing & Iterative Improvement Protocol

**REQUIRED PRACTICE: Test Before Deploy**

As of October 28, 2025, all code changes MUST be validated with automated testing before deployment. Manual back-and-forth testing is inefficient and doesn't scale.

### Testing Infrastructure (Located in `tests/`)

**Available Test Suites:**
1. **`smoke_test.py`** - Quick validation (30 seconds)
   - Run BEFORE every deployment
   - Tests 4 core scenarios
   - Exit code 1 = DO NOT DEPLOY

2. **`golden_test_set.py`** - 11 hand-crafted test cases
   - Educational, technical, executive, professional styles
   - Tables, lists, paragraphs, headings
   - Edge cases (very short, very long text)

3. **`quality_metrics.py`** - Objective scoring system
   - Structure (count, length, parallel formatting)
   - Relevance (keyword coverage, semantic similarity)
   - Style (tone, voice, terminology)
   - Readability (complexity, clarity)
   - Composite score 0-100

4. **`regression_benchmark.py`** - Version comparison
   - Track quality across versions
   - Detect regressions automatically
   - Compare before/after metrics

### Mandatory Testing Workflow

#### BEFORE Making Changes:
```bash
# 1. Establish baseline
python tests/regression_benchmark.py --version v87_baseline

# Stores results in tests/benchmark_results/v87_baseline.json
```

#### AFTER Making Changes:
```bash
# 2. Run smoke test (30 seconds)
python tests/smoke_test.py

# If passes, continue. If fails, fix issues first.

# 3. Run full benchmark
python tests/regression_benchmark.py --version v88_proposed

# 4. Compare versions
python tests/regression_benchmark.py --compare v87_baseline v88_proposed

# Example output:
# Overall Quality: 82.4 → 85.1 (+2.7) ✅
# Structure:       78.5 → 83.2 (+4.7) ✅
# Relevance:       84.2 → 86.8 (+2.6) ✅
#
# REGRESSIONS: 0 tests
# IMPROVEMENTS: 8/11 tests
```

#### DEPLOYMENT CRITERIA:
Deploy ONLY if:
- ✅ Smoke test passes (exit code 0)
- ✅ Overall quality ≥ baseline (no regression)
- ✅ Zero critical test failures
- ✅ Improvements outweigh any minor regressions

### What to Test When

| Change Type | Required Tests | Time Investment |
|-------------|----------------|-----------------|
| **Bug fix** | Smoke test only | 30 seconds |
| **Minor enhancement** | Smoke + affected category | 2 minutes |
| **Major feature** | Full benchmark + comparison | 5-10 minutes |
| **Refactoring** | Full benchmark (regression check) | 5-10 minutes |
| **Before production deploy** | Full benchmark | 5-10 minutes |

### Quality Thresholds (Enforced by Tests)

```python
QUALITY_THRESHOLDS = {
    "overall_quality": 70.0,    # Composite score
    "structure_score": 65.0,    # Bullet formatting
    "relevance_score": 70.0,    # Content relevance
    "style_score": 60.0,        # Style consistency
    "readability_score": 65.0   # Readability
}
```

**DO NOT DEPLOY** if any test falls below these thresholds.

### Adding New Test Cases

When adding new bullet generation features:

```python
# tests/golden_test_set.py

GOLDEN_TEST_SET.append({
    "id": "your_new_test",
    "category": "professional",  # or educational, technical, executive
    "input_text": """Your test input...""",
    "context_heading": "Test Heading",
    "expected_bullets": [
        "Expected bullet 1",
        "Expected bullet 2"
    ],
    "quality_criteria": {
        "min_bullets": 3,
        "avg_word_length": (8, 15),
        "must_contain_keywords": ["keyword1", "keyword2"]
    }
})
```

### Iterative Improvement Process

**Replace manual testing with:**

1. **Make change** → Edit code
2. **Quick validate** → Run smoke test (30s)
3. **Measure impact** → Run benchmark (5m)
4. **Compare** → Automated comparison report
5. **Decision** → Deploy if metrics improve

**Benefits over manual testing:**
- 📊 Quantitative scores (not subjective "looks good")
- 🚨 Automatic regression detection
- ⚡ 90% reduction in testing time
- 📈 Track improvement over time
- 🎯 Data-driven deployment decisions

### Example Usage Session

```bash
# Starting work on v88 - new feature
$ python tests/regression_benchmark.py --version v87_baseline
✅ Baseline established: 82.4/100

# Make code changes...
# ... editing file_to_slides.py ...

# Quick validation
$ python tests/smoke_test.py
✅ All smoke tests passed

# Full validation
$ python tests/regression_benchmark.py --version v88_new_feature
[11/11] Testing completed
✅ Average quality: 85.1/100

# Compare
$ python tests/regression_benchmark.py --compare v87_baseline v88_new_feature

COMPARING VERSIONS: v87_baseline vs v88_new_feature
============================================================
Overall Quality: 82.4 → 85.1 (+2.7) ✅

Breakdown:
  Structure Score:      78.5 → 83.2  ✅ +4.7
  Relevance Score:      84.2 → 86.8  ✅ +2.6
  Style Score:          81.1 → 83.5  ✅ +2.4
  Readability Score:    85.8 → 86.2  ✅ +0.4

✅ IMPROVEMENTS (8 tests):
  - edu_ml_basics: 79.2 → 85.3 (+6.1)
  - tech_microservices: 82.1 → 88.4 (+6.3)
  ...

DECISION: ✅ SAFE TO DEPLOY - Quality improved across all dimensions
```

### Integration with Deployment

**Pre-deployment checklist:**
```bash
# Run BEFORE every `git push heroku main`

# 1. Smoke test (mandatory)
python tests/smoke_test.py || exit 1

# 2. Visual inspection of benchmark results
python tests/regression_benchmark.py --version v88_release

# 3. Compare to last production version
python tests/regression_benchmark.py --compare v87_production v88_release

# 4. If all green, deploy
git push heroku main
```

### CI/CD Integration (Future)

```yaml
# .github/workflows/quality_check.yml (planned)

- name: Run smoke tests
  run: python tests/smoke_test.py

- name: Run regression benchmark
  run: |
    python tests/regression_benchmark.py --version ${GITHUB_SHA}

- name: Quality gate
  run: |
    # Fail if quality drops below threshold
    python tests/check_quality_gate.py --threshold 75
```

### Key Principle: Objective > Subjective

**OLD WAY (Manual Testing):**
- "The bullets look good to me" ❌
- "I think this is better" ❌
- "Seems fine" ❌

**NEW WAY (Automated Testing):**
- "Overall quality: 85.1/100" ✅
- "Relevance improved by 2.6 points" ✅
- "Zero regressions detected" ✅

---

## Key Architecture

### Main Components
- **file_to_slides.py**: Core Flask application handling document conversion logic (8000+ lines)
- **wsgi.py**: Production entry point for Gunicorn/Heroku deployment
- **templates/file_to_slides.html**: Main web interface with Google Drive integration

### Document Processing Flow
1. User selects Google Doc via Google Drive picker OR pastes a Google Docs URL
2. App fetches document content via Google Docs API (authenticated or public)
3. Document is parsed to extract headings (H1-H4) and table/paragraph content
4. Content is processed into bullet points using Claude API or NLTK fallback
5. Presentation is generated as PowerPoint (.pptx) download or Google Slides in Drive
6. User receives download link or Google Slides URL

### Slide Structure Mapping
- H1 → Title slide (presentation title)
- H2 → Section title slide
- H3 → Subsection title slide
- H4 → Individual slide titles
- Table rows OR paragraphs → Content slides with bullet points

## Development Commands

### Run locally
```bash
python wsgi.py
# App runs on http://localhost:5000
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Deploy to Heroku
```bash
git add .
git commit -m "Your commit message"
git push heroku main
```

## Key Dependencies

- **Flask**: Web framework (2.3.3)
- **python-pptx**: PowerPoint generation (0.6.21)
- **anthropic**: Claude API for AI-powered content generation (0.39.0)
- **google-api-python-client**: Google Docs/Slides/Drive API integration (2.108.0)
- **google-auth-oauthlib**: OAuth 2.0 for Google services (1.2.0)
- **nltk/textstat**: Lightweight text analysis for bullet generation fallback
- **gunicorn**: Production WSGI server (21.2.0)

## Important Technical Details

### Google Integration
**Authentication:**
- OAuth 2.0 flow for Google Docs, Drive, and Slides access
- Credentials stored in session (not persistent)
- Routes: `/auth/google` (initiate), `/oauth2callback` (callback), `/api/google-config` (token provider)

**Document Access:**
- Supports both authenticated (private docs) and public Google Docs
- Google Drive Picker API for browsing user's Drive files
- Extracts document ID from Google Docs URLs automatically

### Bullet Point Generation
The app has two modes for generating bullet points:
1. **With Claude API** (user-provided key): Uses Claude 3.5 Sonnet to create concise, complete sentence bullets
2. **Lightweight NLP**: Uses NLTK for basic semantic analysis when no API key provided

### Processing Constraints
- Maximum document size: 50MB
- Supported format: Google Docs only (no file uploads)
- Processing timeout: 10 minutes
- Content fetched and processed in memory (not stored permanently)

### Configuration
**Required Environment Variables (Heroku):**
- `GOOGLE_CREDENTIALS_JSON`: Google OAuth client configuration (JSON string)
- `SECRET_KEY`: Flask session secret

**Optional Environment Variables:**
- `ANTHROPIC_API_KEY`: Server-side Claude API key (if not requiring users to provide their own)
- `GOOGLE_API_KEY`: Google Picker API key (improves Drive picker performance)

**Production:**
- Runs via Gunicorn on Heroku (configured in Procfile)
- Python runtime: 3.11.5 (specified in runtime.txt)
- Session-based authentication state management

## Recent Changes (v54-v55)

### Google-Only Workflow (October 2025)
- **Removed**: File upload functionality (drag-and-drop, .docx files)
- **Added**: Google Drive file picker with OAuth integration
- **Added**: Direct Google Docs URL input (public or authenticated)
- **Added**: `/api/google-config` endpoint for access token management
- **Updated**: UI to unified document selection interface with bordered container
- **Simplified**: Backend upload route to only handle Google Docs URLs

### Output Options
- PowerPoint (.pptx) download
- Google Slides (created directly in user's Drive via API)

### API Key Strategy
**Current approach:** Users provide their own Claude API key
- Optional input field in UI
- Falls back to NLTK/textstat if not provided
- Zero API costs to application owner
- Users get full AI features with their own key

**Alternative approach:** Set `ANTHROPIC_API_KEY` on Heroku
- All users get AI features automatically
- Application owner pays for all API usage (~$0.01-0.05 per document)
- Can remove API key input field from UI

## Input/Output Styling Instructions

- **Your input:** Will appear in **green** in the terminal to distinguish from Claude's response.
- **Claude output:** Will appear in **default terminal color**.
- **Prefix:** Your inputs will be prefixed with `>>` automatically.

### Example:

```
>> My input text (green)
Claude: This is Claude's response (default color)
```