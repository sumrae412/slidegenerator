# Codebase API Key Configuration Analysis

## Executive Summary

The slide generator application currently supports **Claude API (Anthropic)** for AI-powered bullet point generation, with a fallback to lightweight NLP (TF-IDF + TextRank + spaCy). The codebase is **partially prepared for OpenAI integration** - some app files already import OpenAI libraries and have placeholder OpenAI handlers, but the DocumentParser (core parsing engine) does not yet support OpenAI for bullet generation.

---

## 1. Files Handling API Key Configuration

### 1.1 Core Package Files (slide_generator_pkg/)

#### `/home/user/slidegenerator/slide_generator_pkg/document_parser.py` ⭐ **PRIMARY**
- **API Key Parameter**: `claude_api_key` (optional, line 92)
- **Environment Variable Used**: `ANTHROPIC_API_KEY` (line 101)
- **Initialization Flow**:
  ```python
  def __init__(self, claude_api_key=None):
      self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
      self.client = None
      if self.api_key:
          try:
              self.client = anthropic.Anthropic(api_key=self.api_key)
              logger.info("✅ Claude API client initialized")
          except Exception as e:
              logger.error(f"Failed to initialize Claude client: {e}")
              self.client = None
  ```
- **Size**: ~313KB (7123 lines) - Monolithic file
- **Key Methods**:
  - `_call_claude_with_retry()` - Makes API calls with exponential backoff (line 180)
  - `_generate_cache_key()` - Creates cache keys for responses (line 126)
  - `get_cache_stats()` - Returns cache performance metrics (line 166)

#### `/home/user/slidegenerator/slide_generator_pkg/powerpoint_generator.py`
- **API Key Support**: Optional OpenAI client passed to constructor
- **Initialization**: `def __init__(self, openai_client=None)` (line 40)
- **Key Methods**:
  - `create_powerpoint()` - Main slide generation (line 44)
  - Uses OpenAI client if available for image generation and diagrams
- **Current State**: Accepts OpenAI client but doesn't initialize it

#### `/home/user/slidegenerator/slide_generator_pkg/__init__.py`
- **Exports**: DocumentParser with documentation example
- **Example Usage**: Shows `DocumentParser(claude_api_key="your-api-key")` pattern

### 1.2 Application Entry Points (Flask/Streamlit/FastAPI)

#### `/home/user/slidegenerator/file_to_slides.py` ⭐ **MAIN WEB APP**
- **API Keys Handled**:
  - Claude: `claude_api_key` from form input (line 11229)
  - OpenAI: `os.getenv('OPENAI_API_KEY')` (line 8238)
- **Environment Variables**: 
  - `SECRET_KEY`, `FLASK_SECRET_KEY`
  - `GOOGLE_CREDENTIALS_JSON`
  - `OPENAI_API_KEY`
- **Initialization Pattern**:
  ```python
  claude_api_key = request.form.get('claude_key', '').strip()  # Form input
  parser = DocumentParser(claude_api_key=claude_api_key if claude_api_key else None)
  ```
- **OpenAI Integration**: Already integrated for diagram generation
  - Line 8245: `client = openai.OpenAI(api_key=api_key)`

#### `/home/user/slidegenerator/app_stable.py` (Streamlit)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")` (line 80)
- **Client Initialization**: `anthropic.Anthropic(api_key=api_key)` (line 84)
- **Pattern**: Simple environment variable → Anthropic client

#### `/home/user/slidegenerator/app_simple.py` (Streamlit)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")` (line 48)
- **Client Initialization**: `anthropic.Anthropic(api_key=api_key)` (line 52)

#### `/home/user/slidegenerator/app_robust.py` (Streamlit)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")`
- **Pattern**: Same as app_simple.py

#### `/home/user/slidegenerator/app_flask.py` (Flask)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")` (line 55)
- **Client Initialization**: `anthropic.Anthropic(api_key=api_key)` (line 57)
- **Multi-agent Support**: Has both ClaudeAgent and ChatGPTAgent classes

#### `/home/user/slidegenerator/app_fastapi.py` (FastAPI)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")` (line 111)
- **Client Initialization**: `anthropic.Anthropic(api_key=api_key)` (line 113)
- **WebSocket Support**: Real-time progress updates

#### `/home/user/slidegenerator/app_flask_modern.py` (Flask)
- **API Key Retrieval**: Same pattern as other apps
- **Focus**: Modern UI and async support

#### `/home/user/slidegenerator/app_enhanced.py` (Streamlit)
- **API Key Retrieval**: `os.getenv("ANTHROPIC_API_KEY")` (line 37)
- **OpenAI Import**: Yes (line 8)
- **Pattern**: Tries Claude first, has OpenAI fallback structure

#### `/home/user/slidegenerator/app.py` (Streamlit)
- **Purpose**: Dual-agent interface (Claude Code + ChatGPT)
- **API Keys**:
  - Claude Code: `claude_code_sdk` integration
  - OpenAI: `os.getenv("OPENAI_API_KEY")` (line 46)
- **Pattern**: Shows mature OpenAI integration with model selection

#### `/home/user/slidegenerator/direct_run.py` (Direct Python Script)
- **API Keys**:
  - Claude: `os.getenv("ANTHROPIC_API_KEY")` (line 23)
  - OpenAI: `os.getenv("OPENAI_API_KEY")` (line 36)
- **Initialization**:
  ```python
  claude_key = os.getenv("ANTHROPIC_API_KEY")
  self.claude = anthropic.Anthropic(api_key=claude_key)
  
  openai_key = os.getenv("OPENAI_API_KEY")
  self.openai = openai.OpenAI(api_key=openai_key)
  ```
- **Model Selection**: Tries multiple OpenAI models (gpt-4, gpt-3.5-turbo)

#### `/home/user/slidegenerator/docs_to_slides.py`
- **API Key Support**: No integrated API key handling (Google Docs focused)
- **Focus**: Google Docs authentication via OAuth

### 1.3 WSGI Entry Points

#### `/home/user/slidegenerator/wsgi.py`
- **Entry Point**: Imports from `file_to_slides`
- **Environment Variables**: Reads `PORT` for server binding

#### `/home/user/slidegenerator/wsgi_enhanced.py`
- **Entry Point**: Imports from `file_to_slides_enhanced`
- **Environment Variables**: Reads `PORT` for server binding

---

## 2. Environment Variable Patterns

### Currently Used Environment Variables

| Variable Name | Used In | Purpose | Required |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | All apps, DocumentParser | Claude API authentication | Optional* |
| `OPENAI_API_KEY` | file_to_slides.py, direct_run.py, app.py | OpenAI/ChatGPT authentication | Optional* |
| `GOOGLE_CREDENTIALS_JSON` | file_to_slides.py | Google OAuth credentials (JSON) | Optional |
| `GOOGLE_CLIENT_SECRETS_FILE` | file_to_slides.py | Path to Google credentials file | Optional |
| `GOOGLE_API_KEY` | file_to_slides.py (line 11211) | Google API key | Optional |
| `FLASK_SECRET_KEY` | file_to_slides.py, wsgi.py | Flask session secret | Optional |
| `SECRET_KEY` | file_to_slides.py | Flask session secret | Optional |
| `GOOGLE_REDIRECT_URI` | file_to_slides.py | OAuth redirect URL | Optional |
| `FLASK_ENV` | Multiple apps | Environment mode (development/production) | Optional |
| `PORT` | wsgi.py | Server port (default: 5000) | Optional |

*Apps work without these, but full AI features require them.

### How Environment Variables Are Loaded

**Using `load_dotenv()`** (from `python-dotenv` package):
- `/home/user/slidegenerator/app.py` (line 5)
- `/home/user/slidegenerator/app_stable.py` (line 19)
- `/home/user/slidegenerator/app_simple.py` (line 15)
- `/home/user/slidegenerator/app_robust.py`
- `/home/user/slidegenerator/app_flask.py` (line 10)
- `/home/user/slidegenerator/app_fastapi.py` (line 20)
- `/home/user/slidegenerator/app_flask_modern.py` (line 10)
- `/home/user/slidegenerator/app_enhanced.py` (line 7)
- `/home/user/slidegenerator/direct_run.py` (line 8)

**Using `os.getenv()` or `os.environ.get()`** (no .env file needed):
- `os.getenv('ANTHROPIC_API_KEY')` - Direct from environment
- `os.environ.get('GOOGLE_CREDENTIALS_JSON')` - Explicit environ dictionary access

---

## 3. Complete API Key Flow Diagram

### Flow 1: DocumentParser Initialization (Core Module)

```
┌─────────────────────────────────────┐
│  Application Startup                 │
│  (e.g., file_to_slides.py)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  load_dotenv() loads .env file      │
│  (if present) into os.environ       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Form/Request Handler gets API key  │
│  (from user input or ENV)           │
│                                      │
│  api_key = request.form.get(...)    │
│  OR                                  │
│  api_key = os.getenv('...')         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  DocumentParser(claude_api_key=...) │
│  Constructor (line 92)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Inside DocumentParser.__init__:    │
│                                      │
│  self.api_key = claude_api_key or   │
│    os.getenv('ANTHROPIC_API_KEY')   │
└──────────────┬──────────────────────┘
               │
               ▼
         ┌─────┴──────┐
         │             │
         ▼             ▼
    ┌─────────┐   ┌──────────┐
    │ api_key │   │ NO API   │
    │ Found?  │   │ FALLBACK │
    └────┬────┘   │ to NLP   │
         │        └──────────┘
         ▼
┌─────────────────────────────────────┐
│  self.client =                      │
│  anthropic.Anthropic(api_key=...)   │
│  (line 105)                         │
└─────────────────────────────────────┘
```

### Flow 2: Web Application (file_to_slides.py) → DocumentParser

```
User Upload Document with API Key
         │
         ▼
┌──────────────────────────────────┐
│ POST /upload endpoint             │
│ Extracts: claude_key from form   │
│ OR uses ANTHROPIC_API_KEY env    │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ DocumentParser(                  │
│   claude_api_key=user_provided   │
│ )                                │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ parser.parse_file(...)           │
│ Uses Claude API or NLP fallback  │
└──────────────────────────────────┘
```

---

## 4. Current State: OpenAI Integration Status

### Where OpenAI is Already Used

1. **file_to_slides.py** (Line 8238):
   ```python
   api_key = os.getenv('OPENAI_API_KEY')
   client = openai.OpenAI(api_key=api_key)
   ```
   - **Purpose**: DALL-E image generation for slides
   - **Context**: Optional feature when SlideGenerator processes visual prompts

2. **powerpoint_generator.py** (Line 40):
   ```python
   def __init__(self, openai_client=None):
       self.client = openai_client
   ```
   - **Purpose**: Receives initialized OpenAI client for image/diagram generation
   - **How Called**: `SlideGenerator(openai_client=parser.client)`

3. **direct_run.py** (Line 36-50):
   ```python
   openai_key = os.getenv("OPENAI_API_KEY")
   self.openai = openai.OpenAI(api_key=openai_key)
   ```
   - **Purpose**: Code review and refinement using ChatGPT
   - **Models Supported**: gpt-4, gpt-3.5-turbo

4. **Multiple Streamlit apps** (app.py, app_enhanced.py):
   - Import OpenAI library
   - Dual-agent patterns (Claude + ChatGPT)

### Where OpenAI is NOT Yet Integrated

**DocumentParser** (Core Module) - Line 92+
- **Current**: Only supports Claude API for bullet generation
- **Gap**: No OpenAI support for bullet point generation
- **Fallback**: Uses NLP-based extraction (TF-IDF + spaCy)

---

## 5. Where to Add OpenAI API Key Support

### 5.1 DocumentParser Enhancement (Primary Implementation)

**File**: `/home/user/slidegenerator/slide_generator_pkg/document_parser.py`

**Changes Required**:

1. **Constructor (Line 92)**:
   ```python
   def __init__(self, claude_api_key=None, openai_api_key=None):
       # Existing Claude initialization
       self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
       self.client = None
       
       # NEW: OpenAI initialization
       self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
       self.openai_client = None
       if self.openai_api_key:
           try:
               import openai
               self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
               logger.info("✅ OpenAI client initialized")
           except Exception as e:
               logger.error(f"Failed to initialize OpenAI: {e}")
   ```

2. **Add OpenAI API Call Method** (Similar to `_call_claude_with_retry`):
   ```python
   def _call_openai_with_retry(self, **api_params) -> Any:
       """Call OpenAI API with exponential backoff retry logic"""
       # Similar pattern to Claude retry (lines 180-210)
   ```

3. **Update Bullet Generation Logic** (Main parsing method):
   - Add parameter to enable OpenAI choice: `use_openai=False`
   - Try Claude first if available
   - Fall back to OpenAI if Claude unavailable
   - Fall back to NLP if neither available

### 5.2 Application Entry Points

**Files to Update**:

1. **file_to_slides.py** (Line 11229):
   ```python
   openai_key = request.form.get('openai_key', '').strip()  # NEW
   claude_key = request.form.get('claude_key', '').strip()  # Existing
   
   parser = DocumentParser(
       claude_api_key=claude_key or None,
       openai_api_key=openai_key or None  # NEW
   )
   ```

2. **All Streamlit apps** (app_simple.py, app_stable.py, etc.):
   ```python
   # When initializing DocumentParser:
   openai_key = os.getenv("OPENAI_API_KEY")
   claude_key = os.getenv("ANTHROPIC_API_KEY")
   
   parser = DocumentParser(
       claude_api_key=claude_key,
       openai_api_key=openai_key  # NEW
   )
   ```

3. **example_usage.py**:
   - Add new example showing OpenAI usage

### 5.3 slide_generator_pkg/__init__.py

**Update Documentation Example**:
```python
# QuickStart updated:
parser = DocumentParser(
    claude_api_key="your-claude-key",    # Optional
    openai_api_key="your-openai-key"     # Optional
)
```

---

## 6. Configuration File Patterns

### .env File Pattern

```bash
# Claude/Anthropic
ANTHROPIC_API_KEY=sk-ant-v7-xxxxxxxxxxxxxxx

# OpenAI/ChatGPT
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx

# Google APIs
GOOGLE_CREDENTIALS_JSON='{"web": {"client_id": "...", ...}}'
GOOGLE_CLIENT_SECRETS_FILE=credentials.json
GOOGLE_API_KEY=xxxxxxxx
GOOGLE_REDIRECT_URI=http://localhost:5000/callback

# Flask
FLASK_SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Server
PORT=5000
```

### Environment Configuration Hierarchy

1. **Command Line Arguments** (when used)
2. **.env File** (loaded by `load_dotenv()`)
3. **System Environment Variables** (set via export/setenv)
4. **Fallback/Defaults** (hardcoded in app)

---

## 7. Summary: API Key Configuration Locations

### Configuration Sources (By Priority)

| Priority | Source | Files Affected | Example |
|----------|--------|---|---|
| 1 | Constructor Parameter | DocumentParser, SlideGenerator | `DocumentParser(claude_api_key="...")` |
| 2 | Form Input (Web Apps) | file_to_slides.py | `request.form.get('claude_key')` |
| 3 | Environment Variables | All apps | `os.getenv('ANTHROPIC_API_KEY')` |
| 4 | .env File | Via load_dotenv() | Lines in `.env` file |
| 5 | Fallback (NLP) | DocumentParser | Uses TF-IDF if no API key |

### Key Implementation Points

| Component | Claude Support | OpenAI Support | Status |
|-----------|---|---|---|
| **DocumentParser** | ✅ Full | ❌ Partial (NLP fallback only) | **NEEDS UPDATE** |
| **PowerPoint Generator** | ❌ N/A | ✅ For visuals | Complete |
| **file_to_slides.py** | ✅ Form + Env | ✅ Env | Integrated |
| **Direct Python Script** | ✅ Env | ✅ Env | Integrated |
| **Streamlit Apps** | ✅ Env | ❌ Not in core | Needs update |
| **FastAPI App** | ✅ Env | ❌ Not in core | Needs update |

---

## 8. Testing Points for API Key Integration

After implementing OpenAI support, test:

1. **Initialization**:
   - DocumentParser with only Claude API key
   - DocumentParser with only OpenAI API key
   - DocumentParser with both keys
   - DocumentParser with no keys (should use NLP fallback)

2. **Priority/Fallback**:
   - When both Claude and OpenAI are available, which is called first?
   - When Claude fails, does it fall back to OpenAI?
   - When both fail, does it use NLP fallback?

3. **Cache Compatibility**:
   - Are cached responses keyed by model type?
   - Do Claude responses get cached separately from OpenAI?

4. **Error Handling**:
   - Invalid OpenAI API key → graceful fallback
   - OpenAI rate limiting → retry logic
   - Network failures → proper error messages

---

## Summary Table: Files Requiring Changes for OpenAI Integration

| File | Changes Needed | Priority | Impact |
|------|---|---|---|
| `slide_generator_pkg/document_parser.py` | Add openai_api_key parameter, implement OpenAI bullet generation | CRITICAL | Core functionality |
| `slide_generator_pkg/__init__.py` | Update docs/examples | Medium | Documentation |
| `file_to_slides.py` | Add openai_key form input | High | Main web app |
| `app_simple.py` | Pass OPENAI_API_KEY to DocumentParser | High | CLI app |
| `app_stable.py` | Pass OPENAI_API_KEY to DocumentParser | High | Streamlit app |
| `app_robust.py` | Pass OPENAI_API_KEY to DocumentParser | High | Streamlit app |
| `app_flask.py` | Pass OPENAI_API_KEY to DocumentParser | High | Flask app |
| `app_fastapi.py` | Pass OPENAI_API_KEY to DocumentParser | High | FastAPI app |
| `app_flask_modern.py` | Pass OPENAI_API_KEY to DocumentParser | High | Flask modern |
| `example_usage.py` | Add OpenAI example | Low | Documentation |
| `DEPLOYMENT.md` | Document OPENAI_API_KEY config | Medium | Documentation |

