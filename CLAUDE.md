# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Flask web application that converts Google Docs into PowerPoint presentations or Google Slides with AI-generated bullet points and visual prompts. The app integrates with Google Drive for document selection, uses Google OAuth for authentication, and leverages Claude API for intelligent content processing.

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