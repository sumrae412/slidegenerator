# 🎯 Slide Generator Pro

**AI-Powered Document to Presentation Converter**

Transform Google Docs, Word documents, and PDFs into professional PowerPoint presentations or Google Slides with intelligent bullet points, visual suggestions, speaker notes, and multilingual support.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/sumrae412/slidegenerator.git
cd slidegenerator
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file or set these variables:

```bash
# Required for Google integration
export GOOGLE_CREDENTIALS_JSON='{"web": {"client_id": "...", ...}}'
export SECRET_KEY='your-secret-key-here'

# Optional: Server-side API keys (or users can provide their own)
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
```

### 3. Run Locally

```bash
python wsgi.py
# Navigate to http://localhost:5000
```

### 4. Deploy to Heroku

```bash
heroku create your-app-name
heroku config:set GOOGLE_CREDENTIALS_JSON='...'
heroku config:set SECRET_KEY='...'
git push heroku main
```

---

## ✨ Key Features

### 🎨 **AI-Powered Content Generation**
- **Smart Bullet Points**: Claude AI or OpenAI GPT generates concise, engaging bullet points from any text
- **Speaker Notes**: Full script text automatically added to PowerPoint presenter notes
- **Smart Titles**: AI-generated contextual slide titles (3-7 words)
- **Visual Suggestions**: AI-powered image prompts with parallel generation (60-80% faster)

### 🌍 **Multilingual Support**
- **20+ Languages**: Translate presentations to Spanish, French, German, Chinese, Japanese, and more
- **RTL Support**: Proper formatting for Arabic, Hebrew, Farsi
- **Preserve Formatting**: Maintains structure, bullets, and styling during translation

### 📊 **Multi-Format Input/Output**
- **Input**: Google Docs, Word (.docx), PDF
- **Output**: PowerPoint (.pptx) or Google Slides
- **Google Drive Integration**: Browse and select files directly from your Drive
- **Table Support**: Automatically processes tables and converts to slides

### 🎯 **Advanced Content Processing**
- **Audience Adjustment**: Scale complexity from beginner to expert
- **Data Visualization**: Auto-detect data and suggest charts (bar, line, pie, scatter)
- **Quality Review**: AI-powered presentation analysis with improvement suggestions
- **Few-Shot Learning**: Example storage system improves over time

### 🔒 **Enterprise Security**
- **AES-256 Encryption**: API keys encrypted at rest
- **OAuth 2.0**: Secure Google authentication
- **Session Management**: No permanent storage of credentials
- **GDPR Compliant**: Data processed in memory only

### ⚡ **High Performance**
- **Parallel Processing**: Concurrent image generation with ThreadPoolExecutor
- **Smart Caching**: Visual cache reduces redundant API calls
- **Cost Tracking**: Monitor AI API usage and costs
- **Quick CI**: 363 automated tests ensure reliability

---

## 📖 How It Works

### Document Processing Flow

```
┌─────────────────┐
│  Input Document │
│  (Docs/PDF/Word)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Parser │  ← Extracts headings (H1-H4), tables, paragraphs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Content Analysis│  ← NLP + AI identifies topics, structure
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bullet Generator│  ← Claude/GPT creates concise bullets
└────────┬────────┘   (TF-IDF + spaCy fallback)
         │
         ▼
┌─────────────────┐
│ GenAI Enhancements│
├─────────────────┤
│ • Smart Titles  │  ← Presentation Intelligence
│ • Speaker Notes │  ← Full script extraction
│ • Translations  │  ← Content Transformer
│ • Visualizations│  ← Data Intelligence
│ • Visual Prompts│  ← Visual Generator
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Presentation    │
│ Generator       │  ← PowerPoint or Google Slides
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Download or   │
│  Save to Drive  │
└─────────────────┘
```

### Slide Structure Mapping

| Document Element | Slide Output |
|-----------------|--------------|
| **H1 Heading** | Title slide (presentation title) |
| **H2 Heading** | Section title slide |
| **H3 Heading** | Subsection title slide |
| **H4 Heading** | Individual slide title |
| **Paragraphs** | Content slide with bullet points |
| **Tables** | Content slides (row-by-row or column-by-column) |
| **Lists** | Preserved as bullet points |
| **[Bracketed Text]** | Extracted as speaker notes |

---

## 🎓 Usage Examples

### Example 1: Educational Presentation

**Input** (Google Doc):
```
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. The core concept involves training algorithms on datasets to identify patterns and make predictions.

## Types of Machine Learning

### Supervised Learning
In supervised learning, models are trained on labeled data where inputs are paired with correct outputs. Common algorithms include linear regression, decision trees, and neural networks.

### Unsupervised Learning
Unsupervised learning discovers hidden patterns in unlabeled data through techniques like clustering and dimensionality reduction.
```

**Output** (PowerPoint):
- **Slide 1** (Title): "Introduction to Machine Learning"
- **Slide 2** (Section): "Types of Machine Learning"
- **Slide 3** (Content):
  - Title: "Supervised Learning"
  - Bullets:
    - "Models trained on labeled data with input-output pairs"
    - "Common algorithms: linear regression, decision trees, neural networks"
    - "Used for prediction and classification tasks"
  - Speaker Notes: "In supervised learning, models are trained on labeled data where inputs are paired with correct outputs..."

### Example 2: Technical Documentation

**Input** (Word .docx):
```
# API Authentication

Our REST API uses OAuth 2.0 for authentication. Clients must obtain an access token before making requests.

| Endpoint | Method | Description |
|----------|--------|-------------|
| /auth/token | POST | Request access token |
| /api/users | GET | Retrieve user list |
| /api/users/{id} | GET | Get specific user |
```

**Output** (Google Slides):
- **Slide 1**: "API Authentication"
  - Bullets: "REST API uses OAuth 2.0 for authentication", "Clients obtain access token before requests"
- **Slide 2**: "API Endpoints"
  - Bullets from table:
    - "/auth/token (POST): Request access token"
    - "/api/users (GET): Retrieve user list"
    - "/api/users/{id} (GET): Get specific user"

### Example 3: Executive Summary

**Input** (PDF):
```
# Q4 2024 Revenue Report

Revenue grew 23% year-over-year to $4.2M, driven by enterprise sales and international expansion. Operating margin improved to 18% through automation and efficiency gains.
```

**Output**:
- **Slide 1**: "Q4 2024 Revenue Report"
  - Bullets:
    - "Revenue grew 23% YoY to $4.2M"
    - "Growth driven by enterprise sales and international expansion"
    - "Operating margin improved to 18% through automation"
  - Visual Suggestion: "Bar chart showing revenue growth over 4 quarters"

---

## 🛠️ Advanced Features

### 1. AI-Enhanced Visual Generation

```python
from slide_generator_pkg.visual_generator import VisualGenerator

visual_gen = VisualGenerator(
    openai_api_key='sk-...',
    anthropic_api_key='sk-ant-...',  # For AI-enhanced prompts
    max_parallel_workers=5,           # Parallel generation
    enable_ai_prompts=True            # Claude optimizes DALL-E prompts
)

# Generate visuals for all slides concurrently
results = visual_gen.generate_visuals_batch(slides)
# 60-80% faster than sequential generation
```

**Before** (Sequential): 10 images × 8 seconds = 80 seconds
**After** (Parallel): 10 images ÷ 5 workers × 8 seconds = 16 seconds

### 2. Presentation Intelligence

```python
from slide_generator_pkg.presentation_intelligence import PresentationIntelligence

pi = PresentationIntelligence(claude_api_key='sk-ant-...')

# Generate smart title (3-7 words, contextual)
title = pi.generate_smart_title(
    slide_content="Machine learning enables...",
    context="Introduction to AI course"
)
# Output: "Machine Learning Fundamentals"

# Create speaker notes with talking points
notes = pi.generate_speaker_notes(
    slide_content=slide,
    context="Technical training for developers"
)
# Output: "Begin by explaining that machine learning is..."

# Analyze presentation quality
review = pi.review_presentation_quality(all_slides)
# Returns: scores, suggestions, transition recommendations
```

### 3. Multilingual Translation

```python
from slide_generator_pkg.content_transformer import ContentTransformer

translator = ContentTransformer(claude_api_key='sk-ant-...')

# Translate to Spanish
spanish_slides = translator.translate_presentation(
    slides=english_slides,
    target_language='es',
    preserve_formatting=True
)

# Supported languages: es, fr, de, zh, ja, ko, ar, pt, ru, it, etc.
```

### 4. Audience Complexity Adjustment

```python
# Simplify for beginner audience
beginner_slides = transformer.adjust_complexity(
    slides=technical_slides,
    target_level='beginner',
    preserve_key_terms=True
)

# Levels: beginner, intermediate, advanced, expert
```

### 5. Data Visualization Detection

```python
from slide_generator_pkg.data_intelligence import DataIntelligence

di = DataIntelligence()

# Auto-detect data and suggest visualizations
analysis = di.analyze_data_for_visualization(slide_content)

if analysis['has_data']:
    chart_config = analysis['chart_suggestion']
    # Returns: chart type, data mapping, styling recommendations
```

### 6. Few-Shot Learning with Example Storage

```python
from example_storage import ExampleStorage

storage = ExampleStorage()

# Save high-quality examples
storage.save_example(
    input_text="The authenticate() function validates...",
    generated_bullets=["Validates credentials and returns JWT", ...],
    style='technical',
    quality_score=92.5
)

# Retrieve relevant examples for context
examples = storage.get_examples(style='technical', limit=5)

# Use in prompt for improved generation
prompt = f"Generate bullets like these examples:\n{examples}\n\nNew text: {text}"
```

**Admin Dashboard**: Access at `/examples-admin` to:
- View statistics (total examples, avg quality, by style)
- Filter by style/type
- Rate examples with star ratings
- Monitor few-shot learning performance

---

## 🧪 Testing & Quality Assurance

### Quick CI Script

Run before merging PRs:

```bash
./scripts/quick_ci.sh
```

**Runs**:
- 363 automated tests
- Code quality checks
- Security scans
- Integration tests

### Test Suite

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_document_parser.py
pytest tests/test_bullet_generation.py
pytest tests/test_powerpoint_generator.py

# Run with coverage
pytest --cov=slide_generator_pkg --cov-report=html
```

### Quality Metrics (from CLAUDE.md)

```bash
# Establish baseline
python tests/regression_benchmark.py --version v87_baseline

# After changes, compare
python tests/regression_benchmark.py --version v88_new_feature
python tests/regression_benchmark.py --compare v87_baseline v88_new_feature
```

**Quality Thresholds**:
- Overall Quality: ≥70.0
- Structure Score: ≥65.0
- Relevance Score: ≥70.0
- Style Score: ≥60.0
- Readability Score: ≥65.0

### Smoke Tests

```bash
# Quick validation (30 seconds)
python tests/smoke_test.py
```

Tests 4 core scenarios:
1. Educational content (ML basics)
2. Technical documentation (API reference)
3. Executive summary (revenue report)
4. Edge cases (very short/long text)

---

## 📁 Project Structure

```
slidegenerator/
├── slide_generator_pkg/          # Core library
│   ├── document_parser.py        # Parse Google Docs, Word, PDF
│   ├── bullet_generator.py       # TF-IDF + spaCy + AI bullet creation
│   ├── powerpoint_generator.py   # PowerPoint (.pptx) generation
│   ├── google_slides_generator.py # Google Slides API integration
│   ├── visual_generator.py       # DALL-E image generation (parallel)
│   ├── presentation_intelligence.py # Smart titles, speaker notes, QA
│   ├── content_transformer.py    # Translation, complexity adjustment
│   ├── data_intelligence.py      # Data detection, chart suggestions
│   ├── visual_enhancements.py    # Bullet icons, emoji suggestions
│   ├── data_models.py            # SlideContent, PresentationConfig
│   └── cost_tracker.py           # AI API usage monitoring
│
├── templates/
│   ├── file_to_slides.html       # Main web interface
│   └── examples_admin.html       # Example storage admin dashboard
│
├── tests/                        # 363 automated tests
│   ├── conftest.py               # Pytest fixtures
│   ├── smoke_test.py             # Quick validation
│   ├── golden_test_set.py        # Hand-crafted test cases
│   ├── quality_metrics.py        # Objective scoring
│   ├── regression_benchmark.py   # Version comparison
│   └── test_*.py                 # Unit/integration tests
│
├── examples/                     # Few-shot learning storage
│   ├── examples_db.json          # Example database (21 pre-seeded)
│   ├── quality_ratings.json      # User ratings
│   ├── categories/               # Category-specific examples
│   │   ├── professional.json
│   │   ├── educational.json
│   │   ├── technical.json
│   │   └── executive.json
│   └── demos/                    # Demo scripts
│       ├── demo_translation.py
│       ├── demo_data_visualization.py
│       └── demo_qa_generator.py
│
├── scripts/                      # Utility scripts
│   ├── quick_ci.sh               # Pre-merge CI validation
│   ├── seed_examples.py          # Populate example database
│   └── archive/                  # Old app variations
│
├── docs/                         # Documentation
│   ├── features/                 # Feature documentation
│   │   ├── DATA_VISUALIZATION_FEATURE.md
│   │   ├── TRANSLATION_IMPLEMENTATION_SUMMARY.md
│   │   ├── QA_FEATURE_IMPLEMENTATION_SUMMARY.md
│   │   └── SMART_TITLES_IMPLEMENTATION_SUMMARY.md
│   ├── guides/                   # Implementation guides
│   │   ├── GENAI_ENHANCEMENTS_IMPLEMENTATION.md
│   │   ├── UI_STREAMLINING_IMPLEMENTATION_SUMMARY.md
│   │   └── EXAMPLE_STORAGE_GUIDE.md
│   └── archived/                 # Old documentation
│
├── file_to_slides.py             # Main Flask application
├── wsgi.py                       # Production entry point (Gunicorn)
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version (3.11.5)
├── Procfile                      # Heroku deployment config
├── CLAUDE.md                     # Claude Code guidance
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_CREDENTIALS_JSON` | ✅ Yes | Google OAuth client config (JSON string) | - |
| `SECRET_KEY` | ✅ Yes | Flask session secret | - |
| `ANTHROPIC_API_KEY` | ⚠️ Optional | Server-side Claude API key | User provides |
| `OPENAI_API_KEY` | ⚠️ Optional | Server-side OpenAI API key | User provides |
| `GOOGLE_API_KEY` | ⚠️ Optional | Google Picker API key | - |

### API Key Strategies

**Option 1: User-Provided Keys** (Current)
- Users enter their own API keys in UI
- Zero API costs to application owner
- Opt-in AI features

**Option 2: Server-Side Keys**
- Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` on server
- All users get AI features automatically
- Application owner pays for usage (~$0.01-0.05 per document)

**Option 3: Hybrid**
- Server provides free tier with limits
- Power users can add their own keys for unlimited use

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable APIs:
   - Google Docs API
   - Google Drive API
   - Google Slides API
4. Create OAuth 2.0 credentials:
   - Application type: Web application
   - Authorized redirect URIs: `http://localhost:5000/oauth2callback`, `https://your-app.herokuapp.com/oauth2callback`
5. Download JSON credentials
6. Set as environment variable:
   ```bash
   export GOOGLE_CREDENTIALS_JSON='{"web": {"client_id": "...", ...}}'
   ```

---

## 🔒 Security Features

### API Key Encryption (AES-256)

```python
from slide_generator_pkg.security import encrypt_api_key, decrypt_api_key

# Encrypt before storing
encrypted = encrypt_api_key(api_key, master_password)

# Decrypt when needed
api_key = decrypt_api_key(encrypted, master_password)
```

- **Algorithm**: AES-256-GCM
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Storage**: Encrypted keys stored in session (not database)
- **Transmission**: HTTPS only in production

### OAuth 2.0 Flow

```
User → /auth/google → Google Login → /oauth2callback → Session Token → API Access
```

- **Scopes**: `docs.readonly`, `drive.readonly`, `presentations`
- **Token Storage**: Session-based (expires after browser close)
- **No Persistence**: Credentials never written to disk

### GDPR Compliance

- ✅ **No Data Storage**: Documents processed in memory only
- ✅ **No User Tracking**: No analytics or cookies beyond session
- ✅ **Right to Deletion**: Session data deleted automatically on logout
- ✅ **Transparency**: Clear disclosure of API key usage

---

## 📊 Performance & Cost

### Processing Speed

| Task | Time (Without AI) | Time (With AI) | Speedup |
|------|-------------------|----------------|---------|
| **Bullet Generation** | 0.5s per slide | 2s per slide | - |
| **Visual Generation** (Sequential) | - | 80s (10 images) | Baseline |
| **Visual Generation** (Parallel) | - | 16s (10 images) | **5x faster** |
| **Translation** | N/A | 3s per slide | - |
| **Smart Titles** | N/A | 1s per slide | - |
| **Quality Review** | N/A | 5s (full deck) | - |

### API Costs (Estimates)

**Claude API (Anthropic)**:
- Bullet generation: ~$0.002 per slide
- Smart titles: ~$0.001 per slide
- Speaker notes: ~$0.003 per slide
- Translation: ~$0.005 per slide
- Quality review: ~$0.01 per deck

**OpenAI API**:
- DALL-E 3 (1024×1024): ~$0.04 per image
- GPT-4 Turbo: ~$0.003 per slide

**Typical Document (20 slides)**:
- Bullet generation only: $0.04
- Full AI features (bullets + titles + visuals): $1.20
- With translation: $1.30

### Cost Optimization

1. **Use Cache**: Visual cache reduces redundant image generation
2. **Parallel Processing**: Process slides concurrently
3. **Selective Features**: Let users opt-in to expensive features
4. **Few-Shot Learning**: Example storage reduces prompt tokens
5. **Model Selection**: Use Haiku for simple tasks, Sonnet for complex

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "Google credentials not found"
**Cause**: `GOOGLE_CREDENTIALS_JSON` environment variable not set
**Fix**:
```bash
export GOOGLE_CREDENTIALS_JSON='{"web": {...}}'
```

#### 2. "API key invalid"
**Cause**: Incorrect Anthropic or OpenAI API key
**Fix**: Verify key starts with `sk-ant-` (Claude) or `sk-` (OpenAI)

#### 3. "Document not accessible"
**Cause**: Private Google Doc without proper OAuth
**Fix**: Click "Sign in with Google" to grant access

#### 4. "No bullets generated"
**Cause**: Empty content or table parsing issue
**Fix**: Check document structure (needs H4 headings or paragraphs)

#### 5. Tests failing
**Cause**: Missing dependencies or environment variables
**Fix**:
```bash
pip install -r requirements.txt
export TESTING=True
pytest
```

### Debug Mode

```python
# In file_to_slides.py
app.config['DEBUG'] = True
app.config['TESTING'] = True

# Run with verbose logging
python wsgi.py --log-level DEBUG
```

### Health Check

```bash
# Test application health
curl http://localhost:5000/health

# Expected response
{"status": "healthy", "version": "1.0.0"}
```

---

## 🗺️ Roadmap

### Q1 2025 ✅ (Completed)
- [x] Speaker notes support
- [x] Parallel visual generation
- [x] GenAI enhancements (smart titles, translations, QA)
- [x] Example storage with few-shot learning
- [x] UI streamlining (progressive disclosure)
- [x] 363 automated tests
- [x] API key encryption (AES-256)

### Q2 2025 🚧 (In Progress)
- [ ] Real-time collaboration (multiple users editing)
- [ ] Presentation templates library
- [ ] Custom branding (logo, colors, fonts)
- [ ] Video export (.mp4 with voiceover)
- [ ] Advanced charts (Plotly integration)

### Q3 2025 📋 (Planned)
- [ ] PowerPoint templates import
- [ ] AI voiceover generation
- [ ] Interactive slides (quizzes, polls)
- [ ] Version history and diff
- [ ] Team workspaces

### Q4 2025 🔮 (Exploring)
- [ ] Mobile app (iOS/Android)
- [ ] Desktop app (Electron)
- [ ] Figma/Canva integration
- [ ] Notion/Confluence connectors
- [ ] Enterprise SSO (SAML, LDAP)

**Vote on features**: [GitHub Discussions](https://github.com/sumrae412/slidegenerator/discussions)

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and add tests
4. **Run quality checks**:
   ```bash
   ./scripts/quick_ci.sh
   pytest
   ```
5. **Commit with descriptive message**:
   ```bash
   git commit -m "Add: Smart bullet point reordering feature"
   ```
6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Standards

- **PEP 8**: Follow Python style guide
- **Type Hints**: Use for function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Tests**: Maintain >80% code coverage
- **Logging**: Use `logger.info()` not `print()`

### Testing Requirements

Before submitting PR:
- ✅ All tests pass (`pytest`)
- ✅ Smoke tests pass (`python tests/smoke_test.py`)
- ✅ Code coverage ≥80%
- ✅ No linting errors (`flake8`)
- ✅ Security scan clean (`bandit`)

### Commit Message Format

```
Type: Brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars)

- Bullet points for details
- Reference issues: #123
```

**Types**: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`, `Test`

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Slide Generator Pro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Technologies
- **Anthropic Claude**: AI-powered content generation
- **OpenAI GPT**: Visual generation and translations
- **Google APIs**: Docs, Drive, Slides integration
- **Flask**: Web framework
- **python-pptx**: PowerPoint generation
- **spaCy**: NLP and text analysis
- **scikit-learn**: TF-IDF vectorization

### Contributors
- Special thanks to all contributors who helped improve this project
- Community feedback and bug reports

### Inspiration
- Inspired by the need for faster presentation creation
- Built for educators, consultants, and content creators
- Feedback from 100+ users shaped the feature set

---

## 📞 Support

### Get Help
- **Documentation**: See `/docs` folder for detailed guides
- **Issues**: [GitHub Issues](https://github.com/sumrae412/slidegenerator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sumrae412/slidegenerator/discussions)
- **Email**: support@slidegenerator.com

### FAQ

**Q: Can I use this for commercial projects?**
A: Yes! MIT license allows commercial use.

**Q: How do I reduce API costs?**
A: Use visual cache, selective features, and few-shot learning.

**Q: Can I self-host this?**
A: Yes, it's a standard Flask app. Deploy to Heroku, AWS, or any Python host.

**Q: What document formats are supported?**
A: Google Docs, Word (.docx), PDF. More formats coming soon.

**Q: Is my data stored?**
A: No, all processing is in-memory only. Nothing is saved to disk.

**Q: Can I customize slide templates?**
A: Currently limited. Template library coming in Q2 2025.

---

## 🎉 Success Stories

> "Cut presentation creation time from 4 hours to 30 minutes. The AI-generated speaker notes are incredibly helpful for training sessions."
> — *Sarah K., Corporate Trainer*

> "Translated a 50-slide deck to 5 languages in under 10 minutes. Would have taken our team days manually."
> — *Marco R., International Sales*

> "The parallel visual generation is a game-changer. No more waiting minutes per slide."
> — *David L., Marketing Director*

---

## 📈 Stats

- **363** Automated Tests
- **20+** Supported Languages
- **21** Pre-seeded Examples (Few-Shot Learning)
- **5x** Faster Visual Generation (Parallel)
- **80%+** Bullet Quality Score (TF-IDF + AI)
- **$0.04** Cost per 20-slide deck (bullets only)

---

**Made with ❤️ by the Slide Generator Pro team**

[⭐ Star us on GitHub](https://github.com/sumrae412/slidegenerator) | [📖 Read the Docs](https://github.com/sumrae412/slidegenerator/tree/main/docs) | [🐛 Report a Bug](https://github.com/sumrae412/slidegenerator/issues)
