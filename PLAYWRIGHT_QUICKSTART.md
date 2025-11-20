# 🎭 Playwright Testing - Quick Start

**5-Minute Setup Guide** for browser-based E2E tests

---

## ✅ Prerequisites

Since you've already installed the Playwright browsers, you just need to ensure the Python packages are installed.

---

## 🚀 Quick Setup

### Step 1: Install Python Dependencies

```bash
cd /Users/summerrae/claude_code/slidegenerator
pip install playwright pytest-playwright
```

### Step 2: Verify Installation

```bash
# Check Playwright is installed
playwright --version
# Should show: Version 1.55.0 or similar

# Check browsers are installed
playwright install --dry-run
# Should show chromium, firefox, webkit already installed
```

---

## 🎯 Running Tests

### Start Your Flask App

**Terminal 1** - Start the app:
```bash
python wsgi.py
```

Keep this running. You should see:
```
 * Running on http://localhost:5000
```

### Run Playwright Tests

**Terminal 2** - Run tests:

```bash
# Basic run (headless, fast)
pytest tests/e2e/test_browser_workflows.py -v

# See the browser while tests run
pytest tests/e2e/test_browser_workflows.py --headed

# Slow down actions (helpful for debugging)
pytest tests/e2e/test_browser_workflows.py --headed --slowmo 1000

# Run specific test
pytest tests/e2e/test_browser_workflows.py::test_home_page_loads -v

# Test on Firefox instead of Chrome
pytest tests/e2e/test_browser_workflows.py --browser firefox --headed
```

---

## 📊 What You'll See

### Successful Test Run:
```
tests/e2e/test_browser_workflows.py::test_home_page_loads PASSED           [ 5%]
tests/e2e/test_browser_workflows.py::test_page_header_and_branding PASSED  [10%]
tests/e2e/test_browser_workflows.py::test_responsive_layout PASSED         [15%]
...
======================== 20 passed in 45.23s ========================
```

### Screenshots
- Saved automatically to `tests/screenshots/`
- On test failure: `tests/screenshots/FAILED_test_name.png`

---

## 🐛 Quick Debug

If a test fails, use the Playwright Inspector:

```bash
PWDEBUG=1 pytest tests/e2e/test_browser_workflows.py::test_name
```

This opens an interactive debugger where you can:
- ✅ Pause execution
- ✅ Step through actions
- ✅ Inspect elements
- ✅ View network requests

---

## 🎬 Demo Commands

```bash
# Quick smoke test (just verify app loads)
pytest tests/e2e/test_browser_workflows.py::test_home_page_loads --headed

# Test form interactions
pytest tests/e2e/test_browser_workflows.py -k "form" --headed

# Test responsive design
pytest tests/e2e/test_browser_workflows.py::test_responsive_layout --headed

# Test accessibility
pytest tests/e2e/test_browser_workflows.py -k "accessibility" --headed

# Run all tests across all browsers (comprehensive)
pytest tests/e2e/test_browser_workflows.py --browser chromium --browser firefox --browser webkit -v
```

---

## 📂 Files Created

```
slidegenerator/
├── playwright.config.py                    # Configuration
├── tests/
│   └── e2e/
│       ├── conftest.py                     # Fixtures
│       ├── test_browser_workflows.py       # 20 tests
│       └── screenshots/                    # Auto-generated
├── docs/guides/
│   └── PLAYWRIGHT_TESTING_GUIDE.md         # Full documentation
└── requirements.txt                         # Updated with Playwright
```

---

## 🎯 Next Steps

1. **Run your first test**:
   ```bash
   pytest tests/e2e/test_browser_workflows.py::test_home_page_loads --headed
   ```

2. **Explore the tests**:
   Open `tests/e2e/test_browser_workflows.py` to see all 20 tests

3. **Read the full guide**:
   See `docs/guides/PLAYWRIGHT_TESTING_GUIDE.md` for comprehensive documentation

4. **Add to CI/CD**:
   See guide for GitHub Actions / GitLab CI integration examples

---

## 💡 Tips

- **Always run the Flask app first** (`python wsgi.py`) before running tests
- **Use `--headed`** to see what's happening in the browser
- **Use `--slowmo 1000`** to slow down actions (helpful for debugging)
- **Screenshots are your friend** - check `tests/screenshots/` when tests fail
- **Use PWDEBUG=1** for interactive debugging

---

## 🆘 Troubleshooting

### "Connection refused" error
**Problem**: Flask app not running
**Solution**: Start `python wsgi.py` in another terminal

### "Timeout" error
**Problem**: Page took too long to load
**Solution**: Increase timeout in `playwright.config.py` or check app is responding

### "Element not found" error
**Problem**: Selector doesn't match any elements
**Solution**: Use Playwright Inspector (`PWDEBUG=1`) to test selectors

### "Module not found" error
**Problem**: Playwright not installed
**Solution**: `pip install playwright pytest-playwright && playwright install`

---

## 📚 Resources

- **Full Guide**: `docs/guides/PLAYWRIGHT_TESTING_GUIDE.md`
- **Configuration**: `playwright.config.py`
- **Test Fixtures**: `tests/e2e/conftest.py`
- **Playwright Docs**: https://playwright.dev/python/

---

**Happy Testing! 🎭**

You now have a complete browser-based E2E test suite with:
- ✅ 20 comprehensive tests
- ✅ Cross-browser support (Chrome, Firefox, Safari)
- ✅ Mobile/tablet emulation
- ✅ Accessibility testing
- ✅ Performance monitoring
- ✅ Visual regression testing
- ✅ Automatic screenshots
- ✅ Interactive debugging

All tests are documented, organized, and ready to integrate into CI/CD!
