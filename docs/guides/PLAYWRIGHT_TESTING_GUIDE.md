# Playwright Browser Testing Guide

Complete guide to running browser-based end-to-end tests for the Slide Generator application using Playwright.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Running Tests](#running-tests)
4. [Test Structure](#test-structure)
5. [Configuration](#configuration)
6. [Writing Tests](#writing-tests)
7. [Debugging](#debugging)
8. [CI/CD Integration](#cicd-integration)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### What is Playwright?

Playwright is a browser automation framework that allows you to test your web application across multiple browsers (Chromium, Firefox, WebKit/Safari) with a single API.

### Why Playwright?

- **Cross-browser testing**: Test on Chrome, Firefox, and Safari
- **Real browser interactions**: Simulates actual user behavior
- **Fast and reliable**: Parallel execution, auto-waiting for elements
- **Developer-friendly**: Great debugging tools and screenshots
- **Mobile emulation**: Test responsive designs

### Test Coverage

Our Playwright test suite covers:
- ✅ Page load and navigation
- ✅ Form interactions (input, submit, validation)
- ✅ UI component functionality (toggles, dropdowns)
- ✅ Responsive design across devices
- ✅ Accessibility (keyboard navigation, ARIA labels)
- ✅ Performance (page load times, console errors)
- ✅ Visual regression (screenshot comparison)
- ✅ Cross-browser compatibility

---

## 📦 Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `playwright>=1.40.0` - Core Playwright library
- `pytest-playwright>=0.4.0` - Pytest integration

### 2. Install Playwright Browsers

After installing the Python packages, install browser binaries:

```bash
# Install all browsers (Chromium, Firefox, WebKit)
playwright install

# Or install specific browsers
playwright install chromium
playwright install firefox
playwright install webkit
```

**Disk Space**: Each browser requires ~300MB. All three browsers: ~1GB.

### 3. Verify Installation

```bash
# Check Playwright version
playwright --version

# Check pytest-playwright
pytest --co tests/e2e/test_browser_workflows.py
```

---

## 🚀 Running Tests

### Basic Usage

```bash
# Run all Playwright tests
pytest tests/e2e/test_browser_workflows.py -v

# Run specific test
pytest tests/e2e/test_browser_workflows.py::test_home_page_loads -v

# Run with test name pattern
pytest tests/e2e/test_browser_workflows.py -k "form" -v
```

### Show Browser (Headed Mode)

```bash
# See the browser while tests run
pytest tests/e2e/test_browser_workflows.py --headed

# Slow down actions for debugging
pytest tests/e2e/test_browser_workflows.py --headed --slowmo 1000
```

### Cross-Browser Testing

```bash
# Run on Chromium (default)
pytest tests/e2e/test_browser_workflows.py

# Run on Firefox
pytest tests/e2e/test_browser_workflows.py --browser firefox

# Run on WebKit (Safari)
pytest tests/e2e/test_browser_workflows.py --browser webkit

# Run on all browsers
pytest tests/e2e/test_browser_workflows.py --browser chromium --browser firefox --browser webkit
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest tests/e2e/test_browser_workflows.py -n auto

# Run 4 tests concurrently
pytest tests/e2e/test_browser_workflows.py -n 4
```

### Filter by Markers

```bash
# Run only Playwright tests
pytest tests/e2e/test_browser_workflows.py -m playwright

# Run only visual regression tests
pytest tests/e2e/test_browser_workflows.py -m visual

# Skip slow tests
pytest tests/e2e/test_browser_workflows.py -m "not slow"
```

### Generate Test Reports

```bash
# HTML report
pytest tests/e2e/test_browser_workflows.py --html=report.html --self-contained-html

# JUnit XML (for CI/CD)
pytest tests/e2e/test_browser_workflows.py --junitxml=test-results.xml
```

---

## 🗂️ Test Structure

### Directory Layout

```
tests/
├── e2e/
│   ├── conftest.py              # Playwright fixtures
│   ├── test_browser_workflows.py # Main browser tests
│   └── test_data/               # Test data files
├── screenshots/                 # Auto-generated screenshots
└── videos/                      # Test execution videos

playwright.config.py             # Playwright configuration
```

### Test Categories

**1. Page Load Tests**
- `test_home_page_loads`: Verifies main page loads successfully
- `test_page_header_and_branding`: Checks branding elements
- `test_responsive_layout`: Tests across device sizes

**2. Form Interaction Tests**
- `test_google_doc_url_input_visible`: URL input field
- `test_output_format_selection`: Format selection
- `test_api_key_input_field`: API key input
- `test_form_validation_empty_url`: Validation logic

**3. UI Feature Tests**
- `test_progressive_disclosure_api_settings`: Collapsible sections
- `test_google_sign_in_button`: OAuth button
- `test_visual_generation_toggle`: Feature toggles

**4. Submission Tests**
- `test_form_submission_with_valid_data`: Full form flow

**5. Accessibility Tests**
- `test_keyboard_navigation`: Tab navigation
- `test_aria_labels_present`: Screen reader support

**6. Performance Tests**
- `test_page_load_performance`: Load time metrics
- `test_no_console_errors`: JavaScript errors

**7. Visual Regression Tests**
- `test_visual_snapshot_home_page`: Baseline screenshots
- `test_visual_snapshot_form_filled`: Form state snapshots

**8. Cross-Browser Tests**
- `test_cross_browser_compatibility`: Multi-browser support

**9. Error Handling Tests**
- `test_404_page_not_found`: 404 errors
- `test_network_offline_handling`: Offline scenarios

---

## ⚙️ Configuration

### Environment Variables

Set these to customize test behavior:

```bash
# Application URL (default: http://localhost:5000)
export PLAYWRIGHT_BASE_URL="http://localhost:5000"

# Run in headed mode (default: true/headless)
export PLAYWRIGHT_HEADLESS="false"

# Slow down actions in milliseconds (default: 0)
export PLAYWRIGHT_SLOW_MO="1000"

# Browser to use (default: chromium)
export PLAYWRIGHT_BROWSER="firefox"

# Enable video recording (default: false)
export PLAYWRIGHT_VIDEO="true"
```

### Configuration File

Edit `playwright.config.py` to change:

```python
TEST_CONFIG = {
    "base_url": "http://localhost:5000",
    "headless": True,
    "slow_mo": 0,
    "timeout": {
        "default": 30000,  # 30 seconds
        "navigation": 60000,
        "api": 120000,
    },
    "screenshots": {
        "enabled": True,
        "on_failure": True,
        "directory": "tests/screenshots",
    },
}
```

---

## ✍️ Writing Tests

### Basic Test Template

```python
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.playwright
def test_my_feature(page: Page, base_url: str):
    """
    Test description here.

    Steps:
    1. Navigate to page
    2. Interact with element
    3. Verify result
    """
    # Navigate
    page.goto(base_url)

    # Interact
    button = page.locator("button#submit")
    button.click()

    # Assert
    result = page.locator(".result")
    expect(result).to_be_visible()
```

### Using Page Object Pattern

```python
@pytest.mark.playwright
def test_with_page_object(home_page):
    """
    Use the home_page fixture (page object).
    """
    # Page object provides helper methods
    home_page.fill_google_doc_url("https://docs.google.com/document/d/test/edit")
    home_page.select_output_format("pptx")
    home_page.submit_form()
    home_page.wait_for_result()
```

### Taking Screenshots

```python
from playwright.config import take_screenshot

def test_with_screenshot(page: Page, base_url: str):
    page.goto(base_url)

    # Take screenshot
    screenshot_path = take_screenshot(page, "test_name", full_page=True)
    print(f"Screenshot saved: {screenshot_path}")
```

### Waiting for Elements

```python
# Wait for element to be visible
page.locator("button").wait_for(state="visible")

# Wait for network to be idle
page.wait_for_load_state("networkidle")

# Wait for specific timeout
page.wait_for_timeout(1000)  # 1 second
```

### Handling Forms

```python
# Fill text input
page.locator("input[name='email']").fill("user@example.com")

# Check checkbox
page.locator("input[type='checkbox']").check()

# Select dropdown
page.locator("select").select_option("option-value")

# Click button
page.locator("button[type='submit']").click()
```

---

## 🐛 Debugging

### 1. Run Tests in Headed Mode

See the browser while tests execute:

```bash
pytest tests/e2e/test_browser_workflows.py --headed --slowmo 1000
```

### 2. Playwright Inspector

Interactive debugger with pause and step-through:

```bash
PWDEBUG=1 pytest tests/e2e/test_browser_workflows.py
```

Features:
- ✅ Pause execution
- ✅ Step through actions
- ✅ Inspect selectors
- ✅ View network requests

### 3. Screenshots on Failure

Automatically enabled. When a test fails, screenshot is saved to `tests/screenshots/FAILED_test_name.png`.

Check output:
```
📸 Screenshot saved: tests/screenshots/FAILED_test_home_page_loads.png
```

### 4. Video Recording

Enable video recording:

```bash
export PLAYWRIGHT_VIDEO="true"
pytest tests/e2e/test_browser_workflows.py
```

Videos saved to `tests/videos/`.

### 5. Verbose Output

```bash
# Show detailed test output
pytest tests/e2e/test_browser_workflows.py -vv

# Show print statements
pytest tests/e2e/test_browser_workflows.py -s

# Show full tracebacks
pytest tests/e2e/test_browser_workflows.py --tb=long
```

### 6. Debugging Selectors

Test selectors in Playwright Inspector:

```python
# Add this line to pause execution
page.pause()
```

Or use Playwright CLI:

```bash
playwright codegen http://localhost:5000
```

This opens browser and generates code as you interact!

---

## 🔄 CI/CD Integration

### GitHub Actions Example

`.github/workflows/playwright-tests.yml`:

```yaml
name: Playwright Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          playwright install --with-deps chromium

      - name: Start Flask app
        run: |
          python wsgi.py &
          sleep 5

      - name: Run Playwright tests
        run: |
          pytest tests/e2e/test_browser_workflows.py --browser chromium

      - name: Upload screenshots on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-screenshots
          path: tests/screenshots/
```

### GitLab CI Example

`.gitlab-ci.yml`:

```yaml
playwright-tests:
  stage: test
  image: mcr.microsoft.com/playwright/python:v1.40.0-jammy
  script:
    - pip install -r requirements.txt
    - python wsgi.py &
    - sleep 5
    - pytest tests/e2e/test_browser_workflows.py
  artifacts:
    when: on_failure
    paths:
      - tests/screenshots/
    expire_in: 1 week
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running Playwright smoke tests..."
pytest tests/e2e/test_browser_workflows.py -m "not slow" -q

if [ $? -ne 0 ]; then
    echo "❌ Playwright tests failed. Fix issues before committing."
    exit 1
fi
```

---

## 🔧 Troubleshooting

### Issue 1: Playwright not found

**Error**: `ModuleNotFoundError: No module named 'playwright'`

**Solution**:
```bash
pip install playwright pytest-playwright
playwright install
```

### Issue 2: Browser not installed

**Error**: `Executable doesn't exist at /path/to/browser`

**Solution**:
```bash
playwright install chromium
```

### Issue 3: App not running

**Error**: `Connection refused at http://localhost:5000`

**Solution**:
Start Flask app before running tests:
```bash
# Terminal 1
python wsgi.py

# Terminal 2
pytest tests/e2e/test_browser_workflows.py
```

### Issue 4: Timeout errors

**Error**: `TimeoutError: page.goto: Timeout 30000ms exceeded`

**Solution**:
Increase timeout in `playwright.config.py`:
```python
"timeout": {
    "navigation": 120000,  # 2 minutes
}
```

### Issue 5: Element not found

**Error**: `Error: locator.click: Target closed`

**Solution**:
1. Add wait before interaction:
   ```python
   page.locator("button").wait_for(state="visible")
   page.locator("button").click()
   ```

2. Use more specific selector:
   ```python
   # Instead of
   page.locator("button").click()

   # Use
   page.locator("button[type='submit']").click()
   ```

### Issue 6: Tests flaky/unreliable

**Solution**:
1. Use Playwright's auto-waiting (built-in)
2. Avoid manual `time.sleep()`
3. Use `wait_for_load_state("networkidle")`
4. Check for dynamic content:
   ```python
   expect(element).to_be_visible()
   ```

---

## 📚 Additional Resources

### Official Documentation
- [Playwright Python Docs](https://playwright.dev/python/docs/intro)
- [pytest-playwright](https://playwright.dev/python/docs/test-runners)
- [Playwright API Reference](https://playwright.dev/python/docs/api/class-playwright)

### Useful Guides
- [Best Practices](https://playwright.dev/python/docs/best-practices)
- [Selectors Guide](https://playwright.dev/python/docs/selectors)
- [Network Mocking](https://playwright.dev/python/docs/network)
- [Screenshots & Videos](https://playwright.dev/python/docs/screenshots)

### Community
- [Playwright Discord](https://discord.com/invite/playwright-807756831384403968)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/playwright)
- [GitHub Discussions](https://github.com/microsoft/playwright/discussions)

---

## 🎯 Best Practices

1. **Use auto-waiting**: Playwright waits for elements automatically
2. **Prefer user-facing attributes**: Use text, labels, roles over CSS classes
3. **Keep tests independent**: Each test should run standalone
4. **Use fixtures**: Reuse setup code via pytest fixtures
5. **Take screenshots**: Especially on failures for debugging
6. **Test across browsers**: Don't rely on just Chromium
7. **Mobile testing**: Test responsive designs with device emulation
8. **Accessibility**: Include keyboard navigation and ARIA tests
9. **Performance**: Monitor page load times
10. **Visual regression**: Use baseline screenshots to catch UI changes

---

## 📝 Quick Reference

### Common Commands

```bash
# Run all tests
pytest tests/e2e/test_browser_workflows.py -v

# Run in headed mode
pytest tests/e2e/test_browser_workflows.py --headed

# Debug mode
PWDEBUG=1 pytest tests/e2e/test_browser_workflows.py

# Specific browser
pytest tests/e2e/test_browser_workflows.py --browser firefox

# Parallel execution
pytest tests/e2e/test_browser_workflows.py -n auto

# Generate report
pytest tests/e2e/test_browser_workflows.py --html=report.html
```

### Common Assertions

```python
# Visibility
expect(element).to_be_visible()
expect(element).to_be_hidden()

# Text content
expect(element).to_have_text("Expected text")
expect(element).to_contain_text("Partial text")

# Attributes
expect(element).to_have_attribute("href", "/path")
expect(element).to_have_class("active")

# Count
expect(elements).to_have_count(5)

# State
expect(checkbox).to_be_checked()
expect(input).to_be_enabled()
```

---

**Last Updated**: 2025-01-19
**Playwright Version**: 1.40+
**pytest-playwright Version**: 0.4+

For questions or issues, see [troubleshooting](#troubleshooting) or file an issue on GitHub.
