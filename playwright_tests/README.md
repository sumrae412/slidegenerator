# Playwright UI Testing Suite

Comprehensive automated testing for the Document to Slides converter UI.

## Overview

This test suite provides:
- **UI Flow Testing**: Validate core user workflows
- **Visual Regression**: Capture and compare screenshots
- **Performance Metrics**: Measure page load and interaction speed
- **Accessibility Testing**: Keyboard navigation and ARIA compliance
- **UI Complexity Analysis**: Quantify interface clutter

## Installation

```bash
# Install Playwright and dependencies
pip install playwright pytest-playwright

# Install browser
playwright install chromium
```

## Running Tests

### All Tests
```bash
pytest playwright_tests/ -v
```

### Specific Test Categories
```bash
# UI flow tests only
pytest playwright_tests/ -m flow

# Visual regression tests only
pytest playwright_tests/ -m visual

# Performance tests only
pytest playwright_tests/ -m performance
```

### Run Analysis Report
```bash
python playwright_tests/run_analysis.py
```

This will:
1. Run all tests
2. Generate UI complexity metrics
3. Capture baseline screenshots
4. Create a comprehensive markdown report

## Test Files

### `conftest.py`
Pytest configuration and fixtures for the test suite.

### `test_ui_flows.py`
Core functionality tests:
- Homepage loading
- Collapsible sections
- Form validation
- File upload interaction
- Keyboard navigation
- Accessibility compliance

### `test_ui_comparison.py`
Comparison tests for old vs new UI:
- Side-by-side screenshots
- Complexity metrics comparison
- Mobile responsiveness comparison

### `run_analysis.py`
Automated analysis script that generates:
- `ui_analysis_report.md` - Full UI analysis
- `ui_complexity_analysis.json` - Metrics data
- `friction_points.json` - UX issues
- `performance_metrics.json` - Performance data
- `resource_report.json` - Resource loading analysis

## Screenshots

Screenshots are saved in:
```
playwright_tests/screenshots/
├── baseline/          # Baseline screenshots for comparison
├── current/           # Latest test run screenshots
├── diff/              # Visual diff images
└── comparison/        # Old vs New UI comparisons
```

## Metrics Captured

### UI Complexity
- Total buttons
- Total input fields
- Number of cards/containers
- Collapsible sections
- Visible text length
- Heading structure

### Friction Points
- Hidden required fields
- Scroll distance to submit
- Form field count (cognitive load)
- Navigation complexity

### Performance
- Page load time
- DOM Content Loaded
- First Paint
- Time to Interactive
- Resource count and size

### Accessibility
- Focusable elements count
- Keyboard navigation flow
- Form label compliance
- ARIA attributes

## Expected Results

### Old UI (Current)
- **Buttons**: ~25-30
- **Input Fields**: ~15-20
- **Cards**: 8-10
- **Collapsible Sections**: 4-5
- **Scroll Height**: ~3000-4000px
- **Page Load**: 2-3s

### New UI (Streamlined)
- **Buttons**: ~10-12 (60% reduction)
- **Input Fields**: ~7-9 (55% reduction)
- **Cards**: 2-3 (70% reduction)
- **Collapsible Sections**: 1-2 (60% reduction)
- **Scroll Height**: ~1500-2000px (50% reduction)
- **Page Load**: 1-1.5s (40% faster)

## CI/CD Integration

Add to GitHub Actions:

```yaml
name: UI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install playwright pytest-playwright
      - run: playwright install chromium
      - run: python test_server.py &
      - run: sleep 5
      - run: pytest playwright_tests/ -v
      - uses: actions/upload-artifact@v2
        if: always()
        with:
          name: screenshots
          path: playwright_tests/screenshots/
```

## Troubleshooting

### Tests fail with "Connection refused"
Ensure the Flask app is running on port 5000:
```bash
python test_server.py &
sleep 3
pytest playwright_tests/
```

### Screenshots not generating
Check that screenshot directories have write permissions:
```bash
chmod -R 755 playwright_tests/screenshots/
```

### Browser not found
Reinstall Playwright browsers:
```bash
playwright install chromium --force
```

## Contributing

When adding new tests:
1. Use appropriate markers (`@pytest.mark.flow`, `@pytest.mark.visual`, etc.)
2. Add docstrings describing what the test validates
3. Save metrics to JSON for analysis
4. Update this README with new test descriptions

## License

Same as parent project.
