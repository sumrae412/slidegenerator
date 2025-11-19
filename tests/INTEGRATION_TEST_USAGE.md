# Integration Test Suite - Quick Usage Guide

This guide shows how to use the comprehensive format integration test suite in your development workflow.

---

## Quick Start

### Run All Tests (Recommended)

```bash
# Run complete test suite with comprehensive report
python tests/test_format_integration.py --report

# Expected output:
# ================================================================================
# FORMAT INTEGRATION TEST SUITE - COMPREHENSIVE REPORT
# ================================================================================
#
# [19 tests run]
#
# ================================================================================
# TEST SUMMARY
# ================================================================================
# Total Tests Run:     19
# Successes:           19
# Failures:            0
# Errors:              0
# Skipped:             0
#
# Overall Success Rate: 100.0%
# ✅ ALL TESTS PASSED - Integration suite is healthy
```

### Run Specific Test Category

```bash
# Test format parsing only (DOCX, TXT, PDF)
python -m unittest tests.test_format_integration.TestFormatParsing -v

# Test document analysis only
python -m unittest tests.test_format_integration.TestDocumentAnalysis -v

# Test performance only
python -m unittest tests.test_format_integration.TestPerformance -v

# Test error recovery only
python -m unittest tests.test_format_integration.TestErrorRecovery -v
```

---

## Development Workflow Integration

### Before Making Code Changes

```bash
# 1. Run tests to establish baseline
python tests/test_format_integration.py --report > baseline_results.txt

# 2. Note the pass rate and performance
# Total Tests Run:     19
# Overall Success Rate: 100.0%
```

### After Making Code Changes

```bash
# 3. Run tests again to verify no regressions
python tests/test_format_integration.py --report

# 4. Compare results
# - All tests should still pass
# - Performance should not degrade significantly
# - New features should have corresponding tests
```

### Pre-Deployment Checklist

```bash
# Run before every deployment
python tests/test_format_integration.py --report

# ✅ If all tests pass → Safe to deploy
# ❌ If any tests fail → Fix issues before deploying
```

---

## Common Testing Scenarios

### Test a New Document Format

```python
# Add a new test case to test_format_integration.py

def test_new_format_parsing(self):
    """Test NEW_FORMAT → Parse → Slides"""
    test_file = os.path.join(self.temp_dir, 'test.newformat')

    # Create test file
    # ... create file logic ...

    # Parse the document
    doc_structure = self.parser.parse_file(test_file, 'test.newformat')

    # Verify
    self.assertIsInstance(doc_structure, DocumentStructure)
    self.assertGreater(len(doc_structure.slides), 0)
```

### Test a New Feature

```python
# Add test for new content merging feature

def test_new_merging_feature(self):
    """Test advanced content merging"""
    # Create test document with specific structure
    # ... test setup ...

    # Parse with new feature enabled
    doc_structure = self.parser.parse_file(
        test_file,
        'test.docx',
        enable_advanced_merging=True  # New parameter
    )

    # Verify new behavior
    self.assertGreater(len(doc_structure.slides), 0)
    # ... additional assertions ...
```

### Benchmark Performance Changes

```python
# Add performance comparison test

def test_performance_optimization(self):
    """Test that optimization improves speed"""
    # Create large test document
    # ... setup ...

    start_time = time.time()
    doc_structure = self.parser.parse_file(test_file, 'large.docx')
    elapsed_time = time.time() - start_time

    # Verify performance improvement
    self.assertLess(elapsed_time, 30, "Should complete in < 30s")
```

---

## Debugging Failed Tests

### View Detailed Error Output

```bash
# Run with verbose output
python -m unittest tests.test_format_integration.TestFormatParsing.test_docx_parse_to_slides -v

# Example output if failing:
# FAIL: test_docx_parse_to_slides
# AssertionError: 0 not greater than 0 : Should generate slides
#
# This indicates no slides were generated - check parsing logic
```

### Run Single Test for Quick Iteration

```bash
# Test only the failing test while debugging
python -m unittest tests.test_format_integration.TestFormatParsing.test_docx_parse_to_slides -v

# Make changes to code...

# Re-run same test
python -m unittest tests.test_format_integration.TestFormatParsing.test_docx_parse_to_slides -v

# Repeat until passing
```

### Add Debug Print Statements

```python
def test_docx_parse_to_slides(self):
    """Test 1: DOCX → Parse → Slides"""
    # ... test setup ...

    doc_structure = self.parser.parse_file(test_file, 'test_doc.docx')

    # Debug output
    print(f"DEBUG: Got {len(doc_structure.slides)} slides")
    for i, slide in enumerate(doc_structure.slides):
        print(f"  Slide {i}: {slide.title} ({slide.slide_type})")

    # ... assertions ...
```

---

## Test Categories Reference

### 1. Format Parsing Tests
**Tests:** 4
**Purpose:** Validate document format conversion
**Formats:** DOCX, TXT, PDF

```bash
python -m unittest tests.test_format_integration.TestFormatParsing -v
```

**Tests:**
- `test_docx_parse_to_slides` - DOCX → Slides
- `test_txt_parse_to_slides` - TXT → Slides
- `test_pdf_parse_to_slides` - PDF → Slides (conditional)
- `test_mixed_content_tables_and_text` - Mixed content

### 2. Document Analysis Tests
**Tests:** 3
**Purpose:** Validate structure detection and mode suggestion

```bash
python -m unittest tests.test_format_integration.TestDocumentAnalysis -v
```

**Tests:**
- `test_table_dominant_document` - Table detection
- `test_text_dominant_document` - Text detection
- `test_auto_suggestion_accuracy` - Suggestion algorithm

### 3. Content Merging Tests
**Tests:** 2
**Purpose:** Validate paragraph-table merging

```bash
python -m unittest tests.test_format_integration.TestContentMerging -v
```

**Tests:**
- `test_table_with_intro_paragraph` - Intro merging
- `test_table_with_explanation` - Explanation merging

### 4. End-to-End Workflow Tests
**Tests:** 2
**Purpose:** Validate complete user workflows

```bash
python -m unittest tests.test_format_integration.TestEndToEndWorkflow -v
```

**Tests:**
- `test_upload_analyze_parse_generate` - Complete pipeline
- `test_multiple_format_comparison` - Format consistency

### 5. Performance Tests
**Tests:** 2
**Purpose:** Validate scalability and speed

```bash
python -m unittest tests.test_format_integration.TestPerformance -v
```

**Tests:**
- `test_large_document` - 50+ sections (< 60s threshold)
- `test_many_tables` - 20+ tables processing

### 6. Error Recovery Tests
**Tests:** 3
**Purpose:** Validate error handling

```bash
python -m unittest tests.test_format_integration.TestErrorRecovery -v
```

**Tests:**
- `test_unsupported_format` - Invalid file types
- `test_empty_document` - Empty file handling
- `test_malformed_document` - Corrupted file handling

### 7. Regression Tests
**Tests:** 3
**Purpose:** Prevent breaking existing functionality

```bash
python -m unittest tests.test_format_integration.TestRegression -v
```

**Tests:**
- `test_basic_docx_parsing` - Core DOCX functionality
- `test_basic_txt_parsing` - Core TXT functionality
- `test_table_extraction` - Table parsing

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/integration-tests.yml

name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run integration tests
      run: |
        python tests/test_format_integration.py --report

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test_results.txt
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit

#!/bin/bash
echo "Running integration tests..."

python tests/test_format_integration.py --report

if [ $? -ne 0 ]; then
    echo "❌ Integration tests failed - commit aborted"
    exit 1
fi

echo "✅ Integration tests passed"
exit 0
```

---

## Test Data Management

### Using Test Utilities

```python
from tests.test_format_integration import TestUtilities

# Create DOCX for testing
TestUtilities.create_test_docx(
    content_data=[
        {'type': 'heading', 'level': 1, 'text': 'My Test Doc'},
        {'type': 'paragraph', 'text': 'Test content here.'},
        {'type': 'table', 'rows': [
            ['Header 1', 'Header 2'],
            ['Data 1', 'Data 2']
        ]}
    ],
    file_path='my_test.docx'
)

# Create TXT for testing
TestUtilities.create_test_txt(
    content_data=[
        {'type': 'heading', 'text': 'Product List'},
        {'type': 'table', 'rows': [
            ['Product', 'Price'],
            ['Widget', '$10']
        ]}
    ],
    file_path='my_test.txt'
)

# Create PDF for testing (requires reportlab)
TestUtilities.create_test_pdf(
    content_data=[...],
    file_path='my_test.pdf'
)
```

### Cleanup After Tests

```python
import tempfile
import shutil

# Tests automatically clean up temp files
def tearDown(self):
    """Clean up test files"""
    for file_path in self.test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists(self.temp_dir):
        shutil.rmtree(self.temp_dir)
```

---

## Performance Monitoring

### Track Performance Over Time

```bash
# Run tests and save timing data
python tests/test_format_integration.py --report | tee test_results_$(date +%Y%m%d).txt

# Compare performance between versions
# Before changes:
# Ran 19 tests in 1.367s

# After optimization:
# Ran 19 tests in 0.892s
# → 35% improvement!
```

### Identify Slow Tests

```bash
# Run with verbose timing
python -m unittest tests.test_format_integration -v 2>&1 | grep -E "ok|FAIL" | grep -E "\d+\.\d+s"

# Example output:
# test_large_document ... ok (0.67s)
# test_many_tables ... ok (0.17s)
```

---

## Troubleshooting

### PDF Tests Skipped

```bash
# If you see:
# test_pdf_parse_to_slides ... skipped 'PDF parsing libraries not available'

# Fix:
pip install pdfplumber
# OR
pip install PyPDF2
```

### Import Errors

```bash
# If you see:
# ModuleNotFoundError: No module named 'slide_generator_pkg'

# Fix:
# Ensure you're running from project root
cd /home/user/slidegenerator
python tests/test_format_integration.py --report
```

### Temp Directory Issues

```bash
# If tests fail with temp file errors:
# - Check disk space: df -h
# - Check permissions: ls -la /tmp
# - Manually clean: rm -rf /tmp/tmp*
```

---

## Best Practices

### 1. Run Tests Frequently
```bash
# After every code change
python tests/test_format_integration.py --report
```

### 2. Add Tests for New Features
```python
# When adding new functionality, add corresponding tests
def test_my_new_feature(self):
    """Test description"""
    # ... test implementation ...
```

### 3. Keep Tests Fast
```python
# Avoid creating overly large test documents
# Prefer focused, targeted tests
# Use small, representative test data
```

### 4. Document Test Failures
```bash
# When a test fails, document:
# 1. What was expected
# 2. What actually happened
# 3. Steps to reproduce
# 4. Root cause (if known)
```

### 5. Maintain Test Coverage
```bash
# Ensure all critical paths are tested
# Aim for 80%+ coverage of core functionality
# Don't forget edge cases!
```

---

## Support

For issues or questions about the test suite:

1. Check this usage guide
2. Review test documentation in `FORMAT_INTEGRATION_TEST_REPORT.md`
3. Examine test source code: `tests/test_format_integration.py`
4. Check existing test patterns for examples

---

**Last Updated:** November 19, 2025
**Test Suite Version:** 1.0
**Python Version:** 3.11+
