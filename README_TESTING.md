# Testing Guide

## Overview

This project uses `pytest` for automated testing. Tests cover:
- Unit tests for document parsing logic
- Integration tests for API endpoints
- Google Docs URL parsing
- Bullet generation (with and without API keys)
- Slide structure generation

## Running Tests

### Install test dependencies
```bash
pip install pytest pytest-cov
```

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_document_parser.py
```

### Run specific test class
```bash
pytest tests/test_document_parser.py::TestTXTParser
```

### Run specific test
```bash
pytest tests/test_document_parser.py::TestTXTParser::test_heading_detection_h1
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_document_parser.py  # Unit tests for parsing logic
├── test_api_endpoints.py    # Integration tests for Flask routes
└── test_google_docs.py      # Google Docs integration tests
```

## Writing New Tests

### Unit Test Example
```python
def test_my_function():
    """Test description"""
    result = my_function("input")
    assert result == "expected_output"
```

### Integration Test Example
```python
def test_api_endpoint(client):
    """Test API endpoint"""
    response = client.post('/upload', data={'key': 'value'})
    assert response.status_code == 200
```

## Continuous Integration

Tests run automatically on:
- Every push to `main`
- Every pull request

See `.github/workflows/test.yml` for CI configuration.

## Test Coverage Goals

- **Unit tests**: >80% coverage
- **Integration tests**: All critical paths
- **Edge cases**: Empty inputs, invalid data, timeouts

## Running Tests Locally Before Deploy

```bash
# Run all tests
pytest

# If all pass, deploy
git push heroku main
```
