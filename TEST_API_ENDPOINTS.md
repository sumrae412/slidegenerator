# API Endpoint Tests Documentation

## Overview

Comprehensive test suite for all Flask API endpoints created in:
**`/home/user/slidegenerator/tests/integration/test_api_endpoints.py`**

This test file contains **68 test cases** organized into **10 test classes**, providing complete coverage of:
- Index route (GET /)
- Upload route (POST /upload)
- Google OAuth routes
- API key validation
- Encryption key management
- CSP violation reporting
- File download functionality
- Security headers
- Error handling
- Endpoint integration flows

## Test Structure

### Files Created
```
tests/integration/
├── __init__.py                  # Package initialization
└── test_api_endpoints.py        # Main test file (650 lines, 68 tests)
```

## Test Classes & Coverage

### 1. TestIndexRoute (5 tests)
Tests for GET / (main page endpoint)

| Test Name | Purpose |
|-----------|---------|
| `test_index_page_loads` | Verify index page returns 200 status |
| `test_index_page_has_title` | Verify page contains expected HTML |
| `test_index_page_html_structure` | Validate HTML structure |
| `test_index_allows_get_only` | Verify only GET requests accepted |
| `test_index_content_type` | Verify HTML content type header |

### 2. TestUploadRoute (12 tests)
Tests for POST /upload (document conversion endpoint)

| Test Name | Purpose |
|-----------|---------|
| `test_upload_requires_doc_url` | Verify google_docs_url parameter required |
| `test_upload_rejects_empty_input` | Verify empty input rejected |
| `test_upload_returns_json` | Verify JSON response format |
| `test_upload_requires_api_key` | Verify Claude API key required |
| `test_upload_accepts_pptx_format` | Verify PPTX output format supported |
| `test_upload_accepts_google_slides_format` | Verify Google Slides format supported |
| `test_upload_rejects_invalid_url` | Verify invalid URLs handled |
| `test_upload_handles_missing_form_data` | Verify form data validation |
| `test_upload_accepts_encrypted_keys` | Verify encrypted key handling |
| `test_upload_accepts_script_column_parameter` | Verify column parameter support |
| `test_upload_accepts_skip_visuals_parameter` | Verify visual skipping support |
| `test_upload_only_accepts_post` | Verify only POST requests accepted |

### 3. TestGoogleAuthRoutes (8 tests)
Tests for OAuth 2.0 authentication routes

| Test Name | Purpose |
|-----------|---------|
| `test_google_auth_redirect` | Verify OAuth redirect to Google |
| `test_google_auth_endpoint_exists` | Verify /auth/google endpoint exists |
| `test_oauth_callback_without_code` | Handle missing authorization code |
| `test_oauth_callback_with_invalid_code` | Handle invalid authorization code |
| `test_oauth_callback_with_error_parameter` | Handle OAuth errors |
| `test_google_config_endpoint_returns_json` | Verify JSON response for config |
| `test_google_config_endpoint_returns_config` | Verify config data returned |
| `test_google_auth_debug_endpoint` | Verify debug endpoint works |

### 4. TestAPIKeyValidation (10 tests)
Tests for POST /api/validate-key (API key validation)

| Test Name | Purpose |
|-----------|---------|
| `test_validate_key_requires_json` | Verify JSON input required |
| `test_validate_key_endpoint_exists` | Verify endpoint exists |
| `test_validate_key_requires_key_type` | Verify key_type parameter required |
| `test_validate_key_requires_encrypted_key` | Verify encrypted_key parameter required |
| `test_validate_key_returns_json` | Verify JSON response format |
| `test_validate_key_accepts_claude_type` | Verify Claude key type supported |
| `test_validate_key_accepts_openai_type` | Verify OpenAI key type supported |
| `test_validate_key_invalid_key_response` | Verify invalid key handling |
| `test_validate_key_response_structure` | Verify response structure |
| `test_validate_key_only_accepts_post` | Verify only POST requests accepted |

### 5. TestEncryptionKeyEndpoint (8 tests)
Tests for GET /api/encryption-key (encryption key management)

| Test Name | Purpose |
|-----------|---------|
| `test_encryption_key_endpoint_exists` | Verify endpoint exists |
| `test_encryption_key_returns_200` | Verify 200 status returned |
| `test_encryption_key_returns_json` | Verify JSON response format |
| `test_encryption_key_has_key_field` | Verify 'key' field in response |
| `test_encryption_key_is_base64` | Verify key is base64 encoded |
| `test_encryption_key_only_accepts_get` | Verify only GET requests accepted |
| `test_encryption_key_non_empty` | Verify key is non-empty |
| `test_encryption_key_consistent` | Verify key consistency across calls |

### 6. TestCSPReportEndpoint (8 tests)
Tests for POST /api/csp-report (CSP violation reporting)

| Test Name | Purpose |
|-----------|---------|
| `test_csp_report_endpoint_exists` | Verify endpoint exists |
| `test_csp_report_accepts_json` | Verify JSON input accepted |
| `test_csp_report_requires_json` | Verify JSON content type required |
| `test_csp_report_accepts_violations` | Verify CSP violations accepted |
| `test_csp_report_handles_empty_payload` | Handle empty payload |
| `test_csp_report_returns_success` | Verify success response |
| `test_csp_report_only_accepts_post` | Verify only POST requests accepted |
| `test_csp_report_with_complex_violation` | Handle complex violations |

### 7. TestDownloadRoute (9 tests)
Tests for GET /download/<filename> (file download)

| Test Name | Purpose |
|-----------|---------|
| `test_download_endpoint_exists` | Verify endpoint exists |
| `test_download_requires_valid_filename` | Verify filename validation |
| `test_download_rejects_path_traversal` | Prevent path traversal attacks |
| `test_download_handles_missing_file` | Handle missing files |
| `test_download_accepts_pptx_files` | Support PPTX downloads |
| `test_download_rejects_suspicious_filenames` | Reject suspicious filenames |
| `test_download_only_accepts_get` | Verify only GET requests accepted |
| `test_download_filename_must_be_present` | Verify filename required |
| `test_download_accepts_pdf_files` | Support PDF downloads |

### 8. TestSecurityHeaders (2 tests)
Tests for security headers and HTTPS enforcement

| Test Name | Purpose |
|-----------|---------|
| `test_index_has_security_headers` | Verify security headers present |
| `test_upload_validates_content_type` | Verify content type validation |

### 9. TestErrorHandling (3 tests)
Tests for error handling across endpoints

| Test Name | Purpose |
|-----------|---------|
| `test_nonexistent_endpoint_returns_404` | Verify 404 for missing endpoints |
| `test_invalid_method_returns_405` | Verify 405 for invalid methods |
| `test_malformed_json_handled` | Handle malformed JSON |

### 10. TestEndpointIntegration (3 tests)
Integration tests across multiple endpoints

| Test Name | Purpose |
|-----------|---------|
| `test_auth_flow_sequence` | Test typical auth flow sequence |
| `test_upload_with_encryption_flow` | Test upload with encryption |
| `test_encryption_key_for_upload_flow` | Test encryption key usage in upload |

## Test Fixtures

The test file uses pytest fixtures for clean, reusable test setup:

```python
@pytest.fixture
def client():
    """Flask test client for making HTTP requests"""
    # Configured with TESTING=True and ENV='testing'
    
@pytest.fixture
def app_context():
    """Flask application context for operations requiring app context"""
    # For operations like url_for() that need app context
```

## Usage

### Running All Tests
```bash
pytest tests/integration/test_api_endpoints.py -v
```

### Running Specific Test Class
```bash
pytest tests/integration/test_api_endpoints.py::TestIndexRoute -v
```

### Running Specific Test
```bash
pytest tests/integration/test_api_endpoints.py::TestIndexRoute::test_index_page_loads -v
```

### Running with Coverage
```bash
pytest tests/integration/test_api_endpoints.py --cov=file_to_slides --cov-report=html
```

### Running Tests with Output
```bash
pytest tests/integration/test_api_endpoints.py -v -s
```

## Test Dependencies

The test file requires:
- **pytest**: Testing framework
- **unittest.mock**: For mocking external APIs
- **json**: JSON handling
- **tempfile**: Temporary file handling
- **Flask**: Web framework (imported via conftest.py)

Install test dependencies:
```bash
pip install pytest pytest-cov
```

## Mocking Strategy

Tests use `unittest.mock` to mock external dependencies:

```python
# Mock Google API
with patch('file_to_slides.fetch_google_doc_content') as mock_fetch:
    mock_fetch.return_value = ('Test content', 'Test Title')

# Mock API key decryption
with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
    mock_decrypt.return_value = 'decrypted-key'

# Mock Google OAuth flow
with patch('file_to_slides.Flow.from_client_config') as mock_flow:
    mock_flow_instance = MagicMock()
    mock_flow.return_value = mock_flow_instance
```

## Test Coverage Details

### API Endpoints Tested
✓ GET /                            - Index page
✓ POST /upload                     - Document upload/conversion
✓ GET /auth/google                 - OAuth initiation
✓ GET /oauth2callback              - OAuth callback
✓ GET /api/google-config           - Google config endpoint
✓ POST /api/validate-key           - API key validation
✓ GET /api/encryption-key          - Encryption key retrieval
✓ POST /api/csp-report             - CSP violation reporting
✓ GET /download/<filename>         - File download

### Test Categories
- ✓ Endpoint existence (9 tests)
- ✓ HTTP method validation (6 tests)
- ✓ Request parameter validation (15 tests)
- ✓ Response format validation (10 tests)
- ✓ Error handling (12 tests)
- ✓ Security validation (8 tests)
- ✓ Integration flows (3 tests)
- ✓ Edge cases (5 tests)

## Integration with CI/CD

To run tests in CI/CD pipeline:

```bash
#!/bin/bash
cd /home/user/slidegenerator

# Run API endpoint tests
pytest tests/integration/test_api_endpoints.py -v --junitxml=test_results.xml

# Check exit code
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi

echo "All API endpoint tests passed!"
```

## Continuous Improvement

To add new tests:

1. Identify the new endpoint or scenario
2. Add test method to appropriate test class:
   ```python
   def test_new_feature(self, client):
       """Description of what is being tested"""
       response = client.get('/new/endpoint')
       assert response.status_code == 200
   ```
3. Run tests to verify: `pytest tests/integration/test_api_endpoints.py::TestClass::test_new_feature -v`
4. Update this documentation with new test details

## Notes

- Tests use mocking to avoid external API calls
- Tests are isolated and can run in any order
- All tests follow naming convention: `test_<specific_behavior>`
- Fixtures provide setup/teardown for clean tests
- Tests verify both success and error cases

## File Location

**Created:** `/home/user/slidegenerator/tests/integration/test_api_endpoints.py`
**Size:** 650 lines
**Total Tests:** 68
**Test Classes:** 10
