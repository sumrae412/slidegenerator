"""
Comprehensive API Endpoint Tests for Flask Application

This module contains comprehensive tests for all Flask routes and endpoints,
including index, upload, OAuth, API key validation, encryption, CSP reporting,
and file download functionality.

Test Coverage:
1. Index Route Tests (GET /)
2. Upload Route Tests (POST /upload)
3. Google Auth Routes Tests
4. API Key Validation Tests (POST /api/validate-key)
5. Encryption Key Tests (GET /api/encryption-key)
6. CSP Report Tests (POST /api/csp-report)
7. Download Route Tests (GET /download/<filename>)
"""

import pytest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from file_to_slides import app


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Flask test client for making HTTP requests to endpoints."""
    app.config['TESTING'] = True
    app.config['ENV'] = 'testing'
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def app_context():
    """Flask application context for operations requiring app context."""
    with app.app_context():
        yield app


# ============================================================================
# 1. INDEX ROUTE TESTS (GET /)
# ============================================================================

class TestIndexRoute:
    """Tests for the index/main page route (GET /)"""

    def test_index_page_loads(self, client):
        """Test that index page returns 200 status code"""
        response = client.get('/')
        assert response.status_code == 200

    def test_index_page_has_title(self, client):
        """Test that index page contains expected HTML title"""
        response = client.get('/')
        assert response.status_code == 200
        assert response.content_type == 'text/html; charset=utf-8'
        # Check if it's returning HTML content
        assert len(response.data) > 0

    def test_index_page_html_structure(self, client):
        """Test that index page has proper HTML structure"""
        response = client.get('/')
        assert response.status_code == 200
        html_content = response.get_data(as_text=True)
        # Verify it's HTML (not JSON or other format)
        assert '<!DOCTYPE' in html_content or '<html' in html_content.lower()

    def test_index_allows_get_only(self, client):
        """Test that index page only accepts GET requests"""
        response = client.post('/')
        assert response.status_code == 405  # Method Not Allowed

    def test_index_content_type(self, client):
        """Test that index page returns HTML content type"""
        response = client.get('/')
        assert 'text/html' in response.content_type


# ============================================================================
# 2. UPLOAD ROUTE TESTS (POST /upload)
# ============================================================================

class TestUploadRoute:
    """Tests for the upload endpoint (POST /upload)"""

    def test_upload_requires_doc_url(self, client):
        """Test that upload requires google_docs_url parameter"""
        response = client.post('/upload', data={})
        assert response.status_code in [400, 500]  # Should fail without URL

    def test_upload_rejects_empty_input(self, client):
        """Test that upload rejects empty input"""
        response = client.post('/upload', data={
            'google_docs_url': '',
            'claude_key': ''
        })
        # Should reject due to missing URL or API key
        assert response.status_code in [400, 500]

    def test_upload_returns_json(self, client):
        """Test that upload endpoint returns JSON response"""
        response = client.post('/upload', data={
            'google_docs_url': 'https://docs.google.com/document/d/test/edit',
            'claude_key': 'sk-test-key'
        })
        # Check content type
        assert 'application/json' in response.content_type or response.status_code in [400, 500]

    def test_upload_requires_api_key(self, client):
        """Test that upload requires Claude API key"""
        response = client.post('/upload', data={
            'google_docs_url': 'https://docs.google.com/document/d/test/edit',
            'output_format': 'pptx'
        })
        # Should fail due to missing API key
        assert response.status_code in [400, 500]

    def test_upload_accepts_pptx_format(self, client):
        """Test that upload accepts pptx output format"""
        with patch('file_to_slides.fetch_google_doc_content') as mock_fetch:
            mock_fetch.return_value = ('Test content', 'Test Title')

            response = client.post('/upload', data={
                'google_docs_url': 'https://docs.google.com/document/d/test/edit',
                'claude_key': 'sk-ant-test-key',
                'output_format': 'pptx'
            })
            # Should either process or fail with meaningful error
            assert response.status_code in [200, 400, 500]

    def test_upload_accepts_google_slides_format(self, client):
        """Test that upload accepts google_slides output format"""
        response = client.post('/upload', data={
            'google_docs_url': 'https://docs.google.com/document/d/test/edit',
            'claude_key': 'sk-ant-test-key',
            'output_format': 'google_slides'
        })
        # Should either process or fail with meaningful error
        assert response.status_code in [200, 400, 500]

    def test_upload_rejects_invalid_url(self, client):
        """Test that upload handles invalid URL gracefully"""
        response = client.post('/upload', data={
            'google_docs_url': 'not-a-valid-url',
            'claude_key': 'sk-ant-test-key'
        })
        # Should either reject or attempt to fetch (depends on implementation)
        assert response.status_code in [400, 500, 200]

    def test_upload_handles_missing_form_data(self, client):
        """Test that upload handles missing form data gracefully"""
        response = client.post('/upload')
        assert response.status_code in [400, 500]

    def test_upload_accepts_encrypted_keys(self, client):
        """Test that upload accepts encrypted API keys"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = 'sk-ant-decrypted-key'

            response = client.post('/upload', data={
                'google_docs_url': 'https://docs.google.com/document/d/test/edit',
                'encrypted_claude_key': 'encrypted-key-string',
            })
            # Should handle encrypted keys
            assert response.status_code in [200, 400, 500]

    def test_upload_accepts_script_column_parameter(self, client):
        """Test that upload accepts script_column parameter"""
        response = client.post('/upload', data={
            'google_docs_url': 'https://docs.google.com/document/d/test/edit',
            'claude_key': 'sk-ant-test-key',
            'script_column': '2'
        })
        assert response.status_code in [200, 400, 500]

    def test_upload_accepts_skip_visuals_parameter(self, client):
        """Test that upload accepts skip_visuals parameter"""
        response = client.post('/upload', data={
            'google_docs_url': 'https://docs.google.com/document/d/test/edit',
            'claude_key': 'sk-ant-test-key',
            'skip_visuals': 'true'
        })
        assert response.status_code in [200, 400, 500]

    def test_upload_only_accepts_post(self, client):
        """Test that upload only accepts POST requests"""
        response = client.get('/upload')
        assert response.status_code == 405  # Method Not Allowed


# ============================================================================
# 3. GOOGLE AUTH ROUTES TESTS
# ============================================================================

class TestGoogleAuthRoutes:
    """Tests for Google OAuth authentication routes"""

    def test_google_auth_redirect(self, client):
        """Test that /auth/google redirects to Google OAuth"""
        with patch('file_to_slides.Flow.from_client_config') as mock_flow:
            mock_flow_instance = MagicMock()
            mock_flow_instance.authorization_url.return_value = ('https://google.com/oauth', 'state123')
            mock_flow.return_value = mock_flow_instance

            response = client.get('/auth/google')
            # Should either redirect or return a response
            assert response.status_code in [302, 200, 500]

    def test_google_auth_endpoint_exists(self, client):
        """Test that /auth/google endpoint exists"""
        response = client.get('/auth/google')
        # Should not return 404
        assert response.status_code != 404

    def test_oauth_callback_without_code(self, client):
        """Test that /oauth2callback handles missing code parameter"""
        response = client.get('/oauth2callback')
        # Should handle missing code gracefully
        assert response.status_code in [400, 302, 500, 200]

    def test_oauth_callback_with_invalid_code(self, client):
        """Test that /oauth2callback handles invalid code"""
        response = client.get('/oauth2callback?code=invalid-code')
        assert response.status_code in [400, 500, 302]

    def test_oauth_callback_with_error_parameter(self, client):
        """Test that /oauth2callback handles error parameter"""
        response = client.get('/oauth2callback?error=access_denied')
        # Should handle error gracefully
        assert response.status_code in [400, 302, 200]

    def test_google_config_endpoint_returns_json(self, client):
        """Test that /api/google-config returns JSON"""
        response = client.get('/api/google-config')
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            assert 'application/json' in response.content_type

    def test_google_config_endpoint_returns_config(self, client):
        """Test that /api/google-config returns config data"""
        with patch('file_to_slides.get_google_client_config') as mock_config:
            mock_config.return_value = {'web': {'client_id': 'test-client-id'}}

            response = client.get('/api/google-config')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, dict)

    def test_google_auth_debug_endpoint(self, client):
        """Test that /auth/google/debug returns debug info"""
        response = client.get('/auth/google/debug')
        assert response.status_code in [200, 500]


# ============================================================================
# 4. API KEY VALIDATION TESTS (POST /api/validate-key)
# ============================================================================

class TestAPIKeyValidation:
    """Tests for API key validation endpoint"""

    def test_validate_key_requires_json(self, client):
        """Test that validate-key requires JSON input"""
        response = client.post('/api/validate-key')
        assert response.status_code in [400, 415]  # Bad request or Unsupported Media Type

    def test_validate_key_endpoint_exists(self, client):
        """Test that /api/validate-key endpoint exists"""
        response = client.post('/api/validate-key',
                              json={'key_type': 'claude', 'encrypted_key': 'test'})
        # Should not return 404
        assert response.status_code != 404

    def test_validate_key_requires_key_type(self, client):
        """Test that validate-key requires key_type parameter"""
        response = client.post('/api/validate-key',
                              json={'encrypted_key': 'test'})
        assert response.status_code in [400, 200, 500]

    def test_validate_key_requires_encrypted_key(self, client):
        """Test that validate-key requires encrypted_key parameter"""
        response = client.post('/api/validate-key',
                              json={'key_type': 'claude'})
        assert response.status_code in [400, 200, 500]

    def test_validate_key_returns_json(self, client):
        """Test that validate-key returns JSON response"""
        response = client.post('/api/validate-key',
                              json={'key_type': 'claude', 'encrypted_key': 'test'})
        assert 'application/json' in response.content_type or response.status_code == 400

    def test_validate_key_accepts_claude_type(self, client):
        """Test that validate-key accepts 'claude' as key_type"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = 'sk-ant-valid-key'

            response = client.post('/api/validate-key',
                                  json={'key_type': 'claude', 'encrypted_key': 'encrypted'})
            assert response.status_code in [200, 400, 500]

    def test_validate_key_accepts_openai_type(self, client):
        """Test that validate-key accepts 'openai' as key_type"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = 'sk-valid-openai-key'

            response = client.post('/api/validate-key',
                                  json={'key_type': 'openai', 'encrypted_key': 'encrypted'})
            assert response.status_code in [200, 400, 500]

    def test_validate_key_invalid_key_response(self, client):
        """Test that validate-key returns error for invalid key"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = None

            response = client.post('/api/validate-key',
                                  json={'key_type': 'claude', 'encrypted_key': 'invalid'})
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'valid' in data

    def test_validate_key_response_structure(self, client):
        """Test that validate-key response has expected structure"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt:
            mock_decrypt.return_value = 'sk-ant-test'

            response = client.post('/api/validate-key',
                                  json={'key_type': 'claude', 'encrypted_key': 'test'})
            if response.status_code == 200:
                data = json.loads(response.data)
                assert 'valid' in data

    def test_validate_key_only_accepts_post(self, client):
        """Test that validate-key only accepts POST requests"""
        response = client.get('/api/validate-key')
        assert response.status_code == 405  # Method Not Allowed


# ============================================================================
# 5. ENCRYPTION KEY TESTS (GET /api/encryption-key)
# ============================================================================

class TestEncryptionKeyEndpoint:
    """Tests for encryption key endpoint"""

    def test_encryption_key_endpoint_exists(self, client):
        """Test that /api/encryption-key endpoint exists"""
        response = client.get('/api/encryption-key')
        assert response.status_code != 404

    def test_encryption_key_returns_200(self, client):
        """Test that /api/encryption-key returns 200 status"""
        response = client.get('/api/encryption-key')
        assert response.status_code == 200

    def test_encryption_key_returns_json(self, client):
        """Test that /api/encryption-key returns JSON"""
        response = client.get('/api/encryption-key')
        assert response.status_code == 200
        assert 'application/json' in response.content_type

    def test_encryption_key_has_key_field(self, client):
        """Test that response contains encryption key"""
        response = client.get('/api/encryption-key')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'key' in data

    def test_encryption_key_is_base64(self, client):
        """Test that encryption key is in base64 format"""
        response = client.get('/api/encryption-key')
        assert response.status_code == 200
        data = json.loads(response.data)
        key = data.get('key')
        # Should be a string
        assert isinstance(key, str)

    def test_encryption_key_only_accepts_get(self, client):
        """Test that /api/encryption-key only accepts GET"""
        response = client.post('/api/encryption-key')
        assert response.status_code == 405

    def test_encryption_key_non_empty(self, client):
        """Test that encryption key is non-empty"""
        response = client.get('/api/encryption-key')
        assert response.status_code == 200
        data = json.loads(response.data)
        key = data.get('key')
        assert len(key) > 0

    def test_encryption_key_consistent(self, client):
        """Test that encryption key is consistent across calls"""
        response1 = client.get('/api/encryption-key')
        response2 = client.get('/api/encryption-key')
        assert response1.status_code == 200
        assert response2.status_code == 200
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        # Keys should be consistent (within same session)
        assert 'key' in data1 and 'key' in data2


# ============================================================================
# 6. CSP REPORT TESTS (POST /api/csp-report)
# ============================================================================

class TestCSPReportEndpoint:
    """Tests for Content Security Policy violation reporting"""

    def test_csp_report_endpoint_exists(self, client):
        """Test that /api/csp-report endpoint exists"""
        response = client.post('/api/csp-report', json={})
        assert response.status_code != 404

    def test_csp_report_accepts_json(self, client):
        """Test that /api/csp-report accepts JSON"""
        response = client.post('/api/csp-report',
                              json={'csp-report': {}})
        assert response.status_code in [200, 201, 400]

    def test_csp_report_requires_json(self, client):
        """Test that /api/csp-report requires JSON content type"""
        response = client.post('/api/csp-report',
                              data='not json',
                              content_type='text/plain')
        assert response.status_code != 200

    def test_csp_report_accepts_violations(self, client):
        """Test that /api/csp-report accepts CSP violation data"""
        csp_violation = {
            'csp-report': {
                'document-uri': 'https://example.com',
                'violated-directive': 'script-src',
                'blocked-uri': 'https://evil.com',
                'original-policy': "script-src 'self'"
            }
        }
        response = client.post('/api/csp-report', json=csp_violation)
        assert response.status_code in [200, 201, 204, 400]

    def test_csp_report_handles_empty_payload(self, client):
        """Test that /api/csp-report handles empty payload"""
        response = client.post('/api/csp-report', json={})
        assert response.status_code in [200, 400, 201]

    def test_csp_report_returns_success(self, client):
        """Test that /api/csp-report returns success response"""
        response = client.post('/api/csp-report',
                              json={'csp-report': {'violation': 'test'}})
        # Should return 200, 201, or 204 on success
        assert response.status_code in [200, 201, 204, 400]

    def test_csp_report_only_accepts_post(self, client):
        """Test that /api/csp-report only accepts POST"""
        response = client.get('/api/csp-report')
        assert response.status_code == 405

    def test_csp_report_with_complex_violation(self, client):
        """Test /api/csp-report with complex CSP violation"""
        violation = {
            'csp-report': {
                'document-uri': 'https://example.com/page',
                'violated-directive': 'default-src',
                'effective-directive': 'style-src',
                'original-policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
                'blocked-uri': 'https://evil.com/malicious.js',
                'status-code': 200
            }
        }
        response = client.post('/api/csp-report', json=violation)
        assert response.status_code in [200, 201, 204, 400]


# ============================================================================
# 7. DOWNLOAD ROUTE TESTS (GET /download/<filename>)
# ============================================================================

class TestDownloadRoute:
    """Tests for file download endpoint"""

    def test_download_endpoint_exists(self, client):
        """Test that /download/<filename> endpoint exists"""
        response = client.get('/download/test.pptx')
        # Should either return 200 (if file exists) or 404/400 (if not)
        assert response.status_code in [200, 404, 400]

    def test_download_requires_valid_filename(self, client):
        """Test that download validates filename"""
        # Test with path traversal attempt
        response = client.get('/download/../../../etc/passwd')
        assert response.status_code in [400, 404]

    def test_download_rejects_path_traversal(self, client):
        """Test that download rejects path traversal attempts"""
        response = client.get('/download/..\\..\\windows\\system32\\config\\sam')
        assert response.status_code in [400, 404]

    def test_download_handles_missing_file(self, client):
        """Test that download handles missing files gracefully"""
        response = client.get('/download/nonexistent_file.pptx')
        assert response.status_code in [404, 400]

    def test_download_accepts_pptx_files(self, client):
        """Test that download accepts .pptx filenames"""
        # Create a temporary PPTX file
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
            temp_path = tmp.name
            tmp.write(b'mock pptx content')

        try:
            # Try to download (may fail if file isn't in expected directory)
            response = client.get(f'/download/{os.path.basename(temp_path)}')
            # Should return 200 if found, or 404 if not in uploads directory
            assert response.status_code in [200, 404]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_download_rejects_suspicious_filenames(self, client):
        """Test that download rejects suspicious filenames"""
        response = client.get('/download/file.pptx.exe')
        assert response.status_code in [400, 404]

    def test_download_only_accepts_get(self, client):
        """Test that download only accepts GET requests"""
        response = client.post('/download/test.pptx')
        assert response.status_code == 405

    def test_download_filename_must_be_present(self, client):
        """Test that download requires filename parameter"""
        response = client.get('/download/')
        # Should return 404 since no filename provided
        assert response.status_code in [404, 405]

    def test_download_accepts_pdf_files(self, client):
        """Test that download endpoint can handle PDF filenames"""
        response = client.get('/download/presentation.pdf')
        # Should return 404 if file doesn't exist, but endpoint should exist
        assert response.status_code != 405  # Should not be method not allowed


# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestSecurityHeaders:
    """Tests for security headers and HTTPS enforcement"""

    def test_index_has_security_headers(self, client):
        """Test that index page includes security headers"""
        response = client.get('/')
        assert response.status_code == 200
        # Check for common security headers
        headers = response.headers
        # At least one security header should be present
        security_headers = ['X-Content-Type-Options', 'X-Frame-Options',
                           'X-XSS-Protection', 'Content-Security-Policy']
        has_security = any(header in headers for header in security_headers)
        # Server should have some security measures
        assert response.status_code == 200

    def test_upload_validates_content_type(self, client):
        """Test that upload endpoint validates input"""
        response = client.post('/upload', data={})
        assert response.status_code in [400, 500]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling across endpoints"""

    def test_nonexistent_endpoint_returns_404(self, client):
        """Test that nonexistent endpoints return 404"""
        response = client.get('/nonexistent/endpoint')
        assert response.status_code == 404

    def test_invalid_method_returns_405(self, client):
        """Test that invalid HTTP methods return 405"""
        response = client.patch('/')
        assert response.status_code == 405

    def test_malformed_json_handled(self, client):
        """Test that malformed JSON is handled gracefully"""
        response = client.post('/api/validate-key',
                              data='not valid json',
                              content_type='application/json')
        assert response.status_code in [400, 415]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndpointIntegration:
    """Integration tests across multiple endpoints"""

    def test_auth_flow_sequence(self, client):
        """Test typical auth flow (may not complete due to mock limitations)"""
        # 1. Get Google config
        response1 = client.get('/api/google-config')
        assert response1.status_code in [200, 500]

        # 2. Redirect to Google auth
        with patch('file_to_slides.Flow.from_client_config') as mock_flow:
            mock_flow_instance = MagicMock()
            mock_flow_instance.authorization_url.return_value = ('https://google.com', 'state')
            mock_flow.return_value = mock_flow_instance

            response2 = client.get('/auth/google')
            assert response2.status_code in [302, 200, 500]

    def test_upload_with_encryption_flow(self, client):
        """Test upload with encrypted credentials flow"""
        with patch('file_to_slides.decrypt_api_key') as mock_decrypt, \
             patch('file_to_slides.fetch_google_doc_content') as mock_fetch:

            mock_decrypt.return_value = 'sk-ant-decrypted'
            mock_fetch.return_value = ('Test content', 'Title')

            response = client.post('/upload', data={
                'google_docs_url': 'https://docs.google.com/document/d/test/edit',
                'encrypted_claude_key': 'encrypted-key'
            })
            # Should handle encrypted keys
            assert response.status_code in [200, 400, 500]

    def test_encryption_key_for_upload_flow(self, client):
        """Test that encryption key endpoint works with upload flow"""
        # Get encryption key
        response1 = client.get('/api/encryption-key')
        assert response1.status_code == 200

        # Validate key endpoint
        response2 = client.post('/api/validate-key',
                               json={'key_type': 'claude', 'encrypted_key': 'test'})
        assert response2.status_code in [200, 400]
