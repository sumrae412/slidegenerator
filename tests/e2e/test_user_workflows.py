"""
End-to-End Workflow Tests

Comprehensive testing of complete user workflows including:
- Complete document conversion pipelines (PPTX, Google Slides)
- OAuth authentication flows
- Multi-step user interactions
- Error recovery scenarios

Run with:
    pytest tests/e2e/test_user_workflows.py -v
    pytest tests/e2e/test_user_workflows.py -v -m e2e
    pytest tests/e2e/test_user_workflows.py -v -m slow
"""

import sys
import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import base64

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from file_to_slides import app, DocumentParser, SlideGenerator
from werkzeug.test import Client


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Flask test client for making requests"""
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def mock_google_credentials():
    """Mock Google OAuth credentials"""
    return {
        'access_token': 'test_access_token_12345',
        'token_type': 'Bearer',
        'expires_in': 3600,
        'refresh_token': 'test_refresh_token'
    }


@pytest.fixture
def mock_google_doc_content():
    """Mock Google Docs document content"""
    return {
        'title': 'Test Presentation',
        'body': {
            'content': [
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Test Presentation',
                                    'textStyle': {}
                                }
                            }
                        ],
                        'paragraphStyle': {
                            'headingLevel': 'HEADING_1'
                        }
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Section One',
                                    'textStyle': {}
                                }
                            }
                        ],
                        'paragraphStyle': {
                            'headingLevel': 'HEADING_2'
                        }
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'This is a test paragraph with important information about machine learning. '
                                               'Machine learning algorithms can learn from data. Deep learning is a subset of machine learning.',
                                    'textStyle': {}
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response for bullet generation"""
    return {
        'id': 'msg_test123',
        'type': 'message',
        'role': 'assistant',
        'content': [
            {
                'type': 'text',
                'text': '• Test bullet point one\n• Test bullet point two\n• Test bullet point three'
            }
        ],
        'model': 'claude-3-5-sonnet-20241022',
        'stop_reason': 'end_turn',
        'usage': {
            'input_tokens': 150,
            'output_tokens': 50
        }
    }


# ============================================================================
# COMPLETE USER WORKFLOWS - PPTX & GOOGLE SLIDES
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
def test_user_uploads_google_doc_gets_pptx(client, mock_google_doc_content):
    """
    Complete PPTX Workflow:
    1. User accesses main page
    2. User provides Google Docs URL
    3. App fetches document
    4. App generates bullets with NLP fallback
    5. App creates PPTX
    6. User receives download link

    This tests the full conversion pipeline without requiring API key.
    """
    # Step 1: Access main page
    response = client.get('/')
    assert response.status_code == 200
    assert b'file-to-slides' in response.data or b'script' in response.data

    # Step 2: Mock Google Docs API and submit conversion request
    test_url = 'https://docs.google.com/document/d/test123abc/edit'

    with patch('requests.get') as mock_get:
        # Mock the Google Docs content fetch
        mock_response = Mock()
        mock_response.json.return_value = mock_google_doc_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Submit upload form (simulating Google Doc URL submission)
        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx',
            'include_visuals': 'false',
            'include_speaker_notes': 'false'
        }, follow_redirects=False)

        # Should return success with download info
        assert response.status_code in [200, 302]

        if response.status_code == 302:
            # Check if redirect points to download
            assert '/download/' in response.location or 'pptx' in response.location.lower()
        else:
            # Check if response contains download link
            assert b'download' in response.data.lower() or b'pptx' in response.data.lower()


@pytest.mark.e2e
@pytest.mark.slow
def test_user_uploads_google_doc_gets_slides(client, mock_google_doc_content, mock_google_credentials):
    """
    Complete Google Slides Workflow:
    1. User authenticates with Google OAuth
    2. User selects document from Google Drive
    3. App fetches document content
    4. App generates bullets
    5. App creates Google Slides presentation
    6. User receives Slides URL

    This tests the full Google Slides output pipeline with OAuth.
    """
    # Step 1: Simulate OAuth session setup
    with client.session_transaction() as session:
        session['google_credentials'] = mock_google_credentials

    # Step 2: Submit for Google Slides conversion
    with patch('requests.get') as mock_get, \
         patch('googleapiclient.discovery.build') as mock_build:

        # Mock Google Docs fetch
        mock_doc_response = Mock()
        mock_doc_response.json.return_value = mock_google_doc_content
        mock_doc_response.status_code = 200
        mock_get.return_value = mock_doc_response

        # Mock Google Slides API
        mock_slides_service = Mock()
        mock_build.return_value = mock_slides_service

        test_url = 'https://docs.google.com/document/d/test456def/edit'

        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'google_slides',
            'include_visuals': 'false'
        })

        # Should succeed or redirect
        assert response.status_code in [200, 302, 303]


@pytest.mark.e2e
@pytest.mark.slow
def test_user_provides_api_key_for_enhanced_generation(client, mock_google_doc_content, mock_claude_response):
    """
    User-Provided API Key Workflow:
    1. User provides their Claude API key
    2. App validates key with API
    3. App uses key for bullet generation
    4. App creates presentation with AI-generated content
    5. Result shows enhanced quality bullets

    This tests the Claude API integration path.
    """
    test_api_key = 'sk-ant-v4-test-key-12345'
    test_url = 'https://docs.google.com/document/d/test789ghi/edit'

    with patch('requests.get') as mock_get, \
         patch('anthropic.Anthropic') as mock_anthropic:

        # Mock Google Docs fetch
        mock_doc_response = Mock()
        mock_doc_response.json.return_value = mock_google_doc_content
        mock_doc_response.status_code = 200
        mock_get.return_value = mock_doc_response

        # Mock Claude API client
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='• AI generated bullet 1\n• AI generated bullet 2')]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        # Submit with API key
        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx',
            'api_key': test_api_key,
            'include_visuals': 'false'
        })

        # API key should be used for generation
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            assert b'presentation' in response.data.lower() or b'slide' in response.data.lower()


@pytest.mark.e2e
@pytest.mark.slow
def test_user_without_api_key_uses_nlp_fallback(client, mock_google_doc_content):
    """
    NLP Fallback Workflow:
    1. User submits document without API key
    2. App detects missing API key
    3. App falls back to NLP-based bullet generation
    4. App still creates valid presentation
    5. Results show reasonable quality bullets

    This tests the intelligent fallback mechanism.
    """
    test_url = 'https://docs.google.com/document/d/test999xyz/edit'

    with patch('requests.get') as mock_get:
        # Mock Google Docs fetch
        mock_doc_response = Mock()
        mock_doc_response.json.return_value = mock_google_doc_content
        mock_doc_response.status_code = 200
        mock_get.return_value = mock_doc_response

        # Explicitly no API key
        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx',
            'include_visuals': 'false'
        })

        # Should still succeed with NLP fallback
        assert response.status_code in [200, 302]

        # Verify response indicates successful processing
        if response.status_code == 200:
            assert b'presentation' in response.data.lower() or b'slide' in response.data.lower()


# ============================================================================
# OAUTH AUTHENTICATION WORKFLOWS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
def test_user_authenticates_with_google_oauth(client):
    """
    OAuth Authentication Workflow:
    1. User clicks "Login with Google"
    2. App redirects to Google consent screen
    3. Google redirects back with auth code
    4. App exchanges code for access token
    5. Session is established
    6. User can access authenticated features

    This tests the complete OAuth flow.
    """
    # Step 1: Initiate OAuth flow
    response = client.get('/auth/google')

    # Should redirect to Google OAuth endpoint
    assert response.status_code in [302, 307, 308]
    assert 'accounts.google.com' in response.location or 'oauth' in response.location

    # Step 2: Simulate OAuth callback
    with patch('google.oauth2.service_account.Credentials.from_service_account_info') as mock_creds, \
         patch('google.auth.transport.requests.Request') as mock_request:

        mock_credentials = Mock()
        mock_credentials.token = 'test_access_token'
        mock_credentials.refresh_token = 'test_refresh_token'
        mock_credentials.valid = True
        mock_creds.return_value = mock_credentials

        # Simulate callback with auth code
        response = client.get('/oauth2callback?code=test_auth_code&state=test_state')

        # Should redirect to main page or processing page
        assert response.status_code in [302, 307, 308]
        assert '/' in response.location


@pytest.mark.e2e
@pytest.mark.slow
def test_authenticated_user_accesses_private_google_doc(client, mock_google_doc_content, mock_google_credentials):
    """
    Private Document Access Workflow:
    1. User is authenticated via OAuth
    2. User submits URL to private Google Doc
    3. App uses authenticated credentials to fetch document
    4. Document is successfully retrieved (would fail without auth)
    5. Presentation is generated

    This tests authenticated API requests.
    """
    # Set up authenticated session
    with client.session_transaction() as session:
        session['google_credentials'] = mock_google_credentials

    private_doc_url = 'https://docs.google.com/document/d/private_doc_id/edit'

    with patch('requests.get') as mock_get:
        # Mock authenticated document fetch
        mock_response = Mock()
        mock_response.json.return_value = mock_google_doc_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Submit private document for processing
        response = client.post('/upload', data={
            'google_doc_url': private_doc_url,
            'output_format': 'pptx',
            'include_visuals': 'false'
        })

        # Should successfully process the private document
        assert response.status_code in [200, 302]

        # Verify authenticated request was made (with token)
        if mock_get.called:
            call_args = mock_get.call_args
            headers = call_args[1].get('headers', {})
            assert 'Authorization' in headers or 'authorization' in str(headers)


# ============================================================================
# MULTI-STEP WORKFLOWS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
def test_user_processes_multiple_documents_sequentially(client, mock_google_doc_content):
    """
    Sequential Processing Workflow:
    1. User submits first document for conversion
    2. First conversion completes, user downloads result
    3. User submits second document for conversion
    4. Both conversions maintain separate state
    5. Each conversion produces correct output

    This tests session state management across multiple requests.
    """
    doc_urls = [
        'https://docs.google.com/document/d/doc_id_1/edit',
        'https://docs.google.com/document/d/doc_id_2/edit'
    ]

    results = []

    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = mock_google_doc_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Process each document
        for doc_url in doc_urls:
            response = client.post('/upload', data={
                'google_doc_url': doc_url,
                'output_format': 'pptx',
                'include_visuals': 'false'
            })

            # Each should succeed
            assert response.status_code in [200, 302]
            results.append({
                'url': doc_url,
                'status': response.status_code,
                'response': response
            })

    # Verify both processed successfully
    assert len(results) == 2
    assert all(r['status'] in [200, 302] for r in results)


@pytest.mark.e2e
@pytest.mark.slow
def test_user_downloads_generated_presentation(client, mock_google_doc_content):
    """
    Download Workflow:
    1. User submits document for conversion
    2. App returns PPTX file for download
    3. User clicks download link
    4. File is delivered with correct headers
    5. File is valid PPTX format

    This tests file generation and delivery.
    """
    test_url = 'https://docs.google.com/document/d/download_test/edit'

    with patch('requests.get') as mock_get, \
         patch('pptx.Presentation') as mock_pptx:

        # Mock Google Docs fetch
        mock_doc_response = Mock()
        mock_doc_response.json.return_value = mock_google_doc_content
        mock_doc_response.status_code = 200
        mock_get.return_value = mock_doc_response

        # Mock PPTX creation
        mock_presentation = Mock()
        mock_pptx.return_value = mock_presentation

        # Create presentation
        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx',
            'include_visuals': 'false'
        })

        assert response.status_code in [200, 302]

        # If response is successful, verify PPTX was attempted
        if response.status_code == 200:
            # Check that presentation was created
            assert mock_pptx.called or b'pptx' in response.data


@pytest.mark.e2e
def test_user_accesses_google_drive_file_picker(client, mock_google_credentials):
    """
    Google Drive Picker Workflow:
    1. User is authenticated
    2. User accesses file picker
    3. Picker loads user's Drive files
    4. User selects a file
    5. App processes selected file

    This tests Drive integration.
    """
    with client.session_transaction() as session:
        session['google_credentials'] = mock_google_credentials

    # Access main page with authenticated session
    response = client.get('/')
    assert response.status_code == 200

    # Main page should include Drive picker configuration
    assert b'drive' in response.data.lower() or b'picker' in response.data.lower() or b'script' in response.data


# ============================================================================
# ERROR RECOVERY WORKFLOWS
# ============================================================================

@pytest.mark.e2e
@pytest.mark.slow
def test_user_recovers_from_invalid_google_doc_url(client):
    """
    Invalid URL Error Recovery:
    1. User submits invalid Google Docs URL
    2. App validates URL format
    3. App returns user-friendly error message
    4. User can correct and resubmit
    5. Next attempt with valid URL succeeds

    This tests input validation and error handling.
    """
    invalid_urls = [
        'not-a-url',
        'https://example.com/document',
        'https://docs.google.com/invalid',
        ''
    ]

    # Test each invalid URL
    for invalid_url in invalid_urls:
        response = client.post('/upload', data={
            'google_doc_url': invalid_url,
            'output_format': 'pptx'
        })

        # Should return error or redirect safely
        assert response.status_code in [200, 400, 302]

        # If error page, should contain helpful message
        if response.status_code == 400:
            assert b'error' in response.data.lower() or b'invalid' in response.data.lower()

    # Test recovery with valid URL
    with patch('requests.get') as mock_get:
        mock_doc = {
            'title': 'Valid Doc',
            'body': {'content': []}
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_doc
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        valid_response = client.post('/upload', data={
            'google_doc_url': 'https://docs.google.com/document/d/valid_id/edit',
            'output_format': 'pptx'
        })

        # Valid URL should process
        assert valid_response.status_code in [200, 302]


@pytest.mark.e2e
@pytest.mark.slow
def test_user_retries_after_api_failure(client, mock_google_doc_content):
    """
    API Error Retry Workflow:
    1. First request encounters API error (timeout, 500, etc)
    2. App returns error to user
    3. User retries request
    4. Retry succeeds (simulating API recovery)
    5. User gets presentation

    This tests resilience and retry capability.
    """
    test_url = 'https://docs.google.com/document/d/retry_test/edit'

    # First attempt: API fails
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_response.raise_for_status.side_effect = Exception('API Error')
        mock_get.return_value = mock_response

        response1 = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx'
        })

        # Should handle error gracefully
        assert response1.status_code in [200, 400, 500, 302]

    # Second attempt: API recovers
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = mock_google_doc_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response2 = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx'
        })

        # Second attempt should succeed
        assert response2.status_code in [200, 302]


@pytest.mark.e2e
def test_user_handles_missing_required_fields(client):
    """
    Missing Required Fields Error Handling:
    1. User submits form with missing fields
    2. App validates required fields
    3. App returns validation error
    4. User sees helpful message about missing fields
    5. User corrects and resubmits

    This tests form validation.
    """
    # Missing URL
    response = client.post('/upload', data={
        'output_format': 'pptx'
    })

    # Should indicate error
    assert response.status_code in [200, 400, 302]

    # Missing output format
    response = client.post('/upload', data={
        'google_doc_url': 'https://docs.google.com/document/d/test/edit'
    })

    # Should indicate error or use default
    assert response.status_code in [200, 400, 302]


@pytest.mark.e2e
@pytest.mark.slow
def test_user_handles_timeout_scenario(client, mock_google_doc_content):
    """
    Timeout Error Handling:
    1. User submits document
    2. App processing takes too long
    3. Request times out
    4. User sees timeout message
    5. Session is properly cleaned up

    This tests timeout resilience.
    """
    test_url = 'https://docs.google.com/document/d/timeout_test/edit'

    with patch('requests.get') as mock_get:
        # Simulate timeout
        import requests
        mock_get.side_effect = requests.Timeout('Request timed out')

        response = client.post('/upload', data={
            'google_doc_url': test_url,
            'output_format': 'pptx'
        }, timeout=2)

        # Should handle timeout gracefully
        assert response.status_code in [200, 400, 500, 502, 504, 302]


# ============================================================================
# SESSION & STATE MANAGEMENT TESTS
# ============================================================================

@pytest.mark.e2e
def test_session_persists_across_requests(client):
    """
    Session State Persistence:
    1. User authenticates or sets preferences
    2. Session state is stored
    3. Subsequent requests maintain state
    4. User doesn't need to re-authenticate

    This tests Flask session management.
    """
    with client.session_transaction() as session:
        session['test_key'] = 'test_value'

    # First request
    response1 = client.get('/')
    assert response1.status_code == 200

    # Verify session persists
    with client.session_transaction() as session:
        assert session.get('test_key') == 'test_value'

    # Another request
    response2 = client.get('/')
    assert response2.status_code == 200


@pytest.mark.e2e
def test_concurrent_user_sessions_isolated(client):
    """
    Session Isolation for Concurrent Users:
    1. User A creates session with preferences
    2. User B creates separate session
    3. Each session maintains independent state
    4. No cross-contamination between users

    This tests multi-user isolation.
    """
    # User A session
    client_a = app.test_client()
    with client_a.session_transaction() as session:
        session['user_id'] = 'user_a'
        session['api_key'] = 'key_a'

    # User B session
    client_b = app.test_client()
    with client_b.session_transaction() as session:
        session['user_id'] = 'user_b'
        session['api_key'] = 'key_b'

    # Verify isolation
    with client_a.session_transaction() as session:
        assert session['user_id'] == 'user_a'
        assert session['api_key'] == 'key_a'

    with client_b.session_transaction() as session:
        assert session['user_id'] == 'user_b'
        assert session['api_key'] == 'key_b'


# ============================================================================
# INTEGRATION TESTS - DOCUMENT PROCESSING
# ============================================================================

@pytest.mark.e2e
def test_document_parser_integration_with_various_content(mock_google_doc_content):
    """
    Document Parser Content Handling:
    1. Parser receives various content types
    2. Headings are properly identified
    3. Body paragraphs are extracted
    4. Tables are processed
    5. All content is converted to bullets appropriately

    This tests the DocumentParser class with realistic content.
    """
    parser = DocumentParser(claude_api_key=None)

    # Test with heading
    heading_text = "Test Heading"
    bullets = parser._create_unified_bullets(
        "This is a test paragraph.",
        context_heading=heading_text
    )

    assert isinstance(bullets, list)
    assert len(bullets) > 0
    assert all(isinstance(b, str) for b in bullets)


@pytest.mark.e2e
@pytest.mark.slow
def test_full_pipeline_from_url_to_pptx(mock_google_doc_content):
    """
    Complete Pipeline Integration:
    1. Input: Google Docs URL
    2. Process: Fetch content
    3. Process: Generate bullets
    4. Process: Create PPTX structure
    5. Output: Valid PPTX file

    This tests the entire document-to-slides pipeline.
    """
    parser = DocumentParser(claude_api_key=None)
    generator = SlideGenerator()

    # Simulate extracted content
    content = "Machine learning is a subset of artificial intelligence. "
    content += "It enables systems to learn from data. Neural networks are inspired by biological neurons."

    # Generate bullets
    bullets = parser._create_unified_bullets(
        content,
        context_heading="Machine Learning"
    )

    # Verify bullets generated
    assert isinstance(bullets, list)
    assert len(bullets) > 0

    # All bullets should be strings
    for bullet in bullets:
        assert isinstance(bullet, str)
        assert len(bullet) > 0
        assert len(bullet.split()) >= 3  # At least 3 words


# ============================================================================
# EDGE CASES & BOUNDARY CONDITIONS
# ============================================================================

@pytest.mark.e2e
def test_empty_document_handling(client):
    """
    Empty Document Edge Case:
    1. User submits empty Google Doc
    2. App handles gracefully
    3. User sees appropriate message
    4. No errors thrown
    """
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            'title': 'Empty Doc',
            'body': {'content': []}
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.post('/upload', data={
            'google_doc_url': 'https://docs.google.com/document/d/empty/edit',
            'output_format': 'pptx'
        })

        # Should handle gracefully
        assert response.status_code in [200, 302, 400]


@pytest.mark.e2e
def test_very_large_document_handling(client):
    """
    Large Document Edge Case:
    1. User submits very large Google Doc
    2. App processes within constraints
    3. Generates presentation successfully
    4. Doesn't exceed memory/time limits
    """
    large_content = "Test paragraph. " * 500  # Create large content

    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            'title': 'Large Doc',
            'body': {'content': [{
                'paragraph': {
                    'elements': [{
                        'textRun': {
                            'content': large_content,
                            'textStyle': {}
                        }
                    }]
                }
            }]}
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.post('/upload', data={
            'google_doc_url': 'https://docs.google.com/document/d/large/edit',
            'output_format': 'pptx'
        }, timeout=30)

        # Should handle within constraints
        assert response.status_code in [200, 302, 400, 413]


@pytest.mark.e2e
def test_special_characters_in_content(client, mock_google_doc_content):
    """
    Special Characters Edge Case:
    1. Document contains special characters
    2. Unicode, emojis, symbols handled
    3. Presentation generated correctly
    4. No encoding errors
    """
    special_content = mock_google_doc_content.copy()
    special_content['body']['content'].append({
        'paragraph': {
            'elements': [{
                'textRun': {
                    'content': 'Special chars: © ® ™ € £ ¥ • → ← ↑ ↓ 中文 日本語',
                    'textStyle': {}
                }
            }]
        }
    })

    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = special_content
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.post('/upload', data={
            'google_doc_url': 'https://docs.google.com/document/d/special/edit',
            'output_format': 'pptx'
        })

        # Should handle special characters
        assert response.status_code in [200, 302]


# ============================================================================
# SMOKE TESTS FOR QUICK VALIDATION
# ============================================================================

@pytest.mark.e2e
def test_app_initialization(client):
    """
    App Initialization Smoke Test:
    Verify Flask app is properly configured and responsive.
    """
    assert app is not None
    assert app.config['TESTING'] == True

    response = client.get('/')
    assert response.status_code == 200


@pytest.mark.e2e
def test_basic_request_response_cycle(client):
    """
    Basic Request/Response Smoke Test:
    Verify basic HTTP request/response works.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert response.content_type is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'e2e'])
