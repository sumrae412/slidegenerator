"""
Pytest configuration and shared fixtures for slidegenerator tests.

This module provides reusable fixtures for testing Flask app, Google Docs API,
Claude API, document parsing, and file handling.
"""
import pytest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from file_to_slides import app, DocumentParser


# ============================================================================
# FLASK APP FIXTURES
# ============================================================================

@pytest.fixture
def flask_app():
    """
    Fixture that provides the Flask application instance configured for testing.

    Sets up the app with TEST configuration (debug mode, testing mode enabled).
    Use this when you need the app instance directly.

    Yields:
        Flask: Configured Flask application instance
    """
    app.config['TESTING'] = True
    app.config['ENV'] = 'testing'
    yield app


@pytest.fixture
def client(flask_app):
    """
    Fixture that provides a Flask test client for making requests.

    This is the primary fixture for testing Flask routes and endpoints.
    Use this to test HTTP requests/responses without running a live server.

    Example:
        response = client.get('/api/health')
        assert response.status_code == 200

    Yields:
        FlaskClient: Test client for making HTTP requests
    """
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def app_context(flask_app):
    """
    Fixture that provides an application context for Flask operations.

    Some Flask operations (like accessing g, url_for) require an active
    application context. Use this when you need context but aren't making
    HTTP requests.

    Example:
        with app_context:
            url = url_for('index')

    Yields:
        AppContext: Flask application context
    """
    with flask_app.app_context():
        yield flask_app


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_google_docs():
    """
    Fixture that provides a mocked Google Docs API client.

    Returns a Mock object configured to simulate Google Docs API responses
    for document retrieval, metadata access, etc. Use when you need to test
    code that calls the Google Docs API without making actual API calls.

    Returns:
        Mock: Mocked Google Docs API client
    """
    mock_client = Mock()

    # Mock document retrieval
    mock_client.documents.return_value.get.return_value.execute.return_value = {
        'documentId': 'test-doc-id-12345',
        'title': 'Sample Test Document',
        'body': {
            'content': [
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Sample document text',
                                    'textStyle': {}
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }

    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """
    Fixture that provides a mocked Anthropic (Claude) API client.

    Returns a Mock object configured to simulate Claude API responses
    for message generation. Use when testing code that calls Claude API
    for bullet point generation or content processing.

    Returns:
        Mock: Mocked Anthropic/Claude API client
    """
    mock_client = Mock()

    # Mock message creation
    mock_response = Mock()
    mock_response.content = [Mock(text="- First bullet point\n- Second bullet point")]
    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_openai_client():
    """
    Fixture that provides a mocked OpenAI API client.

    Returns a Mock object configured to simulate OpenAI API responses.
    Use when testing code that integrates with OpenAI (if applicable).

    Returns:
        Mock: Mocked OpenAI API client
    """
    mock_client = Mock()

    # Mock chat completion
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Generated content"))]
    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_document():
    """
    Fixture that provides sample document content for testing.

    Returns a dictionary simulating Google Docs API response structure
    with typical document content including headings, paragraphs, and formatting.

    Returns:
        dict: Google Docs-formatted document with headings and content
    """
    return {
        'documentId': 'test-doc-123',
        'title': 'Test Document',
        'body': {
            'content': [
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Introduction to Python',
                                    'textStyle': {'bold': True}
                                }
                            }
                        ],
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_1'
                        }
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Python is a versatile programming language used for web development, data science, and automation. It emphasizes code readability and has a simple syntax that makes it beginner-friendly.',
                                    'textStyle': {}
                                }
                            }
                        ]
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Key Features',
                                    'textStyle': {'bold': True}
                                }
                            }
                        ],
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_2'
                        }
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {
                                'textRun': {
                                    'content': 'Easy to learn and use. Extensive library ecosystem. Strong community support.',
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
def sample_bullets():
    """
    Fixture that provides sample bullet point content for testing.

    Returns a list of pre-generated bullet points in various formats
    for testing bullet formatting, parsing, and rendering.

    Returns:
        list: Sample bullet points with different lengths and styles
    """
    return [
        "Python is a high-level, interpreted programming language",
        "Supports multiple programming paradigms including object-oriented and functional",
        "Extensive standard library reduces need for external dependencies",
        "Active community with thousands of third-party packages available via PyPI",
        "Used across industries: web development, data science, machine learning, automation"
    ]


@pytest.fixture
def google_doc_url():
    """
    Fixture that provides a valid Google Docs URL format for testing.

    Returns a realistic Google Docs URL that can be used for testing
    URL parsing, validation, and document ID extraction.

    Returns:
        str: Valid Google Docs URL format
    """
    return "https://docs.google.com/document/d/1BxiMVs0XRA5nFMweSWJfxlWGgkTvP0sB3H9lBDJLSgc/edit"


# ============================================================================
# PARSER FIXTURES
# ============================================================================

@pytest.fixture
def parser():
    """
    Fixture that provides a DocumentParser instance for testing.

    Creates a parser without API credentials for testing parsing logic
    and document structure extraction. Use this for testing the parser's
    ability to extract content, headings, and structure.

    Returns:
        DocumentParser: Configured document parser instance
    """
    parser_instance = DocumentParser()
    return parser_instance


@pytest.fixture
def parser_with_api_key():
    """
    Fixture that provides a DocumentParser instance with mock API credentials.

    Creates a parser configured with mock Anthropic API credentials for testing
    AI-powered bullet generation. The API key is a test placeholder.

    Returns:
        DocumentParser: Configured parser with mock API credentials
    """
    parser_instance = DocumentParser()
    parser_instance.api_key = "test-api-key-12345"
    return parser_instance


# ============================================================================
# TEMP FILE FIXTURES
# ============================================================================

@pytest.fixture
def temp_pptx_file():
    """
    Fixture that provides a temporary PPTX file for testing.

    Creates a temporary file with .pptx extension that can be used
    for testing PowerPoint generation and file operations. The file
    is automatically cleaned up after the test.

    Example:
        with temp_pptx_file as temp_path:
            # Use temp_path for file operations
            assert os.path.exists(temp_path)

    Yields:
        str: Path to temporary PPTX file
    """
    # Create temporary file with .pptx extension
    with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as tmp:
        temp_path = tmp.name

    yield temp_path

    # Cleanup: remove temporary file after test
    if os.path.exists(temp_path):
        os.remove(temp_path)
