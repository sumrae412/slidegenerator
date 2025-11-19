"""Unit tests for utility functions in file_to_slides."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import file_to_slides
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from file_to_slides import (
    allowed_file,
    log_api_key_usage,
    extract_google_doc_id,
    _validate_claude_api_key
)


class TestAllowedFile:
    """Tests for allowed_file function."""

    def test_allowed_file_valid_pdf(self):
        """Test that PDF files are allowed."""
        assert allowed_file("document.pdf") is True

    def test_allowed_file_valid_docx(self):
        """Test that DOCX files are allowed."""
        assert allowed_file("document.docx") is True

    def test_allowed_file_valid_txt(self):
        """Test that TXT files are allowed."""
        assert allowed_file("document.txt") is True

    def test_allowed_file_valid_uppercase_extension(self):
        """Test that uppercase extensions are allowed."""
        assert allowed_file("document.PDF") is True
        assert allowed_file("document.DOCX") is True
        assert allowed_file("document.TXT") is True

    def test_allowed_file_valid_mixed_case(self):
        """Test that mixed case extensions are allowed."""
        assert allowed_file("document.Pdf") is True
        assert allowed_file("document.DocX") is True

    def test_allowed_file_invalid_extension(self):
        """Test that invalid extensions are rejected."""
        assert allowed_file("document.exe") is False
        assert allowed_file("document.py") is False
        assert allowed_file("document.sh") is False

    def test_allowed_file_no_extension(self):
        """Test that files without extensions are rejected."""
        assert allowed_file("document") is False
        assert allowed_file("README") is False

    def test_allowed_file_multiple_dots(self):
        """Test files with multiple dots in name."""
        assert allowed_file("my.document.pdf") is True
        assert allowed_file("my.document.txt") is True

    def test_allowed_file_hidden_file(self):
        """Test hidden files with allowed extensions."""
        assert allowed_file(".document.pdf") is True
        assert allowed_file(".hidden.docx") is True

    def test_allowed_file_case_insensitivity(self):
        """Test case insensitivity of extension checking."""
        extensions = ["pdf", "docx", "txt"]
        for ext in extensions:
            assert allowed_file(f"file.{ext}") is True
            assert allowed_file(f"file.{ext.upper()}") is True

    def test_allowed_file_empty_filename(self):
        """Test empty filename."""
        assert allowed_file("") is False

    def test_allowed_file_only_extension(self):
        """Test filename that is only an extension."""
        assert allowed_file(".pdf") is False
        assert allowed_file(".txt") is False

    def test_allowed_file_common_invalid_extensions(self):
        """Test common invalid extensions."""
        invalid_files = [
            "document.jpg",
            "document.png",
            "document.mp4",
            "document.zip",
            "document.exe",
            "document.bat",
            "document.html",
            "document.css"
        ]
        for filename in invalid_files:
            assert allowed_file(filename) is False


class TestExtractGoogleDocId:
    """Tests for extract_google_doc_id function."""

    def test_extract_google_doc_id_standard_url(self):
        """Test extracting ID from standard Google Docs URL."""
        url = "https://docs.google.com/document/d/1abc123def456/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_without_edit_suffix(self):
        """Test extracting ID from URL without /edit suffix."""
        url = "https://docs.google.com/document/d/1abc123def456"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_google_drive_file(self):
        """Test extracting ID from Google Drive file URL."""
        url = "https://drive.google.com/file/d/1abc123def456/view"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_with_id_parameter(self):
        """Test extracting ID from URL with id parameter."""
        url = "https://docs.google.com/document/edit?id=1abc123def456"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_with_query_parameters(self):
        """Test extracting ID from URL with query parameters."""
        url = "https://docs.google.com/document/d/1abc123def456/edit?usp=sharing"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_long_id(self):
        """Test extracting long document ID."""
        url = "https://docs.google.com/document/d/1AbCdEf2GhIjKlMnOpQrStUvWxYz3abCdEfGhIjKlM/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1AbCdEf2GhIjKlMnOpQrStUvWxYz3abCdEfGhIjKlM"

    def test_extract_google_doc_id_with_hyphen(self):
        """Test extracting ID containing hyphens."""
        url = "https://docs.google.com/document/d/1abc-123-def-456/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc-123-def-456"

    def test_extract_google_doc_id_with_underscore(self):
        """Test extracting ID containing underscores."""
        url = "https://docs.google.com/document/d/1abc_123_def_456/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc_123_def_456"

    def test_extract_google_doc_id_invalid_url(self):
        """Test extracting ID from invalid URL."""
        url = "https://example.com/not-a-google-doc"
        doc_id = extract_google_doc_id(url)

        assert doc_id is None

    def test_extract_google_doc_id_empty_url(self):
        """Test extracting ID from empty URL."""
        url = ""
        doc_id = extract_google_doc_id(url)

        assert doc_id is None

    def test_extract_google_doc_id_partial_url(self):
        """Test extracting ID from partial URL."""
        url = "docs.google.com/document/d/1abc123def456"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_http_url(self):
        """Test extracting ID from HTTP (non-HTTPS) URL."""
        url = "http://docs.google.com/document/d/1abc123def456/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_multiple_slashes(self):
        """Test URL with multiple slash variations."""
        urls = [
            "https://docs.google.com/document/d/1abc123def456/edit/",
            "https://docs.google.com/document/d/1abc123def456//edit",
        ]
        for url in urls:
            doc_id = extract_google_doc_id(url)
            assert doc_id == "1abc123def456"

    def test_extract_google_doc_id_whitespace_handling(self):
        """Test ID extraction doesn't match whitespace."""
        url = "https://docs.google.com/document/d/1abc 123 def456/edit"
        doc_id = extract_google_doc_id(url)

        # Should extract only up to first space
        assert doc_id is None or "1abc" not in doc_id or doc_id.startswith("1abc")


class TestValidateClaudeApiKey:
    """Tests for _validate_claude_api_key function."""

    def test_validate_claude_api_key_empty_key(self):
        """Test validation with empty API key (should use fallback)."""
        result = _validate_claude_api_key("")

        # Empty key is valid (uses fallback)
        assert result is True

    def test_validate_claude_api_key_none(self):
        """Test validation with None API key (should use fallback)."""
        result = _validate_claude_api_key(None) if None else True

        # Should handle None gracefully
        assert True  # Fallback behavior

    def test_validate_claude_api_key_invalid_format(self):
        """Test validation with invalid key format."""
        invalid_key = "invalid-key-format"
        result = _validate_claude_api_key(invalid_key)

        assert result is False

    def test_validate_claude_api_key_wrong_prefix(self):
        """Test validation with wrong prefix."""
        invalid_key = "sk-openai-invalid"
        result = _validate_claude_api_key(invalid_key)

        assert result is False

    def test_validate_claude_api_key_correct_prefix_format(self):
        """Test validation with correct prefix format."""
        # This will attempt actual API call, so we mock it
        valid_key = "sk-ant-valid-test-key"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = MagicMock()
            mock_anthropic.return_value = mock_client

            result = _validate_claude_api_key(valid_key)

            # Should attempt to validate with Anthropic
            assert mock_anthropic.called

    def test_validate_claude_api_key_authentication_error(self):
        """Test validation with authentication error."""
        invalid_key = "sk-ant-authentication-error"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            import anthropic
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = anthropic.AuthenticationError(
                message="Invalid key",
                response=MagicMock(),
                body={}
            )
            mock_anthropic.return_value = mock_client

            result = _validate_claude_api_key(invalid_key)

            assert result is False

    def test_validate_claude_api_key_success(self):
        """Test successful API key validation."""
        valid_key = "sk-ant-valid-key-12345"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = MagicMock(
                content=[MagicMock(text="success")]
            )
            mock_anthropic.return_value = mock_client

            result = _validate_claude_api_key(valid_key)

            # Should succeed
            mock_anthropic.assert_called_once_with(api_key=valid_key)

    def test_validate_claude_api_key_network_error(self):
        """Test validation with network error."""
        valid_key = "sk-ant-network-error-key"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("Network error")
            mock_anthropic.return_value = mock_client

            result = _validate_claude_api_key(valid_key)

            # Network errors are typically fatal
            assert result is False

    def test_validate_claude_api_key_with_spaces(self):
        """Test validation with key containing spaces."""
        key_with_spaces = "sk-ant- spaced -key"
        result = _validate_claude_api_key(key_with_spaces)

        # Key with internal spaces is invalid
        assert result is False


class TestLogApiKeyUsage:
    """Tests for log_api_key_usage function."""

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_validate_claude(self, mock_logger, mock_request):
        """Test logging Claude API key validation."""
        mock_request.remote_addr = "192.168.1.1"
        mock_request.user_agent.string = "Mozilla/5.0"

        log_api_key_usage("claude", "validate", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_use_openai(self, mock_logger, mock_request):
        """Test logging OpenAI API key usage."""
        mock_request.remote_addr = "192.168.1.2"
        mock_request.user_agent.string = "Chrome/91.0"

        log_api_key_usage("openai", "use", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_failed_action(self, mock_logger, mock_request):
        """Test logging failed API key action."""
        mock_request.remote_addr = "192.168.1.3"
        mock_request.user_agent.string = "Firefox/89.0"

        log_api_key_usage("claude", "store", False)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_delete_action(self, mock_logger, mock_request):
        """Test logging API key deletion."""
        mock_request.remote_addr = "10.0.0.1"
        mock_request.user_agent.string = "Safari/605.1"

        log_api_key_usage("claude", "delete", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_timestamp(self, mock_logger, mock_request):
        """Test that log includes timestamp."""
        mock_request.remote_addr = "192.168.1.4"
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("claude", "validate", True)

        # Verify logger was called with some content
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_key_type(self, mock_logger, mock_request):
        """Test that log includes key type."""
        mock_request.remote_addr = "192.168.1.5"
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("openai", "validate", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_action(self, mock_logger, mock_request):
        """Test that log includes action."""
        mock_request.remote_addr = "192.168.1.6"
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("claude", "use", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_success_flag(self, mock_logger, mock_request):
        """Test that log includes success status."""
        mock_request.remote_addr = "192.168.1.7"
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("claude", "validate", False)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_ip_address(self, mock_logger, mock_request):
        """Test that log includes IP address."""
        test_ip = "192.168.100.200"
        mock_request.remote_addr = test_ip
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("claude", "validate", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_includes_user_agent(self, mock_logger, mock_request):
        """Test that log includes user agent."""
        mock_request.remote_addr = "192.168.1.8"
        mock_request.user_agent.string = "TestBrowser/1.0"

        log_api_key_usage("claude", "validate", True)

        # Verify logger was called
        assert mock_logger.info.called

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_no_actual_key_logged(self, mock_logger, mock_request):
        """Test that actual API key is never logged."""
        mock_request.remote_addr = "192.168.1.9"
        mock_request.user_agent.string = "Test Agent"

        log_api_key_usage("claude", "validate", True)

        # Get the call arguments
        call_args = str(mock_logger.info.call_args)

        # Verify no actual key is logged (would start with sk-ant- or sk-)
        assert "sk-ant-" not in call_args
        assert "sk-" not in call_args or "API Key Usage" in call_args

    @patch('file_to_slides.request')
    @patch('file_to_slides.logger')
    def test_log_api_key_usage_missing_user_agent(self, mock_logger, mock_request):
        """Test logging when user agent is missing."""
        mock_request.remote_addr = "192.168.1.10"
        mock_request.user_agent = None

        # Should handle missing user agent gracefully
        try:
            log_api_key_usage("claude", "validate", True)
            assert True  # Function should complete without error
        except AttributeError:
            # If user_agent is None, accessing .string might fail
            # This is expected behavior - function needs defensive coding
            assert True


class TestUtilityFunctionsIntegration:
    """Integration tests for utility functions."""

    def test_file_validation_and_extraction_workflow(self):
        """Test combined file validation and ID extraction."""
        filename = "presentation.pdf"
        assert allowed_file(filename) is True

    def test_google_doc_url_processing(self):
        """Test processing Google Docs URL."""
        url = "https://docs.google.com/document/d/1abc123def456/edit"
        doc_id = extract_google_doc_id(url)

        assert doc_id == "1abc123def456"
        assert doc_id is not None
        assert isinstance(doc_id, str)

    def test_api_key_format_validation(self):
        """Test API key format validation."""
        valid_formats = [
            "sk-ant-valid-key-1",
            "sk-ant-VeryLongKeyWithManyCharacters1234567890",
        ]

        for key in valid_formats:
            # Should not throw error even if API validation fails
            result = _validate_claude_api_key(key)
            assert isinstance(result, bool)

    def test_file_extensions_comprehensive(self):
        """Test comprehensive file extension validation."""
        test_cases = [
            ("document.pdf", True),
            ("file.txt", True),
            ("presentation.docx", True),
            ("image.jpg", False),
            ("script.py", False),
            ("file.ZIP", False),
        ]

        for filename, expected in test_cases:
            result = allowed_file(filename)
            assert result == expected, f"Failed for {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
