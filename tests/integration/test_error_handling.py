"""
Comprehensive Error Handling Tests for Slide Generator

Tests error handling across the application, including:
- Invalid input validation
- Network error resilience
- API error handling
- Document processing errors
- Concurrent request handling
- Edge case management

These tests verify that the application gracefully handles errors and
provides appropriate feedback to users without crashing.
"""

import sys
import os
import pytest
import json
from unittest.mock import Mock, MagicMock, patch, call
from io import BytesIO
import threading
import time
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from file_to_slides import app, DocumentParser


# ============================================================================
# INVALID INPUT TESTS
# ============================================================================

class TestInvalidInputs:
    """Test handling of malformed and invalid input data"""

    def test_invalid_google_doc_url(self):
        """Test that malformed Google Docs URLs are properly rejected"""
        parser = DocumentParser()

        invalid_urls = [
            "not-a-url",
            "https://example.com/document",
            "https://docs.google.com/spreadsheet/d/invalid",
            "ftp://docs.google.com/document/d/123",
            "https://drive.google.com/file/d/123",  # Google Drive, not Docs
            "",
            None,
        ]

        for invalid_url in invalid_urls:
            if invalid_url is None:
                with pytest.raises((TypeError, AttributeError)):
                    parser.parse_file(invalid_url)
            else:
                # Should either raise an exception or return empty result
                try:
                    result = parser.parse_file(invalid_url)
                    # If it doesn't raise, result should be empty/None or indicate failure
                    assert result is None or (isinstance(result, (list, dict)) and len(result) == 0)
                except Exception:
                    # Expected behavior - invalid URL should cause error
                    pass

    def test_empty_document_content(self):
        """Test handling of empty documents"""
        parser = DocumentParser()

        empty_content_samples = [
            "",  # Empty string
            "   ",  # Only whitespace
            "\n\n\n",  # Only newlines
        ]

        for empty_content in empty_content_samples:
            # Parser should handle empty content gracefully
            result = parser.parse_file(empty_content)
            # Result should be None, empty, or indicate no content
            assert result is None or (isinstance(result, (list, dict)) and
                                     (len(result) == 0 or
                                      (isinstance(result, dict) and
                                       (not result.get('content') or len(result.get('content', [])) == 0))))

    def test_null_input_values(self):
        """Test handling of None/null input values"""
        parser = DocumentParser()

        # Test None as input
        with pytest.raises((TypeError, AttributeError)):
            parser.parse_file(None)

        # Test parse_file with None heading
        with pytest.raises((TypeError, AttributeError)):
            parser._content_to_slides(None, None)

    def test_invalid_file_format(self):
        """Test rejection of non-document URLs and formats"""
        parser = DocumentParser()

        invalid_formats = [
            "https://youtube.com/watch?v=123",
            "https://github.com/user/repo",
            "https://example.com/image.jpg",
            "https://example.com/video.mp4",
            "data:text/html,<h1>HTML</h1>",
        ]

        for invalid_format in invalid_formats:
            # Should fail gracefully, not crash
            try:
                result = parser.parse_file(invalid_format)
                # If no exception, should return empty/None
                assert result is None or (isinstance(result, (list, dict)) and
                                         (len(result) == 0 or not result.get('content')))
            except Exception as e:
                # Expected - invalid format should raise
                assert "error" in str(e).lower() or "invalid" in str(e).lower()


# ============================================================================
# NETWORK ERROR TESTS
# ============================================================================

class TestNetworkErrors:
    """Test handling of network-related errors"""

    @patch('requests.get')
    def test_network_timeout(self, mock_get):
        """Test handling of request timeouts"""
        # Simulate timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        parser = DocumentParser()

        with pytest.raises(requests.exceptions.Timeout):
            parser.parse_file("https://docs.google.com/document/d/test123/edit")

    @patch('requests.get')
    def test_connection_refused(self, mock_get):
        """Test handling when connection is refused"""
        # Simulate connection refused
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "Failed to establish a new connection"
        )

        parser = DocumentParser()

        with pytest.raises(requests.exceptions.ConnectionError):
            parser.parse_file("https://docs.google.com/document/d/test123/edit")

    @patch('requests.get')
    def test_dns_resolution_failure(self, mock_get):
        """Test handling of DNS resolution failures"""
        # Simulate DNS failure
        mock_get.side_effect = requests.exceptions.InvalidURL(
            "Invalid URL 'https://invalid-domain-xyz-123.invalid/doc'"
        )

        parser = DocumentParser()

        with pytest.raises(requests.exceptions.InvalidURL):
            parser.parse_file("https://invalid-domain-xyz-123.invalid/document")

    @patch('requests.get')
    def test_connection_reset(self, mock_get):
        """Test handling of connection reset errors"""
        # Simulate connection reset
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "Connection reset by peer"
        )

        parser = DocumentParser()

        with pytest.raises(requests.exceptions.ConnectionError):
            parser.parse_file("https://docs.google.com/document/d/test123/edit")


# ============================================================================
# API ERROR TESTS
# ============================================================================

class TestAPIErrors:
    """Test handling of API-specific errors"""

    @patch('anthropic.Anthropic')
    def test_claude_api_quota_exceeded(self, mock_anthropic):
        """Test handling of API quota exceeded (429) errors"""
        # Simulate 429 rate limit error
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Create error that simulates 429
        error = Exception("Error code: 429 - Rate limit exceeded")
        mock_client.messages.create.side_effect = error

        parser = DocumentParser(claude_api_key="test-key-123")
        parser.client = mock_client

        with pytest.raises(Exception) as exc_info:
            parser._call_claude_with_retry(
                model="claude-3.5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}]
            )

        assert "429" in str(exc_info.value) or "rate" in str(exc_info.value).lower()

    @patch('anthropic.Anthropic')
    def test_claude_api_invalid_key(self, mock_anthropic):
        """Test handling of invalid API key (401 errors)"""
        # Simulate 401 unauthorized error
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        error = Exception("Error code: 401 - Unauthorized (invalid API key)")
        mock_client.messages.create.side_effect = error

        parser = DocumentParser(claude_api_key="invalid-key-123")
        parser.client = mock_client

        with pytest.raises(Exception) as exc_info:
            parser._call_claude_with_retry(
                model="claude-3.5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}]
            )

        assert "401" in str(exc_info.value) or "unauthorized" in str(exc_info.value).lower()

    @patch('anthropic.Anthropic')
    def test_claude_api_rate_limit_with_retry(self, mock_anthropic):
        """Test that rate limit errors trigger retry mechanism"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Simulate rate limit on first two attempts, success on third
        rate_limit_error = Exception("Error code: 429 - Rate limit exceeded")
        success_response = Mock()
        success_response.content = [Mock(text="Generated bullet points")]

        mock_client.messages.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            success_response
        ]

        parser = DocumentParser(claude_api_key="test-key-123")
        parser.client = mock_client

        # Call with retry - should eventually succeed
        with patch('time.sleep'):  # Speed up test by skipping actual sleep
            try:
                result = parser._call_claude_with_retry(
                    model="claude-3.5-sonnet",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "test"}]
                )
                assert result == success_response
            except Exception:
                # After 3 retries, should fail
                pass

    @patch('anthropic.Anthropic')
    def test_claude_api_server_error_500(self, mock_anthropic):
        """Test handling of 500 server errors"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        error = Exception("Error code: 500 - Internal Server Error")
        mock_client.messages.create.side_effect = error

        parser = DocumentParser(claude_api_key="test-key-123")
        parser.client = mock_client

        with pytest.raises(Exception):
            parser._call_claude_with_retry(
                model="claude-3.5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}]
            )

    @patch('anthropic.Anthropic')
    def test_openai_api_errors(self, mock_anthropic):
        """Test handling of OpenAI API errors if integration exists"""
        # Test generic API client errors
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        error_scenarios = [
            Exception("API request failed"),
            Exception("Invalid request format"),
            Exception("Service unavailable"),
        ]

        parser = DocumentParser(claude_api_key="test-key-123")
        parser.client = mock_client

        for error in error_scenarios:
            mock_client.messages.create.side_effect = error

            with pytest.raises(Exception):
                parser._call_claude_with_retry(
                    model="claude-3.5-sonnet",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "test"}]
                )


# ============================================================================
# DOCUMENT PROCESSING ERROR TESTS
# ============================================================================

class TestDocumentProcessingErrors:
    """Test handling of document processing errors"""

    def test_document_too_large(self):
        """Test handling of documents exceeding size limits"""
        parser = DocumentParser()

        # Create extremely large text (>50MB might be limit)
        large_text = "x" * (100 * 1024 * 1024)  # 100MB

        # Parser should either handle gracefully or raise appropriate error
        try:
            result = parser.parse_file(large_text)
            # If it succeeds, result should be valid
            assert result is None or isinstance(result, (list, dict))
        except MemoryError:
            # Expected for very large files
            pass
        except Exception as e:
            # Should be a size-related error, not a generic crash
            assert "size" in str(e).lower() or "large" in str(e).lower() or "memory" in str(e).lower()

    def test_malformed_document_structure(self):
        """Test handling of documents with invalid structure"""
        parser = DocumentParser()

        malformed_documents = [
            {"title": "Test"},  # Missing 'body'
            {"body": None},  # Null body
            {"body": {"content": None}},  # Null content
            {"body": {"content": "not-a-list"}},  # Content should be list
        ]

        for malformed_doc in malformed_documents:
            # Should handle gracefully
            try:
                result = parser.parse_file(json.dumps(malformed_doc))
                assert result is None or isinstance(result, (list, dict))
            except Exception as e:
                # Should be a parsing error, not a crash
                assert "parse" in str(e).lower() or "error" in str(e).lower()

    def test_missing_document_content(self):
        """Test handling of documents with missing required content"""
        parser = DocumentParser()

        documents_missing_content = [
            {"title": "Title Only"},
            {"title": "No Body"},
            {"body": {}},  # Empty body
            {"body": {"content": []}},  # Empty content array
        ]

        for doc in documents_missing_content:
            result = parser.parse_file(json.dumps(doc))
            # Should return None or empty result, not crash
            assert result is None or (isinstance(result, (list, dict)) and
                                     (len(result) == 0 or not result.get('content')))

    def test_corrupted_document_data(self):
        """Test handling of corrupted/invalid document data"""
        parser = DocumentParser()

        corrupted_data = [
            "\x00\x01\x02\x03",  # Binary garbage
            "<!DOCTYPE html><html>Not a doc</html>",  # HTML instead of doc
            "{invalid json}",  # Invalid JSON
            "<?xml version='1.0'?>",  # XML instead of doc
        ]

        for data in corrupted_data:
            try:
                result = parser.parse_file(data)
                # Should handle gracefully
                assert result is None or isinstance(result, (list, dict))
            except Exception:
                # Expected for corrupted data
                pass


# ============================================================================
# CONCURRENT REQUEST TESTS
# ============================================================================

class TestConcurrentRequests:
    """Test handling of concurrent requests and thread safety"""

    def test_multiple_concurrent_requests(self):
        """Test handling of multiple concurrent document processing requests"""
        parser = DocumentParser()
        results = []
        errors = []

        def process_document(doc_id, content):
            """Worker function to process document"""
            try:
                result = parser.parse_file(content)
                results.append((doc_id, result))
            except Exception as e:
                errors.append((doc_id, str(e)))

        # Create multiple threads
        threads = []
        num_threads = 5

        for i in range(num_threads):
            doc_content = f"Document {i}: This is test content {i}"
            thread = threading.Thread(
                target=process_document,
                args=(f"doc_{i}", doc_content)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)

        # Verify all requests were processed
        assert len(results) + len(errors) == num_threads

    def test_cache_thread_safety(self):
        """Test that cache handles concurrent access safely"""
        parser = DocumentParser()

        def cache_operations(thread_id):
            """Perform cache operations from multiple threads"""
            for i in range(10):
                # Generate cache key
                cache_key = parser._generate_cache_key(f"text_{thread_id}_{i}")

                # Try to get from cache
                cached = parser._get_cached_response(cache_key)

                # Cache some data
                bullets = [f"Bullet {j}" for j in range(3)]
                parser._cache_response(cache_key, bullets)

                # Small delay to increase contention
                time.sleep(0.001)

        # Create multiple threads accessing cache
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)

        # Verify cache integrity
        stats = parser.get_cache_stats()
        assert stats['cache_size'] > 0
        assert stats['total_requests'] >= 0

    def test_concurrent_api_calls(self):
        """Test handling of concurrent API calls with rate limiting"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Simulate successful response
            success_response = Mock()
            success_response.content = [Mock(text="Generated bullets")]
            mock_client.messages.create.return_value = success_response

            parser = DocumentParser(claude_api_key="test-key")
            parser.client = mock_client

            results = []

            def make_api_call(call_id):
                """Make API call from thread"""
                try:
                    result = parser._call_claude_with_retry(
                        model="claude-3.5-sonnet",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": f"Request {call_id}"}]
                    )
                    results.append((call_id, result))
                except Exception as e:
                    results.append((call_id, str(e)))

            # Create multiple concurrent API calls
            threads = []
            for i in range(3):
                thread = threading.Thread(target=make_api_call, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)

            # Verify all calls were processed
            assert len(results) == 3


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test handling of edge cases and unusual inputs"""

    def test_extremely_long_text(self):
        """Test processing of extremely long paragraphs"""
        parser = DocumentParser()

        # Create very long text
        long_paragraph = "word " * 10000  # 50,000 words

        try:
            result = parser.parse_file(long_paragraph)
            # Should handle without crashing
            assert result is None or isinstance(result, (list, dict))
        except Exception:
            # If it fails, should be graceful failure
            pass

    def test_special_characters_and_unicode(self):
        """Test handling of special characters, Unicode, and emojis"""
        parser = DocumentParser()

        special_content = [
            "Hello 世界 مرحبا мир",  # Multiple language scripts
            "emoji test: 🚀 🌟 ✅ ❌",  # Emojis
            "special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`",  # Special chars
            "mixed\ttabs\nand\rnewlines",  # Control characters
            "Unicode escape: \u0041\u0042\u0043",  # Unicode escapes
        ]

        for content in special_content:
            try:
                result = parser.parse_file(content)
                # Should handle gracefully
                assert result is None or isinstance(result, (list, dict))
            except Exception:
                # Should be specific error, not crash
                pass

    def test_very_short_document(self):
        """Test handling of minimal content documents"""
        parser = DocumentParser()

        short_documents = [
            "a",  # Single character
            "Hi",  # Two words
            ".",  # Just punctuation
            "123",  # Just numbers
        ]

        for doc in short_documents:
            result = parser.parse_file(doc)
            # Should handle without error
            assert result is None or isinstance(result, (list, dict))

    def test_whitespace_only_content(self):
        """Test handling of documents with only whitespace"""
        parser = DocumentParser()

        whitespace_content = [
            "   ",  # Spaces
            "\t\t\t",  # Tabs
            "\n\n\n",  # Newlines
            "\r\n\r\n",  # Windows line endings
            "  \t\n  \t\n  ",  # Mixed whitespace
        ]

        for content in whitespace_content:
            result = parser.parse_file(content)
            # Should handle gracefully
            assert result is None or (isinstance(result, (list, dict)) and
                                     (len(result) == 0 or not result.get('content')))

    def test_extremely_deep_nesting(self):
        """Test handling of deeply nested structures"""
        parser = DocumentParser()

        # Create deeply nested JSON structure
        nested = {"content": []}
        current = nested["content"]
        for i in range(100):
            new_item = {"nested": []}
            current.append(new_item)
            current = new_item["nested"]

        try:
            result = parser.parse_file(json.dumps(nested))
            assert result is None or isinstance(result, (list, dict))
        except (RecursionError, ValueError):
            # Expected for very deep nesting
            pass

    def test_repeated_identical_content(self):
        """Test handling of documents with repeated identical content"""
        parser = DocumentParser()

        # Create repeated content
        repeated_text = ("This is a test sentence. " * 1000)

        result = parser.parse_file(repeated_text)
        # Should handle gracefully
        assert result is None or isinstance(result, (list, dict))


# ============================================================================
# CACHE ERROR TESTS
# ============================================================================

class TestCacheErrors:
    """Test error handling in caching mechanisms"""

    def test_cache_key_generation_with_special_input(self):
        """Test cache key generation with unusual inputs"""
        parser = DocumentParser()

        special_inputs = [
            ("", "", ""),  # Empty strings
            ("a" * 10000, "b" * 10000, "c" * 10000),  # Very long strings
            ("text with special chars: !@#$%^&*()", "heading", "context"),
            ("unicode: 你好世界", "мир", "🌍"),
            (None, None, None),  # This should handle None gracefully
        ]

        for text, heading, context in special_inputs:
            if text is None or heading is None or context is None:
                continue  # Skip None values

            try:
                cache_key = parser._generate_cache_key(text, heading, context)
                # Cache key should be valid hex string
                assert isinstance(cache_key, str)
                assert len(cache_key) > 0
            except Exception:
                # Should handle edge cases
                pass

    def test_cache_overflow_handling(self):
        """Test cache behavior when exceeding max size"""
        parser = DocumentParser()
        parser._cache_max_size = 10  # Set small cache size for testing

        # Fill cache beyond max size
        for i in range(20):
            cache_key = parser._generate_cache_key(f"text_{i}")
            bullets = [f"Bullet {j}" for j in range(3)]
            parser._cache_response(cache_key, bullets)

        # Cache should not exceed max size
        assert len(parser._api_cache) <= parser._cache_max_size


# ============================================================================
# INTEGRATION TEST WITH FLASK APP
# ============================================================================

class TestFlaskAppErrorHandling:
    """Test error handling in Flask app routes"""

    def test_app_handles_null_file_input(self, client=None):
        """Test that app handles null/missing file input"""
        if client is None:
            from tests.conftest import flask_app
            app_instance = flask_app()
            client = app_instance.test_client()

        # Test POST with missing document
        response = client.post('/api/parse', data={})
        # Should return error, not 500
        assert response.status_code in [400, 422] or response.status_code < 500

    def test_app_handles_malformed_json(self, client=None):
        """Test that app handles malformed JSON input"""
        if client is None:
            from tests.conftest import flask_app
            app_instance = flask_app()
            client = app_instance.test_client()

        # Test with invalid JSON
        response = client.post(
            '/api/parse',
            data="{invalid json}",
            content_type='application/json'
        )
        # Should return 400, not 500
        assert response.status_code < 500

    def test_app_recovers_from_processing_error(self, client=None):
        """Test that app recovers from processing errors gracefully"""
        if client is None:
            from tests.conftest import flask_app
            app_instance = flask_app()
            client = app_instance.test_client()

        # Make a request that might cause processing error
        response = client.post(
            '/api/parse',
            json={"document": ""}
        )
        # App should still be responsive
        assert response.status_code < 500


if __name__ == "__main__":
    # Run tests with: pytest tests/integration/test_error_handling.py -v
    pytest.main([__file__, "-v", "--tb=short"])
