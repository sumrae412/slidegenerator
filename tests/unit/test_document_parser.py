"""Unit tests for DocumentParser class."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import OrderedDict
import sys
import os

# Add parent directory to path to import file_to_slides
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from file_to_slides import DocumentParser


class TestDocumentParserInitialization:
    """Tests for DocumentParser initialization."""

    def test_parser_initialization_without_api_key(self):
        """Test basic initialization without Claude API key."""
        parser = DocumentParser()

        # Verify parser is initialized
        assert parser is not None
        assert parser.api_key is None or isinstance(parser.api_key, str)
        assert isinstance(parser._api_cache, OrderedDict)
        assert parser._cache_max_size == 1000
        assert parser._cache_hits == 0
        assert parser._cache_misses == 0
        assert len(parser.heading_patterns) > 0

    def test_parser_initialization_with_claude_api_key(self):
        """Test initialization with Claude API key."""
        test_key = "sk-ant-test-key-12345"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            parser = DocumentParser(claude_api_key=test_key)

            # Verify API key is stored
            assert parser.api_key == test_key
            # Client initialization is attempted
            mock_anthropic.assert_called_once_with(api_key=test_key)

    def test_parser_initialization_with_invalid_api_key(self):
        """Test initialization with invalid API key format."""
        test_key = "invalid-key"

        with patch('file_to_slides.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.side_effect = Exception("Invalid API key")

            parser = DocumentParser(claude_api_key=test_key)

            # Parser should still be initialized even with error
            assert parser is not None
            assert parser.api_key == test_key
            # Client should be None due to initialization error
            assert parser.client is None

    def test_parser_cache_initialization(self):
        """Test that cache is properly initialized."""
        parser = DocumentParser()

        assert isinstance(parser._api_cache, OrderedDict)
        assert len(parser._api_cache) == 0
        assert parser._cache_max_size == 1000
        assert parser._cache_hits == 0
        assert parser._cache_misses == 0

    def test_parser_heading_patterns_defined(self):
        """Test that heading patterns are properly defined."""
        parser = DocumentParser()

        assert parser.heading_patterns is not None
        assert isinstance(parser.heading_patterns, list)
        assert len(parser.heading_patterns) >= 4  # At least 4 patterns expected


class TestDocumentParserCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_generation_basic(self):
        """Test basic cache key generation."""
        parser = DocumentParser()

        text = "This is test content"
        cache_key = parser._generate_cache_key(text)

        assert cache_key is not None
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_cache_key_consistency(self):
        """Test that identical inputs generate identical cache keys."""
        parser = DocumentParser()

        text = "Test content for consistency"
        heading = "Test Heading"
        context = "Test Context"

        key1 = parser._generate_cache_key(text, heading, context)
        key2 = parser._generate_cache_key(text, heading, context)

        assert key1 == key2

    def test_cache_key_differentiation(self):
        """Test that different inputs generate different cache keys."""
        parser = DocumentParser()

        text1 = "First test content"
        text2 = "Different test content"

        key1 = parser._generate_cache_key(text1)
        key2 = parser._generate_cache_key(text2)

        assert key1 != key2

    def test_cache_key_with_heading_context(self):
        """Test cache key generation with heading and context."""
        parser = DocumentParser()

        text = "Content"
        key_no_context = parser._generate_cache_key(text)
        key_with_heading = parser._generate_cache_key(text, heading="Title")
        key_with_both = parser._generate_cache_key(text, heading="Title", context="Context")

        # All should be different
        assert key_no_context != key_with_heading
        assert key_with_heading != key_with_both
        assert key_no_context != key_with_both

    def test_cache_key_length(self):
        """Test that cache key has expected length (SHA-256 hex digest)."""
        parser = DocumentParser()

        text = "Test"
        cache_key = parser._generate_cache_key(text)

        # SHA-256 produces 64 hex characters
        assert len(cache_key) == 64


class TestDocumentParserCacheRetrieval:
    """Tests for cache retrieval."""

    def test_get_cached_response_miss(self):
        """Test retrieving non-existent cache entry."""
        parser = DocumentParser()

        cache_key = "non-existent-key"
        response = parser._get_cached_response(cache_key)

        assert response is None
        assert parser._cache_misses == 1

    def test_get_cached_response_hit(self):
        """Test retrieving existing cache entry."""
        parser = DocumentParser()

        text = "Test content"
        bullets = ["Bullet 1", "Bullet 2", "Bullet 3"]
        cache_key = parser._generate_cache_key(text)

        # Store in cache
        parser._cache_response(cache_key, bullets)

        # Retrieve from cache
        response = parser._get_cached_response(cache_key)

        assert response == bullets
        assert parser._cache_hits == 1

    def test_cache_hit_increments_hit_counter(self):
        """Test that cache hits increment the hit counter."""
        parser = DocumentParser()

        text = "Test"
        bullets = ["Test bullet"]
        cache_key = parser._generate_cache_key(text)

        parser._cache_response(cache_key, bullets)

        initial_hits = parser._cache_hits
        parser._get_cached_response(cache_key)

        assert parser._cache_hits == initial_hits + 1

    def test_cache_miss_increments_miss_counter(self):
        """Test that cache misses increment the miss counter."""
        parser = DocumentParser()

        initial_misses = parser._cache_misses
        parser._get_cached_response("non-existent")

        assert parser._cache_misses == initial_misses + 1

    def test_cache_lru_behavior(self):
        """Test that LRU (Least Recently Used) behavior moves accessed items to end."""
        parser = DocumentParser()

        # Add multiple entries
        for i in range(3):
            key = parser._generate_cache_key(f"text{i}")
            parser._cache_response(key, [f"bullet{i}"])

        # Access first entry
        first_key = parser._generate_cache_key("text0")
        parser._get_cached_response(first_key)

        # First key should be moved to end (most recently used)
        cache_keys = list(parser._api_cache.keys())
        assert cache_keys[-1] == first_key


class TestDocumentParserCacheStorage:
    """Tests for cache storage."""

    def test_cache_response_storage(self):
        """Test storing response in cache."""
        parser = DocumentParser()

        text = "Test content"
        bullets = ["Bullet 1", "Bullet 2"]
        cache_key = parser._generate_cache_key(text)

        parser._cache_response(cache_key, bullets)

        assert cache_key in parser._api_cache
        assert parser._api_cache[cache_key] == bullets

    def test_cache_response_multiple_entries(self):
        """Test storing multiple entries in cache."""
        parser = DocumentParser()

        for i in range(5):
            text = f"Test content {i}"
            bullets = [f"Bullet {i}-1", f"Bullet {i}-2"]
            cache_key = parser._generate_cache_key(text)
            parser._cache_response(cache_key, bullets)

        assert len(parser._api_cache) == 5

    def test_cache_eviction_on_max_size(self):
        """Test that oldest entry is evicted when cache reaches max size."""
        parser = DocumentParser()
        parser._cache_max_size = 3

        # Add entries
        first_key = None
        for i in range(4):
            text = f"Test content {i}"
            bullets = [f"Bullet {i}"]
            cache_key = parser._generate_cache_key(text)

            if i == 0:
                first_key = cache_key

            parser._cache_response(cache_key, bullets)

        # Cache should only have 3 entries (oldest evicted)
        assert len(parser._api_cache) == 3
        # First key should be evicted
        assert first_key not in parser._api_cache

    def test_cache_stores_list_of_strings(self):
        """Test that cache stores list of strings correctly."""
        parser = DocumentParser()

        bullets = ["First bullet", "Second bullet", "Third bullet"]
        cache_key = "test-key"

        parser._cache_response(cache_key, bullets)

        retrieved = parser._api_cache[cache_key]
        assert isinstance(retrieved, list)
        assert all(isinstance(b, str) for b in retrieved)
        assert retrieved == bullets


class TestDocumentParserCacheStats:
    """Tests for cache statistics."""

    def test_cache_stats_initial_state(self):
        """Test cache stats in initial state."""
        parser = DocumentParser()

        stats = parser.get_cache_stats()

        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate_percent"] == 0
        assert stats["cache_size"] == 0

    def test_cache_stats_after_hit(self):
        """Test cache stats after a cache hit."""
        parser = DocumentParser()

        text = "Test"
        bullets = ["Bullet"]
        cache_key = parser._generate_cache_key(text)

        parser._cache_response(cache_key, bullets)
        parser._get_cached_response(cache_key)

        stats = parser.get_cache_stats()

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["total_requests"] == 1
        assert stats["cache_size"] == 1

    def test_cache_stats_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        parser = DocumentParser()

        # Add entries and do hits/misses
        for i in range(10):
            text = f"Text {i}"
            bullets = [f"Bullet {i}"]
            cache_key = parser._generate_cache_key(text)
            parser._cache_response(cache_key, bullets)

        # 8 hits, 2 misses
        for i in range(8):
            key = parser._generate_cache_key(f"Text {i}")
            parser._get_cached_response(key)

        parser._get_cached_response("non-existent-1")
        parser._get_cached_response("non-existent-2")

        stats = parser.get_cache_stats()

        assert stats["cache_hits"] == 8
        assert stats["cache_misses"] == 2
        assert stats["total_requests"] == 10
        assert stats["hit_rate_percent"] == 80.0

    def test_cache_stats_format(self):
        """Test that cache stats have proper format and types."""
        parser = DocumentParser()

        stats = parser.get_cache_stats()

        assert isinstance(stats, dict)
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "total_requests" in stats
        assert "hit_rate_percent" in stats
        assert "cache_size" in stats
        assert "estimated_cost_savings" in stats

        assert isinstance(stats["cache_hits"], int)
        assert isinstance(stats["cache_misses"], int)
        assert isinstance(stats["total_requests"], int)
        assert isinstance(stats["hit_rate_percent"], (int, float))
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["estimated_cost_savings"], str)


class TestDocumentParserCreateUnifiedBullets:
    """Tests for create_unified_bullets method."""

    def test_create_unified_bullets_basic(self):
        """Test basic bullet creation with simple text."""
        parser = DocumentParser()

        text = "Python is a programming language. It is widely used in data science."
        result = parser._create_unified_bullets(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(bullet, str) for bullet in result)

    def test_create_unified_bullets_with_context(self):
        """Test bullet creation with heading context."""
        parser = DocumentParser()

        text = "Machine learning uses algorithms to learn from data."
        heading = "Machine Learning Basics"

        result = parser._create_unified_bullets(text, context_heading=heading)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_create_unified_bullets_empty_input(self):
        """Test bullet creation with empty input."""
        parser = DocumentParser()

        result = parser._create_unified_bullets("")

        assert result is not None
        assert isinstance(result, list)

    def test_create_unified_bullets_minimal_input(self):
        """Test bullet creation with minimal input."""
        parser = DocumentParser()

        text = "Short text"
        result = parser._create_unified_bullets(text)

        assert result is not None
        assert isinstance(result, list)

    def test_create_unified_bullets_long_text(self):
        """Test bullet creation with long text."""
        parser = DocumentParser()

        text = """
        This is a comprehensive text about cloud computing. Cloud computing provides
        on-demand access to computing resources. It includes infrastructure, platforms,
        and software services. The main benefits include scalability, cost-effectiveness,
        and accessibility. Different cloud models exist including public, private, and
        hybrid clouds. Major providers include AWS, Azure, and Google Cloud Platform.
        """

        result = parser._create_unified_bullets(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_create_unified_bullets_returns_list_of_strings(self):
        """Test that bullets are strings."""
        parser = DocumentParser()

        text = "This is test content with multiple sentences. Each sentence should become a bullet."
        result = parser._create_unified_bullets(text)

        assert all(isinstance(bullet, str) for bullet in result)
        assert all(len(bullet) > 0 for bullet in result)

    def test_create_unified_bullets_special_characters(self):
        """Test bullet creation with special characters."""
        parser = DocumentParser()

        text = "Data analysis involves statistics, mathematics & programming (R, Python, SQL)!"
        result = parser._create_unified_bullets(text)

        assert isinstance(result, list)
        assert len(result) > 0


class TestDocumentParserIntegration:
    """Integration tests for DocumentParser."""

    def test_parser_full_workflow_without_api_key(self):
        """Test full parser workflow without API key."""
        parser = DocumentParser()

        # Create cache key
        text = "Integration test content"
        cache_key = parser._generate_cache_key(text)

        # Create bullets
        bullets = parser._create_unified_bullets(text)
        assert bullets is not None

        # Cache the response
        parser._cache_response(cache_key, bullets)

        # Retrieve from cache
        cached = parser._get_cached_response(cache_key)
        assert cached == bullets

        # Check stats
        stats = parser.get_cache_stats()
        assert stats["cache_hits"] == 1

    def test_parser_cache_hit_reduces_api_calls(self):
        """Test that cache hits reduce API call count."""
        parser = DocumentParser()

        text = "Test content for cache verification"
        bullets = ["Bullet 1", "Bullet 2"]
        cache_key = parser._generate_cache_key(text)

        # First call - cache miss
        result1 = parser._get_cached_response(cache_key)
        assert result1 is None

        # Store in cache
        parser._cache_response(cache_key, bullets)

        # Second call - cache hit
        result2 = parser._get_cached_response(cache_key)
        assert result2 == bullets

        # Verify stats
        stats = parser.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
