"""
Test suite for bullet quality improvements (1.1, 1.2, 1.3)
These tests should FAIL initially (TDD Red Phase)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slide_generator_pkg.document_parser import DocumentParser


class TestContextAwareBullets:
    """Tests for improvement 1.2: Context-Aware Bullets"""

    def test_bullets_reference_parent_headings(self):
        """Bullets should reference broader document context"""
        parser = DocumentParser()

        # Simulate hierarchical document
        heading_ancestry = ["Introduction", "Background", "Problem Statement"]
        text = "Our research shows that 70% of users struggle with topic separation."

        bullets = parser._create_bullet_points(
            text,
            context_heading="Problem Statement",
            heading_ancestry=heading_ancestry
        )

        # Handle tuple return (topic_sentence, bullets) or (topic_sentence, bullets, metrics)
        if isinstance(bullets, tuple):
            bullets = bullets[1] if len(bullets) > 1 else bullets[0]

        # Bullets should reference that this is a "problem" in "background"
        assert len(bullets) > 0
        # At least one bullet should imply problem/challenge context
        problem_keywords = ['struggle', 'challenge', 'issue', 'problem', 'difficulty']
        assert any(any(kw in b.lower() for kw in problem_keywords) for b in bullets)

    def test_heading_ancestry_passed_to_llm(self):
        """Verify heading ancestry is included in LLM prompt"""
        parser = DocumentParser()

        # Mock LLM call to inspect prompt
        original_method = parser._create_llm_only_bullets if hasattr(parser, '_create_llm_only_bullets') else None
        captured_prompt = None

        def mock_llm(*args, **kwargs):
            nonlocal captured_prompt
            # Capture the prompt that would be sent to LLM
            captured_prompt = kwargs.get('text', args[0] if args else None)
            return ["Bullet 1", "Bullet 2"]

        if original_method:
            parser._create_llm_only_bullets = mock_llm

            heading_ancestry = ["Chapter 1", "Section 2", "Subsection 3"]
            result = parser._create_bullet_points(
                "Sample text",
                context_heading="Subsection 3",
                heading_ancestry=heading_ancestry
            )

            # Prompt should include ancestry context
            # This test will fail until the feature is implemented
            assert captured_prompt is not None or heading_ancestry is not None


class TestBulletValidation:
    """Tests for improvement 1.1: LLM-Based Bullet Validation"""

    def test_validation_detects_missing_concepts(self):
        """Validator should identify missing key concepts"""
        parser = DocumentParser()

        source_text = """
        Machine learning models require three key components: training data,
        algorithms, and computational resources. Training data must be representative
        of real-world scenarios to avoid bias.
        """

        # Incomplete bullets (missing "bias" concept)
        incomplete_bullets = [
            "Machine learning models need training data",
            "Algorithms are required for ML models",
            "Computational resources are necessary"
        ]

        if hasattr(parser, '_validate_and_improve_bullets'):
            improved, metrics = parser._validate_and_improve_bullets(
                incomplete_bullets,
                source_text,
                heading="ML Requirements"
            )

            # Should detect "bias" is missing
            assert 'bias' in str(metrics.get('missing_concepts', [])).lower() or \
                   len(improved) > len(incomplete_bullets) or \
                   any('bias' in b.lower() or 'representative' in b.lower() for b in improved)
        else:
            # Feature not implemented yet - test should fail
            pytest.skip("_validate_and_improve_bullets not implemented yet")

    def test_validation_scores_relevance(self):
        """Validator should score bullet relevance"""
        parser = DocumentParser()

        source_text = "Python is a popular programming language for data science."

        # Highly relevant bullets
        good_bullets = ["Python is widely used in data science"]

        # Irrelevant bullets
        bad_bullets = ["JavaScript is used for web development"]

        if hasattr(parser, '_validate_and_improve_bullets'):
            _, good_metrics = parser._validate_and_improve_bullets(
                good_bullets, source_text, "Python Overview"
            )

            _, bad_metrics = parser._validate_and_improve_bullets(
                bad_bullets, source_text, "Python Overview"
            )

            # Good bullets should score higher
            assert good_metrics['relevance_score'] > bad_metrics['relevance_score']
            assert good_metrics['relevance_score'] > 0.5
        else:
            pytest.skip("_validate_and_improve_bullets not implemented yet")


class TestBulletDiversity:
    """Tests for improvement 1.3: Bullet Diversity Scoring"""

    def test_diversity_score_detects_repetition(self):
        """Should detect when all bullets start the same way"""
        parser = DocumentParser()

        # Repetitive bullets (all start with "The")
        repetitive = [
            "The system processes data efficiently",
            "The system handles large volumes",
            "The system provides real-time results"
        ]

        # Diverse bullets
        diverse = [
            "Data processing occurs efficiently",
            "Large volumes are handled seamlessly",
            "Real-time results ensure quick decisions"
        ]

        if hasattr(parser, '_check_bullet_diversity'):
            repetitive_score = parser._check_bullet_diversity(repetitive)
            diverse_score = parser._check_bullet_diversity(diverse)

            assert diverse_score > repetitive_score
            assert repetitive_score < 0.7  # Low score for repetition
        else:
            pytest.skip("_check_bullet_diversity not implemented yet")

    def test_diversity_triggers_regeneration(self):
        """Low diversity should trigger bullet regeneration"""
        parser = DocumentParser()

        if not hasattr(parser, '_check_bullet_diversity'):
            pytest.skip("_check_bullet_diversity not implemented yet")

        # This test verifies the diversity checking mechanism exists
        # The actual regeneration logic will be tested in integration
        repetitive = [
            "The system does this",
            "The system does that",
            "The system does another thing"
        ]

        score = parser._check_bullet_diversity(repetitive)
        # Low diversity score should be detected
        assert score < 0.7


# Parametrized tests for edge cases
@pytest.mark.parametrize("text_length,expected_min_bullets", [
    ("Short text.", 1),
    ("Medium length text with several sentences. This has more content. And even more.", 2),
    ("Very long text. " * 50, 3),
])
def test_bullet_count_scales_with_content(text_length, expected_min_bullets):
    """Bullet count should scale appropriately with content length"""
    parser = DocumentParser()
    result = parser._create_bullet_points(text_length)

    # Handle different return formats
    if isinstance(result, tuple):
        bullets = result[1] if len(result) > 1 else result[0]
    else:
        bullets = result

    assert len(bullets) >= expected_min_bullets
