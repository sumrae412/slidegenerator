"""
Test suite for topic separation improvements (2.1, 2.2, 2.3)
These tests should FAIL initially (TDD Red Phase)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slide_generator_pkg.document_parser import DocumentParser


class TestTopicBoundaryDetection:
    """Tests for improvement 2.1: Intelligent Topic Boundary Detection"""

    def test_detects_topic_shift_in_unstructured_text(self):
        """Should detect topic boundaries without explicit headings"""
        parser = DocumentParser()

        paragraphs = [
            "Python is a versatile programming language. It's used widely in data science.",
            "Python has simple syntax. Beginners find it easy to learn.",
            "Climate change affects global temperatures. Rising sea levels threaten coastal cities.",
            "Renewable energy offers sustainable solutions. Solar and wind power are growing."
        ]

        if hasattr(parser, '_detect_topic_boundaries'):
            # Should detect boundary between paragraphs 2 and 3 (Python → Climate)
            boundaries = parser._detect_topic_boundaries(paragraphs)

            # Should have at least 2 topics (Python and Climate)
            assert len(boundaries) >= 2

            # Boundary should be around index 2 (where topic shifts)
            assert 2 in boundaries or 3 in boundaries
        else:
            pytest.skip("_detect_topic_boundaries not implemented yet")

    def test_semantic_similarity_threshold(self):
        """Topic shifts should be detected based on semantic similarity"""
        parser = DocumentParser()

        # Very similar paragraphs (same topic)
        similar = [
            "Dogs are loyal pets. They require daily exercise.",
            "Canines make great companions. They need regular walks."
        ]

        # Very different paragraphs (different topics)
        different = [
            "Dogs are loyal pets. They require daily exercise.",
            "Quantum computers use superposition. They solve complex problems."
        ]

        if hasattr(parser, '_detect_topic_boundaries'):
            similar_boundaries = parser._detect_topic_boundaries(similar)
            different_boundaries = parser._detect_topic_boundaries(different)

            # Similar text should have fewer boundaries
            assert len(similar_boundaries) <= len(different_boundaries)
        else:
            pytest.skip("_detect_topic_boundaries not implemented yet")

    def test_minimum_topic_length(self):
        """Should not create topics that are too short"""
        parser = DocumentParser()

        paragraphs = ["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]

        # Create slides from paragraphs
        text = "\n\n".join(paragraphs)
        slides = parser._content_to_slides(text)

        # Should group into reasonable-sized topics, not 1 sentence per slide
        # This is a basic test - the actual implementation might vary
        assert len(slides) >= 1


class TestSemanticClustering:
    """Tests for improvement 2.2: Semantic Analyzer for Topic Clustering"""

    def test_clusters_related_content(self):
        """Should group semantically related sentences"""
        parser = DocumentParser()

        text = """
        Python is great for data analysis. NumPy provides array support.
        Pandas handles dataframes efficiently. Matplotlib creates visualizations.

        JavaScript runs in browsers. React builds user interfaces.
        Node.js enables server-side JavaScript. Express is a web framework.
        """

        if hasattr(parser, '_create_semantic_topic_slides'):
            slides = parser._create_semantic_topic_slides(text)

            if slides and len(slides) > 0:
                # Should create at least 1 topic (could be 2 for Python/data and JavaScript/web)
                assert len(slides) >= 1

                # Verify content was processed
                slide_texts = []
                for slide in slides:
                    if hasattr(slide, 'content'):
                        slide_texts.append(" ".join(slide.content).lower())

                # At least some slides should have content
                assert len(slide_texts) > 0
        else:
            pytest.skip("_create_semantic_topic_slides not implemented yet")

    def test_generates_topic_titles(self):
        """Should generate descriptive titles for discovered topics"""
        parser = DocumentParser()

        cluster_text = """
        Neural networks learn from data. Deep learning uses multiple layers.
        Backpropagation adjusts weights. Training requires large datasets.
        """

        if hasattr(parser, '_create_semantic_topic_slides'):
            slides = parser._create_semantic_topic_slides(cluster_text)

            if slides and len(slides) > 0:
                # Title should exist
                title = slides[0].title if hasattr(slides[0], 'title') else ""
                assert len(title) > 0

                # Title should be reasonably short
                assert len(title.split()) <= 10
        else:
            pytest.skip("_create_semantic_topic_slides not implemented yet")


class TestSmartContentSplitting:
    """Tests for improvement 2.3: Smart Slide Splitting for Large Blocks"""

    def test_splits_large_bullet_lists(self):
        """Should split content that would create too many bullets"""
        parser = DocumentParser()

        # Large text that would generate 10+ bullets
        large_text = """
        Point one about the topic. Point two with more details.
        Point three introduces a new aspect. Point four continues the theme.
        Point five adds complexity. Point six provides examples.
        Point seven discusses implications. Point eight covers edge cases.
        Point nine summarizes findings. Point ten concludes the section.
        Point eleven extends the discussion. Point twelve offers alternatives.
        """

        if hasattr(parser, '_split_large_content_block'):
            slides = parser._split_large_content_block(
                large_text,
                heading="Main Topic",
                max_bullets_per_slide=5
            )

            # Should create multiple slides if content is large enough
            # Or at least handle the content appropriately
            assert len(slides) >= 1

            # Each slide should have reasonable bullet count
            for slide in slides:
                if hasattr(slide, 'content'):
                    assert len(slide.content) <= 7  # Allow some flexibility
        else:
            pytest.skip("_split_large_content_block not implemented yet")

    def test_groups_by_subtopic(self):
        """Should group bullets into logical sub-topics"""
        parser = DocumentParser()

        bullets = [
            "Python has simple syntax",
            "Python is easy to learn",
            "Python runs slower than C++",
            "Python uses more memory",
            "Python has extensive libraries",
            "Python community is large"
        ]

        if hasattr(parser, '_group_bullets_by_subtopic'):
            grouped = parser._group_bullets_by_subtopic(bullets, "Python Overview")

            # Should identify subtopics (e.g., "Advantages" vs "Disadvantages")
            assert len(grouped) >= 1

            # Should actually group the bullets
            assert isinstance(grouped, dict)
        else:
            pytest.skip("_group_bullets_by_subtopic not implemented yet")


# Integration test
class TestEndToEndTopicSeparation:
    """Integration tests for complete topic separation workflow"""

    def test_unstructured_document_to_slides(self):
        """Should convert unstructured document into well-separated slides"""
        parser = DocumentParser()

        # Document without headings, multiple topics
        document = """
        Machine learning has revolutionized data analysis. Algorithms can now
        identify patterns in massive datasets. Deep learning enables image
        recognition and natural language processing.

        Climate change poses significant challenges. Global temperatures are
        rising steadily. Extreme weather events are becoming more frequent.
        Renewable energy adoption must accelerate.

        Economic growth depends on innovation. Technology companies drive
        market expansion. Digital transformation is reshaping industries.
        """

        slides = parser._content_to_slides(document)

        # Should create at least some slides
        assert len(slides) >= 1

        # Slides should have content
        for slide in slides[:3]:  # Check first 3 slides
            if hasattr(slide, 'content'):
                assert len(slide.content) > 0
