"""
Example Storage Manager for Few-Shot Learning

This module manages storage and retrieval of example input/output pairs
for improving bullet generation quality through few-shot learning.

Storage Structure:
- examples/examples_db.json: Main database of examples
- examples/categories/{category}.json: Category-specific examples
- examples/quality_ratings.json: Quality scores for examples

Each example includes:
- input_text: Original content
- generated_bullets: Output bullets
- context_heading: Slide heading/context
- style: professional/educational/technical/executive
- content_type: table/list/heading/paragraph
- quality_score: 0-100 rating (optional)
- metadata: timestamp, user_rating, category tags
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Try to import sklearn for semantic similarity (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    logging.info("sklearn not available - using keyword-based example matching")

logger = logging.getLogger(__name__)


@dataclass
class BulletExample:
    """Represents a single training example for bullet generation."""
    input_text: str
    generated_bullets: List[str]
    context_heading: str
    style: str  # professional, educational, technical, executive
    content_type: str  # table, list, heading, paragraph
    quality_score: Optional[float] = None
    user_rating: Optional[int] = None  # 1-5 stars
    category_tags: List[str] = None
    timestamp: str = None
    example_id: str = None

    def __post_init__(self):
        """Generate ID and timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.example_id is None:
            # Generate unique ID from content hash
            content_hash = hashlib.md5(
                (self.input_text + str(self.generated_bullets)).encode()
            ).hexdigest()[:12]
            self.example_id = f"{self.style}_{content_hash}"
        if self.category_tags is None:
            self.category_tags = []

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BulletExample':
        """Create from dictionary."""
        return cls(**data)


class ExampleStorageManager:
    """Manages storage and retrieval of bullet generation examples."""

    def __init__(self, storage_dir: str = "examples"):
        """
        Initialize the example storage manager.

        Args:
            storage_dir: Directory to store example files
        """
        self.storage_dir = Path(storage_dir)
        self.db_file = self.storage_dir / "examples_db.json"
        self.quality_file = self.storage_dir / "quality_ratings.json"
        self.categories_dir = self.storage_dir / "categories"

        # Create directories if they don't exist
        self.storage_dir.mkdir(exist_ok=True)
        self.categories_dir.mkdir(exist_ok=True)

        # Initialize database files if they don't exist
        if not self.db_file.exists():
            self._save_db([])
        if not self.quality_file.exists():
            self._save_quality_ratings({})

        # In-memory cache
        self._examples_cache = None
        self._last_cache_load = None

    def _save_db(self, examples: List[dict]):
        """Save examples database to file."""
        with open(self.db_file, 'w') as f:
            json.dump(examples, f, indent=2)

    def _load_db(self) -> List[dict]:
        """Load examples database from file."""
        try:
            with open(self.db_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_quality_ratings(self, ratings: dict):
        """Save quality ratings to file."""
        with open(self.quality_file, 'w') as f:
            json.dump(ratings, f, indent=2)

    def _load_quality_ratings(self) -> dict:
        """Load quality ratings from file."""
        try:
            with open(self.quality_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def add_example(self, example: BulletExample) -> str:
        """
        Add a new example to the database.

        Args:
            example: BulletExample instance

        Returns:
            example_id: Unique ID of the added example
        """
        examples = self._load_db()

        # Check for duplicates
        if any(ex['example_id'] == example.example_id for ex in examples):
            logger.warning(f"Example {example.example_id} already exists - skipping")
            return example.example_id

        # Add to database
        examples.append(example.to_dict())
        self._save_db(examples)

        # Save to category file
        self._save_to_category(example)

        # Clear cache
        self._examples_cache = None

        logger.info(f"✅ Added example {example.example_id} ({example.style}, {example.content_type})")
        return example.example_id

    def _save_to_category(self, example: BulletExample):
        """Save example to category-specific file."""
        category = example.style  # Use style as primary category
        category_file = self.categories_dir / f"{category}.json"

        # Load existing category examples
        if category_file.exists():
            with open(category_file, 'r') as f:
                category_examples = json.load(f)
        else:
            category_examples = []

        # Add example if not duplicate
        if not any(ex['example_id'] == example.example_id for ex in category_examples):
            category_examples.append(example.to_dict())
            with open(category_file, 'w') as f:
                json.dump(category_examples, f, indent=2)

    def get_examples_by_style(self, style: str, limit: int = 10) -> List[BulletExample]:
        """
        Get examples matching a specific style.

        Args:
            style: Style to filter by
            limit: Maximum number of examples to return

        Returns:
            List of BulletExample instances
        """
        examples = self._load_db()
        filtered = [ex for ex in examples if ex['style'] == style]

        # Sort by quality score (if available)
        filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        return [BulletExample.from_dict(ex) for ex in filtered[:limit]]

    def get_examples_by_content_type(self, content_type: str, limit: int = 10) -> List[BulletExample]:
        """
        Get examples matching a specific content type.

        Args:
            content_type: Content type to filter by
            limit: Maximum number of examples to return

        Returns:
            List of BulletExample instances
        """
        examples = self._load_db()
        filtered = [ex for ex in examples if ex['content_type'] == content_type]

        # Sort by quality score
        filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        return [BulletExample.from_dict(ex) for ex in filtered[:limit]]

    def get_similar_examples(self, input_text: str, style: str = None,
                            content_type: str = None, limit: int = 3) -> List[BulletExample]:
        """
        Get examples similar to the input text using semantic similarity.

        Args:
            input_text: Text to find similar examples for
            style: Optional style filter
            content_type: Optional content type filter
            limit: Maximum number of examples to return

        Returns:
            List of most similar BulletExample instances
        """
        examples = self._load_db()

        # Apply filters
        if style:
            examples = [ex for ex in examples if ex['style'] == style]
        if content_type:
            examples = [ex for ex in examples if ex['content_type'] == content_type]

        if not examples:
            return []

        # Use semantic similarity if available
        if SIMILARITY_AVAILABLE and len(examples) > 0:
            return self._get_similar_examples_tfidf(input_text, examples, limit)
        else:
            return self._get_similar_examples_keyword(input_text, examples, limit)

    def _get_similar_examples_tfidf(self, input_text: str, examples: List[dict],
                                     limit: int) -> List[BulletExample]:
        """Get similar examples using TF-IDF cosine similarity."""
        try:
            # Extract input texts
            example_texts = [ex['input_text'] for ex in examples]
            all_texts = [input_text] + example_texts

            # Compute TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Compute cosine similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Get top matches
            top_indices = np.argsort(similarities)[-limit:][::-1]

            return [BulletExample.from_dict(examples[i]) for i in top_indices]

        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}, falling back to keyword matching")
            return self._get_similar_examples_keyword(input_text, examples, limit)

    def _get_similar_examples_keyword(self, input_text: str, examples: List[dict],
                                       limit: int) -> List[BulletExample]:
        """Get similar examples using simple keyword overlap."""
        # Extract keywords from input
        input_words = set(input_text.lower().split())

        # Score each example by keyword overlap
        scored_examples = []
        for ex in examples:
            ex_words = set(ex['input_text'].lower().split())
            overlap = len(input_words & ex_words)
            scored_examples.append((overlap, ex))

        # Sort by overlap and take top N
        scored_examples.sort(reverse=True, key=lambda x: x[0])
        return [BulletExample.from_dict(ex) for _, ex in scored_examples[:limit]]

    def rate_example(self, example_id: str, quality_score: float = None,
                     user_rating: int = None):
        """
        Rate an example's quality.

        Args:
            example_id: ID of the example to rate
            quality_score: Objective quality score (0-100)
            user_rating: Subjective user rating (1-5 stars)
        """
        ratings = self._load_quality_ratings()

        if example_id not in ratings:
            ratings[example_id] = {}

        if quality_score is not None:
            ratings[example_id]['quality_score'] = quality_score
        if user_rating is not None:
            ratings[example_id]['user_rating'] = user_rating

        self._save_quality_ratings(ratings)

        # Update example in database
        examples = self._load_db()
        for ex in examples:
            if ex['example_id'] == example_id:
                if quality_score is not None:
                    ex['quality_score'] = quality_score
                if user_rating is not None:
                    ex['user_rating'] = user_rating
        self._save_db(examples)

        logger.info(f"✅ Rated example {example_id}: quality={quality_score}, user_rating={user_rating}")

    def get_stats(self) -> dict:
        """Get statistics about stored examples."""
        examples = self._load_db()

        stats = {
            'total_examples': len(examples),
            'by_style': {},
            'by_content_type': {},
            'avg_quality_score': 0,
            'rated_examples': 0
        }

        for ex in examples:
            # Count by style
            style = ex['style']
            stats['by_style'][style] = stats['by_style'].get(style, 0) + 1

            # Count by content type
            content_type = ex['content_type']
            stats['by_content_type'][content_type] = stats['by_content_type'].get(content_type, 0) + 1

            # Average quality score
            if ex.get('quality_score'):
                stats['avg_quality_score'] += ex['quality_score']
                stats['rated_examples'] += 1

        if stats['rated_examples'] > 0:
            stats['avg_quality_score'] /= stats['rated_examples']

        return stats

    def export_examples_for_testing(self, output_file: str):
        """
        Export examples in format suitable for golden_test_set.py.

        Args:
            output_file: Path to save the exported test cases
        """
        examples = self._load_db()

        test_cases = []
        for ex in examples:
            test_case = {
                'id': ex['example_id'],
                'category': ex['style'],
                'input_text': ex['input_text'],
                'context_heading': ex['context_heading'],
                'expected_bullets': ex['generated_bullets'],
                'quality_criteria': {
                    'min_bullets': max(2, len(ex['generated_bullets']) - 1),
                    'max_bullets': len(ex['generated_bullets']) + 1
                }
            }
            test_cases.append(test_case)

        with open(output_file, 'w') as f:
            json.dump(test_cases, f, indent=2)

        logger.info(f"✅ Exported {len(test_cases)} examples to {output_file}")


# Singleton instance
_storage_manager = None


def get_storage_manager() -> ExampleStorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = ExampleStorageManager()
    return _storage_manager
