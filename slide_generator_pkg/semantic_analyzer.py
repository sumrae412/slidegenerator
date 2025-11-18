"""
Semantic Analyzer for Slide Generator

Handles semantic analysis of document content using both heavy ML and lightweight NLP approaches.
"""

import logging
from typing import List
from collections import defaultdict
import warnings

# Try to import data models from the package
try:
    from .data_models import SemanticChunk
except ImportError:
    from data_models import SemanticChunk

# Semantic analysis libraries - lightweight fallback approach
try:
    import nltk
    import textstat
    LIGHTWEIGHT_SEMANTIC = True

    # Try to download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except (LookupError, OSError):
        logger = logging.getLogger(__name__)
        logger.info("NLTK data not found, attempting download")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as download_error:
            logger.warning(f"Could not download NLTK data: {download_error}")
            LIGHTWEIGHT_SEMANTIC = False
except ImportError:
    LIGHTWEIGHT_SEMANTIC = False
    logging.warning("Lightweight semantic libraries not available - using basic fallback")

# Heavy ML libraries (optional for enhanced semantic analysis)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    import numpy as np
    SEMANTIC_AVAILABLE = True
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.info("Heavy semantic analysis not available - using lightweight approach")

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Handles semantic analysis of document content - supports both heavy and lightweight approaches"""

    def __init__(self):
        self.model = None
        self.initialized = False
        self.use_heavy_analysis = False

        # Try heavy ML approach first
        if SEMANTIC_AVAILABLE:
            try:
                # Use a lightweight model for better performance
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.initialized = True
                self.use_heavy_analysis = True
                logging.info("Semantic analyzer initialized with sentence transformers")
            except Exception as e:
                logging.warning(f"Failed to initialize heavy semantic analyzer: {e}")
                self.initialized = False

        # Fall back to lightweight approach
        if not self.initialized and LIGHTWEIGHT_SEMANTIC:
            try:
                self.initialized = True
                self.use_heavy_analysis = False
                logging.info("Semantic analyzer initialized with lightweight NLTK approach")
            except Exception as e:
                logging.warning(f"Failed to initialize lightweight semantic analyzer: {e}")
                self.initialized = False

    def analyze_chunks(self, text_chunks: List[str]) -> List[SemanticChunk]:
        """Analyze text chunks for semantic content and clustering"""
        if not self.initialized or not text_chunks:
            return [SemanticChunk(text=chunk) for chunk in text_chunks]

        try:
            chunks = []
            for text in text_chunks:
                if len(text.strip()) < 10:  # Skip very short chunks
                    continue

                if self.use_heavy_analysis:
                    # Heavy analysis with sentence transformers
                    embedding = self.model.encode([text])[0]
                    intent = self._classify_intent_heavy(text)
                    importance = self._calculate_importance_heavy(text)
                else:
                    # Lightweight analysis with NLTK
                    embedding = None
                    intent = self._classify_intent_light(text)
                    importance = self._calculate_importance_light(text)

                chunks.append(SemanticChunk(
                    text=text,
                    embedding=embedding,
                    intent=intent,
                    importance_score=importance
                ))

            # Cluster chunks by topic similarity
            if len(chunks) > 2:
                if self.use_heavy_analysis:
                    chunks = self._cluster_chunks_heavy(chunks)
                else:
                    chunks = self._cluster_chunks_light(chunks)

            return chunks

        except Exception as e:
            logging.error(f"Error in semantic analysis: {e}")
            return [SemanticChunk(text=chunk) for chunk in text_chunks]

    def _classify_intent_heavy(self, text: str) -> str:
        """Classify the intent/purpose of a text chunk using heavy analysis"""
        return self._classify_intent_light(text)  # Use same logic for now

    def _classify_intent_light(self, text: str) -> str:
        """Classify the intent/purpose of a text chunk using lightweight analysis"""
        text_lower = text.lower()

        # Intent classification based on content patterns
        if any(word in text_lower for word in ['learn', 'understand', 'explore', 'discover']):
            return 'learning_objective'
        elif any(word in text_lower for word in ['step', 'process', 'method', 'procedure']):
            return 'process_description'
        elif any(word in text_lower for word in ['example', 'for instance', 'such as', 'demonstration']):
            return 'example'
        elif any(word in text_lower for word in ['definition', 'means', 'refers to', 'is defined as']):
            return 'definition'
        elif any(word in text_lower for word in ['benefit', 'advantage', 'feature', 'capability']):
            return 'benefits'
        elif any(word in text_lower for word in ['problem', 'challenge', 'issue', 'difficulty']):
            return 'problem'
        elif any(word in text_lower for word in ['solution', 'approach', 'strategy', 'way to']):
            return 'solution'
        else:
            return 'general_content'

    def _calculate_importance_heavy(self, text: str) -> float:
        """Calculate importance score using heavy analysis"""
        return self._calculate_importance_light(text)  # Use same logic for now

    def _calculate_importance_light(self, text: str) -> float:
        """Calculate importance score using lightweight analysis"""
        score = 0.0
        text_lower = text.lower()

        # Length factor (moderate length preferred)
        length_score = min(len(text) / 200, 1.0) * 0.3
        score += length_score

        # Key term presence
        key_terms = ['important', 'key', 'main', 'primary', 'essential', 'critical', 'fundamental']
        if any(term in text_lower for term in key_terms):
            score += 0.4

        # Technical content indicators
        tech_terms = ['data', 'system', 'process', 'method', 'algorithm', 'framework', 'platform']
        if any(term in text_lower for term in tech_terms):
            score += 0.3

        # Use textstat if available for readability scoring
        if LIGHTWEIGHT_SEMANTIC:
            try:
                readability = textstat.flesch_reading_ease(text)
                if 30 <= readability <= 60:  # Moderate complexity preferred
                    score += 0.2
            except:
                pass  # Skip if textstat fails

        return min(score, 1.0)

    def _cluster_chunks_heavy(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Cluster chunks by topic similarity using heavy ML analysis"""
        try:
            import numpy as np
            from sklearn.cluster import KMeans

            embeddings = np.array([chunk.embedding for chunk in chunks])

            # Determine optimal number of clusters (max 5, min 2)
            n_clusters = min(max(len(chunks) // 3, 2), 5)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Assign cluster labels to chunks
            for i, chunk in enumerate(chunks):
                chunk.topic_cluster = int(cluster_labels[i])

            return chunks

        except Exception as e:
            logging.error(f"Error in heavy clustering: {e}")
            return chunks

    def _cluster_chunks_light(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Cluster chunks by topic similarity using lightweight text analysis"""
        try:
            if not LIGHTWEIGHT_SEMANTIC:
                return chunks

            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize

            # Simple clustering based on common keywords
            stop_words = set(stopwords.words('english'))

            # Extract keywords from each chunk
            chunk_keywords = []
            for chunk in chunks:
                try:
                    words = word_tokenize(chunk.text.lower())
                    keywords = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
                    chunk_keywords.append(set(keywords))
                except:
                    chunk_keywords.append(set())

            # Group chunks by keyword similarity
            clusters = defaultdict(list)
            for i, keywords in enumerate(chunk_keywords):
                # Find best cluster based on keyword overlap
                best_cluster = 0
                max_overlap = 0

                for cluster_id, existing_indices in clusters.items():
                    if not existing_indices:
                        continue

                    # Calculate overlap with existing cluster
                    cluster_keywords = set()
                    for idx in existing_indices:
                        cluster_keywords.update(chunk_keywords[idx])

                    overlap = len(keywords.intersection(cluster_keywords))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_cluster = cluster_id

                if max_overlap == 0:
                    # Create new cluster
                    best_cluster = len(clusters)

                clusters[best_cluster].append(i)

            # Assign cluster labels
            for cluster_id, indices in clusters.items():
                for idx in indices:
                    chunks[idx].topic_cluster = cluster_id

            return chunks

        except Exception as e:
            logging.error(f"Error in lightweight clustering: {e}")
            # Simple fallback: assign clusters based on position
            cluster_size = max(len(chunks) // 3, 1)
            for i, chunk in enumerate(chunks):
                chunk.topic_cluster = i // cluster_size
            return chunks

    def get_slide_break_suggestions(self, chunks: List[SemanticChunk]) -> List[int]:
        """Suggest where to break content into slides based on semantic analysis"""
        if not self.initialized or len(chunks) < 3:
            return []

        suggestions = []

        try:
            for i in range(1, len(chunks)):
                # Check for topic cluster changes
                if (chunks[i-1].topic_cluster != chunks[i].topic_cluster and
                    chunks[i-1].topic_cluster is not None):
                    suggestions.append(i)

                # Check for intent changes that suggest new slides
                intent_changes = [
                    ('definition', 'example'),
                    ('problem', 'solution'),
                    ('learning_objective', 'process_description'),
                    ('general_content', 'benefits')
                ]

                prev_intent = chunks[i-1].intent
                curr_intent = chunks[i].intent

                if any((prev_intent == a and curr_intent == b) or (prev_intent == b and curr_intent == a)
                       for a, b in intent_changes):
                    suggestions.append(i)

            # Remove duplicates and sort
            suggestions = sorted(list(set(suggestions)))

            # Limit number of suggestions to avoid too many slides
            return suggestions[:8]  # Max 8 slide breaks

        except Exception as e:
            logging.error(f"Error generating slide break suggestions: {e}")
            return []
