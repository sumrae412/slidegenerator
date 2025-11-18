"""
Slide Generator Core Module

Core business logic for converting documents to presentation slides.
This module is independent of Flask and can be used in any Python application.

Key Classes:
- DocumentParser: Parses documents and extracts structure
- SlideGenerator: Creates PowerPoint presentations
- GoogleSlidesGenerator: Creates Google Slides presentations
- SemanticAnalyzer: Analyzes content semantically
"""

import os
import logging
import time
import hashlib
import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from math import cos, sin
import requests
import warnings
from collections import OrderedDict

# Semantic analysis libraries - lightweight fallback approach
try:
    import nltk
    import textstat
    from collections import Counter
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

# Heavy ML libraries (optional for enhanced semantic analysis)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SEMANTIC_AVAILABLE = True
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
except ImportError:
    SEMANTIC_AVAILABLE = False

# Document processing libraries
import docx
from docx import Document
import PyPDF2
import markdown
from bs4 import BeautifulSoup

# Presentation generation
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Image generation for sketches
from PIL import Image, ImageDraw, ImageFont
import io

# Anthropic Claude for bullet point generation
import anthropic

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SlideContent:
    """Represents content for a single slide"""
    title: str
    content: List[str]
    slide_type: str = 'content'  # 'title', 'content', 'image', 'bullet'
    heading_level: Optional[int] = None  # Original heading level (1-6)
    subheader: Optional[str] = None  # Bold subheader above bullets
    visual_cues: Optional[List[str]] = None  # Visual/stage direction cues


@dataclass
class DocumentStructure:
    """Represents the parsed structure of a document"""
    title: str
    slides: List[SlideContent]
    metadata: Dict[str, Any]


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text"""
    text: str
    embedding: Optional[object] = None
    topic_cluster: Optional[int] = None
    intent: Optional[str] = None
    importance_score: float = 0.0


# ============================================================================
# GOOGLE DOCS INTEGRATION
# ============================================================================

def extract_google_doc_id(url: str) -> Optional[str]:
    """Extract document ID from Google Docs or Drive URL"""
    patterns = [
        r'/document/d/([a-zA-Z0-9-_]+)',  # docs.google.com/document/d/ID
        r'/file/d/([a-zA-Z0-9-_]+)',       # drive.google.com/file/d/ID
        r'id=([a-zA-Z0-9-_]+)',            # ?id=ID parameter
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def fetch_google_doc_content(doc_id: str, credentials=None) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch content from a Google Doc
    Returns: (content, error_message)
    """
    try:
        if credentials:
            # Use authenticated Google Docs API
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials

            # Rebuild credentials
            creds = Credentials(
                token=credentials['token'],
                refresh_token=credentials.get('refresh_token'),
                token_uri=credentials['token_uri'],
                client_id=credentials['client_id'],
                client_secret=credentials['client_secret'],
                scopes=credentials['scopes']
            )

            service = build('docs', 'v1', credentials=creds)
            document = service.documents().get(documentId=doc_id).execute()

            # Extract text content from document structure
            content = []
            for element in document.get('body', {}).get('content', []):
                # Extract paragraph text
                if 'paragraph' in element:
                    paragraph = element['paragraph']
                    paragraph_style = paragraph.get('paragraphStyle', {})
                    named_style_type = paragraph_style.get('namedStyleType', 'NORMAL_TEXT')

                    # Map heading styles to markdown
                    heading_map = {
                        'HEADING_1': 1, 'HEADING_2': 2, 'HEADING_3': 3,
                        'HEADING_4': 4, 'HEADING_5': 5, 'HEADING_6': 6
                    }
                    heading_level = heading_map.get(named_style_type)

                    # Extract text
                    paragraph_text = ''
                    for elem in paragraph.get('elements', []):
                        if 'textRun' in elem:
                            paragraph_text += elem['textRun']['content']

                    paragraph_text = paragraph_text.strip()

                    if paragraph_text:
                        if heading_level:
                            content.append('#' * heading_level + ' ' + paragraph_text)
                        else:
                            content.append(paragraph_text)

                # Extract table content
                elif 'table' in element:
                    table = element['table']
                    for row in table.get('tableRows', []):
                        row_cells = []
                        for cell in row.get('tableCells', []):
                            cell_text = []
                            for cell_element in cell.get('content', []):
                                if 'paragraph' in cell_element:
                                    for elem in cell_element['paragraph'].get('elements', []):
                                        if 'textRun' in elem:
                                            cell_text.append(elem['textRun']['content'].strip())
                            row_cells.append(' '.join(cell_text))
                        content.append('\t'.join(row_cells))

            return '\n'.join(content), None
        else:
            # Try public export URL
            export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
            response = requests.get(export_url, timeout=30)

            if response.status_code == 200:
                return response.text, None
            elif response.status_code == 403:
                return None, 'Document is not publicly accessible'
            elif response.status_code == 404:
                return None, 'Document not found'
            else:
                return None, f'Failed to fetch document (HTTP {response.status_code})'

    except Exception as e:
        logger.error(f"Error fetching Google Doc: {str(e)}")
        return None, f'Error fetching document: {str(e)}'


# ============================================================================
# SEMANTIC ANALYZER
# ============================================================================

class SemanticAnalyzer:
    """Handles semantic analysis of document content"""

    def __init__(self):
        self.model = None
        self.initialized = False
        self.use_heavy_analysis = False

        # Try heavy ML approach first
        if SEMANTIC_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.initialized = True
                self.use_heavy_analysis = True
                logging.info("Semantic analyzer initialized with sentence transformers")
            except Exception as e:
                logging.warning(f"Failed to initialize heavy semantic analyzer: {e}")

        # Fall back to lightweight approach
        if not self.initialized and LIGHTWEIGHT_SEMANTIC:
            try:
                self.initialized = True
                self.use_heavy_analysis = False
                logging.info("Semantic analyzer initialized with NLTK")
            except Exception as e:
                logging.warning(f"Failed to initialize lightweight semantic analyzer: {e}")

    def analyze_chunks(self, text_chunks: List[str]) -> List[SemanticChunk]:
        """Analyze text chunks for semantic content and clustering"""
        if not self.initialized or not text_chunks:
            return [SemanticChunk(text=chunk) for chunk in text_chunks]

        try:
            chunks = []
            for text in text_chunks:
                if len(text.strip()) < 10:
                    continue

                if self.use_heavy_analysis:
                    embedding = self.model.encode([text])[0]
                    intent = self._classify_intent(text)
                    importance = self._calculate_importance(text)
                else:
                    embedding = None
                    intent = self._classify_intent(text)
                    importance = self._calculate_importance(text)

                chunks.append(SemanticChunk(
                    text=text,
                    embedding=embedding,
                    intent=intent,
                    importance_score=importance
                ))

            # Cluster chunks by topic similarity
            if len(chunks) > 2:
                chunks = self._cluster_chunks(chunks)

            return chunks

        except Exception as e:
            logging.error(f"Error in semantic analysis: {e}")
            return [SemanticChunk(text=chunk) for chunk in text_chunks]

    def _classify_intent(self, text: str) -> str:
        """Classify the intent/purpose of a text chunk"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['learn', 'understand', 'explore']):
            return 'learning_objective'
        elif any(word in text_lower for word in ['step', 'process', 'method']):
            return 'process_description'
        elif any(word in text_lower for word in ['example', 'for instance']):
            return 'example'
        elif any(word in text_lower for word in ['definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['benefit', 'advantage', 'feature']):
            return 'benefits'
        else:
            return 'general_content'

    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score"""
        score = 0.0
        text_lower = text.lower()

        # Length factor
        score += min(len(text) / 200, 1.0) * 0.3

        # Key terms
        if any(term in text_lower for term in ['important', 'key', 'main', 'essential']):
            score += 0.4

        # Technical content
        if any(term in text_lower for term in ['data', 'system', 'process', 'method']):
            score += 0.3

        return min(score, 1.0)

    def _cluster_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Cluster chunks by topic similarity"""
        if self.use_heavy_analysis:
            try:
                embeddings = np.array([chunk.embedding for chunk in chunks])
                n_clusters = min(max(len(chunks) // 3, 2), 5)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                for i, chunk in enumerate(chunks):
                    chunk.topic_cluster = int(cluster_labels[i])

                return chunks
            except Exception as e:
                logging.error(f"Error in clustering: {e}")

        return chunks


# This is a partial extraction. The file is very large (8000+ lines).
# Would you like me to:
# 1. Continue extracting the DocumentParser class and other components?
# 2. Or provide a high-level overview of how to structure the refactoring?

