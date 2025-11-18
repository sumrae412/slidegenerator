"""
Data Models for Slide Generator

Core data structures used throughout the slide generation pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SlideContent:
    """Represents content for a single slide"""
    title: str
    content: List[str]
    slide_type: str = 'content'  # 'title', 'content', 'image', 'bullet'
    heading_level: Optional[int] = None  # Original heading level from DOCX (1-6)
    subheader: Optional[str] = None  # Bold subheader above bullets (extracted topic sentence)
    visual_cues: Optional[List[str]] = None  # Visual/stage direction cues extracted from content


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
    embedding: Optional[object] = None  # Changed from np.ndarray to object for compatibility
    topic_cluster: Optional[int] = None
    intent: Optional[str] = None
    importance_score: float = 0.0
