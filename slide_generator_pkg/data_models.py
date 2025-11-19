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
    speaker_notes: Optional[str] = None  # Full script text for presenter (brackets removed)
    # Visual generation fields
    visual_prompt: Optional[str] = None  # AI-generated DALL-E prompt for this slide
    visual_image_url: Optional[str] = None  # Generated image URL (remote or local path)
    visual_image_path: Optional[str] = None  # Local cached image path
    visual_type: Optional[str] = None  # Visual type: 'technical', 'data', 'concept', etc.


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
