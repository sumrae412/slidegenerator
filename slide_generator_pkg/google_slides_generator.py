"""
Google Slides Generator Module

Handles creation of Google Slides presentations from DocumentStructure objects.
Uses Google Slides API to create presentations with proper formatting and content.
"""

import logging
from typing import Dict, List, Any

from .data_models import DocumentStructure, SlideContent

logger = logging.getLogger(__name__)


class GoogleSlidesGenerator:
    """Handles creation of Google Slides presentations"""

    def __init__(self, credentials=None):
        """Initialize with Google OAuth credentials"""
        self.credentials = credentials
        self.service = None

        if credentials:
            try:
                from googleapiclient.discovery import build
                self.service = build('slides', 'v1', credentials=credentials)
                logger.info("✅ Google Slides service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Slides service: {e}")
                self.service = None

    def create_presentation(self, doc_structure: DocumentStructure) -> Dict[str, Any]:
        """Create a Google Slides presentation from document structure"""
        if not self.service:
            raise Exception("Google Slides service not initialized")

        try:
            # Create a new presentation
            presentation = {
                'title': doc_structure.title or 'Generated Presentation'
            }

            presentation_response = self.service.presentations().create(
                body=presentation
            ).execute()

            presentation_id = presentation_response.get('presentationId')
            logger.info(f"Created Google Slides presentation: {presentation_id}")

            # PHASE 1: Create all new slides (first slide already exists)
            slide_ids = []
            create_slide_requests = []

            for idx, slide in enumerate(doc_structure.slides):
                if idx == 0:
                    # Use the default first slide for title
                    slide_id = presentation_response['slides'][0]['objectId']
                else:
                    # Create new slides
                    slide_id = f'slide_{idx}'
                    create_slide_requests.append({
                        'createSlide': {
                            'objectId': slide_id,
                            'slideLayoutReference': {
                                'predefinedLayout': 'TITLE_AND_BODY'
                            }
                        }
                    })
                slide_ids.append(slide_id)

            # Execute slide creation batch
            if create_slide_requests:
                self.service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': create_slide_requests}
                ).execute()
                logger.info(f"Created {len(create_slide_requests)} new slides")

            # PHASE 2: Add content to all slides using placeholder IDs
            content_requests = []

            for idx, (slide_id, slide) in enumerate(zip(slide_ids, doc_structure.slides)):
                # Add content based on slide type
                if slide.slide_type == 'title':
                    content_requests.extend(
                        self._create_title_slide_requests(slide_id, slide, idx == 0, presentation_id)
                    )
                elif slide.slide_type == 'divider':
                    content_requests.extend(
                        self._create_divider_slide_requests(slide_id, slide, presentation_id)
                    )
                elif slide.slide_type in ['section', 'subsection']:
                    content_requests.extend(
                        self._create_section_slide_requests(slide_id, slide, presentation_id)
                    )
                else:  # content slide
                    content_requests.extend(
                        self._create_content_slide_requests(slide_id, slide, presentation_id)
                    )

            # Execute content insertion batch
            if content_requests:
                self.service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={'requests': content_requests}
                ).execute()
                logger.info(f"Added content to {len(doc_structure.slides)} slides")

            logger.info(f"Successfully created presentation with {len(doc_structure.slides)} slides")

            return {
                'presentation_id': presentation_id,
                'url': f'https://docs.google.com/presentation/d/{presentation_id}/edit',
                'slide_count': len(doc_structure.slides)
            }

        except Exception as e:
            logger.error(f"Error creating Google Slides presentation: {e}")
            raise

    def _get_placeholder_ids(self, presentation_id: str, slide_id: str) -> Dict[str, str]:
        """Get placeholder object IDs for a slide"""
        try:
            presentation = self.service.presentations().get(
                presentationId=presentation_id
            ).execute()

            # Find the slide
            for slide in presentation['slides']:
                if slide['objectId'] == slide_id:
                    placeholders = {}
                    # Look for placeholder shapes
                    for element in slide.get('pageElements', []):
                        if 'shape' in element:
                            shape = element['shape']
                            placeholder = shape.get('placeholder', {})
                            placeholder_type = placeholder.get('type', '')

                            if placeholder_type == 'TITLE' or placeholder_type == 'CENTERED_TITLE':
                                placeholders['title'] = element['objectId']
                            elif placeholder_type == 'SUBTITLE' or placeholder_type == 'BODY':
                                placeholders['body'] = element['objectId']

                    return placeholders
        except Exception as e:
            logger.error(f"Error getting placeholder IDs: {e}")
        return {}

    def _create_title_slide_requests(self, slide_id: str, slide: SlideContent, is_first: bool, presentation_id: str = None) -> List[Dict]:
        """Create requests for title slide"""
        requests = []

        # Use placeholders if we have presentation_id
        if presentation_id:
            placeholders = self._get_placeholder_ids(presentation_id, slide_id)

            if 'title' in placeholders:
                requests.append({
                    'insertText': {
                        'objectId': placeholders['title'],
                        'text': slide.title,
                        'insertionIndex': 0
                    }
                })

            if slide.content and 'body' in placeholders:
                requests.append({
                    'insertText': {
                        'objectId': placeholders['body'],
                        'text': '\n'.join(slide.content),
                        'insertionIndex': 0
                    }
                })

        return requests

    def _create_section_slide_requests(self, slide_id: str, slide: SlideContent, presentation_id: str = None) -> List[Dict]:
        """Create requests for section/subsection slide"""
        return self._create_title_slide_requests(slide_id, slide, False, presentation_id)

    def _create_divider_slide_requests(self, slide_id: str, slide: SlideContent, presentation_id: str = None) -> List[Dict]:
        """Create requests for section divider slide - minimal centered title"""
        requests = []

        # Use placeholders if we have presentation_id
        if presentation_id:
            placeholders = self._get_placeholder_ids(presentation_id, slide_id)

            if 'title' in placeholders:
                requests.append({
                    'insertText': {
                        'objectId': placeholders['title'],
                        'text': slide.title,
                        'insertionIndex': 0
                    }
                })

        return requests

    def _create_content_slide_requests(self, slide_id: str, slide: SlideContent, presentation_id: str = None) -> List[Dict]:
        """Create requests for content slide with bullets"""
        requests = []

        # Get placeholder IDs if we have presentation_id
        if presentation_id:
            placeholders = self._get_placeholder_ids(presentation_id, slide_id)

            # Insert title into title placeholder
            if 'title' in placeholders:
                requests.append({
                    'insertText': {
                        'objectId': placeholders['title'],
                        'text': slide.title,
                        'insertionIndex': 0
                    }
                })

            # Insert content into body placeholder (with optional subheader)
            if ('body' in placeholders) and (slide.content or slide.subheader):
                # Build text with subheader first if present, then bullets
                text_parts = []
                subheader_length = 0

                if slide.subheader:
                    text_parts.append(slide.subheader)
                    text_parts.append('')  # Blank line after subheader
                    subheader_length = len(slide.subheader) + 1  # +1 for newline

                if slide.content:
                    text_parts.extend([f'• {item}' for item in slide.content])

                full_text = '\n'.join(text_parts)

                requests.append({
                    'insertText': {
                        'objectId': placeholders['body'],
                        'text': full_text,
                        'insertionIndex': 0
                    }
                })

                # Make subheader bold if present
                if slide.subheader and subheader_length > 0:
                    requests.append({
                        'updateTextStyle': {
                            'objectId': placeholders['body'],
                            'textRange': {
                                'type': 'FIXED_RANGE',
                                'startIndex': 0,
                                'endIndex': subheader_length
                            },
                            'style': {
                                'bold': True,
                                'fontSize': {
                                    'magnitude': 18,
                                    'unit': 'PT'
                                }
                            },
                            'fields': 'bold,fontSize'
                        }
                    })

        return requests
