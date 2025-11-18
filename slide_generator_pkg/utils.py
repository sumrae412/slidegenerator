"""
Utility Functions for Slide Generator

Helper functions for Google Docs integration, configuration, and other utilities.
"""

import os
import json
import re
import logging
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_google_client_config():
    """Get Google OAuth client configuration from env var or file"""
    # Try environment variable first (for Heroku)
    credentials_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    if credentials_json:
        return json.loads(credentials_json)

    # Fall back to credentials.json file (for local dev)
    credentials_file = os.environ.get('GOOGLE_CLIENT_SECRETS_FILE', 'credentials.json')
    if os.path.exists(credentials_file):
        with open(credentials_file, 'r') as f:
            return json.load(f)

    return None


def extract_google_doc_id(url: str) -> Optional[str]:
    """Extract document ID from Google Docs or Drive URL"""
    # Match various Google Docs and Drive URL patterns
    patterns = [
        r'/document/d/([a-zA-Z0-9-_]+)',  # docs.google.com/document/d/ID
        r'/file/d/([a-zA-Z0-9-_]+)',       # drive.google.com/file/d/ID (from Picker)
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

    Args:
        doc_id: The Google Doc ID
        credentials: Optional OAuth credentials dict with keys:
                    token, refresh_token, token_uri, client_id, client_secret, scopes

    Returns:
        (content, error_message) - content is None if error occurred
    """
    try:
        if credentials:
            # Use authenticated Google Docs API for better formatting
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

            # Extract text content from the document structure (paragraphs AND tables)
            content = []
            for element in document.get('body', {}).get('content', []):
                # Extract paragraph text
                if 'paragraph' in element:
                    paragraph = element['paragraph']

                    # Check if this is a heading by examining the paragraph style
                    paragraph_style = paragraph.get('paragraphStyle', {})
                    named_style_type = paragraph_style.get('namedStyleType', 'NORMAL_TEXT')

                    # Map Google Docs heading styles to markdown heading levels
                    heading_map = {
                        'HEADING_1': 1,
                        'HEADING_2': 2,
                        'HEADING_3': 3,
                        'HEADING_4': 4,
                        'HEADING_5': 5,
                        'HEADING_6': 6
                    }

                    heading_level = heading_map.get(named_style_type)

                    # Extract text from paragraph elements
                    paragraph_text = ''
                    paragraph_elements = paragraph.get('elements', [])
                    for elem in paragraph_elements:
                        if 'textRun' in elem:
                            paragraph_text += elem['textRun']['content']

                    # Strip whitespace and newlines
                    paragraph_text = paragraph_text.strip()

                    # Only add non-empty paragraphs
                    if paragraph_text:
                        # Format headings with markdown syntax
                        if heading_level:
                            content.append('#' * heading_level + ' ' + paragraph_text)
                            logger.info(f"Extracted H{heading_level} heading from Google Doc: {paragraph_text[:50]}...")
                        else:
                            content.append(paragraph_text)

                # Extract table content (tab-delimited, matching .txt export format)
                elif 'table' in element:
                    table = element['table']
                    for row in table.get('tableRows', []):
                        row_cells = []
                        for cell in row.get('tableCells', []):
                            # Extract all text from the cell
                            cell_text = []
                            for cell_element in cell.get('content', []):
                                if 'paragraph' in cell_element:
                                    for elem in cell_element['paragraph'].get('elements', []):
                                        if 'textRun' in elem:
                                            cell_text.append(elem['textRun']['content'].strip())
                            row_cells.append(' '.join(cell_text))

                        # Join cells with tabs to match .txt export format
                        content.append('\t'.join(row_cells))

            return '\n'.join(content), None
        else:
            # Try public export URL (works if document is set to "Anyone with link can view")
            export_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'
            response = requests.get(export_url, timeout=30)

            if response.status_code == 200:
                return response.text, None
            elif response.status_code == 403:
                return None, 'Document is not publicly accessible. Please set sharing to "Anyone with the link can view" or authenticate with Google.'
            elif response.status_code == 404:
                return None, 'Document not found. Please check the URL.'
            else:
                return None, f'Failed to fetch document (HTTP {response.status_code})'

    except Exception as e:
        logger.error(f"Error fetching Google Doc: {str(e)}")
        error_str = str(e).lower()

        # Check if this is a non-Google Docs file (like .docx, .pdf, etc.)
        if 'not supported' in error_str or '400' in error_str:
            return None, 'This file is not a Google Doc. Please convert .docx/.pdf files to Google Docs format first, or use the "Browse Google Drive" button to select a Google Doc.'

        return None, f'Error fetching document: {str(e)}'
