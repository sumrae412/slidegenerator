"""
Utility Functions for Slide Generator

Helper functions for Google Docs integration, configuration, and other utilities.
"""

import os
import json
import re
import logging
import requests
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Tracks API usage and costs for Claude and OpenAI API calls.

    Features:
    - Per-slide and per-document cost tracking
    - Separate tracking for different model types
    - Cache hit/miss tracking for cost savings estimation
    - JSON export for cost reports
    - Real-time cost statistics

    Pricing (as of 2025):
    - Claude Sonnet: $3/1M input, $15/1M output
    - OpenAI GPT-4o: $2.50/1M input, $10/1M output
    - OpenAI GPT-3.5-turbo: $0.50/1M input, $1.50/1M output
    - OpenAI text-embedding-3-small: $0.02/1M tokens
    """

    # Pricing per 1M tokens (in USD)
    PRICING = {
        'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
        'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
        'gpt-4o': {'input': 2.50, 'output': 10.00},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
        'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        'text-embedding-3-small': {'input': 0.02, 'output': 0.00},
        'text-embedding-3-large': {'input': 0.13, 'output': 0.00},
        'text-embedding-ada-002': {'input': 0.10, 'output': 0.00},
    }

    def __init__(self):
        """Initialize cost tracker with empty tracking data"""
        self.reset()

    def reset(self):
        """Reset all tracking data"""
        self.calls = []  # List of all API calls with details
        self.slide_costs = defaultdict(lambda: {
            'calls': [],
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0
        })
        self.cache_hits = 0
        self.cache_misses = 0
        self.session_start = datetime.now()

    def track_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
        slide_id: Optional[str] = None,
        call_type: str = 'chat',
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Track a single API call with token usage and cost.

        Args:
            provider: 'claude' or 'openai'
            model: Model name (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4o')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            cached: Whether this was a cache hit
            slide_id: Optional identifier for the slide being processed
            call_type: Type of call ('chat', 'embedding', 'refinement', etc.)
            success: Whether the API call succeeded
            error: Error message if call failed
        """
        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Track cache statistics
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        # Create call record
        call_record = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'model': model,
            'call_type': call_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'cached': cached,
            'success': success,
            'error': error,
            'slide_id': slide_id
        }

        # Add to global calls list
        self.calls.append(call_record)

        # If associated with a slide, track per-slide costs
        if slide_id:
            self.slide_costs[slide_id]['calls'].append(call_record)
            if not cached:  # Only count non-cached calls toward slide cost
                self.slide_costs[slide_id]['input_tokens'] += input_tokens
                self.slide_costs[slide_id]['output_tokens'] += output_tokens
                self.slide_costs[slide_id]['cost'] += cost

        logger.debug(
            f"Tracked {provider} {model} call: "
            f"{input_tokens} in + {output_tokens} out = ${cost:.4f} "
            f"{'(cached)' if cached else ''}"
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a specific model and token usage"""
        # Normalize model name to match pricing keys
        model_key = model.lower()

        # Find matching pricing (handle version suffixes)
        pricing = None
        for key in self.PRICING:
            if model_key.startswith(key.lower()) or key.lower().startswith(model_key):
                pricing = self.PRICING[key]
                break

        if not pricing:
            logger.warning(f"No pricing data for model '{model}', using Claude Sonnet default")
            pricing = self.PRICING['claude-3-5-sonnet']

        # Cost = (tokens / 1M) * price_per_1M
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost

    def get_total_cost(self, exclude_cached: bool = True) -> float:
        """
        Get total cost across all API calls.

        Args:
            exclude_cached: If True, don't count cached calls in total cost

        Returns:
            Total cost in USD
        """
        if exclude_cached:
            return sum(call['cost'] for call in self.calls if not call['cached'])
        else:
            return sum(call['cost'] for call in self.calls)

    def get_total_tokens(self) -> Dict[str, int]:
        """Get total token usage across all calls"""
        return {
            'input_tokens': sum(call['input_tokens'] for call in self.calls if not call['cached']),
            'output_tokens': sum(call['output_tokens'] for call in self.calls if not call['cached']),
            'total_tokens': sum(call['total_tokens'] for call in self.calls if not call['cached']),
            'cached_tokens_saved': sum(call['total_tokens'] for call in self.calls if call['cached'])
        }

    def get_cost_by_provider(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by provider (Claude vs OpenAI)"""
        breakdown = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0
        })

        for call in self.calls:
            if not call['cached']:
                provider = call['provider']
                breakdown[provider]['calls'] += 1
                breakdown[provider]['input_tokens'] += call['input_tokens']
                breakdown[provider]['output_tokens'] += call['output_tokens']
                breakdown[provider]['cost'] += call['cost']

        return dict(breakdown)

    def get_cost_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by specific model"""
        breakdown = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0
        })

        for call in self.calls:
            if not call['cached']:
                model = call['model']
                breakdown[model]['calls'] += 1
                breakdown[model]['input_tokens'] += call['input_tokens']
                breakdown[model]['output_tokens'] += call['output_tokens']
                breakdown[model]['cost'] += call['cost']

        return dict(breakdown)

    def get_cost_by_call_type(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by call type (chat, embedding, refinement, etc.)"""
        breakdown = defaultdict(lambda: {
            'calls': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0.0
        })

        for call in self.calls:
            if not call['cached']:
                call_type = call['call_type']
                breakdown[call_type]['calls'] += 1
                breakdown[call_type]['input_tokens'] += call['input_tokens']
                breakdown[call_type]['output_tokens'] += call['output_tokens']
                breakdown[call_type]['cost'] += call['cost']

        return dict(breakdown)

    def get_slide_costs(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown per slide"""
        return dict(self.slide_costs)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics and estimated savings"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        # Calculate savings from caching
        cached_cost = sum(call['cost'] for call in self.calls if call['cached'])

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 1),
            'estimated_cost_without_cache': self.get_total_cost(exclude_cached=False),
            'actual_cost_with_cache': self.get_total_cost(exclude_cached=True),
            'cost_savings_usd': cached_cost,
            'cost_savings_percent': round((cached_cost / (self.get_total_cost(exclude_cached=False) or 1)) * 100, 1)
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        tokens = self.get_total_tokens()
        cache_stats = self.get_cache_statistics()

        return {
            'session_start': self.session_start.isoformat(),
            'session_duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'total_cost_usd': round(self.get_total_cost(exclude_cached=True), 4),
            'total_calls': len([c for c in self.calls if not c['cached']]),
            'successful_calls': len([c for c in self.calls if c['success'] and not c['cached']]),
            'failed_calls': len([c for c in self.calls if not c['success'] and not c['cached']]),
            'tokens': tokens,
            'cache_statistics': cache_stats,
            'cost_by_provider': self.get_cost_by_provider(),
            'cost_by_model': self.get_cost_by_model(),
            'cost_by_call_type': self.get_cost_by_call_type(),
            'slides_processed': len(self.slide_costs),
            'avg_cost_per_slide': round(
                self.get_total_cost() / len(self.slide_costs) if self.slide_costs else 0,
                4
            )
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed report including all individual calls"""
        summary = self.get_summary()
        summary['individual_calls'] = self.calls
        summary['slide_breakdown'] = self.get_slide_costs()
        return summary

    def export_to_json(self, filepath: str, detailed: bool = True):
        """
        Export cost tracking data to JSON file.

        Args:
            filepath: Path to output JSON file
            detailed: If True, include all individual calls; if False, summary only
        """
        if detailed:
            data = self.get_detailed_report()
        else:
            data = self.get_summary()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Cost report exported to {filepath}")

    def print_summary(self):
        """Print human-readable cost summary to console"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("API COST TRACKING SUMMARY")
        print("="*70)

        print(f"\nSession Duration: {summary['session_duration_seconds']:.1f} seconds")
        print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Total Calls: {summary['total_calls']} ({summary['successful_calls']} successful, {summary['failed_calls']} failed)")

        print(f"\nToken Usage:")
        print(f"  Input Tokens:  {summary['tokens']['input_tokens']:,}")
        print(f"  Output Tokens: {summary['tokens']['output_tokens']:,}")
        print(f"  Total Tokens:  {summary['tokens']['total_tokens']:,}")
        print(f"  Cached Tokens Saved: {summary['tokens']['cached_tokens_saved']:,}")

        cache = summary['cache_statistics']
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {cache['hit_rate_percent']:.1f}% ({cache['cache_hits']}/{cache['total_requests']} requests)")
        print(f"  Cost Savings: ${cache['cost_savings_usd']:.4f} ({cache['cost_savings_percent']:.1f}%)")

        print(f"\nCost by Provider:")
        for provider, data in summary['cost_by_provider'].items():
            print(f"  {provider.upper()}: ${data['cost']:.4f} ({data['calls']} calls, {data['total_tokens']:,} tokens)")

        print(f"\nCost by Model:")
        for model, data in summary['cost_by_model'].items():
            print(f"  {model}: ${data['cost']:.4f} ({data['calls']} calls)")

        if summary['slides_processed'] > 0:
            print(f"\nSlide Processing:")
            print(f"  Slides Processed: {summary['slides_processed']}")
            print(f"  Avg Cost per Slide: ${summary['avg_cost_per_slide']:.4f}")

        print("\n" + "="*70 + "\n")


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
