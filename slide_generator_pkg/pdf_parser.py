"""
PDF Parser Module

Handles parsing of PDF documents with table extraction and scanned PDF detection.
Supports multiple parsing backends with graceful fallback.
"""

import os
import logging
from typing import Tuple, List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Check PDF library availability with lazy imports
# Don't import at module level to avoid dependency errors
PDFPLUMBER_AVAILABLE = False
PYPDF2_AVAILABLE = False

def _check_pdfplumber():
    """Check if pdfplumber is available (lazy check)"""
    global PDFPLUMBER_AVAILABLE
    try:
        import pdfplumber
        PDFPLUMBER_AVAILABLE = True
        return True
    except Exception as e:
        PDFPLUMBER_AVAILABLE = False
        logger.warning(f"pdfplumber not available: {str(e)}")
        return False

def _check_pypdf2():
    """Check if PyPDF2 is available (lazy check)"""
    global PYPDF2_AVAILABLE
    try:
        import PyPDF2
        PYPDF2_AVAILABLE = True
        return True
    except Exception as e:
        PYPDF2_AVAILABLE = False
        logger.warning(f"PyPDF2 not available: {str(e)}")
        return False


class PDFParser:
    """
    Parses PDF files and extracts text content with table detection.

    Features:
    - Multi-backend support (pdfplumber, PyPDF2)
    - Table extraction as tab-delimited text
    - Scanned PDF detection
    - Page and table metadata tracking
    - Compatible with existing TXT parser pipeline

    Usage:
        parser = PDFParser()
        text_content, metadata = parser.parse_pdf('document.pdf')
    """

    def __init__(self):
        """Initialize PDF parser with available backends (lazy check)"""
        # Check libraries on-demand to avoid import errors
        self.pdfplumber_available = _check_pdfplumber()
        self.pypdf2_available = _check_pypdf2()

        if not self.pdfplumber_available and not self.pypdf2_available:
            logger.warning("No PDF parsing libraries available. Install pdfplumber or PyPDF2.")

    def parse_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main parsing method that tries pdfplumber first, then falls back to PyPDF2.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (text_content, metadata)
            - text_content: Extracted text with tables as tab-delimited format
            - metadata: Dict with page_count, table_count, is_scanned, backend_used

        Raises:
            ValueError: If no PDF parsing libraries are available
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not self.pdfplumber_available and not self.pypdf2_available:
            raise ValueError(
                "No PDF parsing libraries available. Please install pdfplumber or PyPDF2:\n"
                "  pip install pdfplumber\n"
                "  pip install PyPDF2"
            )

        logger.info(f"Parsing PDF: {file_path}")

        # Detect if PDF is scanned (image-based)
        is_scanned = self.detect_scanned_pdf(file_path)
        if is_scanned:
            logger.warning(
                f"PDF appears to be scanned/image-based. Text extraction may be limited. "
                f"Consider using OCR (Optical Character Recognition) for better results."
            )

        # Try pdfplumber first (better table detection)
        if self.pdfplumber_available:
            try:
                text_content, metadata = self._parse_with_pdfplumber(file_path)
                metadata['backend_used'] = 'pdfplumber'
                metadata['is_scanned'] = is_scanned
                logger.info(f"Successfully parsed with pdfplumber: {metadata['page_count']} pages, {metadata['table_count']} tables")
                return text_content, metadata
            except Exception as e:
                logger.warning(f"pdfplumber parsing failed: {e}")
                if self.pypdf2_available:
                    logger.info("Falling back to PyPDF2")
                else:
                    raise

        # Fallback to PyPDF2
        if self.pypdf2_available:
            try:
                text_content, metadata = self._parse_with_pypdf2(file_path)
                metadata['backend_used'] = 'PyPDF2'
                metadata['is_scanned'] = is_scanned
                logger.info(f"Successfully parsed with PyPDF2: {metadata['page_count']} pages")
                return text_content, metadata
            except Exception as e:
                logger.error(f"PyPDF2 parsing failed: {e}")
                raise

        raise ValueError("All PDF parsing attempts failed")

    def _parse_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse PDF using pdfplumber (best for tables).

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (text_content, metadata)
        """
        import pdfplumber

        text_parts = []
        table_count = 0
        page_count = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                logger.debug(f"Processing page {page_num}/{page_count}")

                # Extract tables first
                tables = page.extract_tables()

                if tables:
                    logger.debug(f"Found {len(tables)} table(s) on page {page_num}")
                    for table_num, table in enumerate(tables, start=1):
                        table_count += 1
                        table_text = self._table_to_text(table, page_num, table_num)
                        text_parts.append(table_text)
                        text_parts.append("")  # Blank line after table

                # Extract text content (excluding table areas if possible)
                # Note: pdfplumber's filter_edges parameter can help exclude table text
                # but for simplicity, we'll extract all text and rely on table markers
                page_text = page.extract_text()

                if page_text:
                    # Clean up the text
                    page_text = page_text.strip()
                    if page_text:
                        # Add page marker for debugging
                        text_parts.append(f"# Page {page_num}")
                        text_parts.append(page_text)
                        text_parts.append("")  # Blank line after page

        metadata = {
            'page_count': page_count,
            'table_count': table_count,
            'extraction_method': 'pdfplumber'
        }

        return '\n'.join(text_parts), metadata

    def _parse_with_pypdf2(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Fallback parser using PyPDF2 (no table detection).

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (text_content, metadata)
        """
        import PyPDF2

        text_parts = []
        page_count = 0

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                logger.debug(f"Processing page {page_num}/{page_count}")

                # Extract text from page
                page_text = page.extract_text()

                if page_text:
                    # Clean up the text
                    page_text = page_text.strip()
                    if page_text:
                        # Add page marker for debugging
                        text_parts.append(f"# Page {page_num}")
                        text_parts.append(page_text)
                        text_parts.append("")  # Blank line after page

        metadata = {
            'page_count': page_count,
            'table_count': 0,  # PyPDF2 doesn't detect tables
            'extraction_method': 'PyPDF2',
            'note': 'PyPDF2 does not detect tables - text only'
        }

        return '\n'.join(text_parts), metadata

    def _table_to_text(self, table: List[List[str]], page_num: int, table_num: int) -> str:
        """
        Convert table to tab-delimited text format compatible with _parse_txt().

        Args:
            table: 2D list of table cells from pdfplumber
            page_num: Page number where table appears
            table_num: Table number on the page

        Returns:
            Tab-delimited text representation of table
        """
        if not table or not table[0]:
            return ""

        text_parts = []

        # Add table header comment for metadata
        text_parts.append(f"# Table {table_num} (Page {page_num})")

        # Convert each row to tab-delimited format
        for row in table:
            if row:  # Skip empty rows
                # Clean up cell values (None -> empty string)
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]

                # Join cells with tabs
                row_text = '\t'.join(cleaned_row)

                # Only add non-empty rows
                if row_text.strip():
                    text_parts.append(row_text)

        return '\n'.join(text_parts)

    def detect_scanned_pdf(self, file_path: str, sample_pages: int = 3) -> bool:
        """
        Detect if PDF is scanned/image-based by checking text density.

        A scanned PDF typically has very little extractable text or none at all.
        We sample the first few pages and check average text length.

        Args:
            file_path: Path to PDF file
            sample_pages: Number of pages to sample (default: 3)

        Returns:
            True if PDF appears to be scanned, False otherwise
        """
        if not self.pdfplumber_available and not self.pypdf2_available:
            logger.warning("Cannot detect scanned PDF - no parsing libraries available")
            return False

        try:
            total_text_length = 0
            pages_checked = 0

            # Use pdfplumber if available, otherwise PyPDF2
            if self.pdfplumber_available:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    pages_to_check = min(sample_pages, len(pdf.pages))
                    for i in range(pages_to_check):
                        text = pdf.pages[i].extract_text()
                        if text:
                            total_text_length += len(text.strip())
                        pages_checked += 1

            elif self.pypdf2_available:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages_to_check = min(sample_pages, len(pdf_reader.pages))
                    for i in range(pages_to_check):
                        text = pdf_reader.pages[i].extract_text()
                        if text:
                            total_text_length += len(text.strip())
                        pages_checked += 1

            if pages_checked == 0:
                return True  # No pages could be checked - assume scanned

            # Calculate average text length per page
            avg_text_per_page = total_text_length / pages_checked

            # Threshold: if average text per page is less than 50 characters,
            # it's likely a scanned PDF or mostly images
            threshold = 50
            is_scanned = avg_text_per_page < threshold

            logger.debug(
                f"Scanned PDF detection: {avg_text_per_page:.1f} chars/page "
                f"(threshold: {threshold}) -> {'SCANNED' if is_scanned else 'TEXT-BASED'}"
            )

            return is_scanned

        except Exception as e:
            logger.warning(f"Error detecting scanned PDF: {e}")
            return False  # Assume not scanned if detection fails


# Module-level convenience function
def parse_pdf_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to parse a PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple of (text_content, metadata)

    Example:
        >>> text, metadata = parse_pdf_file('document.pdf')
        >>> print(f"Extracted {metadata['page_count']} pages")
        >>> print(f"Found {metadata['table_count']} tables")
    """
    parser = PDFParser()
    return parser.parse_pdf(file_path)


# Export public API
__all__ = ['PDFParser', 'parse_pdf_file', 'PDFPLUMBER_AVAILABLE', 'PYPDF2_AVAILABLE']
