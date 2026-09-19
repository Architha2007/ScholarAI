"""Input validation module for ScholarAI (Milestone 9).

Provides reusable validation logic for search queries and uploaded PDF files
to ensure production robustness against empty inputs, invalid formats,
and oversized files.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


def validate_query(query: Any, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Validates and sanitizes user search query input.

    Args:
        query: Search query input object or string.
        max_length: Maximum allowed character count.

    Returns:
        Stripped string query.

    Raises:
        ValueError: If query is missing, non-string, blank, or too long.
    """
    if query is None or not isinstance(query, str):
        logger.warning("Invalid query type received: %s", type(query))
        raise ValueError("Query must be a valid non-empty string.")

    cleaned_query = query.strip()
    if not cleaned_query:
        logger.warning("Blank query received.")
        raise ValueError("Query cannot be empty or blank.")

    if len(cleaned_query) > max_length:
        logger.warning(
            "Query length (%d) exceeds limit (%d).",
            len(cleaned_query),
            max_length,
        )
        raise ValueError(
            f"Query exceeds maximum allowed length of {max_length} characters."
        )

    return cleaned_query


def validate_pdf_file(
    pdf_file: Any, max_size_bytes: int = MAX_PDF_SIZE_BYTES
) -> None:
    """Validates an uploaded PDF file stream or object.

    Args:
        pdf_file: File upload object (e.g., BytesIO or Streamlit UploadedFile).
        max_size_bytes: Maximum allowed file size in bytes.

    Raises:
        ValueError: If file is missing, not a PDF, empty (0 bytes),
            or too large.
    """
    if pdf_file is None:
        logger.warning("No PDF file provided for validation.")
        raise ValueError("No PDF file provided.")

    filename = getattr(pdf_file, "name", "document.pdf")
    if filename and not filename.lower().endswith(".pdf"):
        logger.warning("Unsupported file type uploaded: %s", filename)
        raise ValueError(
            "Invalid file type. ScholarAI supports PDF documents (.pdf) only."
        )

    size = getattr(pdf_file, "size", None)
    has_seek = hasattr(pdf_file, "seek") and hasattr(pdf_file, "tell")
    if size is None and has_seek:
        curr_pos = pdf_file.tell()
        pdf_file.seek(0, 2)
        size = pdf_file.tell()
        pdf_file.seek(curr_pos)

    if size == 0:

        logger.warning("Uploaded PDF is 0 bytes: %s", filename)
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    if size is not None and size > max_size_bytes:
        size_mb = size / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        logger.warning(
            "PDF size (%.2f MB) exceeds maximum limit (%.2f MB).",
            size_mb,
            limit_mb,
        )
        raise ValueError(
            f"PDF size ({size_mb:.1f}MB) exceeds maximum limit of "
            f"{limit_mb:.0f}MB."
        )
