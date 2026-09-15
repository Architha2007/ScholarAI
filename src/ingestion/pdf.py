"""Extract plain text and metadata from a PDF file with robust validation."""

from io import BytesIO
import logging
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from src.config.settings import MAX_ANALYSIS_CHARS
from src.utils.validation import validate_pdf_file

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file) -> str:
    """Read a PDF upload and return all page text as one string."""
    return extract_pdf_details(pdf_file)["text"]


def extract_pdf_details(pdf_file) -> dict:
    """Read a PDF upload and return text plus simple document stats.

    Validates PDF file, checks for empty/corrupted/scanned documents, and keeps
    the first MAX_ANALYSIS_CHARS in memory for responsiveness.

    Args:
        pdf_file: Uploaded PDF file stream or object.

    Returns:
        Dict containing text, page_count, and char_count.

    Raises:
        ValueError: If PDF is missing, malformed, empty, or scanned
            without text.
    """

    validate_pdf_file(pdf_file)

    filename = getattr(pdf_file, "name", "document.pdf")
    logger.info("Extracting text from PDF: %s", filename)

    try:
        raw_bytes = pdf_file.read()
        pdf_file.seek(0)
        reader = PdfReader(BytesIO(raw_bytes))
    except (PyPdfError, Exception) as e:
        logger.error("Failed to parse PDF %s: %s", filename, e)
        raise ValueError(
            "The uploaded PDF is malformed or corrupted and could not be "
            "processed."
        ) from e

    page_count = len(reader.pages)
    if page_count == 0:
        logger.warning("PDF has 0 pages: %s", filename)
        raise ValueError("The uploaded PDF contains 0 pages.")

    stored_parts = []
    stored_chars = 0
    total_chars = 0

    try:
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            total_chars += len(text)

            if stored_chars < MAX_ANALYSIS_CHARS:
                remaining = MAX_ANALYSIS_CHARS - stored_chars
                stored_parts.append(text[:remaining])
                stored_chars += min(len(text), remaining)
    except Exception as e:
        logger.error("Error extracting text from page in %s: %s", filename, e)
        raise ValueError(
            "An error occurred while reading text pages from the PDF."
        ) from e

    extracted_text = "\n\n".join(stored_parts)
    if total_chars == 0 or not extracted_text.strip():
        logger.warning(
            "PDF contains no extractable text (likely scanned): %s", filename
        )
        raise ValueError(
            "No extractable text found in PDF. The document may be scanned, "
            "image-only, or password-protected."
        )

    logger.info(
        "Successfully extracted PDF details for %s: %d pages, %d chars.",
        filename,
        page_count,
        total_chars,
    )

    return {
        "text": extracted_text,
        "page_count": page_count,
        "char_count": total_chars,
    }
