"""Extract plain text from a PDF file."""

from io import BytesIO

from pypdf import PdfReader

from rag_store import MAX_ANALYSIS_CHARS


def extract_text_from_pdf(pdf_file) -> str:
    """Read a PDF upload and return all page text as one string."""
    return extract_pdf_details(pdf_file)["text"]


def extract_pdf_details(pdf_file) -> dict:
    """
    Read a PDF upload and return text plus simple document stats.

    For very large PDFs, only the first MAX_ANALYSIS_CHARS are kept in memory
    so the app stays responsive. The full character count is still recorded.
    """
    raw_bytes = pdf_file.read()
    pdf_file.seek(0)

    reader = PdfReader(BytesIO(raw_bytes))
    stored_parts = []
    stored_chars = 0
    total_chars = 0

    for page in reader.pages:
        text = page.extract_text() or ""
        total_chars += len(text)

        if stored_chars < MAX_ANALYSIS_CHARS:
            remaining = MAX_ANALYSIS_CHARS - stored_chars
            stored_parts.append(text[:remaining])
            stored_chars += min(len(text), remaining)

    return {
        "text": "\n\n".join(stored_parts),
        "page_count": len(reader.pages),
        "char_count": total_chars,
    }
