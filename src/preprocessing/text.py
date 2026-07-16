"""Text trimming and processing statistics."""

from src.config.settings import INDEXING_CHAR_LIMIT, MAX_ANALYSIS_CHARS


def get_text_processing_stats(total_chars: int) -> dict:
    """Return how much of a PDF will be analyzed and indexed."""
    return {
        "total_chars": total_chars,
        "chars_analyzed": min(total_chars, MAX_ANALYSIS_CHARS),
        "chars_indexed": min(total_chars, INDEXING_CHAR_LIMIT),
        "is_large_pdf": total_chars > MAX_ANALYSIS_CHARS,
    }


def trim_text_for_analysis(text: str) -> str:
    """Keep only the first portion of text used for Gemini analysis."""
    return text[:MAX_ANALYSIS_CHARS]


def trim_text_for_indexing(text: str) -> str:
    """Keep only the first portion of text used for RAG indexing."""
    return text[:INDEXING_CHAR_LIMIT]
