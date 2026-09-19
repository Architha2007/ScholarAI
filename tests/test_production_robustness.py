"""Unit tests for Milestone 9 Production Robustness features.

Tests query validation, PDF validation, Gemini retry exponential backoff,
FAISS/BM25 retrieval fallbacks, CrossEncoder candidate safety, and structured
error handling.
"""

from io import BytesIO
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("."))

from src.generation.summarizer import analyze_research_paper  # noqa: E402
from src.ingestion.pdf import extract_pdf_details  # noqa: E402
from src.reranking.cross_encoder_reranker import rerank_results  # noqa: E402
from src.retrieval.hybrid_retriever import hybrid_search  # noqa: E402
from src.retrieval.qa import ask_paper_question  # noqa: E402
from src.utils.gemini import call_gemini_with_retry  # noqa: E402
from src.utils.validation import (  # noqa: E402
    validate_pdf_file,
    validate_query,
)


# --- 1. Query Validation Tests ---

def test_validate_query_valid():
    """Valid queries should return stripped text."""
    assert validate_query("  What is transformer?  ") == "What is transformer?"


def test_validate_query_empty():
    """Blank or empty queries should raise ValueError."""
    with pytest.raises(ValueError, match="empty or blank"):
        validate_query("   ")

    with pytest.raises(ValueError, match="non-empty string"):
        validate_query(None)


def test_validate_query_too_long():
    """Queries exceeding length limit should raise ValueError."""
    long_query = "a" * 1001
    with pytest.raises(ValueError, match="maximum allowed length"):
        validate_query(long_query)


# --- 2. PDF Validation Tests ---

def test_validate_pdf_file_non_pdf():
    """Non-PDF extension should raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "document.txt"
    with pytest.raises(ValueError, match="PDF documents"):
        validate_pdf_file(mock_file)


def test_validate_pdf_file_zero_bytes():
    """0-byte file should raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "empty.pdf"
    mock_file.size = 0
    with pytest.raises(ValueError, match="empty"):
        validate_pdf_file(mock_file)


def test_validate_pdf_file_oversized():
    """PDF exceeding size limit should raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "huge.pdf"
    mock_file.size = 60 * 1024 * 1024  # 60MB
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        validate_pdf_file(mock_file)


def test_validate_pdf_file_valid():
    """Valid PDF object should pass without raising."""
    mock_file = MagicMock()
    mock_file.name = "paper.pdf"
    mock_file.size = 2 * 1024 * 1024
    validate_pdf_file(mock_file)  # Should not raise


# --- 3. Ingestion Robustness Tests ---

def test_extract_pdf_details_corrupted():
    """Corrupted PDF stream should raise ValueError."""
    fake_stream = BytesIO(b"Not a real PDF header")
    fake_stream.name = "corrupt.pdf"
    fake_stream.size = len(fake_stream.getvalue())

    with pytest.raises(ValueError, match="malformed or corrupted"):
        extract_pdf_details(fake_stream)


def test_extract_pdf_details_scanned_image_only():

    """Scanned/image-only PDF with 0 text should raise ValueError."""
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "   "  # Pure whitespace
    mock_reader.pages = [mock_page]

    fake_stream = BytesIO(b"%PDF-1.4 scanned content")
    fake_stream.name = "scanned.pdf"
    fake_stream.size = len(fake_stream.getvalue())

    with patch("src.ingestion.pdf.PdfReader", return_value=mock_reader):
        with pytest.raises(ValueError, match="No extractable text"):
            extract_pdf_details(fake_stream)


# --- 4. Gemini API Retry Tests ---

def test_call_gemini_with_retry_success():
    """Successful function call should return result on first try."""
    func = MagicMock(return_value="AI Response")
    res = call_gemini_with_retry(func)
    assert res == "AI Response"
    assert func.call_count == 1


def test_call_gemini_with_retry_recover_after_retry():
    """Should retry after failure and succeed on second attempt."""
    func = MagicMock(side_effect=[Exception("429 Rate Limit"), "OK"])
    res = call_gemini_with_retry(func, initial_delay=0.01)
    assert res == "OK"
    assert func.call_count == 2


def test_call_gemini_with_retry_exhausted():
    """Exhausting retries should raise RuntimeError without leaking key."""
    func = MagicMock(side_effect=Exception("500 Internal Error"))
    with pytest.raises(RuntimeError, match="unavailable or rate-limited"):
        call_gemini_with_retry(func, max_retries=2, initial_delay=0.01)
    assert func.call_count == 2


# --- 5. Retrieval & Reranking Fallback Tests ---

def test_hybrid_search_faiss_failure_fallback():
    """If FAISS search fails, hybrid search should fall back to BM25."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("FAISS crash")

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [
        {"chunk_id": "1", "chunk_text": "text1", "retrieval_score": 0.9}
    ]

    res = hybrid_search(mock_vs, mock_bm25, "test query")
    assert len(res) == 1
    assert res[0]["chunk_id"] == "1"


def test_hybrid_search_both_failed():
    """If both FAISS and BM25 fail, hybrid search should return []."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.side_effect = Exception("FAISS crash")

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.side_effect = Exception("BM25 crash")

    res = hybrid_search(mock_vs, mock_bm25, "test query")
    assert res == []


def test_rerank_results_single_candidate():
    """Single candidate chunk should be returned directly with score."""
    chunks = [{"chunk_id": "1", "chunk_text": "text", "retrieval_score": 0.5}]
    res = rerank_results("query", chunks)
    assert len(res) == 1
    assert res[0]["reranker_score"] == 0.5


def test_rerank_results_inference_failure_fallback():
    """Model inference failure should fallback to original chunks."""

    chunks = [
        {"chunk_id": "1", "chunk_text": "text1", "retrieval_score": 0.8},
        {"chunk_id": "2", "chunk_text": "text2", "retrieval_score": 0.6},
    ]
    with patch(
        "src.reranking.cross_encoder_reranker._get_model"
    ) as mock_get_model:
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("CUDA Out of memory")
        mock_get_model.return_value = mock_model

        res = rerank_results("query", chunks)
        assert len(res) == 2
        assert res[0]["chunk_id"] == "1"


# --- 6. Summarizer & QA Failure Tests ---

def test_analyze_research_paper_empty_text():
    """Empty paper text should raise ValueError."""
    with pytest.raises(ValueError, match="empty paper text"):
        analyze_research_paper("  ")


def test_ask_paper_question_missing_vectorstore():
    """Missing vector store should return clear warning message."""
    res = ask_paper_question(None, "What is the paper about?")
    assert "No paper database is loaded" in res["answer"]
    assert res["source_chunks"] == []
