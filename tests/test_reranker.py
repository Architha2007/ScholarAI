"""Smoke test for Cross-Encoder reranker module."""

import os
import sys
from unittest.mock import patch

# Add project root directory to sys.path to allow imports from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.reranking.cross_encoder_reranker import rerank_results  # noqa: E402


def run_smoke_test() -> None:
    """Executes all test cases for the Cross-Encoder reranking layer."""
    print("=" * 80)
    print("STARTING SCHOLARAI CROSS-ENCODER RERANKER SMOKE TEST")
    print("=" * 80)

    # 1. Prepare query and mock retrieved chunks where lexical rank is
    # imperfect
    query = "FAISS vector database for embedding search"

    mock_chunks = [
        {
            "chunk_id": "chunk_1",
            "chunk_text": "A general guide on relational databases like "
                          "MySQL and PostgreSQL for storing data.",
            "retrieval_score": 18.5,
            "metadata": {"doc_id": "doc_101"}
        },
        {
            "chunk_id": "chunk_2",
            "chunk_text": "How to construct a vector index using FAISS "
                          "(Facebook AI Similarity Search) for fast "
                          "approximate embedding retrieval.",
            "retrieval_score": 12.3,
            "metadata": {"doc_id": "doc_102"}
        },
        {
            "chunk_id": "chunk_3",
            "chunk_text": "A survey of general software engineering "
                          "practices and tool sets.",
            "retrieval_score": 5.1,
            "metadata": {"doc_id": "doc_103"}
        }
    ]

    print("\n--- Initial Retrieval Order (Imperfect Lexical Rank) ---")
    for idx, c in enumerate(mock_chunks):
        print(
            f"{idx + 1}. ID: {c['chunk_id']}, Score: {c['retrieval_score']}, "
            f"Text: '{c['chunk_text']}'"
        )

    # 2. Run the reranker
    print("\n--- Running Reranker ---")
    reranked = rerank_results(query, mock_chunks, top_k=3)

    print("\n--- Reranked Order ---")
    for idx, c in enumerate(reranked):
        print(
            f"{idx + 1}. ID: {c['chunk_id']}, "
            f"Retrieval Score: {c['retrieval_score']}, "
            f"Reranker Score: {c.get('reranker_score')}, "
            f"Text: '{c['chunk_text']}'"
        )

    # Assert that chunk_2 is now ranked first due to high semantic relevance
    assert len(reranked) == 3, "Should return 3 chunks."
    assert reranked[0]["chunk_id"] == "chunk_2", (
        "Chunk 2 (FAISS guide) must be reranked to first place."
    )
    assert "reranker_score" in reranked[0], "Score field must be added."
    print("Semantic reranking test passed!")

    # 3. Edge Cases
    print("\n--- Test Edge Cases (Empty/Null Inputs) ---")
    # Empty chunks
    res_empty = rerank_results(query, [])
    assert res_empty == [], "Should return empty list on empty chunks."

    # Blank query
    res_blank_query = rerank_results("   ", mock_chunks, top_k=2)
    assert len(res_blank_query) == 2, "Should return top_k chunks."
    assert res_blank_query[0]["chunk_id"] == "chunk_1", (
        "Should preserve original order on blank query."
    )
    print("Edge cases tested successfully!")

    # 4. Failure Fallback Test
    print("\n--- Test Failure Fallback ---")
    # Mocking _get_model to return None to simulate load/inference failure
    with patch("src.reranking.cross_encoder_reranker._get_model") as mock_get:
        mock_get.return_value = None
        res_fail = rerank_results(query, mock_chunks, top_k=3)
        assert len(res_fail) == 3, "Should return 3 chunks on fallback."
        assert res_fail[0]["chunk_id"] == "chunk_1", (
            "Fallback must preserve original retrieval order."
        )
        print("Graceful fallback test passed!")

    print("\n" + "=" * 80)
    print("CROSS-ENCODER RERANKER SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
