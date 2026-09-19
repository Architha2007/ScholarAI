"""Standalone smoke test for Query Processor module.

Validates query expansion, decomposition, domain-specific
abbreviation handling, and API failure fallback handling
using real and mocked Gemini calls.
"""

import os
import sys
from unittest.mock import patch

# Add project root directory to sys.path to allow imports from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.retrieval.query_processor import process_query  # noqa: E402


def run_smoke_test() -> None:
    """Executes all test cases for the query processing layer."""
    print("=" * 80)
    print("STARTING SCHOLARAI QUERY PROCESSOR SMOKE TEST")
    print("=" * 80)

    # 1. Simple Query Case
    print("\n--- Test Case 1: Simple Query (Light Expansion) ---")
    q1 = "FAISS vector database search"
    res1 = process_query(q1)
    print(f"Original query: '{q1}'")
    print(f"Processed queries: {res1}")
    assert len(res1) >= 1, "Should return at least the original query."
    assert res1[0] == q1, "First query must be the original query."
    assert len(res1) <= 3, "Should return at most 3 queries."

    # 2. Complex Query Case
    print("\n--- Test Case 2: Complex Query (Decomposition) ---")
    q2 = (
        "How do we construct a FAISS index from document text "
        "and use Gemini to summarize research papers?"
    )
    res2 = process_query(q2)
    print(f"Original query: '{q2}'")
    print(f"Processed queries: {res2}")
    assert len(res2) >= 1, "Should return at least the original query."
    assert res2[0] == q2, "First query must be the original query."
    assert len(res2) <= 3, "Should return at most 3 queries."

    # 3. Abbreviation / Domain-specific Terminology Case
    print("\n--- Test Case 3: Abbreviations / Domain-Specific ---")
    q3 = "RAG vs BM25 for scientific paper retrieval"
    res3 = process_query(q3)
    print(f"Original query: '{q3}'")
    print(f"Processed queries: {res3}")
    assert len(res3) >= 1, "Should return at least the original query."
    assert res3[0] == q3, "First query must be the original query."
    assert len(res3) <= 3, "Should return at most 3 queries."

    # 4. Mocked API Failure Case
    print("\n--- Test Case 4: Mocked LLM Failure Fallback ---")
    q4 = "FAISS search keyword expansion"
    with patch(
        "src.retrieval.query_processor.get_generative_model"
    ) as mock_model:
        mock_model.side_effect = Exception("Simulated Gemini API Error")
        res4 = process_query(q4)
        print(f"Original query: '{q4}'")
        print(f"Processed queries: {res4}")
        assert res4 == [
            q4
        ], "Fallback must return only original query on error."
        print("Fallback test case passed!")

    print("\n" + "=" * 80)
    print("QUERY PROCESSOR SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
