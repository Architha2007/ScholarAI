"""Standalone smoke test for Hybrid Retrieval module with RRF.

This test validates that FAISS and BM25 results are successfully combined
using the RRF algorithm, and that signal merging works on edge cases.
"""

import os
import sys
from typing import List
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

# Add project root directory to sys.path to allow imports from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

# noqa comments bypass E402 imports warning
from src.retrieval.vector_store import (  # noqa: E402
    build_vector_store_from_chunks,
)
from src.retrieval.bm25_retriever import (  # noqa: E402
    build_bm25_index,
    search_bm25,
)
from src.retrieval.hybrid_retriever import hybrid_search  # noqa: E402


class DeterministicMockEmbeddings(Embeddings):
    """Deterministic mock embeddings class for FAISS indexing.

    Generates vectors based on keywords to simulate retrieval behavior
    without requiring external Gemini API requests.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

    def _embed(self, text: str) -> List[float]:
        # 3-dimensional embedding representing:
        # [semantic/dense, library/faiss, technical/terminological]
        vec = [0.0, 0.0, 0.0]
        text_lower = text.lower()
        if (
            "rag" in text_lower
            or "dense" in text_lower
            or "vector" in text_lower
        ):
            vec[0] = 1.0
        if "faiss" in text_lower or "semantic" in text_lower:
            # We give semantic a higher embedding weight to ensure FAISS ranks
            # Chunk 2 higher than Chunk 3 for queries containing both keywords.
            vec[1] = 2.0
        if "terminologicalquery" in text_lower:
            vec[2] = 1.0

        # Normalize vector
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            # Return dummy vector if no match
            vec = [1.0, 0.0, 0.0]
        return vec


def run_smoke_test() -> None:
    """Executes the RRF hybrid retrieval smoke test."""
    print("=" * 80)
    print("STARTING SCHOLARAI HYBRID RETRIEVAL SMOKE TEST (RRF)")
    print("=" * 80)

    # 1. Corpus chunks representing text returned from create_text_chunks()
    # Note: These are raw strings, matching the output of create_text_chunks()
    chunks = [
        "ScholarAI is a RAG assistant. It uses dense vector search.",
        "FAISS performs exceptionally well for capturing complex "
        "semantic concepts.",
        "BM25 looks for exact term matches and keyword-heavy technical "
        "terminology like 'TerminologicalQuery'.",
    ]

    print(f"Indexing {len(chunks)} raw text chunks...")
    for idx, c in enumerate(chunks):
        print(f"  - Chunk {idx + 1}: {c}")
    print()

    # 2. Patch GoogleGenerativeAIEmbeddings and api key to build indexes
    with patch(
        "src.retrieval.vector_store.GoogleGenerativeAIEmbeddings",
        return_value=DeterministicMockEmbeddings(),
    ), patch(
        "src.retrieval.vector_store._get_api_key",
        return_value="mock_key",
    ):
        print("Building vector store (FAISS)...")
        vector_store = build_vector_store_from_chunks(chunks)
        print("Vector store built successfully.\n")

    print("Building BM25 Index...")
    bm25_retriever = build_bm25_index(chunks)
    print("BM25 Index built successfully.\n")

    # 3. Define search queries including the edge case query
    # Query: 'semantic TerminologicalQuery TerminologicalQuery'
    # - BM25 will rank Chunk 3 (Rank 1) because keyword matches twice.
    # - FAISS will rank Chunk 2 (Rank 1) because 'semantic' has double weight.
    test_queries = [
        "TerminologicalQuery dense vector",
        "semantic TerminologicalQuery TerminologicalQuery",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80)

        # A. Execute individual FAISS search
        with patch(
            "src.retrieval.vector_store.GoogleGenerativeAIEmbeddings",
            return_value=DeterministicMockEmbeddings(),
        ), patch(
            "src.retrieval.vector_store._get_api_key",
            return_value="mock_key",
        ):
            faiss_results = vector_store.similarity_search(query, k=3)
            print("\n[FAISS ranking]")
            print(f"{'Rank':<5} | {'Chunk ID':<10} | {'Content'}")
            print("-" * 80)
            for rank, doc in enumerate(faiss_results, 1):
                chunk_id = doc.metadata.get("chunk_id", "?")
                print(f"{rank:<5} | {chunk_id:<10} | {doc.page_content}")

        # B. Execute individual BM25 search
        bm25_results = search_bm25(bm25_retriever, query, top_k=3)
        print("\n[BM25 ranking]")
        print(
            f"{'Rank':<5} | {'Chunk ID':<10} | {'Score':<10} | {'Content'}"
        )
        print("-" * 80)
        for rank, res in enumerate(bm25_results, 1):
            chunk_id = res["chunk_id"]
            score = f"{res['retrieval_score']:.4f}"
            print(
                f"{rank:<5} | {chunk_id:<10} | {score:<10} | "
                f"{res['chunk_text']}"
            )

        # C. Execute Fused search (RRF)
        with patch(
            "src.retrieval.vector_store.GoogleGenerativeAIEmbeddings",
            return_value=DeterministicMockEmbeddings(),
        ), patch(
            "src.retrieval.vector_store._get_api_key",
            return_value="mock_key",
        ):
            fused_results = hybrid_search(
                vector_store, bm25_retriever, query, top_k=3
            )
            print("\n[Fused ranking]")
            print(
                f"{'Rank':<5} | {'Chunk ID':<10} | {'RRF Score':<10} | "
                f"{'Content'}"
            )
            print("-" * 80)
            for rank, res in enumerate(fused_results, 1):
                chunk_id = res["chunk_id"]
                score = f"{res['retrieval_score']:.6f}"
                print(
                    f"{rank:<5} | {chunk_id:<10} | {score:<10} | "
                    f"{res['chunk_text']}"
                )
            print()

        # D. Perform verification checks on the edge case
        if query == "semantic TerminologicalQuery TerminologicalQuery":
            # FAISS ranking: 2 (Rank 1), 3 (Rank 2), 1 (Rank 3)
            # BM25 ranking: 3 (Rank 1), 2 (Rank 2), 1 (Rank 3)
            # Check FAISS ranks:
            faiss_ids = [
                str(doc.metadata.get("chunk_id")) for doc in faiss_results
            ]
            assert faiss_ids[0] == "2", "FAISS should rank Chunk 2 first."
            assert faiss_ids[1] == "3", "FAISS should rank Chunk 3 second."

            # Check BM25 ranks:
            bm25_ids = [str(res["chunk_id"]) for res in bm25_results]
            assert bm25_ids[0] == "3", "BM25 should rank Chunk 3 first."
            assert bm25_ids[1] == "2", "BM25 should rank Chunk 2 second."

            # RRF score check:
            # Chunk 2 and Chunk 3 should both be above Chunk 1.
            fused_ids = [str(r["chunk_id"]) for r in fused_results]
            assert fused_ids[2] == "1", "Chunk 1 should be ranked last."
            assert set(fused_ids[:2]) == {
                "2",
                "3",
            }, "Top 2 should contain Chunk 2 and Chunk 3."
            print("Edge-case rank verification completed: signals fused!")

    # Verification of stable identifier usage
    # IDENTIFIER MAPPING DOCUMENTATION:
    # We use `chunk_id` as the stable key to map and match chunks between
    # FAISS and BM25. In the ScholarAI pipeline, chunk_id is a 1-based integer
    # corresponding to the index of the chunk in the document (idx + 1). In
    # the fusion module, we convert the key to its string representation
    # (e.g. str(chunk_id)) to safely key the dictionary.
    print("=" * 80)
    print("HYBRID RETRIEVAL SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
