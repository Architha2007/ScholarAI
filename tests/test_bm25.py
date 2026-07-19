"""Standalone smoke test for BM25 Sparse Retrieval module.

This script tests the BM25 indexer and search retrieval using mock chunk
objects that resemble the outputs of the create_text_chunks() pipeline.
It verifies indexing, score matching, and serialization.
"""

import os
import sys

# Add project root directory to sys.path to allow imports from src/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

# noqa comments are added to bypass E402 since sys.path modification is required
from src.retrieval.bm25_retriever import (  # noqa: E402
    BM25Retriever,
    build_bm25_index,
    search_bm25,
)


class MockChunk:
    """Mock chunk class representing chunk objects.

    Emulates the structure of chunks produced by the create_text_chunks()
    pipeline, containing id, text, and metadata attributes.
    """

    def __init__(self, chunk_id: str, text: str, metadata: dict = None):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata or {}


def mock_create_text_chunks(
    text: str,
    chunk_size: int = 15,
    chunk_overlap: int = 3,
    doc_id: str = "doc",
) -> list:
    """Mock chunker helper that outputs MockChunk objects for testing."""
    words = text.split()
    chunks = []
    chunk_idx = 0
    i = 0
    step = chunk_size - chunk_overlap

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text_content = " ".join(chunk_words)
        chunks.append(
            MockChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                text=chunk_text_content,
                metadata={
                    "doc_id": doc_id,
                    "start_word_index": i,
                    "end_word_index": min(i + chunk_size, len(words)),
                },
            )
        )
        chunk_idx += 1
        i += step
        if i >= len(words) or step <= 0:
            break

    return chunks


def run_smoke_test() -> None:
    """Executes the BM25 retrieval smoke test pipeline using chunk objects."""
    print("=" * 80)
    print("STARTING SCHOLARAI BM25 SMOKE TEST (OBJECT-BASED)")
    print("=" * 80)

    # 1. Mock document corpus representing research documents
    doc1 = (
        "ScholarAI is a powerful tool designed to build "
        "production-grade RAG pipelines. It supports dense retrieval "
        "using FAISS, sparse retrieval using BM25, and hybrid search. "
        "The system aims to help researchers retrieve academic papers "
        "and summarize them using Gemini."
    )
    doc2 = (
        "Dense retrieval methods use vector embeddings to capture "
        "semantic similarity. FAISS is an efficient library for dense "
        "vector similarity search. It performs exceptionally well "
        "for capturing complex semantic concepts."
    )
    doc3 = (
        "Sparse retrieval algorithms, such as BM25 (Best Matching 25), "
        "look for exact keyword matches. Unlike dense retrieval, BM25 is "
        "highly effective for technical terminology, specific terms, "
        "names, and keyword-heavy search queries."
    )

    # 2. Extract chunks using mock chunking (generates MockChunk objects)
    chunks = []
    chunks.extend(
        mock_create_text_chunks(
            doc1, chunk_size=15, chunk_overlap=3, doc_id="scholarai_intro"
        )
    )
    chunks.extend(
        mock_create_text_chunks(
            doc2, chunk_size=15, chunk_overlap=3, doc_id="dense_retrieval"
        )
    )
    chunks.extend(
        mock_create_text_chunks(
            doc3, chunk_size=15, chunk_overlap=3, doc_id="bm25_sparse"
        )
    )

    print(f"Produced {len(chunks)} chunk objects using MockChunk class.")
    for c in chunks:
        words_count = len(c.text.split())
        print(f"  - Chunk ID: {c.chunk_id} | Words: {words_count}")
    print()

    # 3. Build index
    print("Building BM25 Index...")
    retriever = build_bm25_index(chunks)
    print("BM25 Index built successfully.\n")

    # 4. Verify serialization (load and save index)
    temp_dir = "data/processed"
    os.makedirs(temp_dir, exist_ok=True)
    temp_index_path = os.path.join(temp_dir, "bm25_index_test.pkl")

    print(f"Saving index to {temp_index_path}...")
    retriever.save(temp_index_path)

    print("Loading index back from disk...")
    loaded_retriever = BM25Retriever.load(temp_index_path)
    print("Index loaded successfully.\n")

    # 5. Execute search queries
    test_queries = [
        "FAISS dense retrieval",
        "BM25 keyword matches",
        "ScholarAI summarization with Gemini",
        "nonexistent term search",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80)

        # Retrieve top 3 results
        results = search_bm25(loaded_retriever, query, top_k=3)

        # Print formatted header
        print(
            f"{'Rank':<5} | {'Chunk ID':<25} | {'Score':<10} | "
            f"{'Chunk Preview'}"
        )
        print("-" * 80)

        for rank, res in enumerate(results, 1):
            chunk_id = res["chunk_id"]
            score = f"{res['retrieval_score']:.4f}"
            text_preview = res["chunk_text"]
            if len(text_preview) > 55:
                text_preview = text_preview[:52] + "..."
            text_preview = text_preview.replace("\n", " ")

            print(
                f"{rank:<5} | {chunk_id:<25} | {score:<10} | {text_preview}"
            )
        print()

    # 6. Cleanup temporary file
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)
        print("Cleaned up temporary serialized index file.")

    print("\n" + "=" * 80)
    print("BM25 SMOKE TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
