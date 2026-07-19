"""Hybrid retrieval module for ScholarAI.

This module implements Reciprocal Rank Fusion (RRF) to fuse results from
dense vector retrieval (FAISS) and sparse keyword retrieval (BM25).
"""

from typing import Any, Dict, List


def fuse_results(
    faiss_results: List[Any],
    bm25_results: List[Any],
    k: int = 60
) -> List[Dict[str, Any]]:
    """Fuses ranking results from dense (FAISS) and sparse (BM25) retrievers.

    Uses Reciprocal Rank Fusion (RRF) with parameter k:
        score = sum(1 / (k + rank))
    where rank is the 1-based index in the returned list of results.

    The identifier used to match chunks across results is `chunk_id`.
    It is normalized to a string to ensure type compatibility (e.g.
    integer 1 vs string "1"). The same stable chunk_id identifier
    exists in the FAISS metadata (assigned during indexing in
    build_vector_store_from_chunks) and is aligned in BM25 results.

    Args:
        faiss_results: List of retrieved results from FAISS
                       (LangChain Documents or dicts).
        bm25_results: List of retrieved results from BM25 (dicts).
        k: The constant parameter for Reciprocal Rank Fusion (default 60).

    Returns:
        List of fused search results sorted descending by fusion score.
    """
    rrf_scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, Dict[str, Any]] = {}

    def process_ranking(results: List[Any]) -> None:
        for rank_idx, res in enumerate(results):
            rank = rank_idx + 1

            # Extract fields depending on Document or dict
            if hasattr(res, "page_content") and hasattr(res, "metadata"):
                chunk_id = res.metadata.get("chunk_id")
                text = res.page_content
                metadata = res.metadata
            elif isinstance(res, dict):
                chunk_id = res.get("chunk_id", res.get("id"))
                text = res.get("chunk_text", res.get("text", ""))
                metadata = res.get("metadata", {})
            else:
                continue

            if chunk_id is None or chunk_id == "":
                continue

            # Normalize the stable identifier to string to prevent mismatch
            norm_id = str(chunk_id)

            # Accumulate RRF score
            rrf_scores[norm_id] = (
                rrf_scores.get(norm_id, 0.0) + (1.0 / (k + rank))
            )

            # Store the first occurrence of chunk details
            if norm_id not in chunks_by_id:
                chunks_by_id[norm_id] = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }

    process_ranking(faiss_results)
    process_ranking(bm25_results)

    # Compile results and sort by RRF score descending
    fused_results = []
    for norm_id, score in rrf_scores.items():
        chunk_data = chunks_by_id[norm_id]
        fused_results.append({
            "chunk_id": chunk_data["chunk_id"],
            "chunk_text": chunk_data["text"],
            "retrieval_score": score,
            "metadata": chunk_data["metadata"],
        })

    fused_results.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return fused_results


def hybrid_search(
    vector_store: Any,
    bm25_retriever: Any,
    query: str,
    top_k: int = 5,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """Performs hybrid search by querying FAISS and BM25 and fusing.

    Args:
        vector_store: Built FAISS vector store.
        bm25_retriever: Built BM25Retriever.
        query: Search query string.
        top_k: Maximum number of fused results to return.
        rrf_k: Constant parameter k for Reciprocal Rank Fusion.

    Returns:
        List of fused search results.
    """
    # 1. Query FAISS
    faiss_results = vector_store.similarity_search(query, k=top_k)

    # 2. Query BM25
    from src.retrieval.bm25_retriever import search_bm25
    bm25_results = search_bm25(bm25_retriever, query, top_k=top_k)

    # 3. Fuse and slice
    fused = fuse_results(faiss_results, bm25_results, k=rrf_k)
    return fused[:top_k]
