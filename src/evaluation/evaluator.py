"""Evaluation metrics and generic retriever evaluator for ScholarAI.

This module provides standard Information Retrieval (IR) metrics (Precision@K,
Recall@K, Reciprocal Rank, Mean Reciprocal Rank) and a retriever-agnostic
evaluation framework function `evaluate_retriever()`.
"""

import statistics
import time
from typing import Any, Callable, Dict, List


def normalize_chunk_id(item: Any) -> str:
    """Normalizes various chunk representations to a single string chunk ID.

    Supports:
    - Standard primitive types (str, int)
    - Dictionaries with 'chunk_id', 'id', or metadata['chunk_id']
    - Objects (e.g. LangChain Document) with 'chunk_id', 'id', or metadata
    """
    if isinstance(item, (str, int)):
        return str(item)

    if isinstance(item, dict):
        if "chunk_id" in item and item["chunk_id"] is not None:
            return str(item["chunk_id"])
        if "id" in item and item["id"] is not None:
            return str(item["id"])
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and "chunk_id" in metadata:
            return str(metadata["chunk_id"])
        return str(item)

    # Object handling (e.g. LangChain Document)
    if hasattr(item, "metadata") and isinstance(item.metadata, dict):
        if "chunk_id" in item.metadata:
            return str(item.metadata["chunk_id"])
        if "id" in item.metadata:
            return str(item.metadata["id"])

    for attr in ("chunk_id", "id"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if val is not None:
                return str(val)

    return str(item)


def precision_at_k(
    retrieved_ids: List[Any], relevant_ids: List[Any], k: int = 5
) -> float:
    """Calculates Precision@K for a single query.

    Precision@K = (Number of relevant chunks retrieved in top K) / K

    Args:
        retrieved_ids: List of retrieved chunk IDs or objects.
        relevant_ids: List of ground-truth relevant chunk IDs or objects.
        k: Maximum number of retrieved results to consider (cutoff).

    Returns:
        Float precision score in range [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k_ids = [normalize_chunk_id(x) for x in retrieved_ids[:k]]
    rel_set = {normalize_chunk_id(x) for x in relevant_ids}

    relevant_retrieved_count = sum(1 for cid in top_k_ids if cid in rel_set)
    return float(relevant_retrieved_count) / float(k)


def recall_at_k(
    retrieved_ids: List[Any], relevant_ids: List[Any], k: int = 5
) -> float:
    """Calculates Recall@K for a single query.

    Recall@K = (Number of relevant chunks in top K) / (Total relevant chunks)

    Args:
        retrieved_ids: List of retrieved chunk IDs or objects.
        relevant_ids: List of ground-truth relevant chunk IDs or objects.
        k: Maximum number of retrieved results to consider (cutoff).

    Returns:
        Float recall score in range [0.0, 1.0].
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    top_k_ids = [normalize_chunk_id(x) for x in retrieved_ids[:k]]
    rel_set = {normalize_chunk_id(x) for x in relevant_ids}

    retrieved_rel_count = len(set(top_k_ids) & rel_set)
    return float(retrieved_rel_count) / float(len(rel_set))


def reciprocal_rank(
    retrieved_ids: List[Any], relevant_ids: List[Any]
) -> float:
    """Calculates Reciprocal Rank (RR) for a single query.

    RR = 1 / (1-based rank of first relevant retrieved chunk)

    Args:
        retrieved_ids: List of retrieved chunk IDs or objects.
        relevant_ids: List of ground-truth relevant chunk IDs or objects.

    Returns:
        Float reciprocal rank in range [0.0, 1.0].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    rel_set = {normalize_chunk_id(x) for x in relevant_ids}

    for idx, item in enumerate(retrieved_ids, start=1):
        cid = normalize_chunk_id(item)
        if cid in rel_set:
            return 1.0 / float(idx)

    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """Calculates Mean Reciprocal Rank (MRR) across multiple query scores.

    Args:
        rr_scores: List of float Reciprocal Rank scores.

    Returns:
        Float MRR score.
    """
    if not rr_scores:
        return 0.0
    return float(sum(rr_scores)) / float(len(rr_scores))


def evaluate_retriever(
    retriever_function: Callable[..., List[Any]],
    eval_set: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Generic retriever evaluator function.

    Evaluates any callable retriever against a labeled dataset and computes
    precision@K, recall@K, MRR, and per-query latency measurements.

    Args:
        retriever_function: Callable accepting query string (and top_k)
                            and returning list of retrieved chunks or IDs.
        eval_set: List of evaluation query dicts.
        k: Top-K evaluation cutoff.

    Returns:
        Structured result dictionary with summary metrics and query details.
    """
    per_query_results = []
    p_at_k_scores = []
    r_at_k_scores = []
    rr_scores = []
    latencies = []

    for idx, sample in enumerate(eval_set, start=1):
        query_id = sample.get("id", f"q{idx:03d}")
        question = sample["question"]
        raw_rel = sample.get(
            "relevant_chunk_ids", sample.get("relevant_ids", [])
        )
        relevant_ids = [normalize_chunk_id(r) for r in raw_rel]

        start_time = time.perf_counter()
        try:
            raw_results = retriever_function(question, top_k=k)
        except TypeError:
            raw_results = retriever_function(question)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [
            normalize_chunk_id(res) for res in (raw_results or [])
        ]

        pk = precision_at_k(retrieved_ids, relevant_ids, k=k)
        rk = recall_at_k(retrieved_ids, relevant_ids, k=k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        p_at_k_scores.append(pk)
        r_at_k_scores.append(rk)
        rr_scores.append(rr)

        per_query_results.append({
            "id": query_id,
            "question": question,
            "retrieved_chunk_ids": retrieved_ids[:k],
            "relevant_chunk_ids": relevant_ids,
            f"precision@{k}": round(pk, 4),
            f"recall@{k}": round(rk, 4),
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 3),
        })

    num_queries = len(eval_set)
    avg_p = sum(p_at_k_scores) / float(num_queries) if num_queries > 0 else 0.0
    avg_r = sum(r_at_k_scores) / float(num_queries) if num_queries > 0 else 0.0
    mrr = mean_reciprocal_rank(rr_scores)
    avg_lat = sum(latencies) / float(num_queries) if num_queries > 0 else 0.0
    med_lat = statistics.median(latencies) if num_queries > 0 else 0.0

    return {
        "num_queries": num_queries,
        "k": k,
        "metrics": {
            f"precision@{k}": round(avg_p, 4),
            f"recall@{k}": round(avg_r, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "median_latency_ms": round(med_lat, 3),
        },
        "per_query": per_query_results,
    }
