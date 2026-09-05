"""FAISS + BM25 Hybrid Retrieval Benchmark runner for ScholarAI.

Builds a FAISS vector index and BM25 index over all evaluation corpus chunks,
runs hybrid search (FAISS + BM25 -> RRF -> Top-K) across evaluation_set.json queries,
and saves metrics to data/evaluation/results/hybrid_benchmark.json.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.evaluation.baseline_faiss import (
    build_eval_vector_store,
    get_eval_embeddings,
    load_eval_set,
)
from src.evaluation.eval_corpus import load_all_evaluation_chunks
from src.evaluation.evaluator import evaluate_retriever
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.hybrid_retriever import hybrid_search


def run_hybrid_benchmark(
    data_dir: str = "data/evaluation",
    output_dir: str = "data/evaluation/results",
    k: int = 5,
    rrf_k: int = 60,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs the Hybrid Retrieval (FAISS + BM25 + RRF) benchmark and saves JSON results.

    Args:
        data_dir: Directory containing evaluation_set.json and PDFs.
        output_dir: Directory to write output hybrid_benchmark.json.
        k: Number of retrieved chunks to consider (default: 5).
        rrf_k: Reciprocal Rank Fusion constant parameter (default: 60).
        embeddings: Optional embedding model (defaults to Gemini).

    Returns:
        Dict containing structured benchmark evaluation results.
    """
    eval_set = load_eval_set(data_dir=data_dir)
    eval_chunks = load_all_evaluation_chunks(data_dir=data_dir)

    print(f"Building FAISS vector index over {len(eval_chunks)} evaluation chunks...")
    vector_store = build_eval_vector_store(
        eval_chunks, embeddings=embeddings
    )

    print(f"Building BM25 index over {len(eval_chunks)} evaluation chunks...")
    bm25_retriever = build_bm25_index(eval_chunks)

    def retriever_fn(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return hybrid_search(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            query=query,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    print(f"Evaluating Hybrid Retrieval across {len(eval_set)} queries...")
    results = evaluate_retriever(retriever_fn, eval_set, k=k)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hybrid_benchmark.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Hybrid benchmark evaluation results saved to: {output_path}")
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("SCHOLARAI RETRIEVAL BENCHMARK — HYBRID RETRIEVAL (MILESTONE 6.3)")
    print("=" * 80)

    res = run_hybrid_benchmark()

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Queries Evaluated: {res['num_queries']}")
    print(f"Top-K Cutoff:      {res['k']}")
    metrics = res["metrics"]
    k_val = res["k"]
    p_key = f"precision@{k_val}"
    r_key = f"recall@{k_val}"
    print(f"Precision@{k_val}:      {metrics[p_key]:.4f}")
    print(f"Recall@{k_val}:         {metrics[r_key]:.4f}")
    print(f"MRR:                {metrics['mrr']:.4f}")
    print(f"Avg Latency:        {metrics['avg_latency_ms']:.2f} ms")
    print(f"Median Latency:     {metrics['median_latency_ms']:.2f} ms")
    print("=" * 80)
