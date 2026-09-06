"""Cross-Encoder reranking benchmark runner for ScholarAI.

Builds FAISS vector store and BM25 sparse index over evaluation corpus chunks,
retrieves Top-20 hybrid candidates via Reciprocal Rank Fusion (RRF), reranks
them using SentenceTransformers CrossEncoder (ms-marco-MiniLM-L-6-v2) to Top-5,
evaluates performance against evaluation_set.json, and saves results to
data/evaluation/results/reranking_benchmark.json.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.baseline_faiss import load_eval_set
from src.evaluation.eval_corpus import load_all_evaluation_chunks
from src.evaluation.evaluator import evaluate_retriever
from src.evaluation.hybrid_benchmark import build_hybrid_retriever
from src.reranking.cross_encoder_reranker import rerank_results
from src.retrieval.hybrid_retriever import hybrid_search


def build_reranked_retriever(
    eval_chunks: List[Dict[str, Any]],
    embeddings: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Builds FAISS vector store and BM25 index for candidate retrieval.

    Args:
        eval_chunks: Chunk dicts from load_all_evaluation_chunks.
        embeddings: Optional embedding model instance (defaults to Gemini).

    Returns:
        Tuple of (vector_store, bm25_retriever).
    """
    return build_hybrid_retriever(eval_chunks, embeddings=embeddings)


def run_reranking_benchmark(
    data_dir: str = "data/evaluation",
    output_dir: str = "data/evaluation/results",
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs the Cross-Encoder Reranking benchmark and saves result JSON.

    Args:
        data_dir: Directory containing evaluation_set.json and evaluation PDFs.
        output_dir: Directory to write output reranking_benchmark.json.
        k: Final number of reranked chunks to retrieve (default: 5).
        candidate_k: Number of hybrid candidates to rerank (default: 20).
        rrf_k: Reciprocal Rank Fusion constant parameter (default: 60).
        embeddings: Optional embedding model (defaults to Gemini).

    Returns:
        Dict containing structured benchmark evaluation results.
    """
    eval_set = load_eval_set(data_dir=data_dir)
    eval_chunks = load_all_evaluation_chunks(data_dir=data_dir)

    print(
        f"Building retrievers over {len(eval_chunks)} chunks for reranking..."
    )
    vector_store, bm25_retriever = build_reranked_retriever(
        eval_chunks, embeddings=embeddings
    )

    def reranking_retriever_fn(query: str, top_k: int = 5) -> List[Any]:
        candidates = hybrid_search(
            vector_store,
            bm25_retriever,
            query,
            top_k=candidate_k,
            rrf_k=rrf_k,
        )
        return rerank_results(query, candidates, top_k=top_k)

    print(
        f"Evaluating Reranking benchmark across {len(eval_set)} queries "
        f"(candidates={candidate_k}, final_k={k})..."
    )
    results = evaluate_retriever(reranking_retriever_fn, eval_set, k=k)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reranking_benchmark.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Reranking benchmark evaluation results saved to: {output_path}")
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("SCHOLARAI RETRIEVAL BENCHMARK — CROSS-ENCODER RERANKED (M6.4)")
    print("=" * 80)

    res = run_reranking_benchmark()

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
