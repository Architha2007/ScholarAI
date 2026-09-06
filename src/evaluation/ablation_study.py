"""Retrieval ablation study runner for ScholarAI.

Executes a controlled ablation study comparing 4 retrieval strategies across
the 20-query evaluation set:
1. FAISS-only (Dense Vector)
2. BM25-only (Sparse Keyword)
3. Hybrid RRF (FAISS + BM25, k=60)
4. Hybrid + Cross-Encoder Reranking (RRF Top-20 -> Cross-Encoder Top-5)

Saves summary metrics and per-query details to
data/evaluation/results/ablation_study.json.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.evaluation.baseline_faiss import load_eval_set
from src.evaluation.eval_corpus import load_all_evaluation_chunks
from src.evaluation.evaluator import evaluate_retriever
from src.evaluation.hybrid_benchmark import build_hybrid_retriever
from src.reranking.cross_encoder_reranker import rerank_results
from src.retrieval.hybrid_retriever import hybrid_search


def run_ablation_study(
    data_dir: str = "data/evaluation",
    output_dir: str = "data/evaluation/results",
    k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs controlled ablation study across all 4 retrieval strategies.

    Args:
        data_dir: Directory containing evaluation_set.json and PDFs.
        output_dir: Directory to write ablation_study.json.
        k: Final top-K cutoff (default: 5).
        candidate_k: Candidate K for reranker stage (default: 20).
        rrf_k: RRF constant parameter (default: 60).
        embeddings: Optional embedding model instance.

    Returns:
        Structured ablation study dictionary containing comparison matrix and
        per-strategy evaluation results.
    """
    eval_set = load_eval_set(data_dir=data_dir)
    eval_chunks = load_all_evaluation_chunks(data_dir=data_dir)

    print(
        f"Building retrievers for ablation study "
        f"({len(eval_chunks)} chunks)..."
    )
    vector_store, bm25_retriever = build_hybrid_retriever(
        eval_chunks, embeddings=embeddings
    )

    # Strategy 1: FAISS-only (Dense Vector)
    def faiss_only_fn(query: str, top_k: int = 5) -> List[Any]:
        return vector_store.similarity_search(query, k=top_k)

    # Strategy 2: BM25-only (Sparse Keyword)
    def bm25_only_fn(query: str, top_k: int = 5) -> List[Any]:
        return bm25_retriever.retrieve(query, top_k=top_k)

    # Strategy 3: Hybrid RRF (FAISS + BM25)
    def hybrid_rrf_fn(query: str, top_k: int = 5) -> List[Any]:
        return hybrid_search(
            vector_store,
            bm25_retriever,
            query,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    # Strategy 4: Hybrid + Cross-Encoder Reranking
    def hybrid_reranked_fn(query: str, top_k: int = 5) -> List[Any]:
        candidates = hybrid_search(
            vector_store,
            bm25_retriever,
            query,
            top_k=candidate_k,
            rrf_k=rrf_k,
        )
        return rerank_results(query, candidates, top_k=top_k)

    strategies = [
        ("faiss_only", "1. FAISS-only (Dense Vector)", faiss_only_fn),
        ("bm25_only", "2. BM25-only (Sparse Keyword)", bm25_only_fn),
        ("hybrid_rrf", "3. Hybrid RRF (FAISS + BM25)", hybrid_rrf_fn),
        (
            "hybrid_reranked",
            "4. Hybrid + Cross-Encoder Reranked",
            hybrid_reranked_fn,
        ),
    ]

    strategies_results: Dict[str, Any] = {}
    comparison_matrix: Dict[str, Any] = {}

    for key, name, retriever_fn in strategies:
        print(f"\n--- Evaluating Strategy: {name} ---")
        res = evaluate_retriever(retriever_fn, eval_set, k=k)
        strategies_results[key] = res

        m = res["metrics"]
        p_key = f"precision@{k}"
        r_key = f"recall@{k}"
        comparison_matrix[key] = {
            "name": name,
            f"precision@{k}": m[p_key],
            f"recall@{k}": m[r_key],
            "mrr": m["mrr"],
            "avg_latency_ms": m["avg_latency_ms"],
            "median_latency_ms": m["median_latency_ms"],
        }

    ablation_output = {
        "num_queries": len(eval_set),
        "k": k,
        "comparison_matrix": comparison_matrix,
        "strategies": strategies_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_study.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ablation_output, f, indent=2)

    print(f"\nAblation study results saved to: {output_path}")
    return ablation_output


if __name__ == "__main__":
    print("=" * 80)
    print("SCHOLARAI RETRIEVAL ABLATION STUDY (MILESTONE 6.5)")
    print("=" * 80)

    res = run_ablation_study()
    matrix = res["comparison_matrix"]
    k_val = res["k"]
    p_key = f"precision@{k_val}"
    r_key = f"recall@{k_val}"

    print("\n" + "=" * 80)
    print(f"ABLATION STUDY COMPARATIVE MATRIX (K={k_val})")
    print("=" * 80)
    print(
        f"{'Strategy':<35} | {p_key:<11} | {r_key:<9} | {'MRR':<7} | "
        f"{'Avg (ms)':<9} | {'Med (ms)':<9}"
    )
    print("-" * 90)

    for key, data in matrix.items():
        name = data["name"]
        p_val = data[p_key]
        r_val = data[r_key]
        mrr = data["mrr"]
        avg_lat = data["avg_latency_ms"]
        med_lat = data["median_latency_ms"]
        print(
            f"{name:<35} | {p_val:<11.4f} | {r_val:<9.4f} | {mrr:<7.4f} | "
            f"{avg_lat:<9.2f} | {med_lat:<9.2f}"
        )
    print("=" * 90)
