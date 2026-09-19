"""Embedding model comparison benchmark for ScholarAI (Milestone 8).

Evaluates and compares current production baseline
(models/gemini-embedding-001) against popular open-source
sentence-transformer embedding models across Precision@5, Recall@5, MRR,
embedding generation latency, query latency, and embedding dimension.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.evaluation.baseline_faiss import (
    build_eval_vector_store,
    get_eval_embeddings,
    load_eval_set,
)
from src.evaluation.eval_corpus import load_all_evaluation_chunks
from src.evaluation.evaluator import evaluate_retriever


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible Embeddings wrapper around SentenceTransformer."""

    def __init__(self, model_name: str):
        """Initializes the SentenceTransformer model.

        Args:
            model_name: Hugging Face model identifier string.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings.

        Args:
            texts: List of document text strings.

        Returns:
            List of float vector lists.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string.

        Args:
            text: Query string.

        Returns:
            List of float vector numbers.
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


MODELS_TO_COMPARE = [
    {
        "model_id": "models/gemini-embedding-001",
        "display_name": "Gemini Embedding (gemini-embedding-001)",
        "type": "gemini",
    },
    {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "display_name": "all-MiniLM-L6-v2",
        "type": "sentence_transformer",
    },
    {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "display_name": "all-mpnet-base-v2",
        "type": "sentence_transformer",
    },
    {
        "model_id": "BAAI/bge-base-en-v1.5",
        "display_name": "bge-base-en-v1.5",
        "type": "sentence_transformer",
    },
]


def instantiate_embedding_model(
    model_config: Dict[str, Any]
) -> Embeddings:
    """Instantiates an embedding model object based on model_config dict.

    Args:
        model_config: Configuration dict with model_id and type.

    Returns:
        Embeddings instance compatible with FAISS vector store.
    """
    model_type = model_config.get("type", "sentence_transformer")
    model_id = model_config["model_id"]

    if model_type == "gemini":
        return get_eval_embeddings()

    return SentenceTransformerEmbeddings(model_name=model_id)


def evaluate_single_embedding_model(
    model_config: Dict[str, Any],
    eval_chunks: List[Dict[str, Any]],
    eval_set: List[Dict[str, Any]],
    k: int = 5,
    embeddings_instance: Optional[Embeddings] = None,
) -> Dict[str, Any]:
    """Evaluates single embedding model on indexing & retrieval metrics.

    Args:
        model_config: Model descriptor dict with model_id and display_name.
        eval_chunks: List of document chunk dicts.
        eval_set: List of evaluation query dicts.
        k: Cutoff rank for retrieval evaluation.
        embeddings_instance: Optional pre-instantiated Embeddings object.

    Returns:
        Dict containing model metrics, dimensions, and latencies.
    """
    model_id = model_config["model_id"]
    display_name = model_config.get("display_name", model_id)

    if embeddings_instance is None:
        embeddings_instance = instantiate_embedding_model(model_config)

    # Measure embedding dimension
    sample_vec = embeddings_instance.embed_query("ScholarAI test query")
    embedding_dim = len(sample_vec)

    # Measure embedding generation / indexing latency
    start_gen = time.perf_counter()
    vector_store = build_eval_vector_store(
        eval_chunks, embeddings=embeddings_instance
    )
    end_gen = time.perf_counter()
    embedding_gen_latency_ms = (end_gen - start_gen) * 1000.0

    def retriever_fn(query: str, top_k: int = 5) -> List[Any]:
        return vector_store.similarity_search(query, k=top_k)

    eval_results = evaluate_retriever(retriever_fn, eval_set, k=k)

    metrics = eval_results["metrics"]
    metrics["embedding_generation_latency_ms"] = round(
        embedding_gen_latency_ms, 2
    )

    return {
        "model_id": model_id,
        "display_name": display_name,
        "embedding_dimension": embedding_dim,
        "embedding_generation_latency_ms": round(
            embedding_gen_latency_ms, 2
        ),
        "metrics": metrics,
        "per_query": eval_results["per_query"],
        "status": "success",
    }


def run_embedding_comparison(
    data_dir: str = "data/evaluation",
    output_dir: str = "data/evaluation/results",
    k: int = 5,
    models_config: Optional[List[Dict[str, Any]]] = None,
    custom_embeddings_map: Optional[Dict[str, Embeddings]] = None,
) -> Dict[str, Any]:
    """Runs embedding comparison benchmark across configured models.

    Args:
        data_dir: Directory containing evaluation dataset and PDFs.
        output_dir: Directory to save embedding_comparison.json.
        k: Top-k cutoff metric.
        models_config: Optional list of model configurations to benchmark.
        custom_embeddings_map: Optional map of model_id to Embeddings.

    Returns:
        Structured result dict containing comparison details and matrix.
    """
    if models_config is None:
        models_config = MODELS_TO_COMPARE

    eval_set = load_eval_set(data_dir=data_dir)
    eval_chunks = load_all_evaluation_chunks(data_dir=data_dir)

    print("=" * 80)
    print("SCHOLARAI RETRIEVAL BENCHMARK — EMBEDDING MODEL COMPARISON (M8)")
    print("=" * 80)
    print(f"Loaded {len(eval_chunks)} chunks across evaluation PDFs.")
    print(f"Loaded {len(eval_set)} evaluation queries.")

    model_results: Dict[str, Any] = {}
    summary_matrix: List[Dict[str, Any]] = []

    for cfg in models_config:
        m_id = cfg["model_id"]
        d_name = cfg.get("display_name", m_id)
        print(f"\n[Evaluating Model]: {d_name} ({m_id})...")

        try:
            custom_emb = (
                custom_embeddings_map.get(m_id)
                if custom_embeddings_map
                else None
            )
            res = evaluate_single_embedding_model(
                model_config=cfg,
                eval_chunks=eval_chunks,
                eval_set=eval_set,
                k=k,
                embeddings_instance=custom_emb,
            )
            model_results[m_id] = res

            m = res["metrics"]
            summary_matrix.append({
                "model_id": m_id,
                "display_name": d_name,
                "embedding_dimension": res["embedding_dimension"],
                "embedding_generation_latency_ms": res[
                    "embedding_generation_latency_ms"
                ],
                f"precision@{k}": m.get(f"precision@{k}", 0.0),
                f"recall@{k}": m.get(f"recall@{k}", 0.0),
                "mrr": m.get("mrr", 0.0),
                "avg_latency_ms": m.get("avg_latency_ms", 0.0),
                "median_latency_ms": m.get("median_latency_ms", 0.0),
            })
            print(
                f"  -> Dimension: {res['embedding_dimension']}, "
                f"Precision@{k}: {m.get(f'precision@{k}'):.4f}, "
                f"Recall@{k}: {m.get(f'recall@{k}'):.4f}, "
                f"MRR: {m.get('mrr'):.4f}, "
                f"Query Latency: {m.get('avg_latency_ms'):.2f} ms"
            )
        except Exception as e:
            print(f"  -> ERROR evaluating {d_name}: {e}")
            model_results[m_id] = {
                "model_id": m_id,
                "display_name": d_name,
                "status": "error",
                "error_message": str(e),
            }

    overall_results = {
        "num_queries": len(eval_set),
        "num_chunks": len(eval_chunks),
        "k": k,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": model_results,
        "summary_matrix": summary_matrix,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "embedding_comparison.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nEmbedding comparison results saved to: {output_path}")
    return overall_results


if __name__ == "__main__":
    run_embedding_comparison()
