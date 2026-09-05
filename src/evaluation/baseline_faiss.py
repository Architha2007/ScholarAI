"""FAISS vector store baseline benchmark runner for ScholarAI.

Builds a FAISS vector index over all evaluation corpus chunks using production
embeddings, runs similarity search across evaluation_set.json queries,
and saves metrics to data/evaluation/results/faiss_baseline.json.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.evaluation.eval_corpus import load_all_evaluation_chunks
from src.evaluation.evaluator import evaluate_retriever
from src.utils.gemini import get_gemini_api_key


def get_eval_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instantiates production Google Generative AI embeddings model."""
    api_key = get_gemini_api_key()
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def _embed_batch_with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Executes an embedding call with backoff on 429 quota errors."""
    max_retries = 5
    delay = 3.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if (
                ("429" in err or "resource" in err or "quota" in err)
                and attempt < max_retries - 1
            ):
                print(
                    f"Rate limit hit during embedding. Waiting {delay:.1f}s "
                    f"(retry {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e


def load_eval_set(data_dir: str = "data/evaluation") -> List[Dict[str, Any]]:
    """Loads evaluation queries from evaluation_set.json.

    Returns query records normalized for the evaluator framework.
    """
    dataset_path = os.path.join(data_dir, "evaluation_set.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(data_dir, "eval_dataset.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_queries = data.get("queries", data if isinstance(data, list) else [])

    eval_set = []
    for sample in raw_queries:
        q_id = sample.get("query_id", sample.get("id"))
        question = sample.get("query", sample.get("question"))

        relevant_ids = []
        if "expected_relevant_chunks" in sample:
            for rel in sample["expected_relevant_chunks"]:
                doc_name = rel["doc_id"].replace(".pdf", "")
                c_num = rel["chunk_id"]
                relevant_ids.append(f"{doc_name}:chunk_{c_num}")
        elif "relevant_chunk_ids" in sample:
            relevant_ids = sample["relevant_chunk_ids"]

        eval_set.append({
            "id": q_id,
            "question": question,
            "relevant_chunk_ids": relevant_ids,
        })

    return eval_set


def build_eval_vector_store(
    eval_chunks: List[Dict[str, Any]],
    embeddings: Optional[Any] = None,
    batch_size: int = 5,
) -> FAISS:
    """Builds a FAISS vector store containing all evaluation chunks.

    Args:
        eval_chunks: Chunk dicts from load_all_evaluation_chunks.
        embeddings: Optional embeddings model instance (defaults to Gemini).
        batch_size: Number of texts to embed per API request (default: 5).

    Returns:
        FAISS vector store instance.
    """
    if embeddings is None:
        embeddings = get_eval_embeddings()

    texts = [c["text"] for c in eval_chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "document": c.get("document", ""),
            "local_chunk_id": c.get("local_chunk_id", 0),
        }
        for c in eval_chunks
    ]

    vector_store = None
    for i in range(0, len(texts), batch_size):
        end_idx = i + batch_size
        batch_texts = texts[i:end_idx]
        batch_meta = metadatas[i:end_idx]

        if vector_store is None:
            vector_store = _embed_batch_with_retry(
                FAISS.from_texts,
                texts=batch_texts,
                embedding=embeddings,
                metadatas=batch_meta,
            )
        else:
            _embed_batch_with_retry(
                vector_store.add_texts,
                texts=batch_texts,
                metadatas=batch_meta,
            )

        if end_idx < len(texts) and not hasattr(
            embeddings, "fake_embeddings"
        ):
            time.sleep(1.0)

    if vector_store is None:
        raise ValueError("No chunks provided to build vector store.")

    return vector_store


def run_faiss_baseline(
    data_dir: str = "data/evaluation",
    output_dir: str = "data/evaluation/results",
    k: int = 5,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs the FAISS baseline retrieval benchmark and saves result JSON.

    Args:
        data_dir: Directory containing evaluation_set.json and PDFs.
        output_dir: Directory to write output faiss_baseline.json.
        k: Number of retrieved chunks to consider (default: 5).
        embeddings: Optional embedding model (defaults to Gemini).

    Returns:
        Dict containing structured benchmark evaluation results.
    """
    eval_set = load_eval_set(data_dir=data_dir)
    eval_chunks = load_all_evaluation_chunks(data_dir=data_dir)
    print(f"Building FAISS index over {len(eval_chunks)} evaluation chunks...")

    vector_store = build_eval_vector_store(
        eval_chunks, embeddings=embeddings
    )

    def faiss_retriever(query: str, top_k: int = 5) -> List[Any]:
        return vector_store.similarity_search(query, k=top_k)

    print(f"Evaluating FAISS baseline across {len(eval_set)} queries...")
    results = evaluate_retriever(faiss_retriever, eval_set, k=k)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "faiss_baseline.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"FAISS baseline evaluation results saved to: {output_path}")
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("SCHOLARAI RETRIEVAL BENCHMARK — FAISS BASELINE (MILESTONE 6.2)")
    print("=" * 80)

    res = run_faiss_baseline()

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
