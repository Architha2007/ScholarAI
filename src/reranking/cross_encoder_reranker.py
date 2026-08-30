"""Cross-Encoder reranking module for ScholarAI.

Reranks retrieved chunks using a SentenceTransformers CrossEncoder model to
improve retrieval relevance.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache for the loaded CrossEncoder model
_MODEL_CACHE: Optional[Any] = None
_IMPORT_ERROR: bool = False

try:
    from sentence_transformers import CrossEncoder
except ImportError as e:
    logger.warning(
        "Could not import sentence_transformers: %s. "
        "Reranking will fallback to original order.",
        e
    )
    _IMPORT_ERROR = True


def _get_model() -> Optional[Any]:
    """Helper to lazily load and cache the CrossEncoder model."""
    global _MODEL_CACHE

    if _IMPORT_ERROR:
        return None

    if _MODEL_CACHE is None:
        try:
            # Load the pretrained CrossEncoder model
            _MODEL_CACHE = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        except Exception as e:
            print("MODEL LOAD FAILED:")
            print(repr(e))
            logger.warning("Failed to load CrossEncoder model: %s", e)
            return None

    return _MODEL_CACHE


def rerank_results(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Reranks retrieved chunks using a Cross-Encoder model.

    Scores each chunk based on the similarity between the query and the
    chunk text, adds a 'reranker_score' to the chunk dict, and sorts
    descending by this score.

    Args:
        query: User search query.
        retrieved_chunks: List of dictionaries representing retrieved chunks.
                          Each dictionary must contain 'chunk_text'.
        top_k: The number of top reranked chunks to return.

    Returns:
        List of up to top_k reranked chunk dictionaries, sorted
        descending by reranker_score.
    """
    if not retrieved_chunks:
        return []

    # Handle empty/blank query gracefully by returning original chunks
    if not query or not query.strip():
        return retrieved_chunks[:top_k]

    model = _get_model()
    if model is None:
        return retrieved_chunks[:top_k]

    try:
        # Prepare pairs for Cross-Encoder scoring
        pairs = []
        for chunk in retrieved_chunks:
            # Safely extract text, using common keys
            text = (
                chunk.get("chunk_text") or
                chunk.get("text") or
                chunk.get("page_content") or ""
            )
            pairs.append([query, text])

        # Predict scores
        scores = model.predict(pairs)

        # Build list of chunks with reranker scores
        reranked = []
        for idx, chunk in enumerate(retrieved_chunks):
            # Create a copy to prevent mutation of the original chunks
            chunk_copy = dict(chunk)
            chunk_copy["reranker_score"] = float(scores[idx])
            reranked.append(chunk_copy)

        # Sort descending by reranker_score
        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        print("INFERENCE FAILED:")
        print(repr(e))
        logger.warning("Inference or scoring failed in CrossEncoder: %s", e)
        return retrieved_chunks[:top_k]
