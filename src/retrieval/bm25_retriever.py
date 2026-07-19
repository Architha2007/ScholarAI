"""BM25 sparse retrieval module for ScholarAI.

This module implements the BM25 sparse retriever class and utility functions,
using the rank_bm25 library. It supports indexing, saving/loading the index,
and retrieval with score tracking. It is fully compatible with custom chunk
objects or dictionaries.
"""

import os
import pickle
import re
from typing import Any, Dict, List, Union
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 indexing and query matching.

    Converts to lowercase and extracts sequence of alphanumeric words.

    Args:
        text: Input string to tokenize.

    Returns:
        List of string tokens.
    """
    return re.findall(r"\w+", text.lower())


def _get_chunk_text(chunk: Any) -> str:
    """Extracts text from a chunk object or dictionary."""
    if isinstance(chunk, dict):
        return chunk.get("text", chunk.get("page_content", ""))

    for attr in ("text", "page_content", "content"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            if isinstance(val, str):
                return val
    return ""


def _get_chunk_id(chunk: Any) -> str:
    """Extracts chunk ID from a chunk object or dictionary."""
    if isinstance(chunk, dict):
        return chunk.get("chunk_id", chunk.get("id", ""))

    for attr in ("chunk_id", "id"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            if isinstance(val, (str, int)):
                return str(val)
    return ""


def _get_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """Extracts metadata from a chunk object or dictionary."""
    if isinstance(chunk, dict):
        return chunk.get("metadata", {})

    if hasattr(chunk, "metadata"):
        val = getattr(chunk, "metadata")
        if isinstance(val, dict):
            return val
    return {}


class BM25Retriever:
    """BM25 sparse retriever wrapper class.

    Encapsulates rank_bm25's BM25Okapi model alongside the raw indexed chunks
    to ensure full retrieval information is preserved.
    """

    def __init__(self, chunks: List[Any] = None):
        """Initializes the retriever and builds index if chunks are provided.

        Args:
            chunks: A list of dict chunks or custom chunk objects.
        """
        self.chunks: List[Any] = chunks or []
        self.bm25: Union[BM25Okapi, None] = None
        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: List[Any]) -> None:
        """Builds the BM25 index from a list of document chunks.

        Args:
            chunks: List of document chunks.
        """
        self.chunks = chunks
        tokenized_corpus = [
            tokenize(_get_chunk_text(chunk)) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the BM25 index and returns top_k results.

        Args:
            query: The text query string.
            top_k: The maximum number of results to return.

        Returns:
            A list of dictionary results sorted by score descending:
                - chunk_id: Unique string identifier of the chunk.
                - chunk_text: Raw text of the retrieved chunk.
                - retrieval_score: The calculated BM25 relevance score.
                - metadata: The original chunk metadata dictionary.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Zip scores with their source chunks
        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": _get_chunk_id(chunk),
                "chunk_text": _get_chunk_text(chunk),
                "retrieval_score": float(score),
                "metadata": _get_chunk_metadata(chunk),
            })

        # Sort descending by score
        results.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return results[:top_k]

    def save(self, file_path: str) -> None:
        """Saves the retriever state (chunks + BM25Okapi model) to disk.

        Args:
            file_path: File path to serialize retriever into.
        """
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        data = {
            "chunks": self.chunks,
            "bm25": self.bm25,
        }

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str) -> "BM25Retriever":
        """Loads a saved retriever state from disk.

        Args:
            file_path: File path of the serialized retriever.

        Returns:
            An instance of BM25Retriever with index loaded.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 index not found at: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        retriever = cls()
        retriever.chunks = data["chunks"]
        retriever.bm25 = data["bm25"]
        return retriever


def build_bm25_index(chunks: List[Any]) -> BM25Retriever:
    """Convenience module-level function to construct a BM25 index.

    Args:
        chunks: List of document chunks.

    Returns:
        Built BM25Retriever instance.
    """
    return BM25Retriever(chunks)


def search_bm25(
    retriever: BM25Retriever, query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Convenience module-level function to search a BM25 index.

    Args:
        retriever: The BM25Retriever instance.
        query: Query string.
        top_k: Number of results to retrieve.

    Returns:
        List of retrieval results.
    """
    return retriever.retrieve(query, top_k=top_k)