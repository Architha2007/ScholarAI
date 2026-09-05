"""Evaluation corpus generator for ScholarAI.

This module extracts and chunks the evaluation PDFs deterministically using the
same ingestion and chunking logic used by the ScholarAI production pipeline.
Chunk IDs are qualified with the document name to prevent collisions.
"""

import os
from typing import Any, Dict, List

from src.ingestion.pdf import extract_pdf_details
from src.preprocessing.chunking import create_text_chunks

EVAL_DOCUMENTS = [
    "attention_all_you_need",
    "dense_passage_retrieval",
    "retrieval_augmented_generation",
]


def load_evaluation_corpus(
    data_dir: str = "data/evaluation",
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts text and generates chunks for all evaluation PDFs.

    Args:
        data_dir: Path to the directory containing evaluation PDFs.

    Returns:
        Dict mapping document ID to a list of chunk dictionaries:
            - chunk_id: Qualified string identifier (e.g. 'doc:chunk_1')
            - document: Document ID string
            - local_chunk_id: Integer positional index (1-indexed)
            - text: Raw text chunk string
    """
    corpus: Dict[str, List[Dict[str, Any]]] = {}

    for doc_id in EVAL_DOCUMENTS:
        filename = f"{doc_id}.pdf"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Evaluation PDF not found: {file_path}")

        with open(file_path, "rb") as f:
            pdf_details = extract_pdf_details(f)

        chunk_results = create_text_chunks(pdf_details["text"])
        chunks_text = chunk_results["chunks"]

        doc_chunks = []
        for idx, chunk_str in enumerate(chunks_text, start=1):
            qualified_id = f"{doc_id}:chunk_{idx}"
            doc_chunks.append({
                "chunk_id": qualified_id,
                "document": doc_id,
                "local_chunk_id": idx,
                "text": chunk_str,
            })

        corpus[doc_id] = doc_chunks

    return corpus


def load_all_evaluation_chunks(
    data_dir: str = "data/evaluation",
) -> List[Dict[str, Any]]:
    """Returns a flat list of all chunk records across all evaluation PDFs.

    Args:
        data_dir: Path to the directory containing evaluation PDFs.

    Returns:
        Flat list of chunk dictionaries.
    """
    corpus = load_evaluation_corpus(data_dir=data_dir)
    all_chunks = []
    for doc_id in EVAL_DOCUMENTS:
        all_chunks.extend(corpus.get(doc_id, []))
    return all_chunks
