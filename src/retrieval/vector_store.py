"""
Build a FAISS vector database from research paper text.

This module handles the "indexing" side of RAG:
1. Split long PDF text into smaller chunks
2. Turn each chunk into a vector (embedding) with Gemini
3. Store those vectors in FAISS for fast similarity search
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.preprocessing.chunking import create_text_chunks


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )
    return api_key


def build_vector_store_from_chunks(chunks: list[str]) -> FAISS:
    """Turn pre-created text chunks into a searchable FAISS vector database."""
    print(f"Creating embeddings for {len(chunks)} chunks...")

    print("Using model: models/gemini-embedding-001")
    print("API key loaded:", bool(_get_api_key()))
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=_get_api_key(),
    )

    metadatas = [
        {"chunk_id": index + 1}
        for index in range(len(chunks))
    ]

    return FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
    )


def build_vector_store(text: str) -> dict:
    """
    Create a FAISS vector store from the full paper text.

    Steps:
    1. Chunk the text
    2. Create Gemini embeddings for every chunk
    3. Save embeddings + chunks inside a FAISS index

    Returns a dict with the FAISS object and indexing stats for the UI.
    """
    chunk_result = create_text_chunks(text)
    print(f"Chunks created: {chunk_result['chunks_created']}")

    vector_store = build_vector_store_from_chunks(chunk_result["chunks"])

    return {
        "vector_store": vector_store,
        "chunks_created": chunk_result["chunks_created"],
        "chars_indexed": chunk_result["chars_indexed"],
        "was_truncated": chunk_result["was_truncated"],
    }
