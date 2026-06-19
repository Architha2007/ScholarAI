"""
Build a FAISS vector database from research paper text.

This file handles the "indexing" side of RAG:
1. Split long PDF text into smaller chunks
2. Turn each chunk into a vector (embedding) with Gemini
3. Store those vectors in FAISS for fast similarity search
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# How large each text chunk should be (in characters).
CHUNK_SIZE = 4000

# How much overlap to keep between chunks so context is not lost at boundaries.
CHUNK_OVERLAP = 200

# Maximum characters indexed for the vector database (longer PDFs are trimmed).
INDEXING_CHAR_LIMIT = 50000

# Maximum characters sent to Gemini for full-paper analysis.
MAX_ANALYSIS_CHARS = 100000


def get_text_processing_stats(total_chars: int) -> dict:
    """Return how much of a PDF will be analyzed and indexed."""
    return {
        "total_chars": total_chars,
        "chars_analyzed": min(total_chars, MAX_ANALYSIS_CHARS),
        "chars_indexed": min(total_chars, INDEXING_CHAR_LIMIT),
        "is_large_pdf": total_chars > MAX_ANALYSIS_CHARS,
    }


def trim_text_for_analysis(text: str) -> str:
    """Keep only the first portion of text used for Gemini analysis."""
    return text[:MAX_ANALYSIS_CHARS]


def trim_text_for_indexing(text: str) -> str:
    """Keep only the first portion of text used for RAG indexing."""
    return text[:INDEXING_CHAR_LIMIT]


def _get_api_key() -> str:
    """Read the Gemini API key from the .env file."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )
    return api_key


def _split_text_into_chunks(text: str) -> list[str]:
    """
    Break one long string into smaller overlapping chunks.

    LangChain's RecursiveCharacterTextSplitter tries to split on
    paragraphs, then sentences, then words — which keeps meaning intact.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # split_text returns a simple list of strings.
    return text_splitter.split_text(text)


def create_text_chunks(text: str) -> dict:
    """
    Split paper text into chunks for RAG indexing.

    Returns chunk list plus indexing stats for the UI.
    """
    was_truncated = len(text) > INDEXING_CHAR_LIMIT
    indexed_text = trim_text_for_indexing(text)

    chunks = _split_text_into_chunks(indexed_text)

    if not chunks:
        raise ValueError("No text chunks were created from this PDF.")

    return {
        "chunks": chunks,
        "chunks_created": len(chunks),
        "chars_indexed": len(indexed_text),
        "was_truncated": was_truncated,
    }


def build_vector_store_from_chunks(chunks: list[str]) -> FAISS:
    """Turn pre-created text chunks into a searchable FAISS vector database."""
    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
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
