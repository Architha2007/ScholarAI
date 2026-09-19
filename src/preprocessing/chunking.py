"""Split long paper text into chunks for RAG indexing."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE, INDEXING_CHAR_LIMIT
from src.preprocessing.text import trim_text_for_indexing


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
