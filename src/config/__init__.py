"""Application configuration and constants."""

from src.config.settings import (
    APP_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEXING_CHAR_LIMIT,
    MAX_ANALYSIS_CHARS,
    TOP_K_CHUNKS,
)

__all__ = [
    "APP_MODEL",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "INDEXING_CHAR_LIMIT",
    "MAX_ANALYSIS_CHARS",
    "TOP_K_CHUNKS",
]
