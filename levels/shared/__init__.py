"""
Shared components for RAG training levels.

This package provides common functionality used across all RAG training levels,
organized by domain:
- config: System configuration and settings
- data: Document loading and preprocessing
- retrieval: Embeddings, vector storage, and similarity
- io: Output management and formatting
- utils: Logging and helper utilities
- cache: Embedding cache management
"""

from .config import Config
from .data import load_documents, count_documents
from .retrieval import Embedder, VectorStore, QdrantVectorStore, cosine_similarity
from .io import OutputManager
from .utils import setup_logger
from .cache import EmbeddingCache

__all__ = [
    "Config",
    "load_documents",
    "count_documents",
    "Embedder",
    "VectorStore",
    "QdrantVectorStore",
    "cosine_similarity",
    "OutputManager",
    "setup_logger",
    "EmbeddingCache",
]
