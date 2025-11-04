"""
RAG Utilities - Shared utilities for RAG tools.

This package contains common utilities used across RAG search tools:
- common: Shared initialization, filtering, and formatting utilities
- tree_builder: Document tree structure visualization
"""

from .common import (
    initialize,
    get_embedder,
    get_vector_store,
    get_bm25_search,
    get_collection_name,
    get_documents,
    filter_documents_by_path,
    filter_results_by_path,
    format_results,
    execute_bm25_search,
    execute_semantic_search,
    Embedder,
    QdrantVectorStore,
    BM25Search,
)
from .tree_builder import build_document_tree, get_available_files

__all__ = [
    'initialize',
    'get_embedder',
    'get_vector_store',
    'get_bm25_search',
    'get_collection_name',
    'get_documents',
    'filter_documents_by_path',
    'filter_results_by_path',
    'format_results',
    'execute_bm25_search',
    'execute_semantic_search',
    'build_document_tree',
    'get_available_files',
    'Embedder',
    'QdrantVectorStore',
    'BM25Search',
]
