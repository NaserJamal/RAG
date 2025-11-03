"""
Similarity computation utilities for vector comparison.

Provides efficient implementations of similarity metrics used in RAG systems.
"""

import numpy as np
from typing import Union


def cosine_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and document embeddings.

    This function handles both single document and batch document comparisons.

    Args:
        query_embedding: 1D array of query embedding
        doc_embeddings: 2D array of document embeddings (n_docs x embedding_dim)
                       or 1D array for single document comparison

    Returns:
        1D array of similarity scores (or single score for 1D doc_embeddings)

    Raises:
        ValueError: If input shapes are invalid
    """
    # Handle edge cases
    if query_embedding.size == 0 or doc_embeddings.size == 0:
        return np.array([])

    # Ensure query is 1D
    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D array")

    # Handle 1D doc_embeddings (single document)
    if doc_embeddings.ndim == 1:
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings)
        return np.dot(query_norm, doc_norm)

    # Handle 2D doc_embeddings (multiple documents)
    if doc_embeddings.ndim != 2:
        raise ValueError("doc_embeddings must be 1D or 2D array")

    # Normalize vectors
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(doc_norms, query_norm)

    return similarities


def euclidean_distance(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance between query and document embeddings.

    Smaller distances indicate higher similarity.

    Args:
        query_embedding: 1D array of query embedding
        doc_embeddings: 2D array of document embeddings (n_docs x embedding_dim)

    Returns:
        1D array of distance scores
    """
    if query_embedding.size == 0 or doc_embeddings.size == 0:
        return np.array([])

    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D array")

    if doc_embeddings.ndim == 1:
        return np.linalg.norm(query_embedding - doc_embeddings)

    if doc_embeddings.ndim != 2:
        raise ValueError("doc_embeddings must be 1D or 2D array")

    # Compute Euclidean distance
    distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)

    return distances


def dot_product_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute dot product similarity between query and document embeddings.

    Args:
        query_embedding: 1D array of query embedding
        doc_embeddings: 2D array of document embeddings (n_docs x embedding_dim)

    Returns:
        1D array of similarity scores
    """
    if query_embedding.size == 0 or doc_embeddings.size == 0:
        return np.array([])

    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be 1D array")

    if doc_embeddings.ndim == 1:
        return np.dot(query_embedding, doc_embeddings)

    if doc_embeddings.ndim != 2:
        raise ValueError("doc_embeddings must be 1D or 2D array")

    # Compute dot product
    similarities = np.dot(doc_embeddings, query_embedding)

    return similarities
