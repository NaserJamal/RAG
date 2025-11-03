"""
BM25 keyword search implementation for RAG system.

Provides BM25-based keyword search functionality that can be used
standalone or in hybrid retrieval scenarios.
"""

from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np


class BM25Search:
    """BM25 keyword search for document retrieval."""

    def __init__(self, documents: Optional[List[Dict]] = None):
        """
        Initialize BM25 search.

        Args:
            documents: Optional list of document dictionaries with 'id' and 'content'
        """
        self.documents = documents or []
        self.bm25 = None
        self.tokenized_docs = []
        self._indexed = False

    def index(self, documents: Optional[List[Dict]] = None) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: Optional list of documents (uses constructor documents if not provided)
        """
        if documents is not None:
            self.documents = documents

        if not self.documents:
            raise ValueError("No documents provided for indexing")

        # Tokenize documents (simple whitespace tokenization)
        self.tokenized_docs = [
            doc["content"].lower().split() for doc in self.documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self._indexed = True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search documents using BM25 scoring.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of (document_id, score) tuples sorted by relevance

        Raises:
            ValueError: If index() hasn't been called
        """
        if not self._indexed or self.bm25 is None:
            raise ValueError("BM25 index not built. Call index() first.")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k document indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results list with document IDs and scores
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc_id = self.documents[idx]["id"]
                score = float(scores[idx])
                results.append((doc_id, score))

        return results

    def is_indexed(self) -> bool:
        """Check if the BM25 index has been built."""
        return self._indexed

    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self.documents)


def reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    alpha: float = 0.5,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge search results using Reciprocal Rank Fusion (RRF).

    RRF combines multiple ranked lists by assigning each document a score
    based on its rank in each list: score(d) = sum(1/(k + rank(d)))

    Args:
        vector_results: Results from vector search [(doc_id, score), ...]
        bm25_results: Results from BM25 search [(doc_id, score), ...]
        alpha: Weight for vector vs BM25 (0.0 = pure BM25, 1.0 = pure vector)
        k: RRF constant (typically 60)

    Returns:
        Merged and sorted list of (document_id, fused_score) tuples
    """
    fused_scores = {}

    # Add scores from vector search (weighted by alpha)
    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        rrf_score = alpha / (k + rank)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

    # Add scores from BM25 search (weighted by 1-alpha)
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        rrf_score = (1 - alpha) / (k + rank)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

    # Sort by fused score descending
    sorted_results = sorted(
        fused_scores.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_results
