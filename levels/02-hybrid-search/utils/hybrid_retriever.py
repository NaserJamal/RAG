"""Hybrid retriever combining vector and keyword search."""

from typing import List, Dict, Tuple
from collections import defaultdict
from .vector_retriever import VectorRetriever
from .bm25_retriever import BM25Retriever
from .config import RRF_K


class HybridRetriever:
    """Combines vector and BM25 search using Reciprocal Rank Fusion."""

    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever):
        """
        Initialize the hybrid retriever.

        Args:
            vector_retriever: VectorRetriever instance
            bm25_retriever: BM25Retriever instance
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Search using hybrid approach with Reciprocal Rank Fusion.

        Args:
            query: Search query string
            k: Number of top results to return
            alpha: Weight for semantic vs keyword (0.0 = pure keyword, 1.0 = pure semantic)

        Returns:
            List of (document_id, score) tuples sorted by fused score
        """
        # Get results from both retrievers (retrieve more for better fusion)
        vector_results = self.vector_retriever.search(query, k=k * 2)
        bm25_results = self.bm25_retriever.search(query, k=k * 2)

        # Apply RRF fusion
        fused_scores = self._reciprocal_rank_fusion(
            vector_results, bm25_results, alpha=alpha
        )

        # Sort by fused score and return top-k
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF Formula: score(d) = sum over all rankings of 1/(k + rank(d))

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for semantic (vector) vs keyword (BM25)

        Returns:
            Dictionary mapping document_id to fused score
        """
        fused_scores = defaultdict(float)

        # Add vector search scores (weighted by alpha)
        for rank, (doc_id, score) in enumerate(vector_results, start=1):
            rrf_score = alpha / (RRF_K + rank)
            fused_scores[doc_id] += rrf_score

        # Add BM25 scores (weighted by 1-alpha)
        for rank, (doc_id, score) in enumerate(bm25_results, start=1):
            rrf_score = (1 - alpha) / (RRF_K + rank)
            fused_scores[doc_id] += rrf_score

        return dict(fused_scores)
