"""Evaluation and comparison utilities for chunking strategies."""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder


class ChunkEvaluator:
    """Evaluates and compares chunking strategies."""

    def __init__(self, embedder: Embedder):
        """
        Initialize the evaluator.

        Args:
            embedder: Embedder instance for generating embeddings
        """
        self.embedder = embedder

    def evaluate(self, chunks: List[Dict], query: str, k: int = 3) -> Dict:
        """
        Evaluate chunking strategy by searching with a query.

        Args:
            chunks: List of chunk dictionaries
            query: Search query
            k: Number of results to return

        Returns:
            Dictionary with evaluation metrics and results
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_chunk_size": 0,
                "results": []
            }

        # Generate embeddings for chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.embedder.embed(chunk_texts)

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Compute similarities
        similarities = self._cosine_similarity_batch(query_embedding, chunk_embeddings)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "chunk_id": chunks[idx]["chunk_id"],
                "score": float(similarities[idx]),
                "text": chunks[idx]["text"][:200] + "..." if len(chunks[idx]["text"]) > 200 else chunks[idx]["text"],
                "num_tokens": chunks[idx]["num_tokens"]
            })

        # Calculate metrics
        token_counts = [chunk["num_tokens"] for chunk in chunks]

        return {
            "num_chunks": len(chunks),
            "avg_chunk_size": np.mean(token_counts),
            "min_chunk_size": np.min(token_counts),
            "max_chunk_size": np.max(token_counts),
            "std_chunk_size": np.std(token_counts),
            "results": results,
            "top_score": float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
        }

    @staticmethod
    def _cosine_similarity_batch(query_embedding: np.ndarray,
                                  doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple documents.

        Args:
            query_embedding: 1D array of query embedding
            doc_embeddings: 2D array of document embeddings

        Returns:
            1D array of similarity scores
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    @staticmethod
    def visualize_chunks(chunks: List[Dict], max_width: int = 100) -> str:
        """
        Create ASCII visualization of chunks.

        Args:
            chunks: List of chunk dictionaries
            max_width: Maximum width of visualization

        Returns:
            ASCII art string showing chunk boundaries
        """
        if not chunks:
            return "No chunks to visualize"

        lines = []
        lines.append("\nChunk Visualization:")
        lines.append("=" * max_width)

        for i, chunk in enumerate(chunks):
            chunk_width = min(chunk["num_tokens"] // 10, max_width - 10)
            bar = "█" * chunk_width
            lines.append(f"Chunk {i:2d} [{chunk['num_tokens']:4d} tokens]: {bar}")

        lines.append("=" * max_width)

        return "\n".join(lines)
