"""
Semantic search functionality using vector similarity.

Implements cosine similarity search over embedded document chunks.
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class SemanticSearchEngine:
    """Perform semantic search over document embeddings."""

    def __init__(self):
        """Initialize the search engine."""
        self.embeddings_data = []
        self.embedding_vectors = None

    def index_embeddings(self, embeddings_data: List[Dict]) -> None:
        """
        Index embeddings for search.

        Args:
            embeddings_data: List of embedding dictionaries with 'embedding' and metadata
        """
        self.embeddings_data = embeddings_data

        # Convert to numpy array for efficient computation
        self.embedding_vectors = np.array([
            item['embedding'] for item in embeddings_data
        ])

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.embedding_vectors, axis=1, keepdims=True)
        self.embedding_vectors = self.embedding_vectors / norms

    def cosine_similarity(self, query_vector: List[float]) -> np.ndarray:
        """
        Compute cosine similarity between query and all indexed embeddings.

        Args:
            query_vector: Query embedding vector

        Returns:
            Array of similarity scores
        """
        if self.embedding_vectors is None:
            raise ValueError("No embeddings indexed. Call index_embeddings first.")

        # Normalize query vector
        query_array = np.array(query_vector)
        query_norm = query_array / np.linalg.norm(query_array)

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(self.embedding_vectors, query_norm)

        return similarities

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search for most similar chunks to the query.

        Args:
            query_embedding: Embedded query vector
            top_k: Number of top results to return

        Returns:
            List of dictionaries with search results
        """
        if not self.embeddings_data:
            return []

        # Compute similarities
        similarities = self.cosine_similarity(query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            result = {
                "rank": rank,
                "chunk_id": self.embeddings_data[idx].get("chunk_id", idx + 1),
                "score": float(similarities[idx]),
                "text": self.embeddings_data[idx].get("text", ""),
                "text_hash": self.embeddings_data[idx].get("text_hash", "")
            }
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed embeddings.

        Returns:
            Dictionary with index statistics
        """
        return {
            "total_chunks": len(self.embeddings_data),
            "embedding_dimension": len(self.embeddings_data[0]["embedding"]) if self.embeddings_data else 0,
            "indexed": self.embedding_vectors is not None
        }


class MultiDocumentSearchEngine:
    """Search across multiple documents."""

    def __init__(self):
        """Initialize multi-document search engine."""
        self.documents = {}
        self.all_embeddings = []
        self.search_engine = SemanticSearchEngine()

    def add_document(self, doc_name: str, embeddings_data: List[Dict]) -> None:
        """
        Add a document's embeddings to the index.

        Args:
            doc_name: Name of the document
            embeddings_data: List of embedding dictionaries
        """
        # Add document name to each embedding
        for item in embeddings_data:
            item['document'] = doc_name

        self.documents[doc_name] = len(embeddings_data)
        self.all_embeddings.extend(embeddings_data)

    def build_index(self) -> None:
        """Build the search index from all added documents."""
        if not self.all_embeddings:
            raise ValueError("No documents added. Use add_document first.")

        self.search_engine.index_embeddings(self.all_embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Search across all documents.

        Args:
            query_embedding: Embedded query vector
            top_k: Number of top results to return

        Returns:
            List of search results with document information
        """
        return self.search_engine.search(query_embedding, top_k)

    def get_stats(self) -> Dict:
        """
        Get statistics about all indexed documents.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "chunks_per_document": self.documents,
            "total_chunks": len(self.all_embeddings),
            **self.search_engine.get_stats()
        }
