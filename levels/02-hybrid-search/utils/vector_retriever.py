"""Vector similarity search retriever."""

from typing import List, Dict, Tuple
import numpy as np
from .embedder import Embedder


class VectorRetriever:
    """Retrieves documents using vector similarity search."""

    def __init__(self, documents: List[Dict], embedder: Embedder):
        """
        Initialize the vector retriever.

        Args:
            documents: List of document dictionaries
            embedder: Embedder instance for generating embeddings
        """
        self.documents = documents
        self.embedder = embedder
        self.embeddings = None

    def index(self):
        """Generate and store embeddings for all documents."""
        print("🔢 Generating vector embeddings...")
        doc_texts = [doc["content"] for doc in self.documents]
        self.embeddings = self.embedder.embed(doc_texts)
        print(f"✅ Indexed {len(self.embeddings)} documents")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents using vector similarity.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of (document_id, score) tuples sorted by relevance
        """
        if self.embeddings is None:
            raise ValueError("Must call index() before searching")

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            doc_id = self.documents[idx]["id"]
            score = float(similarities[idx])
            results.append((doc_id, score))

        return results

    @staticmethod
    def _cosine_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: 1D array of query embedding
            doc_embeddings: 2D array of document embeddings

        Returns:
            1D array of similarity scores
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Compute dot product
        similarities = np.dot(doc_norms, query_norm)

        return similarities
