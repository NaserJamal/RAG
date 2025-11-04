"""BM25 keyword search retriever."""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Retrieves documents using BM25 keyword matching."""

    def __init__(self, documents: List[Dict]):
        """
        Initialize the BM25 retriever.

        Args:
            documents: List of document dictionaries
        """
        self.documents = documents
        self.bm25 = None

    def index(self):
        """Build BM25 index from documents."""
        print("📚 Building BM25 index...")

        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc["content"].lower().split() for doc in self.documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"✅ Indexed {len(self.documents)} documents")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents using BM25.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of (document_id, score) tuples sorted by relevance
        """
        if self.bm25 is None:
            raise ValueError("Must call index() before searching")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            doc_id = self.documents[idx]["id"]
            score = float(scores[idx])
            results.append((doc_id, score))

        return results
