"""
Vector search retriever using Qdrant.

This is a wrapper around the shared QdrantVectorStore for Level 02.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder, QdrantVectorStore, EmbeddingCache


class VectorRetriever:
    """Retrieves documents using semantic vector search with Qdrant."""

    def __init__(
        self,
        collection_name: str,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        cache: Optional[EmbeddingCache] = None
    ):
        """
        Initialize the vector retriever.

        Args:
            collection_name: Name of the Qdrant collection
            embedder: Embedder instance for generating embeddings
            vector_store: QdrantVectorStore instance
            cache: Optional EmbeddingCache instance for caching
        """
        self.collection_name = collection_name
        self.embedder = embedder
        self.vector_store = vector_store
        self.cache = cache
        self.documents = []

    def index(self, documents: List[Dict]) -> None:
        """
        Build vector index from documents.

        Args:
            documents: List of document dictionaries with 'id' and 'content'
        """
        self.documents = documents

        # Check if caching is enabled and embeddings are cached
        if self.cache and not self.cache.needs_embedding(
            self.collection_name, documents, self.vector_store
        ):
            print(f"✅ Using cached embeddings (collection already exists with {len(documents)} documents)")
            return

        print("🔢 Generating embeddings and building vector index...")

        doc_contents = [doc["content"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]

        # Generate embeddings
        embeddings = self.embedder.embed(doc_contents)

        # Create collection
        self.vector_store.create_collection(
            collection_name=self.collection_name,
            vector_dim=self.embedder.get_embedding_dimension()
        )

        # Add vectors with metadata
        metadata = [
            {"content": doc["content"], "path": doc.get("path", "")}
            for doc in documents
        ]
        self.vector_store.add_vectors(
            collection_name=self.collection_name,
            vectors=embeddings,
            ids=doc_ids,
            metadata=metadata
        )

        print(f"✅ Indexed {len(documents)} documents in vector store")

        # Mark as cached
        if self.cache:
            self.cache.mark_embedded(self.collection_name, documents)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for documents using vector similarity.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of (document_id, score) tuples sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search in Qdrant
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            top_k=k
        )

        return results
