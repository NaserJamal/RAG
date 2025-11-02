"""
Vector store implementations for RAG system.

Provides an abstract base class and concrete implementations for vector storage
and retrieval, including Qdrant integration.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import Config
from .similarity import cosine_similarity


class VectorStore(ABC):
    """Abstract base class for vector storage and retrieval."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        """
        Create a new collection for storing vectors.

        Args:
            collection_name: Name of the collection
            vector_dim: Dimension of the vectors
        """
        pass

    @abstractmethod
    def add_vectors(
        self,
        collection_name: str,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add vectors to the collection.

        Args:
            collection_name: Name of the collection
            vectors: Array of vectors (n_vectors x vector_dim)
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter_conditions: Optional metadata filters

        Returns:
            List of (id, score) tuples sorted by similarity
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        api_key: str = None,
        use_https: bool = None,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant host (defaults to Config.QDRANT_HOST)
            port: Qdrant port (defaults to Config.QDRANT_PORT)
            api_key: Qdrant API key (defaults to Config.QDRANT_API_KEY)
            use_https: Use HTTPS connection (defaults to Config.QDRANT_USE_HTTPS)
        """
        self.host = host or Config.QDRANT_HOST
        self.port = port or Config.QDRANT_PORT
        self.api_key = api_key or Config.QDRANT_API_KEY
        self.use_https = use_https if use_https is not None else Config.QDRANT_USE_HTTPS

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            https=self.use_https,
        )

    def create_collection(self, collection_name: str, vector_dim: int) -> None:
        """
        Create a new Qdrant collection.

        Args:
            collection_name: Name of the collection
            vector_dim: Dimension of the vectors
        """
        # Delete existing collection if it exists
        if self.collection_exists(collection_name):
            self.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    def add_vectors(
        self,
        collection_name: str,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add vectors to Qdrant collection.

        Args:
            collection_name: Name of the collection
            vectors: Array of vectors (n_vectors x vector_dim)
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        if metadata and len(metadata) != len(ids):
            raise ValueError("Number of metadata entries must match number of IDs")

        points = []
        for idx, (vector, vector_id) in enumerate(zip(vectors, ids)):
            payload = metadata[idx] if metadata else {}
            point = PointStruct(
                id=vector_id,
                vector=vector.tolist(),
                payload=payload,
            )
            points.append(point)

        # Upload in batches for better performance
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in Qdrant.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter_conditions: Optional metadata filters

        Returns:
            List of (id, score) tuples sorted by similarity
        """
        query_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        return [(str(result.id), result.score) for result in results]

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a Qdrant collection.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(collection_name=collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a Qdrant collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def get_vector(self, collection_name: str, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific vector by ID.

        Args:
            collection_name: Name of the collection
            vector_id: ID of the vector

        Returns:
            Vector as numpy array, or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=collection_name, ids=[vector_id]
            )
            if result:
                return np.array(result[0].vector)
            return None
        except Exception:
            return None

    def get_metadata(self, collection_name: str, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific vector.

        Args:
            collection_name: Name of the collection
            vector_id: ID of the vector

        Returns:
            Metadata dictionary, or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=collection_name, ids=[vector_id]
            )
            if result:
                return result[0].payload
            return None
        except Exception:
            return None
