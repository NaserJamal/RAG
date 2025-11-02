"""
Embedding generation utilities using OpenAI API.

Provides a clean interface for generating embeddings for both batch text
processing and single query embeddings.
"""

from typing import List
import numpy as np
from openai import OpenAI

from .config import Config


class Embedder:
    """Handles embedding generation using OpenAI API."""

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the embedder.

        Args:
            model: OpenAI embedding model name (defaults to Config.EMBEDDING_MODEL)
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
        """
        self.model = model or Config.EMBEDDING_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in environment.")

        self.client = OpenAI(api_key=self.api_key)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)

        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            return np.array([])

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query string

        Returns:
            1D NumPy array of query embedding

        Raises:
            ValueError: If query is empty
        """
        if not query:
            raise ValueError("Query string cannot be empty")

        return self.embed([query])[0]

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for the current model.

        Returns:
            Embedding dimension size
        """
        return Config.EMBEDDING_DIMENSION
