"""Embedding generation utilities."""

from typing import List
import numpy as np
from openai import OpenAI
from .config import EMBEDDING_MODEL, OPENAI_API_KEY


class Embedder:
    """Handles embedding generation using OpenAI API."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedder.

        Args:
            model: OpenAI embedding model name
        """
        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)
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
        """
        return self.embed([query])[0]
