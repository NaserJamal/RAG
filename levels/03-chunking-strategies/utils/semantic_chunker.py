"""Semantic chunking implementation using embedding similarity."""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder
from .config import CHUNK_SIZE, SEMANTIC_THRESHOLD

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SemanticChunker:
    """Splits text at semantic boundaries using embedding similarity."""

    def __init__(self, embedder: Embedder, chunk_size: int = CHUNK_SIZE,
                 threshold: float = SEMANTIC_THRESHOLD):
        """
        Initialize the semantic chunker.

        Args:
            embedder: Embedder instance for generating embeddings
            chunk_size: Target size of each chunk in tokens (soft limit)
            threshold: Similarity threshold for detecting topic boundaries
        """
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split text at semantic boundaries.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        # Split into sentences
        sentences = sent_tokenize(text)

        if not sentences:
            return []

        # Generate embeddings for all sentences
        print(f"  Generating embeddings for {len(sentences)} sentences...")
        sentence_embeddings = self.embedder.embed(sentences)

        # Compute similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(
                sentence_embeddings[i],
                sentence_embeddings[i + 1]
            )
            similarities.append(sim)

        # Find split points where similarity drops below threshold
        split_indices = [0]
        current_chunk_tokens = 0

        for i, sim in enumerate(similarities):
            sent_tokens = len(self.tokenizer.encode(sentences[i + 1]))

            # Split if similarity drops below threshold OR chunk size exceeded
            if sim < self.threshold or current_chunk_tokens + sent_tokens > self.chunk_size:
                split_indices.append(i + 1)
                current_chunk_tokens = sent_tokens
            else:
                current_chunk_tokens += sent_tokens

        split_indices.append(len(sentences))

        # Create chunks from split indices
        chunks = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            chunk_tokens = len(self.tokenizer.encode(chunk_text))

            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "num_tokens": chunk_tokens,
                "num_sentences": len(chunk_sentences),
                "method": "semantic"
            })

        return chunks

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1_norm, vec2_norm))
