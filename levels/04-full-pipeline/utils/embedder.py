"""
Embedding generation and caching for document chunks.

Handles creation and persistence of embeddings with smart caching to avoid
redundant API calls.
"""

import sys
from pathlib import Path
import json
import hashlib
from typing import List, Dict, Optional

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder as SharedEmbedder, Config


class EmbeddingManager:
    """Manage embedding generation with persistent caching."""

    def __init__(self, cache_dir: Path):
        """
        Initialize the embedding manager.

        Args:
            cache_dir: Directory for caching embeddings
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SharedEmbedder()
        self.cache_file = self.cache_dir / "embeddings_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load the embedding cache from disk."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save the embedding cache to disk."""
        self.cache_file.write_text(json.dumps(self.cache, indent=2))

    def _compute_hash(self, text: str) -> str:
        """
        Compute a hash for text content.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def _get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding if available.

        Args:
            text_hash: Hash of the text content

        Returns:
            Cached embedding or None
        """
        return self.cache.get(text_hash)

    def _cache_embedding(self, text_hash: str, embedding: List[float]) -> None:
        """
        Cache an embedding.

        Args:
            text_hash: Hash of the text content
            embedding: Embedding vector to cache
        """
        self.cache[text_hash] = embedding
        self._save_cache()

    def embed_chunks(self, chunks: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Generate embeddings for chunks with caching.

        Args:
            chunks: List of text chunks to embed
            show_progress: Whether to show progress information

        Returns:
            List of dictionaries with chunk text, hash, and embedding
        """
        results = []
        new_embeddings_count = 0
        cached_count = 0

        for idx, chunk in enumerate(chunks, 1):
            text_hash = self._compute_hash(chunk)

            # Check cache
            cached_embedding = self._get_cached_embedding(text_hash)

            if cached_embedding is not None:
                embedding = cached_embedding
                cached_count += 1
            else:
                # Generate new embedding
                embedding = self.embedder.embed_query(chunk)
                # Convert to list if it's a numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                self._cache_embedding(text_hash, embedding)
                new_embeddings_count += 1

            results.append({
                "chunk_id": idx,
                "text": chunk,
                "text_hash": text_hash,
                "embedding": embedding
            })

            if show_progress and idx % 10 == 0:
                print(f"  Processed {idx}/{len(chunks)} chunks...")

        if show_progress:
            print(f"  Generated {new_embeddings_count} new embeddings, used {cached_count} cached")

        return results

    def save_embeddings(self, embeddings_data: List[Dict], output_dir: Path) -> Path:
        """
        Save embeddings to disk.

        Args:
            embeddings_data: List of embedding dictionaries
            output_dir: Directory to save embeddings

        Returns:
            Path to the saved embeddings file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        embeddings_file = output_dir / "embeddings.json"

        # Save with metadata
        data = {
            "model": Config.EMBEDDING_MODEL,
            "dimension": self.embedder.get_embedding_dimension(),
            "total_embeddings": len(embeddings_data),
            "embeddings": embeddings_data
        }

        embeddings_file.write_text(json.dumps(data, indent=2))
        return embeddings_file

    def load_embeddings(self, embeddings_file: Path) -> List[Dict]:
        """
        Load embeddings from disk.

        Args:
            embeddings_file: Path to embeddings file

        Returns:
            List of embedding dictionaries
        """
        if not embeddings_file.exists():
            return []

        data = json.loads(embeddings_file.read_text())
        return data.get("embeddings", [])

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedder.get_embedding_dimension()
