"""
Cache management for vector embeddings.

Provides hash-based tracking to avoid re-embedding unchanged documents.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional


class EmbeddingCache:
    """Manages caching of document embeddings based on content hashes."""

    def __init__(self, cache_dir: Path):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.json"

    def _compute_hash(self, documents: list[dict]) -> str:
        """
        Compute hash of all document contents.

        Args:
            documents: List of document dicts with 'content' key

        Returns:
            SHA256 hash of concatenated document contents
        """
        content = "".join(doc["content"] for doc in documents)
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_cache_metadata(self, collection_name: str) -> Optional[dict]:
        """Load cache metadata for a collection."""
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)
                return cache_data.get(collection_name)
        except (json.JSONDecodeError, IOError):
            return None

    def _save_cache_metadata(self, collection_name: str, metadata: dict) -> None:
        """Save cache metadata for a collection."""
        cache_data = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        cache_data[collection_name] = metadata

        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

    def needs_embedding(
        self,
        collection_name: str,
        documents: list[dict],
        vector_store,
    ) -> bool:
        """
        Check if documents need to be embedded.

        Args:
            collection_name: Name of the vector collection
            documents: List of document dicts
            vector_store: Vector store instance to check collection existence

        Returns:
            True if embedding is needed, False if cache is valid
        """
        # Check if collection exists
        if not vector_store.collection_exists(collection_name):
            return True

        # Check document count
        collection_count = vector_store.count_vectors(collection_name)
        if collection_count != len(documents):
            return True

        # Check content hash
        current_hash = self._compute_hash(documents)
        cached_metadata = self._load_cache_metadata(collection_name)

        if not cached_metadata:
            return True

        if cached_metadata.get("content_hash") != current_hash:
            return True

        if cached_metadata.get("document_count") != len(documents):
            return True

        return False

    def mark_embedded(self, collection_name: str, documents: list[dict]) -> None:
        """
        Mark documents as embedded in cache.

        Args:
            collection_name: Name of the vector collection
            documents: List of document dicts that were embedded
        """
        content_hash = self._compute_hash(documents)
        metadata = {
            "content_hash": content_hash,
            "document_count": len(documents),
        }
        self._save_cache_metadata(collection_name, metadata)
