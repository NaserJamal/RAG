"""
Configuration management for RAG training levels.

Centralized configuration for all shared components including API keys,
paths, model settings, and Qdrant connection details.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration for RAG training."""

    # Base paths
    BASE_PATH = Path(__file__).parent.parent.parent
    LEVELS_PATH = BASE_PATH / "levels"

    # Embedding API (OpenAI-compatible)
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small default

    # Qdrant configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_USE_HTTPS: bool = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"

    # Retrieval defaults
    DEFAULT_TOP_K: int = 5

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.EMBEDDING_API_KEY:
            raise ValueError(
                "EMBEDDING_API_KEY not found. Please set it in your .env file or environment."
            )
        if not cls.EMBEDDING_BASE_URL:
            raise ValueError(
                "EMBEDDING_BASE_URL not found. Please set it in your .env file or environment."
            )

    @classmethod
    def get_qdrant_url(cls) -> str:
        """Get the full Qdrant URL."""
        protocol = "https" if cls.QDRANT_USE_HTTPS else "http"
        return f"{protocol}://{cls.QDRANT_HOST}:{cls.QDRANT_PORT}"

    @classmethod
    def get_level_path(cls, level_name: str) -> Path:
        """Get the path for a specific level."""
        return cls.LEVELS_PATH / level_name

    @classmethod
    def get_documents_path(cls, level_name: str) -> Path:
        """Get the documents path for a specific level."""
        return cls.get_level_path(level_name) / "documents"

    @classmethod
    def get_output_path(cls, level_name: str) -> Path:
        """Get the output path for a specific level."""
        output_path = cls.get_level_path(level_name) / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
