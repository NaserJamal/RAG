"""Configuration constants for chunking strategies."""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Config as SharedConfig

# Re-export shared configuration
EMBEDDING_API_KEY = SharedConfig.EMBEDDING_API_KEY
EMBEDDING_BASE_URL = SharedConfig.EMBEDDING_BASE_URL
EMBEDDING_MODEL = SharedConfig.EMBEDDING_MODEL

# Level-specific paths
DOCUMENTS_PATH = SharedConfig.get_documents_path("03-chunking-strategies")
OUTPUT_PATH = SharedConfig.get_output_path("03-chunking-strategies")

# Chunking Configuration
CHUNK_SIZE = 512        # Target chunk size in tokens
CHUNK_OVERLAP = 50      # Overlap between chunks in tokens
SEMANTIC_THRESHOLD = 0.5  # Similarity threshold for semantic chunking

# Retrieval Configuration
TOP_K = 3               # Number of chunks to retrieve
