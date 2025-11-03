"""Configuration constants for hybrid search."""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Config as SharedConfig

# Re-export shared configuration
EMBEDDING_API_KEY = SharedConfig.EMBEDDING_API_KEY
EMBEDDING_BASE_URL = SharedConfig.EMBEDDING_BASE_URL
EMBEDDING_MODEL = SharedConfig.EMBEDDING_MODEL
TOP_K = SharedConfig.DEFAULT_TOP_K

# Level-specific paths
DOCUMENTS_PATH = SharedConfig.get_documents_path("02-hybrid-search")
OUTPUT_PATH = SharedConfig.get_output_path("02-hybrid-search")

# Hybrid Search specific configuration
ALPHA = 0.5  # Weight for semantic search (0.0 = pure keyword, 1.0 = pure semantic)
RRF_K = 60   # Constant for Reciprocal Rank Fusion
