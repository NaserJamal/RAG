"""Configuration constants for hybrid search."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536

# Retrieval Configuration
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Hybrid Search Configuration
ALPHA = 0.5  # Weight for semantic search (0.0 = pure keyword, 1.0 = pure semantic)
RRF_K = 60   # Constant for Reciprocal Rank Fusion

# Paths
BASE_PATH = Path(__file__).parent.parent
DOCUMENTS_PATH = BASE_PATH.parent.parent / "documents"
OUTPUT_PATH = BASE_PATH / "output"
