"""Configuration constants for chunking strategies."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking Configuration
CHUNK_SIZE = 512        # Target chunk size in tokens
CHUNK_OVERLAP = 50      # Overlap between chunks in tokens
SEMANTIC_THRESHOLD = 0.5  # Similarity threshold for semantic chunking

# Retrieval Configuration
TOP_K = 3               # Number of chunks to retrieve

# Paths
BASE_PATH = Path(__file__).parent.parent
DOCUMENTS_PATH = BASE_PATH.parent.parent / "documents"
OUTPUT_PATH = BASE_PATH / "output"
