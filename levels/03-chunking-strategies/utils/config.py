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
CHUNK_OVERLAP = 51      # Overlap between chunks in tokens (10% of CHUNK_SIZE for default)
SEMANTIC_THRESHOLD = 0.5  # Similarity threshold for semantic chunking

# Contextual Retrieval Configuration
# Uses LLM_MODEL from shared config
CONTEXT_INSTRUCTIONS_TEMPLATE = """You are generating context for a document chunk to improve search retrieval.

<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# Retrieval Configuration
TOP_K = 3               # Number of chunks to retrieve
