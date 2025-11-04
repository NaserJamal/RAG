"""Fixed-size chunking implementation.

Split by character/token count (e.g., 512 tokens)
Add overlap between chunks (10-20%)
Simple, fast, predictable
"""

from typing import List, Dict
import tiktoken
from .config import CHUNK_SIZE, CHUNK_OVERLAP


class FixedChunker:
    """Splits text into fixed-size chunks with overlap.

    This is the simplest chunking strategy:
    - Splits at fixed token intervals
    - Adds configurable overlap (typically 10-20%)
    - No awareness of document structure
    - Fast and predictable
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the fixed chunker.

        Args:
            chunk_size: Target size of each chunk in tokens (default: 512)
            chunk_overlap: Number of tokens to overlap between chunks (default: 10% of chunk_size)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Validate overlap is reasonable (10-20% of chunk size)
        if self.chunk_overlap > self.chunk_size * 0.2:
            print(f"Warning: Overlap ({self.chunk_overlap}) is >20% of chunk size ({self.chunk_size})")

    def chunk(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk tokens
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "start_token": start,
                "end_token": end,
                "num_tokens": len(chunk_tokens),
                "method": "fixed",
                "overlap_tokens": self.chunk_overlap if start > 0 else 0
            })

            # Move start position (with overlap)
            start = start + self.chunk_size - self.chunk_overlap

            # Break if we've covered all tokens
            if end >= len(tokens):
                break

        return chunks
