"""Fixed-size chunking implementation."""

from typing import List, Dict
import tiktoken
from .config import CHUNK_SIZE, CHUNK_OVERLAP


class FixedChunker:
    """Splits text into fixed-size chunks with overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the fixed chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split text into fixed-size chunks.

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
                "method": "fixed"
            })

            # Move start position (with overlap)
            start = start + self.chunk_size - self.chunk_overlap

            # Break if we've covered all tokens
            if end >= len(tokens):
                break

        return chunks
