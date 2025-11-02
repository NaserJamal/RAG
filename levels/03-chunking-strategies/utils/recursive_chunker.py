"""Recursive character-based chunking implementation."""

from typing import List, Dict
import re
import tiktoken
from .config import CHUNK_SIZE, CHUNK_OVERLAP


class RecursiveChunker:
    """Splits text recursively by trying separators in order."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Separators in order of preference
        self.separators = [
            "\n\n",      # Paragraphs
            "\n",        # Lines
            ". ",        # Sentences
            "! ",        # Sentences
            "? ",        # Sentences
            "; ",        # Clauses
            ", ",        # Phrases
            " ",         # Words
            ""           # Characters (last resort)
        ]

    def chunk(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split text recursively by trying different separators.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        splits = self._recursive_split(text, self.separators)

        current_chunk = []
        current_tokens = 0

        for split in splits:
            split_tokens = len(self.tokenizer.encode(split))

            # If adding this split would exceed chunk size, save current chunk
            if current_tokens + split_tokens > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "num_tokens": current_tokens,
                    "method": "recursive"
                })

                # Keep overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(self.tokenizer.encode(overlap_text)) if overlap_text else 0

            current_chunk.append(split)
            current_tokens += split_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "num_tokens": current_tokens,
                "method": "recursive"
            })

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text by trying separators in order.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text segments
        """
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Base case: split by characters
            return list(text)

        # Split by current separator
        splits = text.split(separator)

        # Recursively split large segments
        final_splits = []
        for i, split in enumerate(splits):
            if not split:
                continue

            # Add separator back (except for last split)
            if i < len(splits) - 1:
                split = split + separator

            # If split is still too large, recursively split with next separator
            split_tokens = len(self.tokenizer.encode(split))
            if split_tokens > self.chunk_size and remaining_separators:
                final_splits.extend(self._recursive_split(split, remaining_separators))
            else:
                final_splits.append(split)

        return final_splits

    def _get_overlap(self, current_chunk: List[str]) -> str:
        """
        Get overlap text from current chunk.

        Args:
            current_chunk: List of text segments in current chunk

        Returns:
            Overlap text
        """
        chunk_text = "".join(current_chunk)
        tokens = self.tokenizer.encode(chunk_text)

        if len(tokens) <= self.chunk_overlap:
            return chunk_text

        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.tokenizer.decode(overlap_tokens)
