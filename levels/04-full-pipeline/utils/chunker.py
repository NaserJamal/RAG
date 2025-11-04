"""
Text chunking utilities for splitting documents into manageable pieces.

Implements a sliding window chunking strategy with configurable chunk size
and overlap for optimal retrieval performance.
"""

from pathlib import Path
from typing import List, Dict
import json


class TextChunker:
    """Split text into chunks using sliding window approach."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Extract chunk
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Move to next position with overlap
            start += self.chunk_size - self.overlap

            # Prevent infinite loop for very small texts
            if start <= 0:
                break

        return chunks

    def save_chunks(self, chunks: List[str], output_dir: Path) -> List[Path]:
        """
        Save chunks to individual text files.

        Args:
            chunks: List of text chunks
            output_dir: Directory to save chunk files

        Returns:
            List of paths to saved chunk files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []

        for idx, chunk in enumerate(chunks, 1):
            chunk_path = output_dir / f"chunk_{idx:03d}.txt"
            chunk_path.write_text(chunk, encoding='utf-8')
            chunk_paths.append(chunk_path)

        # Also save metadata
        metadata = {
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunk_files": [p.name for p in chunk_paths]
        }

        metadata_path = output_dir / "chunks_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        return chunk_paths

    def process_document(self, text: str, output_dir: Path) -> Dict[str, any]:
        """
        Process a document: chunk it and save to files.

        Args:
            text: Document text to process
            output_dir: Directory to save chunks

        Returns:
            Dictionary with processing results
        """
        chunks = self.chunk_text(text)
        chunk_paths = self.save_chunks(chunks, output_dir)

        return {
            "total_chunks": len(chunks),
            "chunk_paths": chunk_paths,
            "output_dir": output_dir,
            "chunks": chunks
        }
