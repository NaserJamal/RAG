"""Semantic chunking implementation.

Split at natural boundaries (paragraphs, sections, sentences)
Add overlap for context preservation
Respects document structure
"""

import sys
from pathlib import Path
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .config import CHUNK_SIZE, CHUNK_OVERLAP

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class SemanticChunker:
    """Splits text at natural boundaries (paragraphs, sections, sentences) with overlap.

    This strategy:
    - Splits at paragraph boundaries first
    - Falls back to sentence boundaries for large paragraphs
    - Adds overlap between chunks
    - Respects document structure
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size of each chunk in tokens (soft limit)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split text at natural boundaries (paragraphs, sentences) with overlap.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries with metadata
        """
        # Split into paragraphs first (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Fallback: split by single newlines
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk_text = ""
        current_chunk_tokens = 0
        overlap_buffer = []  # Store recent sentences for overlap

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            # If paragraph is too large, split it by sentences
            if para_tokens > self.chunk_size:
                sentences = sent_tokenize(para)

                for sentence in sentences:
                    sent_tokens = len(self.tokenizer.encode(sentence))

                    # If adding this sentence exceeds chunk size, create a chunk
                    if current_chunk_tokens + sent_tokens > self.chunk_size and current_chunk_text:
                        # Create chunk
                        chunks.append(self._create_chunk(
                            current_chunk_text.strip(),
                            doc_id,
                            len(chunks),
                            current_chunk_tokens
                        ))

                        # Start new chunk with overlap from previous chunk
                        current_chunk_text = self._get_overlap_text(overlap_buffer)
                        current_chunk_tokens = len(self.tokenizer.encode(current_chunk_text))
                        overlap_buffer = []

                    # Add sentence to current chunk
                    current_chunk_text += (" " if current_chunk_text else "") + sentence
                    current_chunk_tokens += sent_tokens
                    overlap_buffer.append(sentence)

            else:
                # Paragraph fits within chunk size
                # If adding this paragraph exceeds chunk size, create a chunk
                if current_chunk_tokens + para_tokens > self.chunk_size and current_chunk_text:
                    # Create chunk
                    chunks.append(self._create_chunk(
                        current_chunk_text.strip(),
                        doc_id,
                        len(chunks),
                        current_chunk_tokens
                    ))

                    # Start new chunk with overlap from previous chunk
                    current_chunk_text = self._get_overlap_text(overlap_buffer)
                    current_chunk_tokens = len(self.tokenizer.encode(current_chunk_text))
                    overlap_buffer = []

                # Add paragraph to current chunk
                current_chunk_text += ("\n\n" if current_chunk_text else "") + para
                current_chunk_tokens += para_tokens

                # Split paragraph into sentences for overlap buffer
                para_sentences = sent_tokenize(para)
                overlap_buffer.extend(para_sentences)

        # Add final chunk if there's remaining text
        if current_chunk_text.strip():
            chunks.append(self._create_chunk(
                current_chunk_text.strip(),
                doc_id,
                len(chunks),
                current_chunk_tokens
            ))

        return chunks

    def _get_overlap_text(self, sentence_buffer: List[str]) -> str:
        """
        Get overlap text from recent sentences to prepend to next chunk.

        Args:
            sentence_buffer: List of recent sentences

        Returns:
            Overlap text (up to chunk_overlap tokens)
        """
        if not sentence_buffer:
            return ""

        # Take sentences from the end until we reach overlap token limit
        overlap_text = ""
        overlap_tokens = 0

        for sentence in reversed(sentence_buffer):
            sent_tokens = len(self.tokenizer.encode(sentence))

            if overlap_tokens + sent_tokens <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
                overlap_tokens += sent_tokens
            else:
                break

        return overlap_text.strip()

    def _create_chunk(self, text: str, doc_id: str, chunk_idx: int, num_tokens: int) -> Dict:
        """
        Create a chunk dictionary.

        Args:
            text: Chunk text
            doc_id: Document identifier
            chunk_idx: Index of chunk
            num_tokens: Number of tokens in chunk

        Returns:
            Chunk dictionary with metadata
        """
        return {
            "text": text,
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
            "num_tokens": num_tokens,
            "method": "semantic"
        }
