"""Contextual chunking implementation (LLM-Based Chunk Enrichment).

Uses LLM to add chunk-specific explanatory context to each chunk.
Based on Anthropic's Contextual Retrieval approach.
Significantly improves retrieval accuracy by providing context for each chunk.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
import tiktoken

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Config
from .config import CHUNK_SIZE, CHUNK_OVERLAP, CONTEXT_INSTRUCTIONS_TEMPLATE


class ContextualChunker:
    """Enriches chunks with LLM-generated context for improved retrieval.

    This advanced strategy:
    - First chunks the document (using fixed-size chunking as base)
    - Uses an LLM (Claude) to generate context for each chunk
    - Prepends chunk-specific explanatory context to each chunk
    - Dramatically improves retrieval accuracy (49-67% reduction in failed retrievals)

    Based on Anthropic's Contextual Retrieval research:
    https://www.anthropic.com/news/contextual-retrieval
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        model: Optional[str] = None
    ):
        """
        Initialize the contextual chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            model: LLM model to use for context generation (uses LLM_MODEL from config if not specified)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model or Config.LLM_MODEL
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL
        )

    def chunk(self, text: str, doc_id: str, show_progress: bool = True) -> List[Dict]:
        """
        Split text into chunks and enrich each with LLM-generated context.

        Args:
            text: Full document text
            doc_id: Document identifier
            show_progress: Whether to print progress messages

        Returns:
            List of enriched chunk dictionaries with metadata
        """
        # Step 1: Create base chunks using fixed-size chunking
        base_chunks = self._create_base_chunks(text, doc_id)

        if show_progress:
            print(f"  Created {len(base_chunks)} base chunks")
            print(f"  Generating contextual enrichment for each chunk...")

        # Step 2: Enrich each chunk with context
        enriched_chunks = []
        for i, chunk in enumerate(base_chunks):
            if show_progress and (i + 1) % 5 == 0:
                print(f"    Processed {i + 1}/{len(base_chunks)} chunks...")

            # Generate context for this chunk
            context = self._generate_context(text, chunk["text"])

            # Create enriched chunk
            enriched_text = f"{context}\n\n{chunk['text']}"

            enriched_chunks.append({
                "text": enriched_text,
                "original_text": chunk["text"],
                "context": context,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "start_token": chunk["start_token"],
                "end_token": chunk["end_token"],
                "num_tokens": len(self.tokenizer.encode(enriched_text)),
                "num_tokens_original": chunk["num_tokens"],
                "num_tokens_context": len(self.tokenizer.encode(context)),
                "method": "contextual",
                "overlap_tokens": chunk["overlap_tokens"]
            })

        if show_progress:
            print(f"  ✓ Contextual enrichment complete")

        return enriched_chunks

    def _create_base_chunks(self, text: str, doc_id: str) -> List[Dict]:
        """
        Create base chunks using fixed-size chunking.

        Args:
            text: Text to chunk
            doc_id: Document identifier

        Returns:
            List of base chunk dictionaries
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "start_token": start,
                "end_token": end,
                "num_tokens": len(chunk_tokens),
                "overlap_tokens": self.chunk_overlap if start > 0 else 0
            })

            start = start + self.chunk_size - self.chunk_overlap

            if end >= len(tokens):
                break

        return chunks

    def _generate_context(self, document: str, chunk: str) -> str:
        """
        Generate chunk-specific context using LLM.

        Args:
            document: Full document text
            chunk: Chunk text to generate context for

        Returns:
            Generated context string
        """
        # Prepare the prompt
        prompt = CONTEXT_INSTRUCTIONS_TEMPLATE.format(
            document=document,
            chunk=chunk
        )

        try:
            # Call the LLM to generate context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,  # Context should be concise (50-100 tokens)
                temperature=0  # Deterministic
            )

            # Extract the generated context
            context = response.choices[0].message.content.strip()
            return context

        except Exception as e:
            # If context generation fails, return a minimal context
            print(f"  Warning: Context generation failed: {e}")
            return f"This chunk is from the document."

    def estimate_cost(self, text: str, num_chunks: Optional[int] = None) -> Dict[str, float]:
        """
        Estimate the cost of processing a document with contextual chunking.

        Args:
            text: Document text
            num_chunks: Number of chunks (if known), otherwise estimated

        Returns:
            Dictionary with cost breakdown
        """
        # Count tokens in document
        doc_tokens = len(self.tokenizer.encode(text))

        # Estimate number of chunks
        if num_chunks is None:
            num_chunks = (doc_tokens + self.chunk_size - 1) // (self.chunk_size - self.chunk_overlap)

        # Rough cost estimation (varies by provider)
        # Using typical pricing: ~$0.50 per million input tokens, ~$1.50 per million output tokens
        context_tokens = 100  # Average context length
        instruction_tokens = 100  # Instruction template tokens

        # Cost for each chunk (no caching in this simplified version)
        input_per_chunk = doc_tokens + instruction_tokens
        total_input = num_chunks * input_per_chunk * 0.50 / 1_000_000
        total_output = num_chunks * context_tokens * 1.50 / 1_000_000
        total_cost = total_input + total_output

        return {
            "total_cost_usd": round(total_cost, 6),
            "num_chunks": num_chunks,
            "document_tokens": doc_tokens,
            "input_cost_usd": round(total_input, 6),
            "output_cost_usd": round(total_output, 6),
            "uses_caching": False
        }
