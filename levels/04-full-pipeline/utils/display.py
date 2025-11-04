"""
Display utilities for pretty terminal output.

Provides consistent formatting for headers, steps, results, and errors.
"""

from typing import List, Dict


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(text: str) -> None:
    """Print a step indicator."""
    print(f"\n{text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"✗ {text}")


def display_search_results(query: str, results: List[Dict]) -> None:
    """
    Display search results in a formatted way.

    Args:
        query: The search query
        results: List of search result dictionaries
    """
    print(f"\n{'─' * 70}")
    print(f"Query: {query}")
    print(f"{'─' * 70}\n")

    if not results:
        print("No results found.")
        return

    for result in results:
        rank = result.get('rank', '?')
        score = result.get('score', 0.0)
        doc = result.get('document', 'Unknown')
        chunk_id = result.get('chunk_id', '?')
        text = result.get('text', '')

        # Truncate text for display
        preview = text[:200] + "..." if len(text) > 200 else text

        print(f"[{rank}] Score: {score:.4f}")
        print(f"    Document: {doc} | Chunk: {chunk_id}")
        print(f"    {preview}\n")

    print(f"{'─' * 70}\n")


def display_pipeline_summary(stats: Dict) -> None:
    """
    Display a summary of the pipeline processing.

    Args:
        stats: Dictionary containing pipeline statistics
    """
    print_header("Pipeline Summary")

    print(f"📄 Documents Processed: {stats.get('total_documents', 0)}")
    print(f"📝 Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"🔢 Embedding Dimension: {stats.get('embedding_dimension', 0)}")
    print(f"💾 New Embeddings: {stats.get('new_embeddings', 0)}")
    print(f"♻️  Cached Embeddings: {stats.get('cached_embeddings', 0)}")

    if 'chunks_per_document' in stats:
        print("\nChunks per document:")
        for doc, count in stats['chunks_per_document'].items():
            print(f"  - {doc}: {count} chunks")

    print()
