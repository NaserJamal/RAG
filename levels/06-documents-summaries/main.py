"""
Level 06: Multi-Document Executive Summaries
============================================

Hierarchical Map-Reduce summarization for scaling to hundreds of documents.

This level demonstrates:
- Extracting text from multiple PDFs
- Parallel document summarization (Map phase)
- Hierarchical summary combination (Reduce phase)
- Tree-based batching for scalability
- Executive summary generation

Key concepts:
- Map: Each document summarized independently (parallel)
- Reduce: Summaries combined in batches of 10, recursively (parallel within level)
- Scalability: O(log n) levels for n documents
"""

import asyncio
import sys
from pathlib import Path

# Add shared modules to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.config import Config
from utils.pdf_extractor import PDFExtractor
from utils.summarizer import HierarchicalSummarizer


async def main():
    """Main execution function."""

    # Setup paths
    level_dir = Path(__file__).parent
    documents_dir = level_dir / "documents"
    output_dir = level_dir / "output"

    # Ensure directories exist
    documents_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Level 06: Multi-Document Executive Summaries")
    print(f"{'='*70}\n")

    # Validate configuration
    try:
        Config.validate()
        if not Config.LLM_API_KEY:
            raise ValueError(
                "LLM_API_KEY not found. This level requires an LLM API key."
            )
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease ensure your .env file contains:")
        print("  - EMBEDDING_API_KEY")
        print("  - EMBEDDING_BASE_URL")
        print("  - LLM_API_KEY")
        print("  - LLM_BASE_URL")
        print("  - LLM_MODEL")
        return

    # Check for PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {documents_dir}")
        print("\nTo use this level:")
        print(f"  1. Add PDF files to: {documents_dir}")
        print(f"  2. Run: python {Path(__file__).name}")
        print("\nExample:")
        print(f"  cp /path/to/your/*.pdf {documents_dir}/")
        return

    print(f"📁 Found {len(pdf_files)} PDF files in {documents_dir}\n")

    # Step 1: Extract text from PDFs
    print(f"[Step 1/3] Extracting text from PDFs...")
    extractor = PDFExtractor()

    try:
        extracted_docs = extractor.extract_from_directory(documents_dir)
        print(f"✓ Extracted text from {len(extracted_docs)} documents\n")
    except Exception as e:
        print(f"❌ Error extracting PDFs: {e}")
        return

    # Prepare documents for summarization
    documents = [
        {"filename": filename, "text": text}
        for filename, text in extracted_docs
    ]

    # Step 2: Get user instructions (optional)
    print(f"[Step 2/4] Custom instructions (optional)")
    print("─" * 70)
    print("You can provide custom instructions for the executive summary.")
    print("Examples:")
    print("  • Focus on financial metrics and ROI")
    print("  • Highlight technical challenges and solutions")
    print("  • Summarize in bullet points only")
    print("  • Extract action items and deadlines")
    print("\nPress Enter to skip, or type your instructions:")
    print("─" * 70)

    user_instructions = input("> ").strip()

    if user_instructions:
        print(f"\n✓ Custom instructions received: {user_instructions[:60]}{'...' if len(user_instructions) > 60 else ''}\n")
    else:
        print(f"\n✓ Using default summary format\n")
        user_instructions = None

    # Step 3: Initialize summarizer
    print(f"[Step 3/4] Initializing hierarchical summarizer...")
    print(f"  Model: {Config.LLM_MODEL}")
    print(f"  Batch size: 10 summaries per reduce step")
    print(f"  Max concurrent: 5 LLM calls\n")

    summarizer = HierarchicalSummarizer(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        batch_size=10,
        max_concurrent=5
    )

    # Step 4: Process documents
    print(f"[Step 4/4] Processing documents with hierarchical map-reduce...")

    try:
        result = await summarizer.process_documents(documents, output_dir, user_instructions)

        # Display executive summary
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY")
        if user_instructions:
            print(f"\n## User Instructions:")
            print(f"{user_instructions}")
        print("="*70 + "\n")
        print(result['executive_summary'])
        print("\n" + "="*70)

        # Display cost information
        metadata = result['metadata']
        total_tokens = metadata['total_tokens_used']

        # Estimate costs (approximate for GPT-4-mini)
        # Adjust these rates based on your actual model pricing
        cost_per_1k_tokens = 0.0015  # Example rate
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        print(f"\n📊 Processing Statistics:")
        print(f"  Documents processed: {metadata['total_documents']}")
        print(f"  Hierarchy levels: {metadata['levels_processed']}")
        if user_instructions:
            print(f"  Custom instructions: Yes")
        print(f"  Total tokens used: {total_tokens:,}")
        print(f"  Processing time: {metadata['processing_time_seconds']:.2f}s")
        print(f"  Estimated cost: ${estimated_cost:.4f}")

        print(f"\n💾 Outputs saved to: {output_dir}")
        print(f"  → 01_individual_summaries.json  (each document's summary)")
        print(f"  → 02_level_N_summaries.json     (intermediate reduce steps)")
        print(f"  → 03_executive_summary.txt      (final summary)")
        print(f"  → 04_metadata.json              (processing details)")

        print(f"\n✅ Summarization complete!\n")

    except Exception as e:
        print(f"\n❌ Error during summarization: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    asyncio.run(main())
