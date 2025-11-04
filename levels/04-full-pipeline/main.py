"""
Level 04: Full Pipeline RAG

A complete RAG pipeline demonstrating the full workflow:
PDF → Text Extraction → Chunking → Embedding → Semantic Search

This level shows how all components work together in a production-like setup
with smart caching to avoid redundant processing.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config
from utils import (
    PDFExtractor,
    TextChunker,
    EmbeddingManager,
    MultiDocumentSearchEngine,
    print_header,
    print_step,
    print_success,
    print_error,
    display_search_results,
    display_pipeline_summary
)

LEVEL_NAME = "04-full-pipeline"


class FullPipeline:
    """Orchestrate the complete RAG pipeline."""

    def __init__(self, base_dir: Path):
        """
        Initialize the pipeline.

        Args:
            base_dir: Base directory for the level
        """
        self.base_dir = base_dir
        self.documents_dir = base_dir / "documents"
        self.output_dir = base_dir / "output"
        self.cache_dir = base_dir / "output" / ".cache"

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.chunker = TextChunker(chunk_size=500, overlap=100)
        self.embedding_manager = EmbeddingManager(self.cache_dir)
        self.search_engine = MultiDocumentSearchEngine()

        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'new_embeddings': 0,
            'cached_embeddings': 0,
            'embedding_dimension': 0,
            'chunks_per_document': {}
        }

    def check_documents(self) -> bool:
        """
        Check if there are PDF documents to process.

        Returns:
            True if documents exist, False otherwise
        """
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        return len(pdf_files) > 0

    def process_documents(self) -> None:
        """Extract text from all PDF documents and chunk them."""
        print_step("📄 Step 1: Processing PDF Documents")

        pdf_files = list(self.documents_dir.glob("*.pdf"))
        self.stats['total_documents'] = len(pdf_files)

        print(f"Found {len(pdf_files)} PDF document(s)")

        for pdf_path in sorted(pdf_files):
            print(f"\n  Processing: {pdf_path.name}")

            # Extract text
            doc_dir, text = self.pdf_extractor.process_pdf(pdf_path, self.output_dir)
            doc_name = PDFExtractor.normalize_name(pdf_path.name)

            print_success(f"Extracted {len(text)} characters")

            # Chunk the text
            chunk_result = self.chunker.process_document(text, doc_dir)
            num_chunks = chunk_result['total_chunks']
            self.stats['total_chunks'] += num_chunks
            self.stats['chunks_per_document'][doc_name] = num_chunks

            print_success(f"Created {num_chunks} chunks → {doc_dir}/")

    def generate_embeddings(self) -> None:
        """Generate embeddings for all document chunks."""
        print_step("\n🔢 Step 2: Generating Embeddings")

        # Process each document
        for doc_name in self.stats['chunks_per_document'].keys():
            doc_dir = self.output_dir / doc_name
            print(f"\n  Processing embeddings for: {doc_name}")

            # Check if embeddings already exist
            embeddings_file = doc_dir / "embeddings.json"

            if embeddings_file.exists():
                print_success(f"Loading existing embeddings from cache")
                embeddings_data = self.embedding_manager.load_embeddings(embeddings_file)
                self.stats['cached_embeddings'] += len(embeddings_data)
            else:
                # Load chunks
                chunk_files = sorted(doc_dir.glob("chunk_*.txt"))
                chunks = [f.read_text() for f in chunk_files]

                # Generate embeddings
                embeddings_data = self.embedding_manager.embed_chunks(chunks, show_progress=True)
                self.stats['new_embeddings'] += len(embeddings_data)

                # Save embeddings
                self.embedding_manager.save_embeddings(embeddings_data, doc_dir)
                print_success(f"Saved embeddings to {embeddings_file}")

            # Add to search engine
            self.search_engine.add_document(doc_name, embeddings_data)

        # Build search index
        self.search_engine.build_index()
        self.stats['embedding_dimension'] = self.embedding_manager.get_embedding_dimension()

        print_success("\n✓ All embeddings indexed for search")

    def interactive_search(self) -> None:
        """Run interactive search loop."""
        print_step("\n🔍 Step 3: Semantic Search")
        print("\nYou can now search across all documents!")
        print("Commands:")
        print("  - Type your query and press Enter to search")
        print("  - Press Enter on empty line to exit")
        print("  - Type 'stats' to see pipeline statistics\n")

        while True:
            try:
                query = input("Enter your query: ").strip()

                if not query:
                    print("\n👋 Goodbye!")
                    break

                if query.lower() == 'stats':
                    display_pipeline_summary(self.stats)
                    continue

                # Generate query embedding
                query_embedding = self.embedding_manager.embedder.embed_query(query)

                # Search
                results = self.search_engine.search(query_embedding, top_k=5)

                # Display results
                display_search_results(query, results)

                # Save results
                self._save_search_results(query, results)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print_error(f"Error during search: {str(e)}")

    def _save_search_results(self, query: str, results: List[Dict]) -> None:
        """
        Save search results to output directory.

        Args:
            query: Search query
            results: List of search results
        """
        import json
        from datetime import datetime

        results_file = self.output_dir / "search_history.jsonl"

        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": results
        }

        # Append to JSONL file
        with results_file.open('a') as f:
            f.write(json.dumps(result_entry) + '\n')

    def run(self) -> None:
        """Execute the full pipeline."""
        # Check for documents
        if not self.check_documents():
            print_error(f"No PDF documents found in {self.documents_dir}/")
            print(f"\nPlease add PDF files to: {self.documents_dir}/")
            print("Then run this script again.\n")
            return

        # Step 1: Process PDFs and chunk
        self.process_documents()

        # Step 2: Generate embeddings
        self.generate_embeddings()

        # Display summary
        display_pipeline_summary(self.stats)

        # Step 3: Interactive search
        self.interactive_search()


def main():
    """Main execution function."""
    try:
        Config.validate()
    except ValueError as e:
        print_error(f"Configuration error: {e}")
        print("\nPlease ensure your .env file is properly configured.")
        return

    level_dir = Config.get_level_path(LEVEL_NAME)

    print_header("Level 04: Full Pipeline RAG")
    print("This level demonstrates the complete RAG workflow:")
    print("  PDF → Text Extraction → Chunking → Embedding → Search\n")

    pipeline = FullPipeline(level_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
