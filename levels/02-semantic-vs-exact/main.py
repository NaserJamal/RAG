"""
Level 02: Vector vs BM25 Search Comparison

Demonstrates the difference between semantic (vector) search and keyword (BM25) search.
Vector search understands meaning and context, while BM25 excels at exact matches.

Example: Searching for an Emirates ID shows BM25's strength in exact keyword matching,
while vector search may return semantically similar but irrelevant results.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config, Embedder, QdrantVectorStore, load_documents, OutputManager, EmbeddingCache

from utils.config import DOCUMENTS_PATH, OUTPUT_PATH, TOP_K
from utils.vector_retriever import VectorRetriever
from utils.bm25_retriever import BM25Retriever

COLLECTION_NAME = "level_02_dual_search"


def format_results(results: List[tuple], documents: List[Dict]) -> List[Dict]:
    """Format retriever results into display-friendly structure."""
    doc_map = {doc["id"]: doc for doc in documents}

    formatted = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_map.get(doc_id, {})
        content = doc.get("content", "").strip()
        formatted.append({
            "rank": rank,
            "score": float(score),
            "document_id": doc_id,
            "path": doc.get("path", ""),
            "preview": content[:200] + "..." if len(content) > 200 else content
        })

    return formatted


def print_comparison(
    vector_results: List[Dict],
    bm25_results: List[Dict]
):
    """Print side-by-side comparison of results with previews."""
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100 + "\n")

    print("Vector Search (Semantic - understands context):")
    print("-" * 100)
    for result in vector_results:
        doc_id = result["document_id"][:50]
        print(f"  #{result['rank']} - {doc_id:50s} Score: {result['score']:.4f}")
        if result['rank'] == 1:
            preview = result['preview'].replace('\n', ' ')[:150]
            print(f"      Preview: {preview}...")

    print("\nBM25 Search (Keyword - excels at exact matches):")
    print("-" * 100)
    for result in bm25_results:
        doc_id = result["document_id"][:50]
        print(f"  #{result['rank']} - {doc_id:50s} Score: {result['score']:.4f}")
        if result['rank'] == 1:
            preview = result['preview'].replace('\n', ' ')[:150]
            print(f"      Preview: {preview}...")

    print()


def main():
    """Main execution function."""
    # Validate configuration
    Config.validate()

    # Initialize components
    cache_dir = Config.SHARED_PATH / "data"
    embedder = Embedder()
    vector_store = QdrantVectorStore()
    output_manager = OutputManager(OUTPUT_PATH)
    cache = EmbeddingCache(cache_dir)

    print("=" * 100)
    print("Level 02: Vector vs BM25 Search Comparison")
    print("=" * 100)

    # Load all documents (both generic and citizens)
    print("\n📄 Loading documents...")

    # Load generic documents
    documents = load_documents(DOCUMENTS_PATH)

    # Also load citizen documents
    citizens_path = Path(__file__).parent / "citizens"
    if citizens_path.exists() and citizens_path.is_dir():
        for doc_file in citizens_path.glob("*.txt"):
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "id": f"citizens/{doc_file.name}",
                        "path": str(doc_file),
                        "content": content,
                    })
            except Exception as e:
                print(f"Warning: Failed to load {doc_file}: {e}")

    print(f"✅ Loaded {len(documents)} documents\n")

    # Initialize retrievers
    print("🔧 Initializing retrievers...")
    vector_retriever = VectorRetriever(COLLECTION_NAME, embedder, vector_store, cache)
    bm25_retriever = BM25Retriever(documents)

    # Index documents
    vector_retriever.index(documents)
    bm25_retriever.index()
    print()

    # Run two queries to showcase different strengths
    queries = [
        {
            "query": "How do I submit travel expenses?",
            "description": "Semantic query - Vector search excels here",
            "k": TOP_K
        },
        {
            "query": "784-1992-7856432-1",
            "description": "Exact ID match - BM25 excels here",
            "k": 4  # Only show citizen documents
        }
    ]

    all_results = []
    for query_info in queries:
        query = query_info["query"]
        print(f"🔍 Query {queries.index(query_info) + 1}: '{query}'")
        print(f"   → {query_info['description']}\n")

        # Get results from both methods
        vector_results = format_results(vector_retriever.search(query, k=query_info["k"]), documents)
        bm25_results = format_results(bm25_retriever.search(query, k=query_info["k"]), documents)

        # Print comparison
        print_comparison(vector_results, bm25_results)

        all_results.append({
            "query": query,
            "description": query_info["description"],
            "vector_results": vector_results,
            "bm25_results": bm25_results
        })

    # Save results
    print("💾 Saving results...")

    # Save all results
    output_manager.save_results("all_queries_results", {
        "queries": all_results,
        "metadata": {
            "vector_store": "qdrant",
            "collection": COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "total_documents": len(documents)
        }
    })

    # Create comparison text file
    comparison_lines = []
    for idx, result in enumerate(all_results, 1):
        comparison_lines.append(f"\n{'='*100}")
        comparison_lines.append(f"QUERY {idx}: {result['query']}")
        comparison_lines.append(f"Description: {result['description']}")
        comparison_lines.append('='*100 + '\n')

        comparison_text = output_manager.format_comparison(
            query=result['query'],
            comparisons={
                "vector_search": result['vector_results'],
                "bm25_search": result['bm25_results']
            }
        )
        comparison_lines.append(comparison_text)

    output_manager.save_text("comparison.txt", "\n".join(comparison_lines))

    print(f"✅ Results saved to {OUTPUT_PATH}/")
    print("   - all_queries_results.json")
    print("   - comparison.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
