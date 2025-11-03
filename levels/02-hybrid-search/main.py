"""
Level 02: Hybrid Search

Combines semantic (vector) search with keyword (BM25) search using Reciprocal Rank Fusion.
Uses Qdrant for vector storage and BM25 for keyword matching.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config, Embedder, QdrantVectorStore, load_documents, OutputManager, EmbeddingCache

from utils.config import DOCUMENTS_PATH, OUTPUT_PATH, TOP_K, ALPHA
from utils.vector_retriever import VectorRetriever
from utils.bm25_retriever import BM25Retriever
from utils.hybrid_retriever import HybridRetriever

COLLECTION_NAME = "level_02_hybrid_search"


def format_results(results: List[tuple], documents: List[Dict]) -> List[Dict]:
    """Format retriever results into display-friendly structure."""
    doc_map = {doc["id"]: doc for doc in documents}

    formatted = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_map.get(doc_id, {})
        content = doc.get("content", "")
        formatted.append({
            "rank": rank,
            "score": float(score),
            "document_id": doc_id,
            "path": doc.get("path", ""),
            "preview": content[:500] + "..." if len(content) > 500 else content
        })

    return formatted


def print_comparison(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    hybrid_results: List[Dict]
):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100 + "\n")

    print("Vector Search (Semantic):")
    print("-" * 100)
    for result in vector_results:
        doc_id = result["document_id"][:60]
        print(f"  #{result['rank']} - {doc_id:60s} Score: {result['score']:.4f}")

    print("\nBM25 Search (Keyword):")
    print("-" * 100)
    for result in bm25_results:
        doc_id = result["document_id"][:60]
        print(f"  #{result['rank']} - {doc_id:60s} Score: {result['score']:.4f}")

    print("\nHybrid Search (RRF Fusion):")
    print("-" * 100)
    for result in hybrid_results:
        doc_id = result["document_id"][:60]
        print(f"  #{result['rank']} - {doc_id:60s} Score: {result['score']:.4f}")

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
    print("Level 02: Hybrid Search with Qdrant and BM25")
    print("=" * 100)

    # Load documents
    print("\n📄 Loading documents...")
    documents = load_documents(DOCUMENTS_PATH)
    print(f"✅ Loaded {len(documents)} documents\n")

    # Initialize retrievers
    print("🔧 Initializing retrievers...")
    vector_retriever = VectorRetriever(COLLECTION_NAME, embedder, vector_store, cache)
    bm25_retriever = BM25Retriever(documents)

    # Index documents
    vector_retriever.index(documents)
    bm25_retriever.index()
    print()

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    # Example query
    query = "How do I submit travel expenses?"
    print(f"🔍 Searching for: '{query}'\n")

    # Get results from all methods
    vector_results = format_results(vector_retriever.search(query, k=TOP_K), documents)
    bm25_results = format_results(bm25_retriever.search(query, k=TOP_K), documents)
    hybrid_results = format_results(hybrid_retriever.search(query, k=TOP_K, alpha=ALPHA), documents)

    # Print comparison
    print_comparison(vector_results, bm25_results, hybrid_results)

    # Save results
    print("💾 Saving results...")

    # Save individual method results
    output_manager.save_results("vector_results", {
        "query": query,
        "method": "vector_search",
        "results": vector_results,
        "metadata": {
            "vector_store": "qdrant",
            "collection": COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL
        }
    })

    output_manager.save_results("bm25_results", {
        "query": query,
        "method": "bm25_search",
        "results": bm25_results
    })

    output_manager.save_results("hybrid_results", {
        "query": query,
        "method": "hybrid_search",
        "results": hybrid_results,
        "metadata": {
            "alpha": ALPHA,
            "fusion_method": "reciprocal_rank_fusion"
        }
    })

    # Create comparison text file
    comparison_text = output_manager.format_comparison(
        query=query,
        comparisons={
            "vector_search": vector_results,
            "bm25_search": bm25_results,
            "hybrid_search": hybrid_results
        }
    )
    output_manager.save_text("comparison.txt", comparison_text)

    print(f"✅ Results saved to {OUTPUT_PATH}/")
    print("   - vector_results.json")
    print("   - bm25_results.json")
    print("   - hybrid_results.json")
    print("   - comparison.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
