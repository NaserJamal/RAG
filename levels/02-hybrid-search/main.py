"""
Level 02: Hybrid Search
Combines semantic (vector) search with keyword (BM25) search using Reciprocal Rank Fusion.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from utils.config import DOCUMENTS_PATH, OUTPUT_PATH, TOP_K, ALPHA
from utils.embedder import Embedder
from utils.vector_retriever import VectorRetriever
from utils.bm25_retriever import BM25Retriever
from utils.hybrid_retriever import HybridRetriever


def load_documents() -> List[Dict[str, str]]:
    """Load all text documents from the documents directory."""
    documents = []

    for doc_dir in DOCUMENTS_PATH.iterdir():
        if not doc_dir.is_dir():
            continue

        for doc_file in doc_dir.glob("*.txt"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "id": f"{doc_dir.name}/{doc_file.name}",
                    "path": str(doc_file),
                    "content": content
                })

    return documents


def format_results(results: List[tuple], documents: List[Dict]) -> List[Dict]:
    """Format retriever results into display-friendly structure."""
    doc_map = {doc["id"]: doc for doc in documents}

    formatted = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = doc_map.get(doc_id, {})
        formatted.append({
            "rank": rank,
            "score": float(score),
            "document": doc_id,
            "path": doc.get("path", ""),
            "preview": doc.get("content", "")[:500] + "..." if len(doc.get("content", "")) > 500 else doc.get("content", "")
        })

    return formatted


def save_results(query: str, vector_results: List[Dict], bm25_results: List[Dict],
                hybrid_results: List[Dict], output_dir: Path):
    """Save all search results to output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Save individual results
    with open(output_dir / "vector_results.json", 'w') as f:
        json.dump({
            "query": query,
            "timestamp": timestamp,
            "method": "vector_search",
            "results": vector_results
        }, f, indent=2)

    with open(output_dir / "bm25_results.json", 'w') as f:
        json.dump({
            "query": query,
            "timestamp": timestamp,
            "method": "bm25_search",
            "results": bm25_results
        }, f, indent=2)

    with open(output_dir / "hybrid_results.json", 'w') as f:
        json.dump({
            "query": query,
            "timestamp": timestamp,
            "method": "hybrid_search",
            "results": hybrid_results
        }, f, indent=2)

    # Create comparison text file
    with open(output_dir / "comparison.txt", 'w') as f:
        f.write(f"Query: {query}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'=' * 100}\n\n")

        f.write("VECTOR SEARCH RESULTS (Semantic)\n")
        f.write(f"{'-' * 100}\n")
        for result in vector_results:
            f.write(f"#{result['rank']} - {result['document']} (Score: {result['score']:.4f})\n")
        f.write("\n")

        f.write("BM25 RESULTS (Keyword)\n")
        f.write(f"{'-' * 100}\n")
        for result in bm25_results:
            f.write(f"#{result['rank']} - {result['document']} (Score: {result['score']:.4f})\n")
        f.write("\n")

        f.write("HYBRID RESULTS (Combined with RRF)\n")
        f.write(f"{'-' * 100}\n")
        for result in hybrid_results:
            f.write(f"#{result['rank']} - {result['document']} (Score: {result['score']:.4f})\n")
        f.write("\n")


def print_comparison(vector_results: List[Dict], bm25_results: List[Dict], hybrid_results: List[Dict]):
    """Print side-by-side comparison of results."""
    print(f"\n{'=' * 100}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 100}\n")

    print("Vector Search (Semantic):")
    print("-" * 100)
    for result in vector_results:
        print(f"  #{result['rank']} - {result['document']:60s} Score: {result['score']:.4f}")

    print("\nBM25 Search (Keyword):")
    print("-" * 100)
    for result in bm25_results:
        print(f"  #{result['rank']} - {result['document']:60s} Score: {result['score']:.4f}")

    print("\nHybrid Search (RRF Fusion):")
    print("-" * 100)
    for result in hybrid_results:
        print(f"  #{result['rank']} - {result['document']:60s} Score: {result['score']:.4f}")

    print()


def main():
    """Main execution function."""
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} documents\n")

    # Initialize retrievers
    print("🔧 Initializing retrievers...")
    embedder = Embedder()
    vector_retriever = VectorRetriever(documents, embedder)
    bm25_retriever = BM25Retriever(documents)

    # Index documents
    vector_retriever.index()
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
    save_results(query, vector_results, bm25_results, hybrid_results, OUTPUT_PATH)
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
