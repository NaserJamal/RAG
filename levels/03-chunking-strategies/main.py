"""
Level 03: Chunking Strategies
Compares different text chunking approaches for optimal retrieval.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from utils.config import DOCUMENTS_PATH, OUTPUT_PATH, TOP_K
from utils.embedder import Embedder
from utils.fixed_chunker import FixedChunker
from utils.recursive_chunker import RecursiveChunker
from utils.semantic_chunker import SemanticChunker
from utils.chunk_evaluator import ChunkEvaluator


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


def save_chunks(chunks: List[Dict], filename: str, output_dir: Path):
    """Save chunks to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / filename, 'w') as f:
        json.dump(chunks, f, indent=2)


def save_evaluation(evaluation: Dict, filename: str, output_dir: Path):
    """Save evaluation results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / filename, 'w') as f:
        json.dump(evaluation, f, indent=2)


def save_comparison(query: str, evaluations: Dict[str, Dict], output_dir: Path):
    """Save comparison of chunking strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison metrics
    comparison = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "strategies": {}
    }

    for strategy, eval_data in evaluations.items():
        comparison["strategies"][strategy] = {
            "num_chunks": eval_data["num_chunks"],
            "avg_chunk_size": round(eval_data["avg_chunk_size"], 1),
            "top_score": round(eval_data["top_score"], 4)
        }

    with open(output_dir / "comparison_metrics.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    # Create text visualization
    with open(output_dir / "chunk_visualization.txt", 'w') as f:
        f.write(f"Chunking Strategies Comparison\n")
        f.write(f"Query: {query}\n")
        f.write(f"{'=' * 100}\n\n")

        # Metrics table
        f.write("Strategy Metrics:\n")
        f.write(f"{'-' * 100}\n")
        f.write(f"{'Strategy':<15} {'Chunks':>8} {'Avg Size':>10} {'Top Score':>12}\n")
        f.write(f"{'-' * 100}\n")

        for strategy, eval_data in evaluations.items():
            f.write(f"{strategy:<15} {eval_data['num_chunks']:>8} "
                   f"{eval_data['avg_chunk_size']:>10.1f} "
                   f"{eval_data['top_score']:>12.4f}\n")

        f.write(f"{'-' * 100}\n\n")

        # Top results for each strategy
        for strategy, eval_data in evaluations.items():
            f.write(f"\n{strategy.upper()} - Top Results:\n")
            f.write(f"{'-' * 100}\n")
            for result in eval_data["results"]:
                f.write(f"  #{result['rank']} (Score: {result['score']:.4f}, "
                       f"{result['num_tokens']} tokens)\n")
                f.write(f"  {result['text']}\n\n")


def print_comparison_table(evaluations: Dict[str, Dict]):
    """Print comparison table to console."""
    print(f"\n{'=' * 100}")
    print("CHUNKING STRATEGIES COMPARISON")
    print(f"{'=' * 100}\n")

    print(f"{'Strategy':<15} {'Chunks':>8} {'Avg Size':>10} {'Min':>8} {'Max':>8} {'Top Score':>12}")
    print(f"{'-' * 100}")

    for strategy, eval_data in evaluations.items():
        print(f"{strategy:<15} {eval_data['num_chunks']:>8} "
              f"{eval_data['avg_chunk_size']:>10.1f} "
              f"{eval_data['min_chunk_size']:>8.0f} "
              f"{eval_data['max_chunk_size']:>8.0f} "
              f"{eval_data['top_score']:>12.4f}")

    print(f"{'-' * 100}\n")


def main():
    """Main execution function."""
    print("📄 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} documents\n")

    # Select one document for detailed analysis
    # Choose the comprehensive ML document for interesting chunking behavior
    test_doc = next(
        (doc for doc in documents if "ml-systems-design" in doc["id"]),
        documents[0]  # Fallback to first document
    )

    print(f"📋 Analyzing document: {test_doc['id']}")
    print(f"   Length: {len(test_doc['content'])} characters\n")

    # Initialize embedder and evaluator
    embedder = Embedder()
    evaluator = ChunkEvaluator(embedder)

    # Initialize chunkers
    fixed_chunker = FixedChunker()
    recursive_chunker = RecursiveChunker()
    semantic_chunker = SemanticChunker(embedder)

    # Process with each chunking strategy
    print("🔪 Chunking document...")

    print("\n1. Fixed-size chunking...")
    fixed_chunks = fixed_chunker.chunk(test_doc["content"], test_doc["id"])
    print(f"   Created {len(fixed_chunks)} chunks")

    print("\n2. Recursive chunking...")
    recursive_chunks = recursive_chunker.chunk(test_doc["content"], test_doc["id"])
    print(f"   Created {len(recursive_chunks)} chunks")

    print("\n3. Semantic chunking...")
    semantic_chunks = semantic_chunker.chunk(test_doc["content"], test_doc["id"])
    print(f"   Created {len(semantic_chunks)} chunks")

    # Save chunks
    print("\n💾 Saving chunks...")
    save_chunks(fixed_chunks, "fixed_chunks.json", OUTPUT_PATH)
    save_chunks(recursive_chunks, "recursive_chunks.json", OUTPUT_PATH)
    save_chunks(semantic_chunks, "semantic_chunks.json", OUTPUT_PATH)

    # Evaluate with a query
    query = "How do you handle model training and deployment?"
    print(f"\n🔍 Evaluating with query: '{query}'")

    print("\n   Evaluating fixed chunks...")
    fixed_eval = evaluator.evaluate(fixed_chunks, query, k=TOP_K)

    print("   Evaluating recursive chunks...")
    recursive_eval = evaluator.evaluate(recursive_chunks, query, k=TOP_K)

    print("   Evaluating semantic chunks...")
    semantic_eval = evaluator.evaluate(semantic_chunks, query, k=TOP_K)

    # Save evaluations
    print("\n💾 Saving evaluations...")
    evaluations = {
        "fixed": fixed_eval,
        "recursive": recursive_eval,
        "semantic": semantic_eval
    }

    for strategy, eval_data in evaluations.items():
        save_evaluation(eval_data, f"{strategy}_evaluation.json", OUTPUT_PATH)

    save_comparison(query, evaluations, OUTPUT_PATH)

    # Print comparison
    print_comparison_table(evaluations)

    # Print visualizations
    print("\nChunk Size Distributions:")
    for strategy, chunks in [("Fixed", fixed_chunks), ("Recursive", recursive_chunks),
                              ("Semantic", semantic_chunks)]:
        print(f"\n{strategy}:")
        viz = evaluator.visualize_chunks(chunks)
        print(viz)

    # Print top results
    print(f"\n{'=' * 100}")
    print("TOP RETRIEVED CHUNKS")
    print(f"{'=' * 100}\n")

    for strategy, eval_data in evaluations.items():
        print(f"{strategy.upper()}:")
        for result in eval_data["results"][:2]:  # Show top 2
            print(f"  #{result['rank']} - Score: {result['score']:.4f}")
            print(f"  {result['text'][:150]}...\n")

    print("\n✅ Analysis complete!")
    print(f"\nOutput files saved to {OUTPUT_PATH}/:")
    print("   - fixed_chunks.json")
    print("   - recursive_chunks.json")
    print("   - semantic_chunks.json")
    print("   - fixed_evaluation.json")
    print("   - recursive_evaluation.json")
    print("   - semantic_evaluation.json")
    print("   - comparison_metrics.json")
    print("   - chunk_visualization.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
