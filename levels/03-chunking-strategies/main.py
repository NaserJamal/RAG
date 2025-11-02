"""
Level 03: Chunking Strategies

Compares different text chunking approaches for optimal retrieval.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import Config, Embedder, load_documents, OutputManager

from utils.config import DOCUMENTS_PATH, OUTPUT_PATH, TOP_K
from utils.fixed_chunker import FixedChunker
from utils.recursive_chunker import RecursiveChunker
from utils.semantic_chunker import SemanticChunker
from utils.chunk_evaluator import ChunkEvaluator


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
    # Validate configuration
    Config.validate()

    # Initialize components
    embedder = Embedder()
    output_manager = OutputManager(OUTPUT_PATH)

    print("=" * 100)
    print("Level 03: Chunking Strategies Comparison")
    print("=" * 100)

    # Load documents
    print("\n📄 Loading documents...")
    documents = load_documents(DOCUMENTS_PATH)
    print(f"✅ Loaded {len(documents)} documents\n")

    # Select one document for detailed analysis
    test_doc = next(
        (doc for doc in documents if "ml-systems-design" in doc["id"]),
        documents[0]  # Fallback to first document
    )

    print(f"📋 Analyzing document: {test_doc['id']}")
    print(f"   Length: {len(test_doc['content'])} characters\n")

    # Initialize evaluator
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
    output_manager.save_results("fixed_chunks", {"chunks": fixed_chunks})
    output_manager.save_results("recursive_chunks", {"chunks": recursive_chunks})
    output_manager.save_results("semantic_chunks", {"chunks": semantic_chunks})

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
        output_manager.save_results(f"{strategy}_evaluation", eval_data)

    # Save comparison metrics
    comparison = {
        "query": query,
        "strategies": {}
    }

    for strategy, eval_data in evaluations.items():
        comparison["strategies"][strategy] = {
            "num_chunks": eval_data["num_chunks"],
            "avg_chunk_size": round(eval_data["avg_chunk_size"], 1),
            "top_score": round(eval_data["top_score"], 4)
        }

    output_manager.save_results("comparison_metrics", comparison)

    # Create visualization text
    viz_lines = [
        "Chunking Strategies Comparison",
        f"Query: {query}",
        "=" * 100,
        "",
        "Strategy Metrics:",
        "-" * 100,
        f"{'Strategy':<15} {'Chunks':>8} {'Avg Size':>10} {'Top Score':>12}",
        "-" * 100,
    ]

    for strategy, eval_data in evaluations.items():
        viz_lines.append(
            f"{strategy:<15} {eval_data['num_chunks']:>8} "
            f"{eval_data['avg_chunk_size']:>10.1f} "
            f"{eval_data['top_score']:>12.4f}"
        )

    viz_lines.append("-" * 100)
    viz_lines.append("")

    # Top results for each strategy
    for strategy, eval_data in evaluations.items():
        viz_lines.append(f"\n{strategy.upper()} - Top Results:")
        viz_lines.append("-" * 100)
        for result in eval_data["results"]:
            viz_lines.append(
                f"  #{result['rank']} (Score: {result['score']:.4f}, "
                f"{result['num_tokens']} tokens)"
            )
            viz_lines.append(f"  {result['text']}\n")

    output_manager.save_text("chunk_visualization.txt", "\n".join(viz_lines))

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
