"""
Level 01: Basic Vector Search

A simple implementation of vector similarity search using OpenAI embeddings
and Qdrant vector database.
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    Config,
    Embedder,
    QdrantVectorStore,
    load_documents,
    OutputManager,
)
from utils import print_header, print_step, print_success, get_user_query, display_results

LEVEL_NAME = "01-basic-vector-search"
COLLECTION_NAME = "level_01_basic_search"


def setup_vector_store(embedder, documents_path):
    """Initialize vector store with documents."""
    vector_store = QdrantVectorStore()

    print_step("📄 Loading documents...")
    documents = load_documents(documents_path)
    print_success(f"Loaded {len(documents)} documents")

    print_step("🔢 Generating embeddings...")
    doc_contents = [doc["content"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    embeddings = embedder.embed(doc_contents)
    print_success(f"Generated {len(embeddings)} embeddings")

    print_step("🗄️  Setting up Qdrant collection...")
    vector_store.create_collection(
        collection_name=COLLECTION_NAME,
        vector_dim=embedder.get_embedding_dimension()
    )
    print_success(f"Collection '{COLLECTION_NAME}' created")

    print_step("📥 Adding vectors to Qdrant...")
    metadata = [{"content": doc["content"], "path": doc["path"]} for doc in documents]
    vector_store.add_vectors(
        collection_name=COLLECTION_NAME,
        vectors=embeddings,
        ids=doc_ids,
        metadata=metadata
    )
    print_success(f"Added {len(embeddings)} vectors to Qdrant")

    return vector_store


def search_query(query, embedder, vector_store):
    """Execute search for a query and return formatted results."""
    query_embedding = embedder.embed_query(query)
    search_results = vector_store.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        top_k=Config.DEFAULT_TOP_K
    )

    results = []
    for rank, (doc_id, score) in enumerate(search_results, 1):
        doc_metadata = vector_store.get_metadata(COLLECTION_NAME, doc_id)
        content = doc_metadata.get("content", "") if doc_metadata else ""
        preview = content[:500] + "..." if len(content) > 500 else content

        results.append({
            "rank": rank,
            "score": float(score),
            "document_id": doc_id,
            "path": doc_metadata.get("path", "") if doc_metadata else "",
            "preview": preview
        })

    return results


def save_results(query, results, output_manager, output_path):
    """Save search results to files."""
    print_step("💾 Saving results...")

    output_data = {
        "query": query,
        "results": results,
        "metadata": {
            "retrieval_method": "vector_search",
            "embedding_model": Config.EMBEDDING_MODEL,
            "vector_store": "qdrant",
            "collection": COLLECTION_NAME,
            "top_k": Config.DEFAULT_TOP_K
        }
    }

    output_manager.save_results("results", output_data)

    readable_text = output_manager.format_search_results(
        query=query,
        results=results,
        title="Level 01: Basic Vector Search Results"
    )
    output_manager.save_text("query_log.txt", readable_text)

    print_success(f"Results saved to {output_path}/")
    print("   - results.json")
    print("   - query_log.txt")


def main():
    """Main execution function."""
    Config.validate()

    documents_path = Config.get_documents_path(LEVEL_NAME)
    output_path = Config.get_output_path(LEVEL_NAME)

    embedder = Embedder()
    output_manager = OutputManager(output_path)

    print_header("Level 01: Basic Vector Search with Qdrant")

    # Setup vector store once
    vector_store = setup_vector_store(embedder, documents_path)

    # Interactive query loop
    while True:
        query = get_user_query()

        if not query:
            break

        results = search_query(query, embedder, vector_store)
        display_results(results, query)
        save_results(query, results, output_manager, output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
