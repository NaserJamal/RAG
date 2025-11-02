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

LEVEL_NAME = "01-basic-vector-search"
COLLECTION_NAME = "level_01_basic_search"


def main():
    """Main execution function."""
    # Validate configuration
    Config.validate()

    # Setup paths
    documents_path = Config.get_documents_path(LEVEL_NAME)
    output_path = Config.get_output_path(LEVEL_NAME)

    # Initialize components
    embedder = Embedder()
    vector_store = QdrantVectorStore()
    output_manager = OutputManager(output_path)

    print("=" * 80)
    print("Level 01: Basic Vector Search with Qdrant")
    print("=" * 80)

    # Load documents
    print("\n📄 Loading documents...")
    documents = load_documents(documents_path)
    print(f"✅ Loaded {len(documents)} documents")

    # Generate embeddings
    print("\n🔢 Generating embeddings...")
    doc_contents = [doc["content"] for doc in documents]
    doc_ids = [doc["id"] for doc in documents]
    embeddings = embedder.embed(doc_contents)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Create vector store collection
    print("\n🗄️  Setting up Qdrant collection...")
    vector_store.create_collection(
        collection_name=COLLECTION_NAME,
        vector_dim=embedder.get_embedding_dimension()
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Add vectors to store
    print("\n📥 Adding vectors to Qdrant...")
    metadata = [{"content": doc["content"], "path": doc["path"]} for doc in documents]
    vector_store.add_vectors(
        collection_name=COLLECTION_NAME,
        vectors=embeddings,
        ids=doc_ids,
        metadata=metadata
    )
    print(f"✅ Added {len(embeddings)} vectors to Qdrant")

    # Example query
    query = "What is the vacation policy?"
    print(f"\n🔍 Searching for: '{query}'")

    # Generate query embedding and search
    query_embedding = embedder.embed_query(query)
    search_results = vector_store.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        top_k=Config.DEFAULT_TOP_K
    )
    print(f"✅ Found {len(search_results)} results")

    # Format results
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

    # Display results
    print("\n" + "=" * 80)
    print("Top Results:")
    print("=" * 80 + "\n")

    for result in results:
        print(f"#{result['rank']} - {result['document_id']} (Score: {result['score']:.4f})")
        print(f"   {result['preview'][:150]}...")
        print()

    # Save results
    print("💾 Saving results...")

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

    # Save human-readable format
    readable_text = output_manager.format_search_results(
        query=query,
        results=results,
        title="Level 01: Basic Vector Search Results"
    )
    output_manager.save_text("query_log.txt", readable_text)

    print(f"✅ Results saved to {output_path}/")
    print(f"   - results.json")
    print(f"   - query_log.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
