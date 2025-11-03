"""
Qdrant Search Tool - Semantic search over document collection.

Provides semantic search functionality for the AI agent to retrieve
relevant documents from the Qdrant vector database.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder, QdrantVectorStore, load_documents, Config
from core.tool_system import registry


# Initialize shared components (lazy loading on first call)
_embedder = None
_vector_store = None
_collection_name = "agentic_rag_docs"
_initialized = False


def _initialize():
    """Initialize embedder and vector store (called on first use)."""
    global _embedder, _vector_store, _initialized

    if _initialized:
        return

    try:
        # Validate configuration
        Config.validate()

        # Initialize embedder and vector store
        _embedder = Embedder()
        _vector_store = QdrantVectorStore()

        # Load and index documents if collection doesn't exist
        if not _vector_store.collection_exists(_collection_name):
            print(f"📚 Initializing document collection '{_collection_name}'...")

            # Load documents from shared directory
            documents_path = Config.get_documents_path()
            documents = load_documents(documents_path)

            if not documents:
                raise ValueError(f"No documents found in {documents_path}")

            # Generate embeddings
            doc_contents = [doc["content"] for doc in documents]
            doc_ids = [doc["id"] for doc in documents]
            embeddings = _embedder.embed(doc_contents)

            # Create collection and add vectors
            _vector_store.create_collection(
                collection_name=_collection_name,
                vector_dim=_embedder.get_embedding_dimension()
            )

            metadata = [
                {"content": doc["content"], "path": doc.get("path", "")}
                for doc in documents
            ]
            _vector_store.add_vectors(
                collection_name=_collection_name,
                vectors=embeddings,
                ids=doc_ids,
                metadata=metadata
            )

            print(f"✅ Indexed {len(documents)} documents")

        _initialized = True

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Qdrant search: {str(e)}")


@registry.register(
    name="search_documents",
    description="Search for relevant documents using semantic similarity. Use this tool when you need to find information from the document collection to answer user questions.",
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents. Should be a clear, specific question or topic."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of most relevant documents to return (default: 3, max: 10)",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            }
        }
    }
)
def search_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search for relevant documents using semantic similarity.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3, max: 10)

    Returns:
        Dictionary containing:
        - results: List of relevant documents with content and scores
        - query: The original query
        - result_count: Number of results returned
    """
    try:
        # Initialize on first use
        _initialize()

        # Validate and clamp top_k
        top_k = max(1, min(top_k, 10))

        # Generate query embedding
        query_embedding = _embedder.embed_query(query)

        # Search in Qdrant
        search_results = _vector_store.search(
            collection_name=_collection_name,
            query_vector=query_embedding,
            top_k=top_k
        )

        # Format results with full content and metadata
        results = []
        for doc_id, score in search_results:
            metadata = _vector_store.get_metadata(_collection_name, doc_id)
            if metadata:
                results.append({
                    "document_id": doc_id,
                    "relevance_score": round(float(score), 4),
                    "content": metadata.get("content", ""),
                    "path": metadata.get("path", "")
                })

        return {
            "query": query,
            "result_count": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": []
        }
