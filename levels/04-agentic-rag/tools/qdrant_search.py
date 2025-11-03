"""
Qdrant Search Tool - Semantic and keyword search over document collection.

Provides semantic (vector), keyword (BM25), and hybrid search functionality
for the AI agent to retrieve relevant documents from the Qdrant vector database.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Embedder, QdrantVectorStore, load_documents, Config
from shared.retrieval.bm25 import BM25Search, reciprocal_rank_fusion
from core.tool_system import registry


# Initialize shared components (lazy loading on first call)
_embedder = None
_vector_store = None
_bm25_search = None
_documents = None
_collection_name = "agentic_rag_docs"
_initialized = False


def _initialize():
    """Initialize embedder, vector store, and BM25 search (called on first use)."""
    global _embedder, _vector_store, _bm25_search, _documents, _initialized

    if _initialized:
        return

    try:
        # Validate configuration
        Config.validate()

        # Initialize embedder and vector store
        _embedder = Embedder()
        _vector_store = QdrantVectorStore()

        # Load documents from shared directory
        documents_path = Config.get_documents_path()
        _documents = load_documents(documents_path)

        if not _documents:
            raise ValueError(f"No documents found in {documents_path}")

        # Initialize BM25 search
        _bm25_search = BM25Search(_documents)
        _bm25_search.index()

        # Load and index documents if collection doesn't exist
        if not _vector_store.collection_exists(_collection_name):
            print(f"📚 Initializing document collection '{_collection_name}'...")

            # Generate embeddings
            doc_contents = [doc["content"] for doc in _documents]
            doc_ids = [doc["id"] for doc in _documents]
            embeddings = _embedder.embed(doc_contents)

            # Create collection and add vectors
            _vector_store.create_collection(
                collection_name=_collection_name,
                vector_dim=_embedder.get_embedding_dimension()
            )

            metadata = [
                {"content": doc["content"], "path": doc.get("path", "")}
                for doc in _documents
            ]
            _vector_store.add_vectors(
                collection_name=_collection_name,
                vectors=embeddings,
                ids=doc_ids,
                metadata=metadata
            )

            print(f"✅ Indexed {len(_documents)} documents")

        _initialized = True

    except Exception as e:
        raise RuntimeError(f"Failed to initialize search system: {str(e)}")


@registry.register(
    name="search_documents",
    description="Search for relevant documents using semantic similarity (vector search). Best for conceptual queries and finding semantically related content. Use this when understanding meaning is more important than exact keyword matches.",
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


@registry.register(
    name="bm25_search",
    description="Search for documents using BM25 keyword matching. Best for exact term matches, acronyms, technical terms, or when you need precise keyword-based retrieval. Use this when specific terminology matters.",
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query with keywords to match in documents."
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
def bm25_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search for relevant documents using BM25 keyword matching.

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

        # Search using BM25
        search_results = _bm25_search.search(query, top_k=top_k)

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
            "search_type": "bm25",
            "result_count": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"BM25 search failed: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": []
        }


@registry.register(
    name="hybrid_search",
    description="Search using both semantic similarity and keyword matching (hybrid approach). Combines the benefits of both vector and BM25 search using Reciprocal Rank Fusion. Best for comprehensive retrieval that needs both conceptual understanding and exact term matching.",
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant documents."
            },
            "top_k": {
                "type": "integer",
                "description": "Number of most relevant documents to return (default: 3, max: 10)",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            },
            "alpha": {
                "type": "number",
                "description": "Weight balance between vector and BM25 search (0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced). Default: 0.5",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0
            }
        }
    }
)
def hybrid_search(query: str, top_k: int = 3, alpha: float = 0.5) -> Dict[str, Any]:
    """
    Search for relevant documents using hybrid approach (vector + BM25).

    Combines semantic similarity and keyword matching using Reciprocal Rank Fusion
    to provide comprehensive retrieval that leverages both approaches.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3, max: 10)
        alpha: Weight for vector vs BM25 (0.0 = pure BM25, 1.0 = pure vector)

    Returns:
        Dictionary containing:
        - results: List of relevant documents with content and fused scores
        - query: The original query
        - result_count: Number of results returned
    """
    try:
        # Initialize on first use
        _initialize()

        # Validate and clamp parameters
        top_k = max(1, min(top_k, 10))
        alpha = max(0.0, min(alpha, 1.0))

        # Get results from both search methods (retrieve more for better fusion)
        query_embedding = _embedder.embed_query(query)
        vector_results = _vector_store.search(
            collection_name=_collection_name,
            query_vector=query_embedding,
            top_k=top_k * 2
        )
        bm25_results = _bm25_search.search(query, top_k=top_k * 2)

        # Fuse results using RRF
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, alpha=alpha
        )

        # Take top-k and format with full content and metadata
        results = []
        for doc_id, score in fused_results[:top_k]:
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
            "search_type": "hybrid",
            "alpha": alpha,
            "result_count": len(results),
            "results": results
        }

    except Exception as e:
        return {
            "error": f"Hybrid search failed: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": []
        }
