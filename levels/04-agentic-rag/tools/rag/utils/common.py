"""
Common shared utilities for RAG search tools.

Provides centralized initialization and configuration for all search tools.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from shared import Embedder, QdrantVectorStore, load_documents, Config
from shared.retrieval.bm25 import BM25Search

# Re-export for convenience
__all__ = [
    'Embedder', 'QdrantVectorStore', 'BM25Search',
    'initialize', 'get_embedder', 'get_vector_store', 'get_bm25_search',
    'get_collection_name', 'get_documents', 'filter_documents_by_path',
    'filter_results_by_path', 'format_results'
]


# Shared state (lazy initialization)
_embedder = None
_vector_store = None
_bm25_search = None
_documents = None
_collection_name = "agentic_rag_docs"
_initialized = False


def initialize():
    """
    Initialize embedder, vector store, and BM25 search (called on first use).

    Raises:
        RuntimeError: If initialization fails
    """
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
                {"content": doc["content"], "path": doc.get("path", ""), "doc_id": doc["id"]}
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


def get_embedder() -> Embedder:
    """Get the initialized embedder instance."""
    initialize()
    return _embedder


def get_vector_store() -> QdrantVectorStore:
    """Get the initialized vector store instance."""
    initialize()
    return _vector_store


def get_bm25_search() -> BM25Search:
    """Get the initialized BM25 search instance."""
    initialize()
    return _bm25_search


def get_collection_name() -> str:
    """Get the Qdrant collection name."""
    return _collection_name


def get_documents() -> List[Dict[str, Any]]:
    """Get all loaded documents."""
    initialize()
    return _documents


def filter_documents_by_path(file_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Filter documents by file path or folder path.

    Supports both exact file matching and folder prefix matching:
    - 'company-kb/vacation-policy.txt' -> exact file match
    - 'company-kb' -> all files under company-kb folder
    - 'company-kb/hr/policies' -> all files under nested folder

    Args:
        file_path: Optional file or folder path to filter

    Returns:
        Filtered list of documents, or all documents if file_path is None
    """
    initialize()

    if not file_path:
        return _documents

    # Check if it's an exact file match (ends with .txt)
    if file_path.endswith('.txt'):
        return [doc for doc in _documents if doc["id"] == file_path]

    # Otherwise, treat as folder - match all files under this path
    prefix = file_path if file_path.endswith('/') else file_path + '/'
    return [doc for doc in _documents if doc["id"].startswith(prefix)]


def filter_results_by_path(
    results: List[tuple],
    file_path: Optional[str]
) -> List[tuple]:
    """
    Filter search results by file path or folder path.

    Applies post-filtering to search results to support folder-level filtering.
    Used for semantic search where we filter after vector retrieval.

    Args:
        results: List of (doc_id, score) tuples from search
        file_path: Optional file or folder path to filter

    Returns:
        Filtered list of (doc_id, score) tuples
    """
    if not file_path:
        return results

    # Check if it's an exact file match (ends with .txt)
    if file_path.endswith('.txt'):
        return [(doc_id, score) for doc_id, score in results if doc_id == file_path]

    # Otherwise, treat as folder - match all files under this path
    prefix = file_path if file_path.endswith('/') else file_path + '/'
    return [(doc_id, score) for doc_id, score in results if doc_id.startswith(prefix)]


def format_results(
    search_results: List[tuple],
    vector_store: QdrantVectorStore,
    collection_name: str
) -> List[Dict[str, Any]]:
    """
    Format search results with full content and metadata.

    Args:
        search_results: List of (doc_id, score) tuples
        vector_store: Vector store instance
        collection_name: Name of the collection

    Returns:
        List of formatted result dictionaries
    """
    results = []
    for doc_id, score in search_results:
        # Skip results with exactly zero relevance (no match at all)
        # Note: Small positive scores are still relevant, especially in single-file searches
        if score == 0:
            continue

        metadata = vector_store.get_metadata(collection_name, doc_id)
        if metadata:
            results.append({
                "document_id": doc_id,
                "relevance_score": round(float(score), 4),
                "content": metadata.get("content", ""),
                "path": metadata.get("path", "")
            })

    return results
