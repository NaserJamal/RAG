"""
Semantic Search Tool - Vector-based similarity search.

Provides semantic (vector) search functionality for the AI agent to retrieve
relevant documents from the Qdrant vector database using embedding similarity.
"""

from typing import Dict, Any, Optional
from core.tool_system import registry
from tools.rag.utils import (
    get_embedder,
    get_vector_store,
    get_collection_name,
    filter_results_by_path,
    format_results
)


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
            },
            "file_path": {
                "type": "string",
                "description": "Optional: Filter to a specific file or folder. Examples: 'company-kb/vacation-policy.txt' (exact file), 'company-kb' (all files in folder), 'company-kb/hr/policies' (nested folder). If not provided, searches all documents."
            }
        }
    }
)
def search_documents(query: str, top_k: int = 3, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for relevant documents using semantic similarity.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3, max: 10)
        file_path: Optional file or folder path to filter (e.g., 'company-kb' or 'company-kb/vacation-policy.txt')

    Returns:
        Dictionary containing:
        - results: List of relevant documents with content and scores
        - query: The original query
        - result_count: Number of results returned
        - file_filter: File/folder path filter applied (if any)
    """
    try:
        # Validate and clamp top_k
        top_k = max(1, min(top_k, 10))

        # Get initialized components
        embedder = get_embedder()
        vector_store = get_vector_store()
        collection_name = get_collection_name()

        # Generate query embedding
        query_embedding = embedder.embed_query(query)

        # Search in Qdrant (retrieve more results if filtering by folder)
        # We'll filter after retrieval to support folder-level filtering
        search_limit = top_k * 3 if file_path and not file_path.endswith('.txt') else top_k

        search_results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=search_limit
        )

        # Apply folder/file filtering if specified
        if file_path:
            search_results = filter_results_by_path(search_results, file_path)
            # Take only top_k after filtering
            search_results = search_results[:top_k]

        # Format results with full content and metadata
        results = format_results(search_results, vector_store, collection_name)

        response = {
            "query": query,
            "result_count": len(results),
            "results": results
        }

        if file_path:
            response["file_filter"] = file_path

        return response

    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": []
        }
