"""
BM25 Search Tool - Keyword-based search.

Provides BM25 keyword matching functionality for the AI agent to retrieve
documents using exact term matching and traditional information retrieval.
"""

from typing import Dict, Any, Optional
from core.tool_system import registry
from tools.rag.common import (
    get_bm25_search,
    get_vector_store,
    get_collection_name,
    filter_documents_by_file,
    format_results,
    BM25Search
)


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
            },
            "file_path": {
                "type": "string",
                "description": "Optional: Filter search to a specific file (e.g., 'company-kb/expense-reimbursement.txt'). If not provided, searches all documents."
            }
        }
    }
)
def bm25_search(query: str, top_k: int = 3, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for relevant documents using BM25 keyword matching.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3, max: 10)
        file_path: Optional file path to filter search (e.g., 'company-kb/expense-reimbursement.txt')

    Returns:
        Dictionary containing:
        - results: List of relevant documents with content and scores
        - query: The original query
        - search_type: Type of search performed
        - result_count: Number of results returned
        - file_filter: File path filter applied (if any)
    """
    try:
        # Validate and clamp top_k
        top_k = max(1, min(top_k, 10))

        # Get components
        vector_store = get_vector_store()
        collection_name = get_collection_name()

        # Handle file-specific search
        if file_path:
            # Filter documents to only the specified file
            filtered_docs = filter_documents_by_file(file_path)

            if not filtered_docs:
                return {
                    "error": f"File not found: {file_path}",
                    "query": query,
                    "search_type": "bm25",
                    "result_count": 0,
                    "results": []
                }

            # Create a new BM25 index for just this file
            file_bm25 = BM25Search(filtered_docs)
            file_bm25.index()
            search_results = file_bm25.search(query, top_k=top_k)
        else:
            # Use global BM25 index
            bm25 = get_bm25_search()
            search_results = bm25.search(query, top_k=top_k)

        # Format results with full content and metadata
        results = format_results(search_results, vector_store, collection_name)

        response = {
            "query": query,
            "search_type": "bm25",
            "result_count": len(results),
            "results": results
        }

        if file_path:
            response["file_filter"] = file_path

        return response

    except Exception as e:
        return {
            "error": f"BM25 search failed: {str(e)}",
            "query": query,
            "search_type": "bm25",
            "result_count": 0,
            "results": []
        }
