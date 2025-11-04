"""
Unified Search Tool - Supports both semantic and keyword-based search.

Provides a single search interface that can perform either:
- Semantic (vector-based) search using embeddings (default)
- BM25 keyword search for exact term matching
"""

from typing import Dict, Any, Optional
from core.tool_system import registry
from tools.rag.utils import execute_semantic_search, execute_bm25_search


@registry.register(
    name="search",
    description="Search documents using semantic similarity (default) or keyword matching. Use semantic search for concepts and topics, or keyword search (bm25=true) for exact terms and IDs.",
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
            "file_path": {
                "type": "string",
                "description": "Optional: Filter to a specific file or folder. Examples: 'company-kb/vacation-policy.txt' (exact file), 'company-kb' (all files in folder), 'company-kb/hr/policies' (nested folder). If not provided, searches all documents."
            },
            "bm25": {
                "type": "boolean",
                "description": "Use BM25 keyword search instead of semantic search (default: false). Enable for exact keyword/term matching.",
                "default": False
            }
        }
    }
)
def search(
    query: str,
    top_k: int = 3,
    file_path: Optional[str] = None,
    bm25: bool = False
) -> Dict[str, Any]:
    """
    Search for relevant documents using semantic or keyword-based search.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3, max: 10)
        file_path: Optional file or folder path to filter
        bm25: Use BM25 keyword search instead of semantic search (default: False)

    Returns:
        Dictionary containing:
        - results: List of relevant documents with content and scores
        - query: The original query
        - search_type: Type of search performed ('semantic' or 'bm25')
        - result_count: Number of results returned
        - file_filter: File/folder path filter applied (if any)
    """
    if bm25:
        return execute_bm25_search(query, top_k, file_path)
    else:
        return execute_semantic_search(query, top_k, file_path)
