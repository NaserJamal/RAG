"""
Document loading utilities for RAG system.

Provides functionality to load text documents from the documents directory,
maintaining consistent structure across all training levels.
"""

from pathlib import Path
from typing import List, Dict


def load_documents(documents_path: Path) -> List[Dict[str, str]]:
    """
    Load all text documents from the specified documents directory.

    Args:
        documents_path: Path to the documents directory

    Returns:
        List of document dictionaries with 'id', 'path', and 'content' keys

    Raises:
        FileNotFoundError: If documents_path does not exist
    """
    if not documents_path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {documents_path}")

    if not documents_path.is_dir():
        raise ValueError(f"Documents path is not a directory: {documents_path}")

    documents = []

    for doc_dir in documents_path.iterdir():
        if not doc_dir.is_dir():
            continue

        for doc_file in doc_dir.glob("*.txt"):
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "id": f"{doc_dir.name}/{doc_file.name}",
                        "path": str(doc_file),
                        "content": content,
                    })
            except Exception as e:
                print(f"Warning: Failed to load {doc_file}: {e}")
                continue

    return documents


def count_documents(documents_path: Path) -> int:
    """
    Count the number of text documents in the documents directory.

    Args:
        documents_path: Path to the documents directory

    Returns:
        Number of .txt files found
    """
    if not documents_path.exists() or not documents_path.is_dir():
        return 0

    count = 0
    for doc_dir in documents_path.iterdir():
        if doc_dir.is_dir():
            count += len(list(doc_dir.glob("*.txt")))

    return count
