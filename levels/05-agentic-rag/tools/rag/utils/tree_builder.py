"""
Document Tree Builder - Generate tree structure visualization of knowledge base.

Provides utility to generate a tree-like representation of the document structure
for inclusion in system prompts.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def build_document_tree(documents_path: Path) -> str:
    """
    Build a tree structure visualization of the documents directory.

    Args:
        documents_path: Path to the documents directory

    Returns:
        String representation of the directory tree structure

    Example output:
        documents/
        ├── company-kb
        │   ├── expense-reimbursement.txt
        │   ├── product-cloudstore-overview.txt
        │   └── vacation-policy.txt
        └── technical-docs
            ├── api-authentication-guide.txt
            └── python-sdk-quickstart.txt
    """
    if not documents_path.exists() or not documents_path.is_dir():
        return f"{documents_path.name}/ (not found)"

    # Collect all directories and files
    structure: Dict[str, List[str]] = {}

    for doc_dir in sorted(documents_path.iterdir()):
        if not doc_dir.is_dir():
            continue

        # Get all .txt files in this directory
        files = sorted([f.name for f in doc_dir.glob("*.txt")])
        structure[doc_dir.name] = files

    if not structure:
        return f"{documents_path.name}/ (empty)"

    # Build the tree representation
    lines = [f"{documents_path.name}/"]

    dir_names = list(structure.keys())
    for dir_idx, dir_name in enumerate(dir_names):
        is_last_dir = dir_idx == len(dir_names) - 1
        dir_prefix = "└── " if is_last_dir else "├── "
        lines.append(f"{dir_prefix}{dir_name}")

        files = structure[dir_name]
        for file_idx, file_name in enumerate(files):
            is_last_file = file_idx == len(files) - 1

            # Determine indentation based on whether this is the last directory
            if is_last_dir:
                indent = "    "
            else:
                indent = "│   "

            file_prefix = "└── " if is_last_file else "├── "
            lines.append(f"{indent}{file_prefix}{file_name}")

    return "\n".join(lines)


def get_available_files(documents_path: Path) -> List[str]:
    """
    Get a list of all available file paths in the knowledge base.

    Args:
        documents_path: Path to the documents directory

    Returns:
        List of file paths in the format 'directory/filename.txt'
    """
    if not documents_path.exists() or not documents_path.is_dir():
        return []

    file_paths = []
    for doc_dir in sorted(documents_path.iterdir()):
        if not doc_dir.is_dir():
            continue

        for doc_file in sorted(doc_dir.glob("*.txt")):
            file_paths.append(f"{doc_dir.name}/{doc_file.name}")

    return file_paths
