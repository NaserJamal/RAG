"""
Output management utilities for RAG system results.

Provides consistent formatting and saving of retrieval results across
all training levels.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class OutputManager:
    """Manages saving and formatting of RAG system outputs."""

    def __init__(self, output_path: Path):
        """
        Initialize the output manager.

        Args:
            output_path: Directory path where outputs will be saved
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        filename: str,
        results: Dict[str, Any],
        include_timestamp: bool = True,
    ) -> Path:
        """
        Save results to a JSON file.

        Args:
            filename: Name of the output file (without extension)
            results: Dictionary of results to save
            include_timestamp: Whether to add a timestamp to the results

        Returns:
            Path to the saved file
        """
        if include_timestamp:
            results["timestamp"] = datetime.now().isoformat()

        output_file = self.output_path / f"{filename}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        return output_file

    def save_text(self, filename: str, content: str) -> Path:
        """
        Save text content to a file.

        Args:
            filename: Name of the output file (with extension)
            content: Text content to save

        Returns:
            Path to the saved file
        """
        output_file = self.output_path / filename

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        return output_file

    def format_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        title: str = "Search Results",
    ) -> str:
        """
        Format search results as human-readable text.

        Args:
            query: The search query
            results: List of result dictionaries with 'rank', 'score', etc.
            title: Title for the results section

        Returns:
            Formatted string representation
        """
        lines = [
            "=" * 80,
            title,
            "=" * 80,
            f"\nQuery: {query}\n",
            f"Found {len(results)} results\n",
            "-" * 80,
        ]

        for result in results:
            rank = result.get("rank", "?")
            score = result.get("score", 0.0)
            doc_id = result.get("document_id", "unknown")

            lines.append(f"\n[Rank {rank}] Score: {score:.4f}")
            lines.append(f"Document: {doc_id}")

            if "content" in result:
                content = result["content"]
                preview = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"Preview: {preview}")

            if "metadata" in result:
                lines.append(f"Metadata: {result['metadata']}")

            lines.append("-" * 80)

        return "\n".join(lines)

    def format_comparison(
        self,
        query: str,
        comparisons: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        """
        Format comparison of multiple retrieval methods.

        Args:
            query: The search query
            comparisons: Dictionary mapping method names to result lists

        Returns:
            Formatted comparison string
        """
        lines = [
            "=" * 80,
            "Retrieval Method Comparison",
            "=" * 80,
            f"\nQuery: {query}\n",
        ]

        for method_name, results in comparisons.items():
            lines.append(f"\n{method_name.upper()}")
            lines.append("-" * 80)

            for result in results:
                rank = result.get("rank", "?")
                score = result.get("score", 0.0)
                doc_id = result.get("document_id", "unknown")
                lines.append(f"[{rank}] {doc_id} (Score: {score:.4f})")

            lines.append("")

        return "\n".join(lines)

    def format_statistics(
        self,
        stats: Dict[str, Any],
        title: str = "Statistics",
    ) -> str:
        """
        Format statistics as human-readable text.

        Args:
            stats: Dictionary of statistics
            title: Title for the statistics section

        Returns:
            Formatted statistics string
        """
        lines = [
            "=" * 80,
            title,
            "=" * 80,
            "",
        ]

        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def save_multiple_outputs(
        self,
        outputs: Dict[str, Any],
        prefix: str = "",
    ) -> List[Path]:
        """
        Save multiple outputs with optional filename prefix.

        Args:
            outputs: Dictionary mapping filenames to content
            prefix: Optional prefix for all filenames

        Returns:
            List of paths to saved files
        """
        saved_paths = []

        for filename, content in outputs.items():
            full_filename = f"{prefix}_{filename}" if prefix else filename

            if isinstance(content, dict):
                path = self.save_results(full_filename, content)
            elif isinstance(content, str):
                path = self.save_text(
                    full_filename if "." in full_filename else f"{full_filename}.txt",
                    content,
                )
            else:
                # Convert to JSON for other types
                path = self.save_results(full_filename, {"data": content})

            saved_paths.append(path)

        return saved_paths
