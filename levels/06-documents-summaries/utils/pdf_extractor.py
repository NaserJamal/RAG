"""
PDF text extraction module using PyMuPDF.

Simplified version for document summarization - extracts full text
without chunking or directory organization.
"""

import pymupdf
from pathlib import Path
from typing import List, Tuple


class PDFExtractor:
    """Extract text content from PDF files."""

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be opened or read
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = pymupdf.open(pdf_path)
            text_content = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(text)

            doc.close()

            full_text = "\n\n".join(text_content)

            if not full_text.strip():
                raise ValueError(f"No text content extracted from {pdf_path.name}")

            return full_text

        except Exception as e:
            raise ValueError(f"Error extracting text from {pdf_path.name}: {str(e)}")

    def extract_from_directory(self, pdf_dir: Path) -> List[Tuple[str, str]]:
        """
        Extract text from all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            List of tuples: (filename, extracted_text)
        """
        results = []
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")

        for pdf_path in sorted(pdf_files):
            text = self.extract_text(pdf_path)
            results.append((pdf_path.name, text))

        return results
