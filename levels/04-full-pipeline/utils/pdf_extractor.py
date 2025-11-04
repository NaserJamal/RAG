"""
PDF text extraction module using PyMuPDF.

Handles extraction of text from PDF documents and organizes the output
into normalized directory structures.
"""

import pymupdf
from pathlib import Path
from typing import List, Tuple
import re


class PDFExtractor:
    """Extract text content from PDF files."""

    @staticmethod
    def normalize_name(filename: str) -> str:
        """
        Normalize a filename for use as a directory name.

        Args:
            filename: Original filename (with or without extension)

        Returns:
            Normalized name suitable for directory usage
        """
        # Remove .pdf extension if present
        name = filename.lower()
        if name.endswith('.pdf'):
            name = name[:-4]

        # Replace spaces and special characters with underscores
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)

        return name.strip('_')

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

    def process_pdf(self, pdf_path: Path, output_base_dir: Path) -> Tuple[Path, str]:
        """
        Process a single PDF: extract text and prepare for chunking.

        Args:
            pdf_path: Path to the PDF file
            output_base_dir: Base directory for output

        Returns:
            Tuple of (document_dir, extracted_text)
        """
        # Create normalized directory name
        doc_name = self.normalize_name(pdf_path.name)
        doc_dir = output_base_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Extract text
        text = self.extract_text(pdf_path)

        # Save raw extracted text
        raw_text_path = doc_dir / "raw_text.txt"
        raw_text_path.write_text(text, encoding='utf-8')

        return doc_dir, text

    def process_directory(self, pdf_dir: Path, output_base_dir: Path) -> List[Tuple[str, Path, str]]:
        """
        Process all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files
            output_base_dir: Base directory for output

        Returns:
            List of tuples: (doc_name, doc_dir, extracted_text)
        """
        results = []
        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")

        for pdf_path in sorted(pdf_files):
            doc_dir, text = self.process_pdf(pdf_path, output_base_dir)
            doc_name = self.normalize_name(pdf_path.name)
            results.append((doc_name, doc_dir, text))

        return results
