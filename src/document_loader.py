"""
MedQueryAI - Document Loader
Handles ingestion of medical documents (PDF, TXT) with metadata extraction.
"""

import src.utils  # Windows console UTF-8 fix

import os
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document


def clean_medical_text(text: str) -> str:
    """
    Clean and normalize raw medical document text.
    Fixes common OCR artifacts and normalizes whitespace
    while preserving medical formatting.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix common OCR artifacts in medical texts
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # camelCase splits
    text = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", text)  # collapse 3+ newlines to 2
    text = re.sub(r"[ \t]+", " ", text)  # normalize spaces/tabs

    # Normalize medical abbreviations spacing
    text = re.sub(r"(\d)\s*(mg|mL|mcg|kg|mmHg|bpm|mmol|ng|pg|mIU)", r"\1 \2", text)

    return text.strip()


def load_text_file(file_path: str) -> list[Document]:
    """
    Load a plain text medical document.

    Args:
        file_path: Path to the .txt file

    Returns:
        List containing a single Document with content and metadata
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    cleaned_text = clean_medical_text(raw_text)

    doc = Document(
        page_content=cleaned_text,
        metadata={
            "source": path.name,
            "source_path": str(path.absolute()),
            "file_type": "txt",
            "file_size_kb": round(path.stat().st_size / 1024, 2),
        },
    )

    return [doc]


def load_pdf_file(file_path: str) -> list[Document]:
    """
    Load a PDF medical document using PyPDF.
    Each page becomes a separate Document with page metadata.

    Args:
        file_path: Path to the .pdf file

    Returns:
        List of Documents, one per page
    """
    from pypdf import PdfReader

    path = Path(file_path)
    reader = PdfReader(str(path))

    documents = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_medical_text(raw_text)

        if not cleaned_text.strip():
            continue  # skip empty pages

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": path.name,
                "source_path": str(path.absolute()),
                "file_type": "pdf",
                "page_number": page_num,
                "total_pages": len(reader.pages),
            },
        )
        documents.append(doc)

    return documents


def load_document(file_path: str) -> list[Document]:
    """
    Load a medical document from a file path.
    Supports PDF and TXT formats.

    Args:
        file_path: Path to the document

    Returns:
        List of Documents with content and metadata

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        return load_text_file(file_path)
    elif extension == ".pdf":
        return load_pdf_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. Supported: .pdf, .txt"
        )


def load_directory(
    dir_path: str, extensions: Optional[list[str]] = None
) -> list[Document]:
    """
    Load all supported documents from a directory.

    Args:
        dir_path: Path to directory containing documents
        extensions: Optional list of extensions to filter (e.g., ['.pdf', '.txt'])

    Returns:
        List of all Documents loaded from the directory
    """
    if extensions is None:
        extensions = [".pdf", ".txt"]

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {dir_path}")

    all_documents = []
    loaded_files = []
    skipped_files = []

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() in extensions:
            try:
                docs = load_document(str(file_path))
                all_documents.extend(docs)
                loaded_files.append(file_path.name)
            except Exception as e:
                skipped_files.append((file_path.name, str(e)))

    print(f"[OK] Loaded {len(loaded_files)} files -> {len(all_documents)} document(s)")
    for name in loaded_files:
        print(f"   + {name}")

    if skipped_files:
        print(f"[WARN] Skipped {len(skipped_files)} files:")
        for name, error in skipped_files:
            print(f"   - {name}: {error}")

    return all_documents
