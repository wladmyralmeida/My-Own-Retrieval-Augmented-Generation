"""Suporte simples a .md, .txt, .pdf (com metadados de página/arquivo)."""
import os
from typing import List, Dict
from pypdf import PdfReader

def load_text_files(folder: str) -> List[Dict]:
    """Load .txt and .md files as documents with metadata."""
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                docs.append({"page_content": text, "metadata": {"source": path}})
    return docs

def load_pdf_files(folder: str) -> List[Dict]:
    """Load PDFs splitting by page, attach page metadata."""
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append({
                            "page_content": text,
                            "metadata": {"source": path, "page": i + 1}
                        })
    return docs

def load_all_documents(folder: str) -> List[Dict]:
    """Return a list of {page_content, metadata} dicts."""
    return load_text_files(folder) + load_pdf_files(folder)
