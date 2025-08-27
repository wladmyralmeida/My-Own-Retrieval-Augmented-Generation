"""Simple loaders for .md, .txt, .pdf with resilient parsing and metadata."""
from __future__ import annotations

import os
import io
import time
import logging
from typing import List, Dict, Optional

# Try to use PyMuPDF (fitz) first; fallback to pypdf if unavailable
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

try:
    from pypdf import PdfReader
    _HAVE_PYPDF = True
except Exception:
    _HAVE_PYPDF = False

log = logging.getLogger("loaders")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _file_meta(path: str) -> Dict:
    """Collect basic file metadata."""
    try:
        st = os.stat(path)
        return {
            "source": path,
            "ext": os.path.splitext(path)[1].lower(),
            "bytes": st.st_size,
            "mtime": int(st.st_mtime),
        }
    except Exception:
        return {"source": path, "ext": os.path.splitext(path)[1].lower()}


def _read_text(path: str, max_chars: Optional[int] = None) -> str:
    """Read text file safely, optionally truncating to max_chars."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        data = fh.read()
    if max_chars and len(data) > max_chars:
        return data[:max_chars]
    return data


def load_text_files(folder: str, max_chars_per_file: Optional[int] = None) -> List[Dict]:
    """Load .txt and .md files as documents with metadata."""
    docs: List[Dict] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".txt", ".md")):
                path = os.path.join(root, f)
                try:
                    text = _read_text(path, max_chars_per_file)
                    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                    if not text:
                        continue
                    meta = _file_meta(path)
                    docs.append({"page_content": text, "metadata": meta})
                except Exception as e:
                    log.warning(f"Skipping text file due to error: {path} ({e})")
    log.info(f"Loaded {len(docs)} text/md documents from {folder}")
    return docs


def _load_pdf_with_pymupdf(path: str, max_pages: Optional[int] = None) -> List[Dict]:
    """Load PDF pages via PyMuPDF."""
    docs: List[Dict] = []
    doc = fitz.open(path)  # type: ignore[name-defined]
    try:
        n_pages = len(doc)
        limit = min(n_pages, max_pages) if max_pages else n_pages
        base_meta = _file_meta(path)
        for i in range(limit):
            page = doc.load_page(i)
            # Use text extractor; "text" mode preserves layout reasonably
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue
            meta = dict(base_meta)
            meta["page"] = i + 1
            docs.append({"page_content": text, "metadata": meta})
        return docs
    finally:
        doc.close()


def _load_pdf_with_pypdf(path: str, max_pages: Optional[int] = None) -> List[Dict]:
    """Load PDF pages via pypdf as a fallback."""
    docs: List[Dict] = []
    reader = PdfReader(path)  # type: ignore[name-defined]
    n_pages = len(reader.pages)
    limit = min(n_pages, max_pages) if max_pages else n_pages
    base_meta = _file_meta(path)
    for i in range(limit):
        try:
            page = reader.pages[i]
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            meta = dict(base_meta)
            meta["page"] = i + 1
            docs.append({"page_content": text, "metadata": meta})
        except Exception as e:
            log.warning(f"Skipping PDF page {i+1} in {path} due to error: {e}")
    return docs


def load_pdf_files(folder: str, max_pages_per_pdf: Optional[int] = None) -> List[Dict]:
    """Load PDFs splitting by page, attach page/file metadata. Uses PyMuPDF if available."""
    docs: List[Dict] = []
    use_pymupdf = _HAVE_PYMUPDF
    if not _HAVE_PYMUPDF and not _HAVE_PYPDF:
        log.warning("Neither PyMuPDF nor pypdf is available; PDF files will be skipped.")
        return docs

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, f)
            try:
                if use_pymupdf:
                    pages = _load_pdf_with_pymupdf(path, max_pages_per_pdf)
                else:
                    pages = _load_pdf_with_pypdf(path, max_pages_per_pdf)
                docs.extend(pages)
                log.info(f"Loaded {len(pages)} pages from {path}")
            except Exception as e:
                # If PyMuPDF failed, try pypdf as a fallback for this file
                if use_pymupdf and _HAVE_PYPDF:
                    log.warning(f"PyMuPDF failed for {path} ({e}); retrying with pypdf...")
                    try:
                        pages = _load_pdf_with_pypdf(path, max_pages_per_pdf)
                        docs.extend(pages)
                        log.info(f"Loaded {len(pages)} pages (fallback) from {path}")
                    except Exception as e2:
                        log.error(f"Failed to load PDF {path} with both backends: {e2}")
                else:
                    log.error(f"Failed to load PDF {path}: {e}")
    return docs


def load_all_documents(folder: str, *, max_pages_per_pdf: Optional[int] = None,
                       max_chars_per_text: Optional[int] = None) -> List[Dict]:
    """Return a list of {page_content, metadata} dicts for txt/md/pdf."""
    start = time.time()
    texts = load_text_files(folder, max_chars_per_text)
    pdfs = load_pdf_files(folder, max_pages_per_pdf)
    docs = texts + pdfs
    log.info(f"Total loaded docs: {len(docs)} in {time.time() - start:.2f}s")
    return docs
