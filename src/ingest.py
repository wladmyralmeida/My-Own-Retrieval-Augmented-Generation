# src/ingest.py
"""Chunk + FAISS index build with robust logs, optional local embeddings fallback,
dedup by hash, and corpus/metas dump for future hybrid search."""
from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Optional local embeddings (fallback)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

# OpenAI SDK exceptions (for rate limit/backoff)
import openai

from config import DATA_DIR, INDEX_DIR, OPENAI_EMBEDDING
from loaders import load_all_documents

# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

# ---- env knobs ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
CORPUS_PATH = os.getenv("CORPUS_PATH", str(Path(INDEX_DIR).parent / "corpus.json"))
METAS_PATH = os.getenv("METAS_PATH", str(Path(INDEX_DIR).parent / "metas.json"))


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_docs(raw_docs: List[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    base_docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]
    return splitter.split_documents(base_docs)


def build_openai_embedder() -> OpenAIEmbeddings:
    # You already define the model via OPENAI_EMBEDDING (config.py/.env)
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING)


def build_local_embedder():
    if not _HAVE_HF:
        raise RuntimeError(
            "langchain-huggingface is not installed. Install with: pip install langchain-huggingface"
        )
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    log.info(f"🧠 Using local embeddings: {model_name}")
    from langchain_huggingface import HuggingFaceEmbeddings  # local import for mypy
    return HuggingFaceEmbeddings(model_name=model_name)


def maybe_fallback_embedder(e: Exception):
    """Decide whether to fallback to local embeddings."""
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
    if use_local:
        return build_local_embedder()
    return None


def embed_with_backoff(texts: List[str], embedder, max_attempts: int = 5) -> List[List[float]]:
    """Try to generate embeddings with exponential backoff when hitting rate limit."""
    attempt = 0
    while True:
        try:
            return embedder.embed_documents(texts)
        except openai.RateLimitError:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = 2 ** attempt
            log.warning(f"⏳ Rate limit/quota. Retrying in {delay}s (attempt {attempt}/{max_attempts})...")
            time.sleep(delay)
        except Exception:
            raise


def _dedup(texts: List[str], metas: List[dict]) -> Tuple[List[str], List[dict]]:
    """Deduplicate chunks by SHA-256 of content while keeping first metadata occurrence."""
    seen = set()
    out_texts: List[str] = []
    out_metas: List[dict] = []
    for t, m in zip(texts, metas):
        h = _doc_hash(t)
        if h in seen:
            continue
        seen.add(h)
        m = dict(m)
        m["hash"] = h
        out_texts.append(t)
        out_metas.append(m)
    return out_texts, out_metas


def _dump_corpus_and_metas(texts: List[str], metas: List[dict]) -> None:
    """Persist corpus/metas for future hybrid search (BM25+RRF) or debugging."""
    try:
        Path(CORPUS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        with open(METAS_PATH, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False)
        log.info(f"📝 Dumped corpus -> {CORPUS_PATH} and metas -> {METAS_PATH}")
    except Exception:
        log.exception("Failed to dump corpus/metas")


def main() -> None:
    data_dir = Path(DATA_DIR)
    index_dir = Path(INDEX_DIR)

    log.info(f"📚 Loading documents from: {data_dir}")
    raw_docs = load_all_documents(str(data_dir))

    if not raw_docs:
        log.warning("⚠️ No documents found. Put your notes under data/ (or adjust DATA_DIR).")
        return

    log.info(f"✂️ Chunking {len(raw_docs)} docs ...")
    docs = chunk_docs(raw_docs)
    log.info(f"🧩 Generated {len(docs)} chunks.")

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # Dedup chunks
    texts, metadatas = _dedup(texts, metadatas)
    log.info(f"🧼 After dedup: {len(texts)} unique chunks.")

    # 1) Try OpenAI embeddings first
    embedder = None
    try:
        if not os.getenv("OPENAI_API_KEY"):
            log.warning("🔑 OPENAI_API_KEY missing; skipping OpenAI embeddings.")
            raise openai.RateLimitError("no-key")  # force fallback if enabled
        embedder = build_openai_embedder()
        log.info(f"🔢 Building embeddings (OpenAI: {os.getenv('OPENAI_EMBEDDING') or 'default'}) ...")
        _ = embed_with_backoff(texts, embedder)
    except openai.RateLimitError:
        log.error("🚫 OpenAI quota/rate limit reached (or missing key).")
        fallback = maybe_fallback_embedder(openai.RateLimitError("rate"))
        if fallback:
            embedder = fallback
        else:
            log.error("❌ No local fallback enabled. Set USE_LOCAL_EMBEDDINGS=1 or adjust billing.")
            raise
    except Exception:
        log.exception("❌ Failed to generate embeddings with OpenAI.")
        fallback = maybe_fallback_embedder(Exception("err"))
        if fallback:
            embedder = fallback
        else:
            raise

    if embedder is None:
        log.error("❌ No embedder available.")
        sys.exit(1)

    # Build FAISS
    try:
        log.info("🏗️ Building FAISS index ...")
        vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
    except Exception:
        log.exception("❌ Failed to build FAISS index.")
        raise

    # Save index + corpus/metas
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
        _dump_corpus_and_metas(texts, metadatas)
        log.info(f"✅ Index saved to: {index_dir}")
    except Exception:
        log.exception("❌ Failed to save FAISS index.")
        raise


if __name__ == "__main__":
    # NOTE: pypdf may log warnings like 'Ignoring wrong pointing object' due to imperfect PDFs.
    # If noisy, consider switching to PyMuPDF in the loader (already supported).
    main()
