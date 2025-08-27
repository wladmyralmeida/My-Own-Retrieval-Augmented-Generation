# src/ingest.py
"""Faz chunking + indexação FAISS e persiste em storage/index."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import DATA_DIR, INDEX_DIR, OPENAI_EMBEDDING
from loaders import load_all_documents


def ensure_path(p: str | Path) -> Path:
    """Normalize to Path."""
    return p if isinstance(p, Path) else Path(p)


def chunk_docs(raw_docs: List[dict], chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    """Chunk documents mantendo metadados."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    base_docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]
    return splitter.split_documents(base_docs)


def main() -> None:
    data_dir = ensure_path(DATA_DIR)
    index_dir = ensure_path(INDEX_DIR)

    # garante pastas (caso config.py não tenha feito)
    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"📚 Loading documents from: {data_dir}")
    raw_docs = load_all_documents(data_dir)

    if not raw_docs:
        print("⚠️ No documents found. Put your notes in the /storage/data folder (or DATA_DIR).")
        return

    print(f"✂️  Chunking {len(raw_docs)} docs ...")
    docs = chunk_docs(raw_docs)
    print(f"🧩 Gerados {len(docs)} chunks.")

    # checagem rápida da key (evita falhas silenciosas)
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY não encontrada no ambiente. Ex.: export OPENAI_API_KEY='sk-...'")
        return

    print("🔢 Building embeddings & FAISS index ...")
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING)
    vs = FAISS.from_texts(
        texts=[d.page_content for d in docs],
        embedding=embeddings,
        metadatas=[d.metadata for d in docs],
    )

    print(f"💾 Saving index to: {index_dir}")
    vs.save_local(str(index_dir))  # FAISS espera str para o path
    print("✅ Done.")


if __name__ == "__main__":
    main()
