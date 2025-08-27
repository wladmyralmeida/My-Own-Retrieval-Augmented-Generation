# src/ingest.py
"""Faz chunking + indexação FAISS e persiste em storage/index com logs robustos."""
from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# fallback local
from langchain_huggingface import HuggingFaceEmbeddings

# OpenAI SDK para capturar RateLimitError
import openai

from config import DATA_DIR, INDEX_DIR, OPENAI_EMBEDDING
from loaders import load_all_documents

# ---- logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ingest")

def chunk_docs(raw_docs: List[dict], chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    base_docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw_docs]
    return splitter.split_documents(base_docs)

def build_openai_embedder() -> OpenAIEmbeddings:
    # você já define o modelo via OPENAI_EMBEDDING (config.py/.env)
    # dá para reduzir lotes se quiser: OpenAIEmbeddings(batch_size=64, ...)
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING)

def build_local_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    log.info(f"🧠 Usando embeddings locais: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

def maybe_fallback_embedder(e: Exception):
    """Decide se devemos cair para embeddings locais."""
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
    if use_local:
        return build_local_embedder()
    return None

def embed_with_backoff(texts: List[str], embedder, max_attempts: int = 5) -> list[list[float]]:
    """Tenta gerar embeddings com backoff exponencial em 429/limite."""
    attempt = 0
    while True:
        try:
            # Para LangChain Embeddings, embed_documents retorna List[List[float]]
            return embedder.embed_documents(texts)
        except openai.RateLimitError as err:
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = 2 ** attempt
            log.warning(f"⏳ Rate limit/quota. Tentando novamente em {delay}s (tentativa {attempt}/{max_attempts})...")
            time.sleep(delay)
        except Exception:
            # Repropaga outros erros
            raise

def main() -> None:
    data_dir = Path(DATA_DIR)
    index_dir = Path(INDEX_DIR)

    log.info(f"📚 Loading documents from: {data_dir}")
    raw_docs = load_all_documents(data_dir)

    if not raw_docs:
        log.warning("⚠️ No documents found. Put your notes in storage/data (ou ajuste DATA_DIR).")
        return

    log.info(f"✂️  Chunking {len(raw_docs)} docs ...")
    docs = chunk_docs(raw_docs)
    log.info(f"🧩 Gerados {len(docs)} chunks.")

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # 1) tenta com OpenAI
    embedder = None
    try:
        if not os.getenv("OPENAI_API_KEY"):
            log.warning("🔑 OPENAI_API_KEY ausente; pulando OpenAI embeddings.")
            raise openai.RateLimitError("no-key")  # força fallback se habilitado
        embedder = build_openai_embedder()
        log.info(f"🔢 Building embeddings (OpenAI: {os.getenv('OPENAI_EMBEDDING') or 'default'}) ...")
        _ = embed_with_backoff(texts, embedder)
    except openai.RateLimitError as e:
        log.error("🚫 Quota/Rate limit do OpenAI atingido (ou sem chave).")
        fallback = maybe_fallback_embedder(e)
        if fallback:
            embedder = fallback
        else:
            log.error("❌ Sem fallback local habilitado. Defina USE_LOCAL_EMBEDDINGS=1 ou ajuste seu billing.")
            raise
    except Exception as e:
        log.exception("❌ Falha gerando embeddings com OpenAI.")
        fallback = maybe_fallback_embedder(e)
        if fallback:
            embedder = fallback
        else:
            raise

    # 2) (re)gera embeddings efetivamente com o embedder final e constrói o FAISS
    if embedder is None:
        log.error("❌ Nenhum embedder disponível.")
        sys.exit(1)

    # Gera embeddings e cria FAISS
    try:
        # Observação: FAISS.from_texts chama embedder.embed_documents internamente,
        # mas como já testamos acima, seguimos direto. Se quiser usar seu cache, pode
        # gerar embeddings manualmente e usar FAISS.from_embeddings.
        log.info("🏗️ Construindo índice FAISS ...")
        vs = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
    except Exception:
        log.exception("❌ Falha ao construir o índice FAISS.")
        raise

    # Salva índice
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
        log.info(f"✅ Índice salvo em: {index_dir}")
    except Exception:
        log.exception("❌ Falha ao salvar o índice FAISS.")
        raise

if __name__ == "__main__":
    # dica: mensagens 'Ignoring wrong pointing object' vêm do pypdf.
    # São avisos de PDFs imperfeitos; se incomodar, troque para PyMuPDF no loader.
    main()
