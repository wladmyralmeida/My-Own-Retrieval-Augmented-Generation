# src/rag_chain.py
# -*- coding: utf-8 -*-
"""
RAG helpers: FAISS loading, smart retriever (multi-query + compression),
formatted prompt with mode/language, and rich citations (snippet/score).
Env:
- USE_LOCAL_EMBEDDINGS=1
- LOCAL_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
- TOP_K=6
- FETCH_K=12
- MMR_LAMBDA=0.5
- SMART_RETRIEVER=1  (multi-query + contextual compression)
"""
from __future__ import annotations

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document



# Toggle via env: SCRUB_OUTPUT=0 desliga
SCRUB_OUTPUT = os.getenv("SCRUB_OUTPUT", "1") == "1"

# Amplie conforme sua realidade (tokens, bearer, emails, senhas, chaves, etc.)
SECRET_PATTERNS = [
    re.compile(r'(?i)\b(senha|password|pass|pwd)\s*[:=]\s*\S+'),
    re.compile(r'(?i)\b(token|api[_-]?key|secret)\s*[:=]\s*[A-Za-z0-9_\-\.]{6,}'),
    re.compile(r'(?i)\bbearer\s+[A-Za-z0-9\-_\.]+'),
    re.compile(r'(?i)\b(x-userid|x-tenantid)\s*[:=]\s*\S+'),
    # Remova se quiser manter emails visíveis nos trechos
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
]

def scrub(text: str) -> str:
    """Mask sensitive tokens in output/context."""
    if not SCRUB_OUTPUT or not text:
        return text
    for pat in SECRET_PATTERNS:
        text = pat.sub('[redacted]', text)
    return text

# Optional local embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

# Smart retriever bits
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

from src.config import INDEX_DIR, OPENAI_MODEL, OPENAI_EMBEDDING

log = logging.getLogger("rag_chain")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SYSTEM_PROMPT = """You MUST answer ONLY using the provided context.
If the answer is not in the context, reply in Portuguese: "Não encontrei essa informação nas suas anotações."
- Keep hallucinations low.
- Always finish with a short "Fontes" section listing file path and page (if available).
- Output language: {language}
- Output style mode: {mode} (brief|detailed|json)
"""

USER_PROMPT = """Pergunta: {question}

Regras de saída por modo:
- brief: 3–6 bullets claros e objetivos.
- detailed: resposta estruturada com seções curtas.
- json: responda em JSON válido, com chave "answer" e listas quando fizer sentido.

Use linguagem natural em {language}.
"""


# ------------------------
# Embeddings selection
# ------------------------
def _build_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING)

def _build_local_embeddings():
    if not _HAVE_HF:
        raise RuntimeError("langchain-huggingface is not installed. pip install langchain-huggingface")
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    log.info(f"🧠 Using local embeddings: {model_name}")
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    return HuggingFaceEmbeddings(model_name=model_name)

def _choose_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
    return _build_local_embeddings() if use_local else _build_openai_embeddings()


# ------------------------
# Vector store
# ------------------------
def load_vectorstore() -> FAISS:
    """Load FAISS index from disk with matching embeddings."""
    embeddings = _choose_embeddings()
    index_path = Path(INDEX_DIR).expanduser().resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at: {index_path}")
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    return vs


# ------------------------
# Formatting utilities
# ------------------------
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Formats docs like:
    [/path/file.pdf (p.3)]
    Content...
    ---
    [/path/file.md]
    Content...
    """
    parts: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "desconhecido")
        page = meta.get("page")
        head = f"[{src}" + (f" (p.{page})]" if page else "]")
        # 🔒 scrub no conteúdo enviado ao LLM
        content = scrub((d.page_content or ""))
        parts.append(f"{head}\n{content}")
    return "\n\n---\n\n".join(parts)


def extract_snippets(docs_scores: List[Tuple[Document, float]]):
    """Produce compact citations with snippet & score."""
    out: List[Dict[str, Any]] = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        src = meta.get("source", "desconhecido")
        page = meta.get("page")
        text = (doc.page_content or "").strip().replace("\n", " ")
        # 🔒 scrub no snippet devolvido para o cliente
        snippet = scrub(text)
        if len(snippet) > 260:
            snippet = snippet[:260] + "…"
        out.append({
            "source": src,
            "page": page,
            "score": float(score),
            "snippet": snippet
        })
    return out


# ------------------------
# Retriever & Chain
# ------------------------
def build_retriever(
    vs: FAISS,
    *,
    k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
) -> ContextualCompressionRetriever | Any:
    """Build a retriever. If SMART_RETRIEVER=1, use MultiQuery + ContextualCompression."""
    # Defaults via env
    _k = int(os.getenv("TOP_K", str(k or 6)))
    _fk = int(os.getenv("FETCH_K", str(fetch_k or 12)))
    _lm = float(os.getenv("MMR_LAMBDA", str(mmr_lambda or 0.5)))
    base = vs.as_retriever(search_type="mmr", search_kwargs={"k": _k, "fetch_k": _fk, "lambda_mult": _lm})

    if os.getenv("SMART_RETRIEVER", "1") != "1":
        return base

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, timeout=15, max_retries=1)
    mqr = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    compressor = LLMChainExtractor.from_llm(llm)
    smart = ContextualCompressionRetriever(base_retriever=mqr, base_compressor=compressor)
    return smart


def build_chain(retriever) -> Any:
    """Return a LCEL chain that takes {question, language, mode} and produces a string answer."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, timeout=15, max_retries=1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    # LCEL: retrieve docs separately, format context, then prompt -> llm -> string
    chain = RunnableMap({
        "docs": lambda x: retriever.invoke(x["question"]),
        "question": RunnablePassthrough(),
        "language": RunnablePassthrough(),
        "mode": RunnablePassthrough(),
    }) | RunnableMap({
        "context": lambda x: format_docs_for_prompt(x["docs"]),
        "question": lambda x: x["question"],
        "language": lambda x: x["language"],
        "mode": lambda x: x["mode"],
    }) | (prompt | llm | StrOutputParser())

    return chain
