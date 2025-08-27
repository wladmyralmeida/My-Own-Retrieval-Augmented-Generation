# -*- coding: utf-8 -*-
"""
RAG helpers: carregamento do FAISS, construção de cadeia LCEL com citações e fallback
para embeddings locais (HuggingFace) quando configurado via env.

Env úteis:
- USE_LOCAL_EMBEDDINGS=1
- LOCAL_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
- TOP_K=5
- FETCH_K=12
- MMR_LAMBDA=0.5
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Embeddings locais (fallback opcional)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

from src.config import INDEX_DIR, OPENAI_MODEL, OPENAI_EMBEDDING

log = logging.getLogger("rag_chain")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


SYSTEM_PROMPT = (
    "Você é um assistente focado que DEVE responder APENAS usando o contexto fornecido.\n"
    "- Se a resposta não estiver no contexto, diga: \"Não encontrei essa informação nas suas anotações.\".\n"
    "- Seja conciso, estruturado e responda em português.\n"
    "- Sempre inclua uma pequena seção \"Fontes\" citando os índices entre colchetes usados no texto (ex.: [1], [2]).\n"
    "- Evite inventar (alucinar)."
)

USER_PROMPT = (
    "Pergunta: {question}\n\n"
    "Contexto (cada bloco possui um índice entre colchetes para citação):\n"
    "{context}\n\n"
    "Requisitos:\n"
    "- Responda em português de forma objetiva.\n"
    "- Cite no corpo da resposta com [n] quando usar trechos.\n"
    "- Finalize com uma seção 'Fontes' listando esses [n]."
)


# ------------------------
# Embeddings
# ------------------------

def _build_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING)

def _build_local_embeddings() -> HuggingFaceEmbeddings:
    if not _HAVE_HF:
        raise RuntimeError(
            "langchain-huggingface não está instalado. "
            "Instale com: pip install langchain-huggingface"
        )
    model_name = os.getenv("LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    log.info(f"🧠 Usando embeddings locais: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

def _choose_embeddings():
    """Seleciona embeddings OpenAI ou locais com base nas envs."""
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
    if use_local:
        return _build_local_embeddings()
    return _build_openai_embeddings()


# ------------------------
# Vector store
# ------------------------

def load_vectorstore() -> FAISS:
    """Carrega o índice FAISS do disco com embeddings adequadas."""
    embeddings = _choose_embeddings()
    index_path = Path(INDEX_DIR).expanduser().resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Índice não encontrado em: {index_path}")
    vs = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vs


# ------------------------
# Utilidades de formatação
# ------------------------

def _format_docs_for_prompt(docs: Iterable) -> str:
    """
    Concatena docs no formato:
    [1] /path/arquivo.pdf (p.3)
    Conteúdo...
    ---
    [2] /path/arquivo.md
    Conteúdo...
    """
    parts: List[str] = []
    for idx, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "desconhecido")
        page = meta.get("page")
        head = f"[{idx}] {src}" + (f" (p.{page})" if page else "")
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _sources_from_docs(docs: List) -> List[Dict[str, Any]]:
    """Extrai fontes normalizadas para retornar na API."""
    out: List[Dict[str, Any]] = []
    for idx, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        out.append({
            "id": idx,
            "source": meta.get("source", "desconhecido"),
            "page": meta.get("page"),
            # alguns retrievers devolvem .score em metadata; se houver, retornamos
            "score": meta.get("score"),
        })
    return out


# ------------------------
# Cadeia LCEL
# ------------------------

def build_chain(
    vs: FAISS,
    *,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
):
    """
    Monta uma cadeia LCEL que retorna:
      {"answer": <str>, "sources": <list>}
    com retriever MMR configurável por env/parâmetro.
    """
    # Defaults via env
    k = int(os.getenv("TOP_K", str(top_k or 5)))
    fk = int(os.getenv("FETCH_K", str(fetch_k or 12)))
    lm = float(os.getenv("MMR_LAMBDA", str(mmr_lambda or 0.5)))

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fk, "lambda_mult": lm},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    # 1) Em paralelo, coletamos os documentos e repassamos a pergunta.
    # 2) Formatamos os docs para o prompt.
    # 3) Passamos prompt -> llm -> parser de string.
    # 4) Empacotamos resposta + fontes normais.
    gather = RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough()
    )

    chain = (
        gather
        | {
            "context": lambda x: _format_docs_for_prompt(x["docs"]),
            "question": lambda x: x["question"],
            "docs": lambda x: x["docs"],  # carregamos junto para extrair fontes mais tarde
        }
        | {
            "answer": (prompt | llm | StrOutputParser()),
            "sources": lambda x: _sources_from_docs(list(x["docs"])),
        }
    )

    return chain
