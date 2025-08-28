# src/server.py
# -*- coding: utf-8 -*-
"""FastAPI: /ask with structured sources (snippet/score) and /reindex to rebuild index."""
from __future__ import annotations

import os
import logging
from subprocess import run
from typing import Literal

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from src.rag_chain import (
    load_vectorstore,
    build_retriever,
    build_chain,
    extract_snippets,
)

log = logging.getLogger("server")
app = FastAPI(title="Agente RAG das Minhas Anotações")

# ---- Auth (optional) ----
API_KEY = os.getenv("RAG_API_KEY", "")

def _ensure_auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---- Bootstrap ----
_vs = load_vectorstore()
_retriever = build_retriever(_vs)
_chain = build_chain(_retriever)


class AskBody(BaseModel):
    question: str
    top_k: int | None = None
    mode: Literal["brief", "detailed", "json"] = "brief"
    language: Literal["pt-BR", "en-US"] = "pt-BR"
    # stream: bool = False  # placeholder if you later add SSE


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reindex")
async def reindex(x_api_key: str | None = Header(default=None, convert_underscores=False)):
    """Run ingest, reload vectorstore/retriever/chain without restarting the server."""
    _ensure_auth(x_api_key)
    r = run(["python", "src/ingest.py"])
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail="Reindex failed")
    global _vs, _retriever, _chain
    _vs = load_vectorstore()
    _retriever = build_retriever(_vs)
    _chain = build_chain(_retriever)
    return {"ok": True}


@app.post("/ask")
async def ask(body: AskBody, x_api_key: str | None = Header(default=None, convert_underscores=False)):
    """
    Returns:
    {
      "answer": "<text>",
      "sources": [
        {"source": "<path>", "page": 3, "score": 0.42, "snippet": "<...>"},
        ...
      ]
    }
    """
    _ensure_auth(x_api_key)
    try:
        # 1) Sources with score/snippet from plain similarity (stable & fast)
        k = body.top_k or int(os.getenv("TOP_K", "6"))
        docs_scores = _vs.similarity_search_with_score(body.question, k=k)
        sources = extract_snippets(docs_scores)

        # 2) If caller overrides k, rebuild retriever with that k
        chain = _chain
        if body.top_k:
            retr = build_retriever(_vs, k=body.top_k)
            chain = build_chain(retr)

        # 3) Ask chain (uses smart retriever + compressed context)
        answer = await chain.ainvoke({"question": body.question, "language": body.language, "mode": body.mode})

        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error in /ask")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {e.__class__.__name__}: {e}")
