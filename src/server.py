# -*- coding: utf-8 -*-
"""API HTTP com /ask."""
from __future__ import annotations

import logging
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_chain import load_vectorstore, build_chain

log = logging.getLogger("server")
app = FastAPI(title="Agente RAG das Minhas Anotações")

# Carrega uma vez na inicialização
_vs = load_vectorstore()
_chain = build_chain(_vs)  # usa TOP_K/FETCH_K/MMR_LAMBDA se definidos no .env

class AskBody(BaseModel):
    question: str
    top_k: int | None = None  # opcional para sobrescrever em runtime

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask(body: AskBody):
    """
    Envia a pergunta para a cadeia RAG e retorna:
    { "answer": "<texto>", "sources": [{id, source, page, score?}, ...] }
    """
    try:
        chain = _chain
        # se quiser permitir override de k por requisição:
        if body.top_k:
            chain = build_chain(_vs, top_k=body.top_k)

        result = await chain.ainvoke(body.question)
        # result já é {"answer": str, "sources": [...]}
        return result
    except Exception as e:
        log.exception("Erro no /ask")
        return {"error": f"Falha ao processar a pergunta: {e.__class__.__name__}: {e}"}
