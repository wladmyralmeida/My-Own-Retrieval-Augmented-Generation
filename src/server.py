"""API HTTP com /ask."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from rag_chain import load_vectorstore, build_chain

app = FastAPI(title="Agente RAG das Minhas Anotações")
_vs = load_vectorstore()
_chain = build_chain(_vs)

class AskBody(BaseModel):
    question: str

@app.post("/ask")
async def ask(body: AskBody):
    # Run chain and return text output
    result = await _chain.ainvoke(body.question)
    # result is an AIMessage; its .content contains the text
    return {"answer": result.content}
