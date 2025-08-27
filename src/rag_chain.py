"""Define o retriever, prompt e a cadeia RAG."""
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

from config import INDEX_DIR, OPENAI_MODEL, OPENAI_EMBEDDING

SYSTEM_PROMPT = """You are a focused assistant that MUST answer ONLY using the provided context.
- If the answer is not in the context, say: "Não encontrei essa informação nas suas anotações."
- Be concise, structured, and in Portuguese.
- Always include a short "Fontes" section with file path and page (if available).
"""

USER_PROMPT = """Pergunta: {question}

Contexto:
{context}

Responda em português e cite as fontes no final.
"""

def load_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING)
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def format_docs(docs: List[Dict[str, Any]]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "desconhecido")
        page = meta.get("page")
        tag = f"{src}" + (f" (p.{page})" if page else "")
        parts.append(f"[{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def build_chain(vs: FAISS):
    retriever = vs.as_retriever(
        search_type="mmr",  # diverse results
        search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5},
    )

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    chain = RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }) | prompt | llm

    return chain
