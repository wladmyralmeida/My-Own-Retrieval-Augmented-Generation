# src/config.py
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in your .env file.")

# === Paths ===
# Use defaults sensatos e crie as pastas se não existirem
DATA_DIR = Path(os.getenv("DATA_DIR", "storage/data")).expanduser().resolve()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "storage/index")).expanduser().resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# (Opcional) verificação de escrita — falha cedo se não der para gravar
if not os.access(DATA_DIR, os.W_OK):
    raise RuntimeError(f"Sem permissão de escrita em {DATA_DIR}")
if not os.access(INDEX_DIR, os.W_OK):
    raise RuntimeError(f"Sem permissão de escrita em {INDEX_DIR}")
