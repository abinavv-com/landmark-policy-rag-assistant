"""
rag.py — Retrieval engine for the Policy & Support RAG Assistant.

Public interface (this is the contract other modules, e.g. ``main.py``,
should depend on):

- ``index_exists(index_path=None) -> bool``
    Check whether the embedding index has already been built.

- ``build_index(...)``
    Re-exported from ``ingest.py`` so callers needing a one-shot "make sure
    the index exists" step don't need to import ``ingest`` directly.

- ``retrieve_all(query: str, index_path=None, client=None) -> list[dict]``
    Embed ``query`` with the same OpenAI embedding model used at ingest
    time, compute cosine similarity against every stored chunk vector
    (plain numpy — no vector DB needed at this corpus size), and return
    EVERY chunk (not just a top-k slice) as a list of dicts, sorted
    descending by score:

        {
          "id": str,              # e.g. "returns-and-exchange-policy.md#Refund Methods#0"
          "score": float,         # cosine similarity, higher is better
          "source": str,          # e.g. "returns-and-exchange-policy.md"
          "section": str,         # e.g. "Refund Methods"
          "text": str,            # full chunk text
        }

    This is the "full activation" computation a neural-network-style
    frontend graph needs: a score for every node, not just the ones that
    ended up grounding the answer.

- ``retrieve(query: str, k: int = 3, index_path=None, client=None) -> list[dict]``
    Thin wrapper around ``retrieve_all``: returns just the top-k highest
    scoring chunks (``retrieve_all(...)[:k]``). This is what should be fed
    to the LLM as grounding context.

    Raises a ``RuntimeError`` with a clear message if the index has not
    been built yet (call ``build_index()`` first, or let ``main.py`` do so
    on startup).

Requires ``OPENAI_API_KEY`` in the environment for the query-embedding
call. Building the index itself (``ingest.build_index``) also requires it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Support both "run as a package" (e.g. `uvicorn backend.main:app` from the
# parent directory) and "run from inside backend/" (plain module imports),
# mirroring the same dual-import pattern used in main.py.
try:
    from .ingest import (
        DEFAULT_INDEX_PATH,
        EMBEDDING_MODEL,
        build_index,  # re-exported for convenience: `from rag import build_index`
    )
except ImportError:  # pragma: no cover - fallback for non-package execution
    from ingest import (
        DEFAULT_INDEX_PATH,
        EMBEDDING_MODEL,
        build_index,
    )

__all__ = ["retrieve", "retrieve_all", "index_exists", "build_index", "load_index"]


def index_exists(index_path: str | Path = DEFAULT_INDEX_PATH) -> bool:
    """Return True if the embedding index file has already been built."""
    return Path(index_path).exists()


def load_index(index_path: str | Path = DEFAULT_INDEX_PATH) -> list[dict]:
    """Load the persisted chunk+embedding records from index_path.

    Kept as its own function (rather than inlined into ``retrieve``) so a
    future switch to an .npz-backed store only requires changing this one
    loader, not every call site.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise RuntimeError(
            f"No index found at {index_path}. Call build_index() first "
            "(requires OPENAI_API_KEY in the environment), or run "
            "`python ingest.py` from the backend/ directory."
        )
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _embed_query(query: str, client=None) -> np.ndarray:
    """Embed a single query string with the same model used at ingest time."""
    if client is None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. It is required "
                "to embed the query for retrieval."
            )
        client = OpenAI(api_key=api_key)

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return np.array(response.data[0].embedding, dtype=np.float32)


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a matrix of row vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


def retrieve_all(
    query: str,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    client=None,
) -> list[dict]:
    """Score EVERY stored chunk against ``query`` — the "full activation" set.

    Embeds ``query`` once via OpenAI's ``text-embedding-3-small`` (the same
    model used to build the index), scores every stored chunk by cosine
    similarity, and returns all of them as a list of dicts sorted by
    descending score:

        [{"id": "...md#Section#0", "score": 0.83, "source": "...md",
          "section": "...", "text": "..."}, ...]

    This is the same computation ``retrieve()`` uses internally, just
    without truncating to top-k — intended for a frontend that wants to
    render every chunk's activation score (e.g. a neural-network-style
    graph), not only the ones that ended up grounding the answer.

    Raises RuntimeError if the index has not been built yet, or if
    OPENAI_API_KEY is missing from the environment.
    """
    records = load_index(index_path)
    if not records:
        return []

    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    query_vec = _embed_query(query, client=client)
    scores = _cosine_similarity(query_vec, embeddings)

    ranked_idx = np.argsort(-scores)

    return [
        {
            "id": records[i].get("id", f"{records[i]['source']}#{records[i]['section']}#{i}"),
            "score": float(scores[i]),
            "source": records[i]["source"],
            "section": records[i]["section"],
            "text": records[i]["text"],
        }
        for i in ranked_idx
    ]


def retrieve(
    query: str,
    k: int = 3,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    client=None,
) -> list[dict]:
    """Retrieve the top-k most relevant chunks for ``query``.

    Thin wrapper around ``retrieve_all``: returns just the highest-scoring
    ``k`` chunks (``retrieve_all(...)[:k]``), sorted descending by score.

    Raises RuntimeError if the index has not been built yet, or if
    OPENAI_API_KEY is missing from the environment.
    """
    return retrieve_all(query, index_path=index_path, client=client)[:k]
