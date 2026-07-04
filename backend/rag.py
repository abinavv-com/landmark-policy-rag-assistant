"""
rag.py — Multi-stage retrieval engine for the Policy & Support RAG Assistant.

Public interface (this is the contract other modules, e.g. ``main.py``,
should depend on):

- ``index_exists(index_path=None) -> bool``
    Check whether the embedding index has already been built.

- ``build_index(...)``
    Re-exported from ``ingest.py`` so callers needing a one-shot "make sure
    the index exists" step don't need to import ``ingest`` directly.

- ``embedding_stage(query: str, index_path=None, client=None) -> list[dict]``
    STAGE 1. Embed ``query`` with the same OpenAI embedding model used at
    ingest time, compute cosine similarity against every stored chunk
    vector (plain numpy — no vector DB needed at this corpus size), and
    return EVERY chunk (not just a top-k slice) as a list of dicts, sorted
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
    ended up grounding the answer. ``retrieve_all`` is kept as an alias for
    backward compatibility.

- ``rerank_stage(query: str, candidates: list[dict], top_n: int = 25) -> list[dict]``
    STAGE 2. Takes the top ``top_n`` candidates surviving Stage 1 (embedding
    similarity) and re-scores them with a genuinely different, cheap signal:
    a TF-IDF-weighted keyword/term-overlap score computed over the query and
    the candidate set (a BM25-lite; see ``_keyword_score`` docstring for the
    exact formula). The final ``rerank_score`` blends the two signals:

        rerank_score = 0.65 * normalized_embedding_score
                     + 0.35 * normalized_keyword_score

    Returns the candidates re-sorted descending by ``rerank_score``, with
    ``embedding_score``, ``keyword_score``, and ``rerank_score`` all present
    on each dict alongside the original fields.

- ``select_context(reranked: list[dict], k: int = 5) -> list[dict]``
    STAGE 3. Takes the top-k of the Stage 2 output as the final grounding
    context handed to the LLM.

- ``retrieve(query: str, k: int = 3, index_path=None, client=None) -> list[dict]``
    Backward-compatible convenience wrapper that runs the full pipeline
    (Stage 1 -> Stage 2 -> Stage 3) and returns the top-k chunks.

    Raises a ``RuntimeError`` with a clear message if the index has not
    been built yet (call ``build_index()`` first, or let ``main.py`` do so
    on startup).

Requires ``OPENAI_API_KEY`` in the environment for the query-embedding
call. Building the index itself (``ingest.build_index``) also requires it.
Stage 2 (``rerank_stage``) and its keyword scoring are pure Python/numpy and
require no API key at all.
"""

from __future__ import annotations

import json
import math
import os
import re
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

__all__ = [
    "retrieve",
    "retrieve_all",
    "embedding_stage",
    "rerank_stage",
    "select_context",
    "index_exists",
    "build_index",
    "load_index",
]


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


def embedding_stage(
    query: str,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    client=None,
) -> list[dict]:
    """STAGE 1 — score EVERY stored chunk against ``query`` by cosine similarity.

    Embeds ``query`` once via OpenAI's ``text-embedding-3-small`` (the same
    model used to build the index), scores every stored chunk by cosine
    similarity, and returns all of them as a list of dicts sorted by
    descending score:

        [{"id": "...md#Section#0", "score": 0.83, "source": "...md",
          "section": "...", "text": "..."}, ...]

    This is the "full activation" set — a score for every node, not only
    the ones that end up grounding the answer. Downstream, ``rerank_stage``
    takes the top slice of this list and re-scores it with a different
    signal (Stage 2).

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


# Backward-compatible alias: this is the exact same function that used to be
# called retrieve_all(). Kept so existing callers/tests importing
# rag.retrieve_all keep working unchanged.
retrieve_all = embedding_stage


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization used by the keyword stage."""
    return _TOKEN_RE.findall(text.lower())


def _keyword_score(query: str, candidates: list[dict]) -> list[float]:
    """Score each candidate chunk's keyword/term overlap with ``query``.

    This is a real, from-scratch TF-IDF-weighted overlap score (a "BM25-lite"),
    deliberately independent of the embedding model so it acts as a genuinely
    different second signal in ``rerank_stage``:

    1. Tokenize the query and every candidate's text (lowercase, alphanumeric
       tokens only).
    2. Compute inverse document frequency (IDF) for each query term across the
       candidate set (the "corpus" for this rerank pass is the Stage-1
       survivors being reranked):

           idf(t) = ln((N + 1) / (df(t) + 1)) + 1

       where N is the number of candidates and df(t) is the number of
       candidates whose text contains term t at least once. (This is the
       standard smoothed IDF formula used by scikit-learn's TfidfVectorizer;
       the +1 smoothing avoids division by zero / negative IDFs for terms
       that appear in every candidate.)
    3. For each candidate, compute term frequency tf(t, d) = count of term t
       in that candidate's tokenized text, and sum tf(t, d) * idf(t) over all
       *unique* query terms t. This rewards chunks that repeat query terms
       (tf) but weights rarer, more distinctive query terms more heavily
       (idf) than common ones that appear in most candidates.
    4. Length-normalize by dividing by sqrt(number of tokens in the
       candidate), to avoid unfairly favoring very long chunks purely for
       having more raw term occurrences.

    Returns a list of raw (un-normalized) keyword scores, one per candidate,
    in the same order as ``candidates``. Higher is better. A candidate with
    zero query-term overlap scores exactly 0.0.
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return [0.0] * len(candidates)

    candidate_tokens = [_tokenize(c.get("text", "")) for c in candidates]
    n_candidates = len(candidates)

    # Document frequency of each query term across the candidate set.
    doc_freq: dict[str, int] = {}
    for term in query_terms:
        doc_freq[term] = sum(1 for tokens in candidate_tokens if term in tokens)

    idf = {
        term: math.log((n_candidates + 1) / (doc_freq[term] + 1)) + 1.0
        for term in query_terms
    }

    scores: list[float] = []
    for tokens in candidate_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for tok in tokens:
            if tok in query_terms:
                tf[tok] = tf.get(tok, 0) + 1
        raw = sum(count * idf[term] for term, count in tf.items())
        scores.append(raw / math.sqrt(len(tokens)))

    return scores


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of floats into [0, 1]. Constant lists -> 0.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank_stage(query: str, candidates: list[dict], top_n: int = 25) -> list[dict]:
    """STAGE 2 — re-score the top ``top_n`` Stage-1 survivors with keyword overlap.

    Takes the highest-scoring ``top_n`` chunks from ``embedding_stage`` (the
    embedding-similarity survivors) and blends their embedding score with an
    independent TF-IDF-weighted keyword-overlap score (``_keyword_score``):

        rerank_score = 0.65 * normalized_embedding_score
                     + 0.35 * normalized_keyword_score

    Both signals are min-max normalized across the ``top_n`` candidate set
    before blending, so the two differently-scaled scores (cosine similarity
    vs. TF-IDF overlap) contribute comparably to the final rank.

    Returns the candidates re-sorted descending by ``rerank_score``. Each
    returned dict keeps its original fields (``id``, ``source``, ``section``,
    ``text``) plus ``embedding_score`` (renamed from Stage 1's ``score``),
    ``keyword_score`` (raw, un-normalized), and ``rerank_score`` (the final
    blended, sorted-on value).
    """
    top_candidates = candidates[:top_n]
    if not top_candidates:
        return []

    embedding_scores = [c["score"] for c in top_candidates]
    keyword_scores = _keyword_score(query, top_candidates)

    norm_embedding = _normalize(embedding_scores)
    norm_keyword = _normalize(keyword_scores)

    reranked = []
    for c, emb_score, kw_score, n_emb, n_kw in zip(
        top_candidates, embedding_scores, keyword_scores, norm_embedding, norm_keyword
    ):
        rerank_score = 0.65 * n_emb + 0.35 * n_kw
        reranked.append(
            {
                "id": c["id"],
                "source": c["source"],
                "section": c["section"],
                "text": c["text"],
                "embedding_score": float(emb_score),
                "keyword_score": float(kw_score),
                "rerank_score": float(rerank_score),
            }
        )

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked


def select_context(reranked: list[dict], k: int = 5) -> list[dict]:
    """STAGE 3 — take the top-k Stage-2 survivors as the final grounding context.

    Thin slice of ``rerank_stage``'s output: ``reranked[:k]``. Kept as its
    own named function (rather than inlined at call sites) so the pipeline
    stages stay individually testable and the "3 stages then LLM" structure
    is explicit in the code, not just in comments.
    """
    return reranked[:k]


def retrieve(
    query: str,
    k: int = 3,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    client=None,
) -> list[dict]:
    """Backward-compatible convenience wrapper: run the full pipeline, return top-k.

    Runs Stage 1 (``embedding_stage``) -> Stage 2 (``rerank_stage`` over the
    top 25 Stage-1 survivors) -> Stage 3 (``select_context`` for the final
    top-k). Anything that used to call ``retrieve(query, k=3)`` for a plain
    top-k list of grounding chunks keeps working, now backed by the richer
    pipeline. Note the returned dicts now carry ``embedding_score`` /
    ``keyword_score`` / ``rerank_score`` instead of a single ``score`` field.

    Raises RuntimeError if the index has not been built yet, or if
    OPENAI_API_KEY is missing from the environment.
    """
    stage1 = embedding_stage(query, index_path=index_path, client=client)
    stage2 = rerank_stage(query, stage1, top_n=25)
    return select_context(stage2, k=k)
