"""
FastAPI serving layer for the Policy & Support RAG Assistant demo.

This app is intentionally a thin orchestration layer: it does not do any
retrieval or embedding logic itself. Retrieval and index-building live in
sibling modules built in parallel:

    backend/rag.py    -> retrieve(query, k=3) -> list[dict], index_exists() -> bool
    backend/ingest.py -> build_index() -> None

If those modules are not present yet (e.g. still being built by a parallel
agent), this file will fail to import until they land. Its own syntax is
independently valid and can be checked with:

    python -m py_compile backend/main.py
"""

from __future__ import annotations

import logging
import os
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("policy_rag_assistant")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Imports from the parallel-built modules.
#
# We support both "run as a package" (uvicorn backend.main:app, relative
# imports work) and "run from inside backend/" (plain module imports) so this
# file keeps working regardless of exactly how the other agent's modules are
# invoked/tested.
# ---------------------------------------------------------------------------
try:
    from . import rag  # type: ignore
    from . import ingest  # type: ignore
except ImportError:  # pragma: no cover - fallback for non-package execution
    import rag  # type: ignore
    import ingest  # type: ignore


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    """Simple mutable holder for startup-derived state."""

    index_ready: bool = False
    startup_error: Optional[str] = None


app_state = AppState()


def _try_prepare_index() -> None:
    """
    Ensure the retrieval index exists, building it if necessary.

    Any failure here (most commonly: missing OPENAI_API_KEY, or an OpenAI API
    error while embedding documents) is caught and logged. We deliberately do
    NOT let this crash the server — the API should still come up so /health
    and /ask can report a clear, structured error to the client instead of the
    process failing to bind.
    """
    try:
        if rag.index_exists():
            logger.info("Policy RAG index already exists — skipping build.")
            app_state.index_ready = True
            app_state.startup_error = None
            return

        if not os.environ.get("OPENAI_API_KEY"):
            msg = (
                "OPENAI_API_KEY is not set in the environment. "
                "Skipping index build; /ask will return a clear error until "
                "the key is configured and the server is restarted."
            )
            logger.warning(msg)
            app_state.index_ready = False
            app_state.startup_error = msg
            return

        logger.info("No existing index found — building index from docs/ ...")
        ingest.build_index()

        app_state.index_ready = rag.index_exists()
        if app_state.index_ready:
            logger.info("Policy RAG index built successfully.")
            app_state.startup_error = None
        else:
            msg = "ingest.build_index() completed but no index was found afterwards."
            logger.error(msg)
            app_state.startup_error = msg

    except Exception as exc:  # noqa: BLE001 - intentionally broad: startup must not crash
        msg = f"Failed to prepare RAG index on startup: {exc}"
        logger.exception(msg)
        app_state.index_ready = False
        app_state.startup_error = msg


@asynccontextmanager
async def lifespan(_: FastAPI):
    _try_prepare_index()
    yield
    # No teardown needed for this local demo.


app = FastAPI(
    title="Policy & Support RAG Assistant",
    description=(
        "Demo REST API showing retrieval-augmented generation over Landmark "
        "Group policy/support documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: wide open. This is a local prototype/demo, not a production service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_INDEX_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class AskRequest(BaseModel):
    question: str
    history: Optional[list[ChatMessage]] = None


class Citation(BaseModel):
    source: str
    section: str


class RetrievedChunk(BaseModel):
    text: str
    source: str
    section: str
    score: float


class ActivationChunk(BaseModel):
    id: str
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    activation: list[ActivationChunk]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


SYSTEM_PROMPT_TEMPLATE = """You are a Policy & Support assistant for Landmark Group retail brands.

Answer the user's question using ONLY the retrieved policy context provided below.
Do not use outside knowledge and do not make up policy details that are not present
in the context. If the context does not contain enough information to answer the
question, say so clearly in your answer instead of guessing.

Always cite the source document name(s) you drew the answer from.

Respond with a strict JSON object matching this exact shape and nothing else:
{{
  "answer": "<your grounded answer as plain text, citing source document names inline>",
  "citations": [{{"source": "<document name>", "section": "<section name or heading>"}}]
}}

Retrieved context:
{context}
"""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.get('source', 'unknown')} "
            f"| Section: {chunk.get('section', 'unknown')}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(parts) if parts else "(no context retrieved)"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "index_ready": bool(app_state.index_ready)}


@app.get("/")
async def frontend_root() -> Any:
    if FRONTEND_INDEX_PATH.exists():
        return FileResponse(FRONTEND_INDEX_PATH)
    return _error("Frontend index.html was not bundled with the deployment.", status.HTTP_500_INTERNAL_SERVER_ERROR)


_TITLE_RE = re.compile(r"^#\s+(.*)$", re.MULTILINE)


@app.get("/documents")
async def documents() -> Any:
    """
    Structural view of the docs/ corpus for the frontend's graph/explorer.

    Reuses ingest.chunk_markdown_file (the same paragraph-granular splitter
    used to build the retrieval index) so the graph the user sees matches
    exactly what /ask and /ask's "activation" scores retrieve against — the
    chunk "id" values returned here are the SAME ids used in
    rag.retrieve_all()/the /ask "activation" list, so the frontend can build
    one consistent node graph and match activation scores back onto it by
    "id". No separate parsing logic to drift out of sync.

    BREAKING CHANGE from the previous shape: each document's chunk list is
    now under the key "chunks" (was "sections"), and each entry has an "id"
    field (previously absent) in addition to "section" (previously
    "title") and "text" (previously "content").

    Returns:
        [
          {
            "filename": "returns-and-exchange-policy.md",
            "title": "Returns and Exchange Policy",
            "chunks": [
              {
                "id": "returns-and-exchange-policy.md#Standard Return Window#0",
                "section": "Standard Return Window",
                "text": "..."
              },
              ...
            ]
          },
          ...
        ]
    """
    docs_dir = ingest.DEFAULT_DOCS_DIR
    results: list[dict[str, Any]] = []
    try:
        for md_path in sorted(docs_dir.glob("*.md")):
            raw = md_path.read_text(encoding="utf-8")
            title_match = _TITLE_RE.match(raw.strip())
            title = title_match.group(1).strip() if title_match else md_path.stem

            chunks = ingest.chunk_markdown_file(md_path)
            doc_chunks = [
                {"id": c["id"], "section": c["section"], "text": c["text"]}
                for c in chunks
            ]
            results.append(
                {"filename": md_path.name, "title": title, "chunks": doc_chunks}
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load documents for /documents")
        return _error(f"Failed to load documents: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)

    return results


@app.post("/ask")
async def ask(request: AskRequest) -> Any:
    question = (request.question or "").strip()
    if not question:
        return _error("Field 'question' is required and cannot be empty.", status.HTTP_400_BAD_REQUEST)

    if not os.environ.get("OPENAI_API_KEY"):
        return _error(
            "OPENAI_API_KEY is not configured on the server. Set it in the "
            "environment and restart the server to enable /ask.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if not app_state.index_ready:
        detail = app_state.startup_error or "Retrieval index is not ready yet."
        return _error(
            f"Policy index is not ready: {detail}",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # --- Retrieval -------------------------------------------------------
    # Score every chunk once via retrieve_all() (a single query-embedding
    # call), then slice the top-k off that same ranked list for grounding.
    # This avoids embedding the query twice just to get both a top-k answer
    # context and a full "activation" list for the frontend's graph.
    ASK_TOP_K = 3
    try:
        all_scored_chunks = rag.retrieve_all(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrieval failed for question: %s", question)
        return _error(f"Retrieval failed: {exc}", status.HTTP_502_BAD_GATEWAY)

    if not all_scored_chunks:
        return _error(
            "No relevant policy content could be retrieved for this question.",
            status.HTTP_404_NOT_FOUND,
        )

    raw_chunks = all_scored_chunks[:ASK_TOP_K]
    activation = [
        {
            "id": c["id"],
            "source": c["source"],
            "section": c["section"],
            "score": c["score"],
        }
        for c in all_scored_chunks
    ]

    # --- OpenAI call -----------------------------------------------------
    try:
        from openai import OpenAI  # imported here so a missing package only
        # breaks /ask, not the whole app import.
    except ImportError:
        return _error(
            "The 'openai' package is not installed on the server.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize OpenAI client")
        return _error(f"Failed to initialize OpenAI client: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=_build_context_block(raw_chunks))

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if request.history:
        for turn in request.history:
            if turn.role in ("user", "assistant") and turn.content:
                messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": question})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI chat completion call failed")
        return _error(f"OpenAI request failed: {exc}", status.HTTP_502_BAD_GATEWAY)

    try:
        raw_content = completion.choices[0].message.content
        parsed = json.loads(raw_content)
        answer = parsed.get("answer")
        citations = parsed.get("citations", [])
        if not isinstance(answer, str) or not isinstance(citations, list):
            raise ValueError("Model response JSON did not match the expected shape.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse OpenAI response as expected JSON")
        return _error(
            f"Model returned a malformed response: {exc}",
            status.HTTP_502_BAD_GATEWAY,
        )

    # Normalize citations defensively — never let a bad model response 500 us.
    clean_citations = []
    for c in citations:
        if isinstance(c, dict):
            clean_citations.append(
                {"source": str(c.get("source", "")), "section": str(c.get("section", ""))}
            )

    return {
        "answer": answer,
        "citations": clean_citations,
        "retrieved_chunks": raw_chunks,
        "activation": activation,
    }


@app.get("/{full_path:path}")
async def frontend_fallback(full_path: str) -> Any:
    # Browser refreshes should return the single-page frontend. Unknown API
    # style paths still get a JSON 404 so client mistakes are visible.
    if full_path.startswith("api/"):
        return _error("Not found.", status.HTTP_404_NOT_FOUND)
    if FRONTEND_INDEX_PATH.exists():
        return FileResponse(FRONTEND_INDEX_PATH)
    return _error("Frontend index.html was not bundled with the deployment.", status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort safety net: never let an unhandled exception leak a raw stack
    trace to the client. Anything that reaches here is a bug we should still
    look at (it's logged), but the client always gets clean JSON.
    """
    logger.exception("Unhandled exception")
    return _error(f"Internal server error: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)
