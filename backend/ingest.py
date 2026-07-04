"""
ingest.py — Corpus chunking + embedding pipeline for the Policy & Support RAG Assistant.

Purpose
-------
Reads every ``*.md`` file in ``docs/``, splits each document into semantically
coherent chunks along its markdown ``##`` section headers (so a chunk is
roughly "one policy section", not an arbitrary character-count slice), embeds
each chunk with OpenAI's ``text-embedding-3-small`` model, and persists the
result to disk so embeddings are computed once and reused across app restarts
and queries.

Storage format
--------------
We write a single ``backend/index.json`` file containing a list of records:

    {
      "id": "returns-and-exchange-policy.md#Refund Methods#0",
      "source": "returns-and-exchange-policy.md",
      "section": "Refund Methods",
      "text": "<chunk text — a whole short section, or one paragraph of a longer one>",
      "embedding": [0.0123, -0.0456, ...]   # 1536 floats for text-embedding-3-small
    }

A single JSON file (rather than an .npz + sidecar JSON pair) was chosen
deliberately: this corpus is small (a handful of documents, a few dozen
chunks total), so the simplicity of one human-readable, git-diffable file
outweighs the marginal load-time/size efficiency of a binary .npz array.
If the corpus grows into the hundreds of documents, switching to an
``.npz`` vector store + sidecar metadata JSON (to avoid re-parsing floats
from JSON text on every load) would be the natural next step — the loader
in ``rag.py`` isolates that decision behind a single ``load_index()`` call
so making that change later would not require touching call sites.

Public interface
-----------------
- ``chunk_markdown_file(path) -> list[dict]``: pure, no-API-call chunking
  logic, unit-testable without an OpenAI key.
- ``chunk_all_docs(docs_dir) -> list[dict]``: chunks every .md file in a
  directory.
- ``embed_chunks(chunks, client=None) -> list[dict]``: attaches an
  "embedding" field to each chunk dict via the OpenAI embeddings API.
- ``build_index(docs_dir=None, index_path=None) -> list[dict]``: end-to-end
  entry point — chunk + embed + write index.json. This is the function
  ``main.py`` should call on startup if ``index.json`` is missing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = BACKEND_DIR.parent / "docs"
DEFAULT_INDEX_PATH = (
    Path("/tmp/policy-rag-index.json")
    if os.environ.get("VERCEL")
    else BACKEND_DIR / "index.json"
)

EMBEDDING_MODEL = "text-embedding-3-small"

# A section header line looks like: "## Refund Methods"
_SECTION_HEADER_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)

# Blank-line paragraph boundary (one or more blank lines, allowing trailing
# whitespace on the "blank" line).
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# Sections with more than this many paragraphs are split further into one
# chunk per paragraph, so long sections get real paragraph-level
# granularity instead of being retrieved/scored as one large blob.
_MAX_PARAGRAPHS_PER_CHUNK = 2


def _split_into_paragraphs(text: str) -> list[str]:
    """Split ``text`` on blank-line boundaries into non-empty paragraphs."""
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]


def _make_section_chunks(source: str, section_title: str, section_text: str, body: str) -> list[dict]:
    """Turn one ``##`` section into one or more paragraph-granular chunks.

    ``section_text`` is the full section text (heading line + body,
    including any folded-in doc preamble) — used verbatim as the chunk
    text when the section is short enough to stay a single chunk, so it
    keeps reading coherently on its own.

    ``body`` is the section content with the heading line stripped off —
    used as the basis for paragraph splitting when the section is longer
    than ``_MAX_PARAGRAPHS_PER_CHUNK`` paragraphs.

    Every chunk gets a stable, unique ``id`` of the form
    ``"<filename>#<section title>#<index>"``.
    """
    paragraphs = _split_into_paragraphs(body)

    if len(paragraphs) <= _MAX_PARAGRAPHS_PER_CHUNK:
        return [
            {
                "id": f"{source}#{section_title}#0",
                "source": source,
                "section": section_title,
                "text": section_text,
            }
        ]

    return [
        {
            "id": f"{source}#{section_title}#{idx}",
            "source": source,
            "section": section_title,
            "text": paragraph,
        }
        for idx, paragraph in enumerate(paragraphs)
    ]


def chunk_markdown_file(path: str | Path) -> list[dict]:
    """Split one markdown file into paragraph-granular chunks.

    Splitting starts at top-level ``##`` headers (the natural "policy
    section" boundary in this corpus). The document's ``#`` title (if any)
    is treated as a preamble and folded into the first section so it is not
    lost. Any section longer than ``_MAX_PARAGRAPHS_PER_CHUNK`` paragraphs
    (split on blank lines) is further broken into one chunk per paragraph,
    so long sections get real retrieval granularity instead of being
    scored as a single large blob; short sections stay as one chunk
    (heading + body together) so they still read coherently on their own.

    Returns a list of dicts: ``{"id": <stable unique id>, "source":
    <filename>, "section": <heading title, shared by all paragraph-chunks
    under it>, "text": <chunk text>}``. No API calls are made here — this
    function is pure text processing and is exercised directly by
    ``test_retrieval.py`` without requiring ``OPENAI_API_KEY``.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    source = path.name

    matches = list(_SECTION_HEADER_RE.finditer(raw))

    chunks: list[dict] = []

    if not matches:
        # No ## headers found; treat the whole file as a single chunk.
        title_match = re.match(r"^#\s+(.*)$", raw.strip(), re.MULTILINE)
        section_title = title_match.group(1).strip() if title_match else source
        text = raw.strip()
        if text:
            chunks.extend(_make_section_chunks(source, section_title, text, text))
        return chunks

    # Anything before the first "##" (the doc title + any preamble text)
    # is treated as front matter and folded into the first section chunk
    # so it isn't dropped.
    preamble = raw[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        section_text = raw[start:end].strip()
        body = raw[match.end():end].strip()

        if i == 0 and preamble:
            section_text = f"{preamble}\n\n{section_text}"
            body = f"{preamble}\n\n{body}" if body else preamble

        if section_text:
            chunks.extend(_make_section_chunks(source, section_title, section_text, body))

    return chunks


def chunk_all_docs(docs_dir: str | Path = DEFAULT_DOCS_DIR) -> list[dict]:
    """Chunk every ``*.md`` file in ``docs_dir``, in sorted filename order."""
    docs_dir = Path(docs_dir)
    all_chunks: list[dict] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        all_chunks.extend(chunk_markdown_file(md_path))
    return all_chunks


def _get_openai_client():
    """Lazily construct an OpenAI client using OPENAI_API_KEY from the environment.

    Imported lazily so that chunking (and its tests) work without the
    ``openai`` package's network dependencies being exercised, and so a
    missing API key only fails at the point embeddings are actually needed.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment. Set it before "
            "building the embedding index, e.g.:\n"
            "  PowerShell: $env:OPENAI_API_KEY = '...'\n"
            "  bash:       export OPENAI_API_KEY=...\n"
        )
    return OpenAI(api_key=api_key)


def embed_chunks(chunks: list[dict], client=None) -> list[dict]:
    """Embed each chunk's text with the OpenAI embeddings API.

    Mutates and returns the same list of chunk dicts, adding an
    "embedding" key (list[float]) to each. Batches all chunk texts into a
    single API call where possible (OpenAI's embeddings endpoint accepts a
    list of inputs), falling back to per-chunk calls only if the batch call
    fails.
    """
    if not chunks:
        return chunks

    client = client or _get_openai_client()
    texts = [c["text"] for c in chunks]

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        for chunk, item in zip(chunks, response.data):
            chunk["embedding"] = item.embedding
    except Exception:
        # Fall back to one request per chunk if a single batched call fails
        # (e.g. a provider-side batch size limit).
        for chunk in chunks:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk["text"])
            chunk["embedding"] = response.data[0].embedding

    return chunks


def build_index(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[dict]:
    """End-to-end entry point: chunk all docs, embed them, and persist to index_path.

    This is the function ``main.py`` (the FastAPI wrapper) should call on
    first startup if ``index.json`` does not exist yet. Requires
    ``OPENAI_API_KEY`` to be set in the environment. Returns the list of
    chunk dicts (each including its "embedding") that were written to disk.
    """
    chunks = chunk_all_docs(docs_dir)
    chunks = embed_chunks(chunks)

    index_path = Path(index_path)
    index_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return chunks


if __name__ == "__main__":
    result = build_index()
    print(f"Built index with {len(result)} chunks -> {DEFAULT_INDEX_PATH}")
