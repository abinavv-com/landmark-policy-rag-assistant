"""
test_retrieval.py — Tests for the chunking and retrieval logic.

Run with:  python backend/test_retrieval.py
(or from inside backend/:  python test_retrieval.py)

Two tiers of coverage:

1. Chunking tests (always run, no API calls, no OPENAI_API_KEY needed):
   verifies chunk_markdown_file()/chunk_all_docs() split each doc into a
   reasonable number of section-based chunks with correct metadata.

2. Retrieval tests (only run if OPENAI_API_KEY is set in the environment):
   builds the real embedding index and runs a handful of real queries,
   asserting the expected source document surfaces in the top result.
   If no key is set, these are skipped with a clear printed message
   rather than failing the test run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import ingest  # noqa: E402


DOCS_DIR = BACKEND_DIR.parent / "docs"

# Core docs that must always be present, regardless of how large the corpus grows.
CORE_DOC_FILES = [
    "loyalty-program-faq.md",
    "returns-and-exchange-policy.md",
    "shipping-and-delivery-faq.md",
    "size-and-fit-guide.md",
    "store-and-online-service-faq.md",
    "warranty-policy.md",
]

# The full expected set is derived from disk, not hardcoded, so it stays correct
# as the corpus grows (currently 51 docs and counting).
EXPECTED_DOC_FILES = sorted(p.name for p in DOCS_DIR.glob("*.md"))


def test_docs_present() -> None:
    print("\n=== Test: docs/ contains the core corpus files ===")
    found = sorted(p.name for p in DOCS_DIR.glob("*.md"))
    missing = [f for f in CORE_DOC_FILES if f not in found]
    assert not missing, f"Missing core doc files: {missing}"
    print(f"PASS: found {len(found)} doc files (>= {len(CORE_DOC_FILES)} core files present)")


def test_chunking_is_reasonable() -> None:
    print("\n=== Test: chunking produces reasonable paragraph-granular chunks ===")
    total_chunks = 0
    for filename in EXPECTED_DOC_FILES:
        path = DOCS_DIR / filename
        chunks = ingest.chunk_markdown_file(path)

        # Paragraph-level splitting of longer sections should yield
        # meaningfully more chunks per doc than the old one-chunk-per-##
        # section scheme (which landed in the 3-12 range per doc). Not one
        # giant blob, not one-chunk-per-line.
        assert 6 <= len(chunks) <= 25, (
            f"{filename}: expected 6-25 chunks, got {len(chunks)}"
        )

        ids_seen = set()
        sections_to_ids: dict[str, list[str]] = {}
        for chunk in chunks:
            assert chunk["source"] == filename
            assert chunk["section"], f"{filename}: chunk missing section title"
            assert chunk["text"], f"{filename}: chunk has empty text"
            assert chunk["id"], f"{filename}: chunk missing id"
            assert chunk["id"] not in ids_seen, (
                f"{filename}: duplicate chunk id {chunk['id']!r}"
            )
            ids_seen.add(chunk["id"])
            assert chunk["id"].startswith(f"{filename}#{chunk['section']}#"), (
                f"{filename}: id {chunk['id']!r} does not match "
                f"'<filename>#<section>#<index>' convention"
            )
            sections_to_ids.setdefault(chunk["section"], []).append(chunk["id"])
            # Each chunk should be more than a single line/header fragment.
            assert len(chunk["text"].split()) >= 3, (
                f"{filename}: chunk '{chunk['id']}' looks too small "
                f"({len(chunk['text'].split())} words)"
            )

        total_chunks += len(chunks)
        print(f"  {filename}: {len(chunks)} chunks -> "
              f"{[c['id'] for c in chunks]}")

    print(f"PASS: {total_chunks} total chunks across {len(EXPECTED_DOC_FILES)} docs")
    # Real paragraph-level granularity should push the corpus well above
    # the old ~27-chunk (one-chunk-per-##-section) baseline.
    assert total_chunks > 50, (
        f"expected a meaningful increase over the old ~27-chunk baseline, "
        f"got {total_chunks}"
    )


def test_chunk_ids_unique_and_sections_consistent() -> None:
    print("\n=== Test: chunk ids are globally unique; sections are consistent ===")
    all_chunks = ingest.chunk_all_docs(DOCS_DIR)
    all_ids = [c["id"] for c in all_chunks]
    assert len(all_ids) == len(set(all_ids)), "duplicate chunk ids found across corpus"

    # All chunks belonging to the same (source, section) pair must share
    # the exact same "section" string (paragraph sub-chunks of one section
    # are grouped under one shared heading).
    by_key: dict[tuple[str, str], set[str]] = {}
    for c in all_chunks:
        key = (c["source"], c["id"].rsplit("#", 1)[0])
        by_key.setdefault(key, set()).add(c["section"])
    for key, section_values in by_key.items():
        assert len(section_values) == 1, (
            f"chunks under id-prefix {key} disagree on 'section': {section_values}"
        )

    print(f"PASS: {len(all_ids)} unique chunk ids; sections consistent within each id group")


def test_chunk_all_docs_matches_per_file_sum() -> None:
    print("\n=== Test: chunk_all_docs() matches sum of per-file chunking ===")
    all_chunks = ingest.chunk_all_docs(DOCS_DIR)
    per_file_total = sum(
        len(ingest.chunk_markdown_file(DOCS_DIR / f)) for f in EXPECTED_DOC_FILES
    )
    assert len(all_chunks) == per_file_total
    print(f"PASS: chunk_all_docs() returned {len(all_chunks)} chunks")


def test_rerank_stage_keyword_scoring_discriminates() -> None:
    """
    Isolates rag.rerank_stage()'s keyword-scoring signal from the embedding
    signal: requires no OPENAI_API_KEY and no index, since it feeds
    synthetic candidates with a shared, fixed embedding "score" directly
    into rerank_stage().

    If keyword_score were a placeholder (e.g. always 0, or a copy of the
    embedding score), a chunk with heavy exact keyword overlap with the
    query and a chunk with none would tie. This test asserts they don't:
    the high-overlap chunk must score strictly higher on keyword_score
    (and therefore on rerank_score, since embedding_score is identical
    across all three candidates).
    """
    print("\n=== Test: rerank_stage() keyword scoring discriminates on term overlap (no API key needed) ===")
    import rag  # local import: numpy-only, no openai/network dependency for this path

    query = "warranty appliance repair"

    candidates = [
        {
            "id": "doc-a.md#High Overlap#0",
            "source": "doc-a.md",
            "section": "High Overlap",
            # Repeats every query term multiple times.
            "text": (
                "This warranty covers appliance repair. If your appliance "
                "needs repair, the warranty applies to appliance repair costs."
            ),
            "score": 0.5,  # identical embedding_score across all candidates
        },
        {
            "id": "doc-b.md#Partial Overlap#1",
            "source": "doc-b.md",
            "section": "Partial Overlap",
            # Shares only one query term ("warranty"), no repetition.
            "text": "The warranty period begins on the date of purchase.",
            "score": 0.5,
        },
        {
            "id": "doc-c.md#No Overlap#2",
            "source": "doc-c.md",
            "section": "No Overlap",
            # Shares zero query terms.
            "text": "Loyalty points can be redeemed in store or online at checkout.",
            "score": 0.5,
        },
    ]

    reranked = rag.rerank_stage(query, candidates, top_n=25)
    assert len(reranked) == 3, f"expected all 3 candidates back, got {len(reranked)}"

    by_id = {c["id"]: c for c in reranked}
    high = by_id["doc-a.md#High Overlap#0"]
    partial = by_id["doc-b.md#Partial Overlap#1"]
    none_ = by_id["doc-c.md#No Overlap#2"]

    print(f"  high overlap:    keyword_score={high['keyword_score']:.4f}  rerank_score={high['rerank_score']:.4f}")
    print(f"  partial overlap: keyword_score={partial['keyword_score']:.4f}  rerank_score={partial['rerank_score']:.4f}")
    print(f"  no overlap:      keyword_score={none_['keyword_score']:.4f}  rerank_score={none_['rerank_score']:.4f}")

    # All three share the same input embedding_score (0.5), so any spread in
    # keyword_score / rerank_score must come purely from the new keyword
    # signal doing real, non-placeholder work.
    assert high["embedding_score"] == partial["embedding_score"] == none_["embedding_score"] == 0.5

    assert high["keyword_score"] > partial["keyword_score"] > 0.0, (
        "expected strictly more keyword overlap to score strictly higher"
    )
    assert none_["keyword_score"] == 0.0, (
        f"expected zero query-term overlap to score exactly 0.0, got {none_['keyword_score']}"
    )
    assert high["rerank_score"] > partial["rerank_score"] > none_["rerank_score"], (
        "rerank_score must reflect the keyword_score ordering when embedding_score is tied"
    )

    # Final sort order must follow rerank_score descending.
    assert [c["id"] for c in reranked] == [high["id"], partial["id"], none_["id"]], (
        f"rerank_stage() did not sort by rerank_score descending: "
        f"{[c['id'] for c in reranked]}"
    )

    print("PASS: rerank_stage() keyword scoring genuinely discriminates on term overlap")


def test_select_context_slices_top_k() -> None:
    print("\n=== Test: select_context() takes the top-k of a pre-ranked list (no API key needed) ===")
    import rag

    reranked = [
        {"id": f"chunk-{i}", "source": "x.md", "section": "S", "text": "t",
         "embedding_score": 1.0 - i * 0.1, "keyword_score": 0.0,
         "rerank_score": 1.0 - i * 0.1}
        for i in range(10)
    ]
    selected = rag.select_context(reranked, k=5)
    assert len(selected) == 5
    assert [c["id"] for c in selected] == [f"chunk-{i}" for i in range(5)]
    print("PASS: select_context() returns exactly the top-k slice")


def test_retrieval_with_real_embeddings() -> None:
    print("\n=== Test: real retrieval (requires OPENAI_API_KEY) ===")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "SKIP: OPENAI_API_KEY is not set in the environment. "
            "Skipping embedding-dependent retrieval assertions — this is "
            "expected in environments without API access and is not a "
            "test failure."
        )
        return

    import rag  # imported here so its numpy/openai deps are only required
    # when we actually intend to exercise the API-backed path.

    index_path = BACKEND_DIR / "index.json"
    print("Building real embedding index from docs/ ...")
    ingest.build_index(docs_dir=DOCS_DIR, index_path=index_path)
    print(f"Index built at {index_path}")

    test_cases = [
        (
            "Can I return a sale item from Max after 20 days?",
            "returns-and-exchange-policy.md",
        ),
        (
            "What is the warranty on Home Centre appliances?",
            "warranty-policy.md",
        ),
        (
            "Do Max and Lifestyle use the same size chart?",
            "size-and-fit-guide.md",
        ),
        (
            "How long does delivery take to Saudi Arabia?",
            "shipping-and-delivery-faq.md",
        ),
        (
            "Can I redeem Max points at Home Centre?",
            "loyalty-program-faq.md",
        ),
    ]

    for query, expected_source in test_cases:
        results = rag.retrieve(query, k=3, index_path=index_path)
        assert results, f"No results returned for query: {query!r}"
        top_sources = [r["source"] for r in results]
        print(f"  Query: {query!r}")
        print(f"    Top-{len(results)} sources: {top_sources} "
              f"(scores: {[round(r['score'], 3) for r in results]})")
        assert expected_source in top_sources, (
            f"Expected '{expected_source}' in top results for query "
            f"{query!r}, got {top_sources}"
        )

    print(f"PASS: {len(test_cases)} real retrieval queries returned expected sources")


def main() -> None:
    test_docs_present()
    test_chunking_is_reasonable()
    test_chunk_all_docs_matches_per_file_sum()
    test_chunk_ids_unique_and_sections_consistent()
    test_rerank_stage_keyword_scoring_discriminates()
    test_select_context_slices_top_k()
    test_retrieval_with_real_embeddings()
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
