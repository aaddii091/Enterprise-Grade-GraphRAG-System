"""Pipeline 3 - hybrid retrieval and generation.

Vector search and graph traversal run against the same corpus and are fused into
one context block before llama3.1 answers. Entity keys are produced with the same
`canonical_key` used at write time in pipeline 2 - if those two ever diverge, graph
retrieval returns nothing and the system quietly degrades to plain vector RAG.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src import config, graph_store, ollama_client, vector_store
from src.text_utils import canonical_key

log = logging.getLogger(__name__)

ENTITY_SYSTEM = "You extract entity names. You emit JSON only, never commentary."

ENTITY_PROMPT = """Identify the named entities in the question below.

Return a JSON object: {{"entities": ["...", "..."]}}

Include organisations, people, products, places, dates and named metrics.
Exclude generic words like "revenue", "company" or "trend" unless they are part of
a proper name. If there are none, return {{"entities": []}}.

QUESTION: {query}
"""

ANSWER_SYSTEM = (
    "You answer strictly from the supplied context. If the context does not "
    "contain the answer, say so plainly and name what is missing - never fill the "
    "gap from general knowledge. Cite sources as (filename p.N) when the context "
    "provides them."
)

ANSWER_PROMPT = """{context}

QUESTION: {query}

Answer using only the context above."""

# Leave room for the question and the answer inside num_ctx. ~4 chars/token.
MAX_CONTEXT_CHARS = max(2000, (config.NUM_CTX - 1200) * 4)


def _extract_query_entities(query: str) -> list[str]:
    """Entity keys for the graph lookup, normalised exactly as pipeline 2 did."""
    try:
        raw = ollama_client.generate(
            config.TEXT_MODEL,
            ENTITY_PROMPT.format(query=query),
            system=ENTITY_SYSTEM,
            fmt="json",
            temperature=0.0,
        )
        parsed = json.loads(raw)
        names = parsed.get("entities", []) if isinstance(parsed, dict) else parsed
        if not isinstance(names, list):
            names = []
    except Exception as exc:
        log.warning("Entity extraction failed (%s); falling back to raw tokens.", exc)
        names = []

    if not names:
        # Crude fallback: capitalised runs in the question. Better a weak graph
        # probe than none at all.
        names = [w.strip(".,?!") for w in query.split() if w[:1].isupper()]

    keys, seen = [], set()
    for name in names:
        key = canonical_key(str(name))
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def retrieve(query: str) -> dict[str, Any]:
    """Step A + B: vector search and graph traversal."""
    # Step A - vector
    try:
        passages = vector_store.search(query, top_k=config.TOP_K)
    except Exception as exc:
        log.warning("Vector search failed: %s", exc)
        passages = []

    # Step B - graph
    keys = _extract_query_entities(query)
    paths: list[str] = []
    if keys:
        try:
            paths = graph_store.find_paths(keys)
            # Graph -> vector: pull the passages those entities came from, so a
            # question answerable only via a hop still gets its supporting text.
            extra_ids = [
                cid
                for cid in graph_store.chunk_ids_for_keys(keys)
                if cid not in {p["chunk_id"] for p in passages}
            ]
            passages.extend(vector_store.fetch_by_ids(extra_ids[:2]))
        except Exception as exc:
            log.warning("Graph search failed: %s", exc)

    return {"passages": passages, "paths": paths, "entity_keys": keys}


def fuse_context(retrieved: dict[str, Any]) -> str:
    """Step C: one labelled, length-bounded context block."""
    sections: list[str] = []

    if retrieved["passages"]:
        lines = ["### Retrieved passages"]
        for i, hit in enumerate(retrieved["passages"], start=1):
            page = f" p.{hit['page_no']}" if hit.get("page_no") else ""
            score = f" score={hit['score']:.3f}" if hit.get("score") is not None else ""
            lines.append(f"\n[{i}] ({hit.get('source', 'unknown')}{page}{score})")
            lines.append(hit.get("text", ""))
        sections.append("\n".join(lines))

    if retrieved["paths"]:
        lines = ["### Knowledge graph paths"]
        lines.extend(f"- {p}" for p in retrieved["paths"])
        sections.append("\n".join(lines))

    if not sections:
        return "### Context\n(No relevant information was retrieved.)"

    context = "\n\n".join(sections)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[context truncated]"
    return context


def answer(query: str, *, show_context: bool = False) -> dict[str, Any]:
    """Full A -> B -> C -> D for a single question. Does not unload the model."""
    retrieved = retrieve(query)
    context = fuse_context(retrieved)

    if show_context:
        print("\n" + "=" * 70)
        print(context)
        print("=" * 70 + "\n")

    response = ollama_client.generate(
        config.TEXT_MODEL,
        ANSWER_PROMPT.format(context=context, query=query),
        system=ANSWER_SYSTEM,
        temperature=0.1,
    )
    return {
        "query": query,
        "answer": response,
        "context": context,
        "n_passages": len(retrieved["passages"]),
        "n_paths": len(retrieved["paths"]),
        "entity_keys": retrieved["entity_keys"],
    }


def chat(query: str, *, show_context: bool = False) -> dict[str, Any]:
    """One-shot query. Loads the model, answers, evicts it."""
    ollama_client.require_models(config.TEXT_MODEL)
    vector_store.verify()
    graph_store.verify()
    try:
        result = answer(query, show_context=show_context)
        print(f"\n{result['answer']}\n")
        log.info(
            "Grounded on %d passage(s) and %d graph path(s).",
            result["n_passages"],
            result["n_paths"],
        )
        return result
    finally:
        graph_store.close()
        ollama_client.unload(config.TEXT_MODEL)


def repl(*, show_context: bool = False) -> None:
    """Interactive session - the model stays warm across turns, evicted once at exit."""
    ollama_client.require_models(config.TEXT_MODEL)
    vector_store.verify()
    graph_store.verify()

    print("GraphRAG chat. Ctrl-D or 'exit' to quit.\n")
    try:
        while True:
            try:
                query = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                break
            try:
                result = answer(query, show_context=show_context)
                print(f"\nbot > {result['answer']}\n")
                print(
                    f"      [{result['n_passages']} passage(s), "
                    f"{result['n_paths']} graph path(s)]\n"
                )
            except Exception as exc:
                print(f"\n[error] {exc}\n")
    finally:
        graph_store.close()
        ollama_client.unload(config.TEXT_MODEL)
