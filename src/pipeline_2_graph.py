"""Pipeline 2 - knowledge graph and vector index construction.

Reads the cache produced by pipeline 1, chunks it, and writes each chunk down two
paths: an embedding into Qdrant, and LLM-extracted triples into Neo4j. The two are
joined by chunk_id - stored in the Qdrant payload and on every Entity node - which
is what makes hybrid retrieval possible in pipeline 3.

Loads llama3.1 (8B) and evicts it on the way out.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tqdm import tqdm

from src import config, graph_store, ollama_client, state, vector_store
from src.text_utils import canonical_key, chunk_id as make_chunk_id, display_name, sanitize_relation

log = logging.getLogger(__name__)

EXTRACTION_SYSTEM = (
    "You are a precise information extraction engine. You emit JSON only. "
    "You never add commentary, explanation, or markdown fences."
)

EXTRACTION_PROMPT = """Extract knowledge from the text below.

Return a JSON object with a single key "triples", whose value is a list of objects
with exactly the keys "source", "relation", and "target".

Rules:
- "source" and "target" must be concrete named entities (people, organisations,
  products, places, dates, metrics). Never pronouns, never whole sentences.
- "relation" must be a short uppercase verb phrase with underscores, e.g.
  ACQUIRED, REPORTED_REVENUE_OF, HEADQUARTERED_IN.
- Extract only facts stated in the text. Invent nothing.
- If the text contains no extractable facts, return {{"triples": []}}.

TEXT:
\"\"\"
{chunk}
\"\"\"
"""

REPAIR_PROMPT = """The following was supposed to be a JSON object of the form
{{"triples": [{{"source": "...", "relation": "...", "target": "..."}}]}}
but could not be parsed. Return the corrected JSON object and nothing else.

MALFORMED:
\"\"\"
{raw}
\"\"\"
"""

# Chunks per state flush. Frequent enough that a crash costs little, rare enough
# that we are not rewriting state.json on every chunk.
FLUSH_EVERY_CHUNKS = 20


def _parse_triples(raw: str) -> list[dict[str, str]] | None:
    """Accept either {"triples": [...]} or a bare [...]; reject anything else."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        candidate = parsed.get("triples")
        if candidate is None:
            # Some replies come back as a single triple object.
            if {"source", "relation", "target"} <= set(parsed):
                candidate = [parsed]
            else:
                return None
    elif isinstance(parsed, list):
        candidate = parsed
    else:
        return None

    if not isinstance(candidate, list):
        return None
    return [t for t in candidate if isinstance(t, dict)]


def _normalise(raw_triples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Canonicalise entities and relation types, dropping unusable triples."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for triple in raw_triples:
        source = str(triple.get("source", "")).strip()
        target = str(triple.get("target", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        if not source or not target:
            continue

        skey, tkey = canonical_key(source), canonical_key(target)
        if not skey or not tkey or skey == tkey:
            # Self-loops carry no retrievable information.
            continue
        # Runaway extractions: a whole sentence masquerading as an entity.
        if len(skey) > 120 or len(tkey) > 120:
            continue

        rel = sanitize_relation(relation)
        signature = (skey, rel, tkey)
        if signature in seen:
            continue
        seen.add(signature)

        out.append(
            {
                "skey": skey,
                "sname": display_name(source),
                "rel": rel,
                "tkey": tkey,
                "tname": display_name(target),
            }
        )
    return out


def _extract_triples(chunk_text: str) -> list[dict[str, str]]:
    """One LLM call, one repair retry, then give up and dead-letter."""
    raw = ollama_client.generate(
        config.TEXT_MODEL,
        EXTRACTION_PROMPT.format(chunk=chunk_text),
        system=EXTRACTION_SYSTEM,
        fmt="json",
        temperature=0.0,
        timeout=config.TEXT_TIMEOUT_S,
    )
    parsed = _parse_triples(raw)

    if parsed is None:
        log.debug("Unparseable extraction, attempting repair.")
        repaired = ollama_client.generate(
            config.TEXT_MODEL,
            REPAIR_PROMPT.format(raw=raw[:4000]),
            system=EXTRACTION_SYSTEM,
            fmt="json",
            temperature=0.0,
            timeout=config.TEXT_TIMEOUT_S,
        )
        parsed = _parse_triples(repaired)

    if parsed is None:
        raise ValueError(f"Could not parse triples from model output: {raw[:200]!r}")

    return _normalise(parsed)


def build_index(*, rebuild: bool = False) -> dict[str, Any]:
    """Index every cached chunk that is not already indexed."""
    config.ensure_dirs()

    cache = state.load_cache()
    if not cache:
        raise RuntimeError(
            "data/processed/document_cache.json is empty.\n"
            "  Run:  python src/main.py --ingest data/raw/<file>.pdf"
        )

    ollama_client.require_models(config.TEXT_MODEL)
    vector_store.verify()
    graph_store.verify()
    graph_store.ensure_schema()

    run_state = state.load_state()
    done = set() if rebuild else state.indexed_chunk_set(run_state)

    # Build the full work list first so the progress bar is honest.
    splitter = vector_store.splitter()
    work: list[dict[str, Any]] = []
    for doc_hash, doc in cache.items():
        for page in doc.get("pages", []):
            text = (page.get("text") or "").strip()
            if not text:
                continue
            for piece in splitter.split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                cid = make_chunk_id(doc_hash, page["page_no"], piece)
                if cid in done:
                    continue
                work.append(
                    {
                        "chunk_id": cid,
                        "text": piece,
                        "metadata": {
                            "chunk_id": cid,
                            "doc_hash": doc_hash,
                            "source": doc.get("source", ""),
                            "filename": doc.get("filename", ""),
                            "page_no": page["page_no"],
                        },
                    }
                )

    if not work:
        log.info("Nothing to index - every cached chunk is already present.")
        return {"chunks_indexed": 0, "triples_written": 0, "skipped": len(done)}

    log.info("Indexing %d new chunk(s); %d already done.", len(work), len(done))

    triples_written = 0
    indexed = 0
    try:
        for i, item in enumerate(tqdm(work, desc="chunks", unit="ch"), start=1):
            cid = item["chunk_id"]

            # --- Vector path ------------------------------------------------
            try:
                node = vector_store.build_node(cid, item["text"], item["metadata"])
                vector_store.add_nodes([node])
            except Exception as exc:
                log.warning("Vector write failed for %s: %s", cid[:8], exc)
                state.record_failure(
                    "vector",
                    {"chunk_id": cid, "page": item["metadata"]["page_no"],
                     "error": str(exc)},
                )
                continue

            # --- Graph path -------------------------------------------------
            try:
                triples = _extract_triples(item["text"])
                triples_written += graph_store.upsert_triples(triples, cid)
            except Exception as exc:
                log.warning("Triple extraction failed for %s: %s", cid[:8], exc)
                state.record_failure(
                    "graph",
                    {"chunk_id": cid, "page": item["metadata"]["page_no"],
                     "error": str(exc)},
                )
                # The chunk is searchable by vector even without triples, so it
                # counts as done; only the graph enrichment was lost.

            done.add(cid)
            indexed += 1

            if i % FLUSH_EVERY_CHUNKS == 0:
                state.commit_indexed_chunks(run_state, done)
                state.save_state(run_state)

        state.commit_indexed_chunks(run_state, done)
        state.save_state(run_state)
    finally:
        state.commit_indexed_chunks(run_state, done)
        state.save_state(run_state)
        graph_store.close()
        ollama_client.unload(config.TEXT_MODEL)

    log.info(
        "Indexed %d chunk(s); %d relationship(s) merged.", indexed, triples_written
    )
    return {
        "chunks_indexed": indexed,
        "triples_written": triples_written,
        "skipped": len(work) - indexed,
    }
