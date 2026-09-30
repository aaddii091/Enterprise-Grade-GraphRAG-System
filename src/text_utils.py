"""Hashing, chunk identity, and entity normalisation.

The normalisation helpers here are the single most safety-critical code in the
project: pipeline_2 writes Entity.key with `canonical_key`, and pipeline_3 looks
entities up by the same property. If the two ever diverge, graph retrieval
silently returns zero paths and the system degrades to plain vector RAG without
raising a single error. They therefore live in exactly one place.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from pathlib import Path

# Stable namespace so chunk UUIDs are reproducible across runs and machines.
_CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "graphrag.enterprise.chunks")

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_REL_ALLOWED_RE = re.compile(r"[^A-Z0-9_]")

# Dropped from the head/tail of entity names so "Apple Inc." and "Apple" unify.
_CORPORATE_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "plc", "corp", "corporation",
    "co", "company", "gmbh", "ag", "sa", "nv", "bv", "holdings", "group",
}
_LEADING_ARTICLES = {"the", "a", "an"}


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large PDFs don't land in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(doc_hash: str, page_no: int, text: str) -> str:
    """Deterministic chunk identity, formatted as a UUID.

    Qdrant only accepts unsigned ints or UUIDs as point IDs - a raw hex digest is
    rejected - so a UUID5 over the content gives us both a legal ID and free
    idempotency on re-runs.
    """
    seed = f"{doc_hash}:{page_no}:{text_hash(text)}"
    return str(uuid.uuid5(_CHUNK_NAMESPACE, seed))


def canonical_key(name: str) -> str:
    """Fold an entity mention to its match key.

    'Apple Inc.', 'apple  inc', 'The Apple Inc' -> 'apple'
    Returns '' for anything that normalises away entirely; callers must skip those.
    """
    if not name:
        return ""

    # Strip accents so "Nestle" and "Nestlé" converge.
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))

    # Collapse acronym dots BEFORE the general punctuation pass. Otherwise
    # "S.A." becomes the two tokens "s a", never matches the suffix list, and
    # "Nestle S.A." ends up as a different node from "Nestle SA".
    folded = folded.replace(".", "")

    folded = _PUNCT_RE.sub(" ", folded.lower())
    tokens = [t for t in _WS_RE.split(folded) if t]

    while tokens and tokens[0] in _LEADING_ARTICLES:
        tokens.pop(0)
    while tokens and tokens[-1] in _CORPORATE_SUFFIXES:
        tokens.pop()

    return " ".join(tokens).strip()


def display_name(name: str) -> str:
    """Human-facing form of an entity: collapsed whitespace, original casing."""
    return _WS_RE.sub(" ", name).strip()


def sanitize_relation(relation: str) -> str:
    """Coerce an LLM-authored relation into a legal Cypher relationship type.

    Cypher cannot parameterise relationship types, so these strings reach the
    database through apoc.merge.relationship and must be constrained here.
    """
    if not relation:
        return "RELATED_TO"
    folded = unicodedata.normalize("NFKD", relation)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = _REL_ALLOWED_RE.sub("_", folded.upper().replace(" ", "_"))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"REL_{cleaned}" if cleaned else "RELATED_TO"
    return cleaned[:64]
