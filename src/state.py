"""Run state, document cache, and the dead-letter log.

Ingestion is resumable and idempotent: vision inference is the most expensive
thing this system does, so a crash on page 40 of 50 must never force a re-run of
pages 1-39.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from src import config

log = logging.getLogger(__name__)

_EMPTY_STATE: dict[str, Any] = {"documents": {}, "indexed_chunks": []}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s (%s); starting from empty.", path.name, exc)
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write via temp file + rename.

    Ingest flushes periodically; a plain truncating write that dies mid-flush
    would leave a corrupt cache and destroy the resumability we are paying for.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# state.json
# --------------------------------------------------------------------------- #
def load_state() -> dict[str, Any]:
    state = _read_json(config.STATE_FILE, dict(_EMPTY_STATE))
    state.setdefault("documents", {})
    state.setdefault("indexed_chunks", [])
    return state


def save_state(state: dict[str, Any]) -> None:
    _write_json_atomic(config.STATE_FILE, state)


def document_is_complete(state: dict[str, Any], doc_hash: str) -> bool:
    return state["documents"].get(doc_hash, {}).get("status") == "complete"


def mark_document(
    state: dict[str, Any],
    doc_hash: str,
    *,
    source: str,
    pages: int,
    status: str,
    visuals: int = 0,
) -> None:
    from datetime import datetime, timezone

    state["documents"][doc_hash] = {
        "source": source,
        "pages": pages,
        "visuals": visuals,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def indexed_chunk_set(state: dict[str, Any]) -> set[str]:
    return set(state.get("indexed_chunks", []))


def commit_indexed_chunks(state: dict[str, Any], chunk_ids: set[str]) -> None:
    state["indexed_chunks"] = sorted(chunk_ids)


# --------------------------------------------------------------------------- #
# document_cache.json
# --------------------------------------------------------------------------- #
def load_cache() -> dict[str, Any]:
    return _read_json(config.DOCUMENT_CACHE, {})


def save_cache(cache: dict[str, Any]) -> None:
    _write_json_atomic(config.DOCUMENT_CACHE, cache)


# --------------------------------------------------------------------------- #
# dead_letter.json
# --------------------------------------------------------------------------- #
def record_failure(stage: str, detail: dict[str, Any]) -> None:
    """Log a non-fatal failure so a bad page or an unparseable LLM reply can be
    inspected later instead of aborting the run."""
    from datetime import datetime, timezone

    entries = _read_json(config.DEAD_LETTER_FILE, [])
    if not isinstance(entries, list):
        entries = []
    entries.append(
        {
            "stage": stage,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **detail,
        }
    )
    _write_json_atomic(config.DEAD_LETTER_FILE, entries)


def dead_letter_count() -> int:
    entries = _read_json(config.DEAD_LETTER_FILE, [])
    return len(entries) if isinstance(entries, list) else 0
