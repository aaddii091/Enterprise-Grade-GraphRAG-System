"""Thin, shared Ollama wrapper.

All three pipelines need the same three things: generate, *verified* unload, and a
health check with an actionable error. Duplicating the unload-verification loop is
how it ends up subtly wrong in two of the three copies, so it lives here once.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import requests

from src import config

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Ollama is unreachable, or returned something we cannot use."""


def _url(path: str) -> str:
    return f"{config.OLLAMA_BASE_URL}{path}"


def health() -> list[str]:
    """Return the list of locally available model names.

    Raises with remediation text rather than letting a ConnectionRefused surface
    forty pages into an ingest run.
    """
    try:
        resp = requests.get(_url("/api/tags"), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}.\n"
            "  Install:  brew install ollama\n"
            "  Start:    brew services start ollama\n"
            f"  Detail:   {exc}"
        ) from exc

    return [m.get("name", "") for m in resp.json().get("models", [])]


def require_models(*models: str) -> None:
    """Fail early if a model this run depends on has not been pulled."""
    available = health()
    # Ollama reports "llama3.1:latest" for a "llama3.1" pull; compare on the stem.
    stems = {name.split(":", 1)[0] for name in available}
    missing = [m for m in models if m.split(":", 1)[0] not in stems]
    if missing:
        pulls = "\n".join(f"  ollama pull {m}" for m in missing)
        raise OllamaError(f"Missing Ollama model(s): {', '.join(missing)}\n{pulls}")


def loaded_models() -> list[str]:
    """Models currently resident in RAM, per /api/ps."""
    try:
        resp = requests.get(_url("/api/ps"), timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        # A failed introspection must never break a pipeline; assume nothing loaded.
        return []
    return [m.get("name", "") for m in resp.json().get("models", [])]


def generate(
    model: str,
    prompt: str,
    *,
    images: Sequence[str] | None = None,
    system: str | None = None,
    fmt: str | None = None,
    num_ctx: int | None = None,
    temperature: float = 0.0,
    timeout: int = 600,
    retries: int = 2,
) -> str:
    """Single-shot, non-streaming completion.

    `images` are base64-encoded PNG strings (no data: prefix).
    `fmt="json"` puts Ollama into constrained-JSON decoding - it guarantees valid
    syntax, though not sensible content.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx if num_ctx is not None else config.NUM_CTX,
            "temperature": temperature,
        },
    }
    if images:
        payload["images"] = list(images)
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(_url("/api/generate"), json=payload, timeout=timeout)
            if resp.status_code >= 400:
                # Ollama puts the real reason in the body ("unknown model
                # architecture", "model requires more system memory", ...).
                # raise_for_status() alone discards it and reports a bare
                # "500 Server Error", which is useless for diagnosis.
                detail = ""
                try:
                    detail = resp.json().get("error", "")
                except ValueError:
                    detail = resp.text[:300]
                raise OllamaError(
                    f"Ollama returned {resp.status_code} for model "
                    f"{model!r}: {detail or '(no detail)'}"
                )
            return resp.json().get("response", "").strip()
        except OllamaError as exc:
            # An architecture mismatch or a missing model will not fix itself on
            # retry; fail immediately rather than burning three attempts per page.
            raise
        except requests.Timeout as exc:
            raise OllamaError(
                f"{model!r} exceeded {timeout}s on one call. Retrying replays the "
                f"same input, so this is not retried."
            ) from exc
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                backoff = 2 ** attempt
                log.warning(
                    "Ollama generate failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    retries + 1,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

    raise OllamaError(f"generate() failed after {retries + 1} attempts: {last_exc}")


def unload(model: str, *, settle: bool = True) -> bool:
    """Evict `model` from RAM and VERIFY it is gone.

    The spec's `keep_alive: 0` POST is a request, not a guarantee. On a 16 GB
    machine the next pipeline stage cannot afford to assume it worked, so we poll
    /api/ps until the model disappears rather than sleeping and hoping.

    Returns True if the model was confirmed evicted.
    """
    try:
        requests.post(
            _url("/api/generate"),
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except requests.RequestException as exc:
        log.warning("Unload request for %s failed: %s", model, exc)

    stem = model.split(":", 1)[0]
    deadline = time.monotonic() + config.UNLOAD_TIMEOUT_S
    while time.monotonic() < deadline:
        resident = {n.split(":", 1)[0] for n in loaded_models()}
        if stem not in resident:
            log.info("Unloaded %s from RAM.", model)
            if settle:
                # Give macOS a moment to actually reclaim the unified memory.
                time.sleep(config.MEMORY_SETTLE_S)
            return True
        time.sleep(1)

    log.warning(
        "%s still resident after %ds. The next stage may swap - consider running "
        "stages in separate invocations.",
        model,
        config.UNLOAD_TIMEOUT_S,
    )
    if settle:
        time.sleep(config.MEMORY_SETTLE_S)
    return False


def unload_all() -> None:
    """Evict everything Ollama currently holds."""
    for name in loaded_models():
        if name:
            unload(name, settle=False)
    time.sleep(config.MEMORY_SETTLE_S)
