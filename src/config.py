"""Central configuration.

Every tunable lives here or in .env. Nothing else in the codebase reads os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DOCUMENT_CACHE = PROCESSED_DIR / "document_cache.json"
STATE_FILE = PROCESSED_DIR / "state.json"
DEAD_LETTER_FILE = PROCESSED_DIR / "dead_letter.json"

# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# llama3.2-vision (architecture "mllama") is NOT runnable on Ollama 0.34.x -
# the engine has no loader for it and fails at model-load time. Vision archs
# this build supports: gemma3, qwen3vl.
VISION_MODEL = _env("VISION_MODEL", "qwen2.5vl:3b")
TEXT_MODEL = _env("TEXT_MODEL", "llama3.1")

# Ollama defaults num_ctx to 2048, which silently TRUNCATES our chunks and makes
# triple extraction look randomly broken. Always pass this explicitly.
NUM_CTX = _env_int("NUM_CTX", 8192)
# Vision calls carry one image plus two sentences - they do not need the 8k
# window that triple extraction does, and a smaller KV cache directly shrinks
# resident memory for the largest model in the pipeline.
VISION_NUM_CTX = _env_int("VISION_NUM_CTX", 2048)

# Seconds to wait for a model to disappear from /api/ps after an unload request.
UNLOAD_TIMEOUT_S = _env_int("UNLOAD_TIMEOUT_S", 30)

# Per-call ceilings. Text chunks run ~50s; anything past 3 minutes is a model
# looping on pathological input (number-dense tables), not slow progress - so
# fail fast, dead-letter it and move on. Vision genuinely is slow, so it keeps
# a generous ceiling.
TEXT_TIMEOUT_S = _env_int("TEXT_TIMEOUT_S", 180)
VISION_TIMEOUT_S = _env_int("VISION_TIMEOUT_S", 600)
# Settle time for macOS to reclaim unified memory once the model is gone.
# The spec calls for >= 5s between pipeline stages; this is the tail of it.
MEMORY_SETTLE_S = _env_int("MEMORY_SETTLE_S", 5)

# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "graphrag_chunks")

# --------------------------------------------------------------------------- #
# Neo4j
# --------------------------------------------------------------------------- #
NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", "password123")

# --------------------------------------------------------------------------- #
# Embeddings / chunking
# --------------------------------------------------------------------------- #
# ~130 MB resident. Deliberately small: it has to coexist with an 8B text model.
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = _env_int("EMBED_DIM", 384)

CHUNK_SIZE = _env_int("CHUNK_SIZE", 768)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 96)
TOP_K = _env_int("TOP_K", 3)

# --------------------------------------------------------------------------- #
# Vision extraction
# --------------------------------------------------------------------------- #
CROP_DPI = _env_int("CROP_DPI", 150)
# pdfplumber's default x_tolerance of 3 fuses adjacent words on tightly-kerned
# PDFs ("MonthlyReport", "August1997"), which wrecks both embeddings and triple
# extraction. 1.5 is the least aggressive value that separates them cleanly.
X_TOLERANCE = float(os.environ.get("X_TOLERANCE", "1.5"))
# Regions smaller than this (in PDF points) are rules, bullets and logos. Sending
# them to an 11B vision model is pure cost.
MIN_CROP_POINTS = _env_int("MIN_CROP_POINTS", 40)
# Hard ceiling on visual elements per page, so one pathological page can't stall a run.
MAX_VISUALS_PER_PAGE = _env_int("MAX_VISUALS_PER_PAGE", 8)
# pdfplumber's find_tables() keys on ruling lines. Documents that lay tables out
# with whitespace and filled rects (financial reports, very often) slip past it,
# while multi-column body text gets flagged AS a table. Digit density separates
# the two reliably: a numeric table is digit-dense, prose is not. Regions below
# this are dropped rather than sent to an 11 GB vision model to re-read text
# extract_text() already captured.
MIN_TABLE_DIGIT_PCT = float(os.environ.get("MIN_TABLE_DIGIT_PCT", "15"))
# If nothing survives that filter but the page as a whole is numeric, the table
# was missed by detection entirely - send the whole page instead.
PAGE_TABLE_DIGIT_PCT = float(os.environ.get("PAGE_TABLE_DIGIT_PCT", "20"))

# --------------------------------------------------------------------------- #
# Graph search
# --------------------------------------------------------------------------- #
GRAPH_MAX_HOPS = _env_int("GRAPH_MAX_HOPS", 2)
GRAPH_PATH_LIMIT = _env_int("GRAPH_PATH_LIMIT", 50)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
