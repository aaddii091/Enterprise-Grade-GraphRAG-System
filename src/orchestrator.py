"""Single-command orchestration of the three pipeline stages.

Each stage runs as its OWN SUBPROCESS, not as a function call. That is the whole
point of this module. Chaining the pipelines in-process would mean:

  - a failed unload (it is best-effort, 30s timeout) lets stage 2 start while the
    7.9 GB vision model is still resident;
  - torch, the embedding model and pdfplumber's buffers accumulate in one RSS
    that never shrinks, because Python returns freed arenas to the OS reluctantly.

Process exit is the only guarantee that memory actually comes back. Between
stages we additionally gate on /api/ps being empty before launching the next.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src import config, ollama_client
from src.main import _port_open  # single definition of the port check

log = logging.getLogger(__name__)

MAIN = Path(__file__).resolve().parent / "main.py"

# Neo4j needs a while before bolt accepts connections, even once the container is up.
DB_READY_TIMEOUT_S = 120
# How long to wait for Ollama to give the RAM back between stages.
GATE_TIMEOUT_S = 45
# Docker Desktop is a GUI app and is slow to hand over a usable daemon.
DOCKER_DESKTOP_TIMEOUT_S = 120


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class RunReport:
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def render(self) -> str:
        width = max((len(s.name) for s in self.stages), default=10)
        lines = ["", "Run summary", "-" * (width + 26)]
        for s in self.stages:
            mark = "ok  " if s.ok else "FAIL"
            lines.append(f"  {mark}  {s.name:<{width}}  {s.seconds:6.1f}s  {s.detail}")
        total = sum(s.seconds for s in self.stages)
        lines.append("-" * (width + 26))
        lines.append(f"  total {total:.1f}s")
        return "\n".join(lines)


def _run_stage(name: str, argv: list[str]) -> StageResult:
    """Spawn one stage of main.py and stream its output live."""
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}\n", flush=True)
    started = time.monotonic()
    # No capture: tqdm bars and log lines go straight to the user's terminal.
    proc = subprocess.run(
        [sys.executable, str(MAIN), *argv],
        cwd=str(config.PROJECT_ROOT),
    )
    elapsed = time.monotonic() - started
    ok = proc.returncode == 0
    detail = "" if ok else f"exit code {proc.returncode}"
    return StageResult(name=name, ok=ok, seconds=elapsed, detail=detail)


def _memory_gate(label: str) -> None:
    """Refuse to start the next stage until Ollama has released its models."""
    resident = ollama_client.loaded_models()
    if not resident:
        return

    log.info("%s: %s still resident; requesting eviction.", label, ", ".join(resident))
    ollama_client.unload_all()

    deadline = time.monotonic() + GATE_TIMEOUT_S
    while time.monotonic() < deadline:
        if not ollama_client.loaded_models():
            log.info("%s: RAM released.", label)
            return
        time.sleep(2)

    # Not fatal - Ollama will evict on its own keep_alive timer, and the next
    # stage may simply be slower. But the user should know why.
    log.warning(
        "%s: models still resident after %ds. The next stage may swap.",
        label,
        GATE_TIMEOUT_S,
    )


def _docker_daemon_up() -> bool:
    """True if the Docker daemon is reachable (not merely installed)."""
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_databases(auto_start: bool) -> StageResult:
    """Bring Qdrant and Neo4j up if they are not already listening."""
    started = time.monotonic()
    qdrant_up = _port_open(config.QDRANT_URL, 6333)
    neo4j_up = _port_open(config.NEO4J_URI, 7687)

    if qdrant_up and neo4j_up:
        return StageResult("databases", True, time.monotonic() - started, "already up")

    if not auto_start:
        return StageResult(
            "databases", False, time.monotonic() - started,
            "not running (start with: docker compose up -d)",
        )

    # `docker compose up` needs the daemon, and the CLI cannot start it - on
    # macOS that is Docker Desktop, a GUI app. Launch it and wait rather than
    # failing the run after a long ingest has already completed.
    if not _docker_daemon_up():
        print("\nDocker daemon is not running; launching Docker Desktop ...", flush=True)
        subprocess.run(["open", "-a", "Docker"], capture_output=True)
        deadline = time.monotonic() + DOCKER_DESKTOP_TIMEOUT_S
        while time.monotonic() < deadline:
            if _docker_daemon_up():
                break
            time.sleep(3)
        else:
            return StageResult(
                "databases", False, time.monotonic() - started,
                f"Docker Desktop did not start within {DOCKER_DESKTOP_TIMEOUT_S}s",
            )

    print("\nStarting Qdrant and Neo4j (docker compose up -d) ...", flush=True)
    try:
        proc = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=str(config.PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return StageResult(
            "databases", False, time.monotonic() - started, "docker not found on PATH"
        )

    if proc.returncode != 0:
        return StageResult(
            "databases", False, time.monotonic() - started,
            (proc.stderr or proc.stdout).strip().splitlines()[-1][:120],
        )

    # Neo4j accepts bolt well after the container reports running.
    print("Waiting for databases to accept connections ...", flush=True)
    deadline = time.monotonic() + DB_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if _port_open(config.QDRANT_URL, 6333) and _port_open(config.NEO4J_URI, 7687):
            return StageResult(
                "databases", True, time.monotonic() - started, "started"
            )
        time.sleep(3)

    return StageResult(
        "databases", False, time.monotonic() - started,
        f"not ready after {DB_READY_TIMEOUT_S}s (check: docker compose logs)",
    )


def _resolve_pdfs(target: str | None) -> list[Path]:
    """A file, a directory of PDFs, or everything in data/raw/."""
    if target:
        path = Path(target).expanduser().resolve()
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.pdf"))
        raise FileNotFoundError(f"No such file or directory: {path}")
    return sorted(config.RAW_DIR.glob("*.pdf"))


def run_all(
    target: str | None = None,
    *,
    ask: str | None = None,
    force: bool = False,
    rebuild: bool = False,
    auto_start_db: bool = True,
    show_context: bool = False,
) -> RunReport:
    """Ingest -> build index -> (optionally) answer, as separate processes."""
    report = RunReport()

    pdfs = _resolve_pdfs(target)
    if not pdfs:
        where = target or str(config.RAW_DIR)
        raise RuntimeError(
            f"No PDFs found in {where}.\n"
            "  Put one in data/raw/ or pass a path: --run path/to/file.pdf"
        )

    print(f"\nProcessing {len(pdfs)} PDF(s): {', '.join(p.name for p in pdfs)}")

    # --- Stage 1: vision ingest (one process per PDF) -----------------------
    # The databases are deliberately NOT started yet. The vision model can be
    # ~12 GB resident; adding a 4 GB Docker VM on a 16 GB machine guarantees
    # swapping. Ingest needs no database (it preflights with need_db=False), so
    # the containers start only once the vision model has been evicted.
    # Per-PDF processes, so a corrupt file costs one document rather than the run.
    for pdf in pdfs:
        result = _run_stage(
            f"1/3  ingest  {pdf.name}",
            ["--ingest", str(pdf)] + (["--force"] if force else []),
        )
        report.stages.append(result)
        if not result.ok:
            log.warning("Ingest failed for %s; continuing with the rest.", pdf.name)
        _memory_gate("after ingest")

    if not any(s.ok for s in report.stages if s.name.startswith("1/3")):
        log.error("Every ingest failed; nothing to index.")
        return report

    # --- Stage 1.5: databases (only now, with the vision model evicted) -----
    db = _ensure_databases(auto_start_db)
    report.stages.append(db)
    if not db.ok:
        return report

    # --- Stage 2: index -----------------------------------------------------
    result = _run_stage(
        "2/3  build-index",
        ["--build-index"] + (["--rebuild"] if rebuild else []),
    )
    report.stages.append(result)
    _memory_gate("after build-index")
    if not result.ok:
        return report

    # --- Stage 3: answer (only if a question was supplied) ------------------
    if ask:
        argv = ["--chat", ask] + (["--show-context"] if show_context else [])
        report.stages.append(_run_stage("3/3  chat", argv))
        _memory_gate("after chat")

    return report
