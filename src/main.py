"""Orchestration CLI.

The stages are deliberately separate invocations. On 16 GB of unified memory an
11B vision model and an 8B text model cannot both be resident, so nothing here
chains pipelines automatically - each command loads one model, does its work, and
evicts it before returning.

    python src/main.py --run data/raw/report.pdf     # all three stages
    python src/main.py --ingest data/raw/report.pdf
    python src/main.py --build-index
    python src/main.py --chat "What was the Q3 revenue trend?"
    python src/main.py --status
    python src/main.py --reset
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

# Allow `python src/main.py` as well as `python -m src.main`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ollama_client, state  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at INFO and drown out our own output.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Neo4j emits a DEPRECATION notice per apoc.coll.toSet call and dumps the whole
    # query with it - three per chunk, which buries every real message in the run.
    # They are advisory only; -v still shows them.
    if not verbose:
        logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def _port_open(url: str, default_port: int) -> bool:
    parsed = urlparse(url if "//" in url else f"//{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _preflight(need_db: bool) -> None:
    """Check the environment before doing anything expensive."""
    problems: list[str] = []

    try:
        ollama_client.health()
    except ollama_client.OllamaError as exc:
        problems.append(str(exc))

    if need_db:
        if not _port_open(config.QDRANT_URL, 6333):
            problems.append(
                f"Qdrant is not listening at {config.QDRANT_URL}.\n"
                "  Start it:  docker compose up -d qdrant"
            )
        if not _port_open(config.NEO4J_URI, 7687):
            problems.append(
                f"Neo4j is not listening at {config.NEO4J_URI}.\n"
                "  Start it:  docker compose up -d neo4j"
            )

    if problems:
        print("\nPreflight failed:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}\n", file=sys.stderr)
        sys.exit(1)


def _peak_rss_mb() -> float:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Darwin reports bytes; Linux reports kilobytes.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except Exception:
        return 0.0


def cmd_status() -> None:
    from src import graph_store, vector_store

    run_state = state.load_state()
    docs = run_state.get("documents", {})

    print("\nGraphRAG status")
    print("-" * 52)
    print(f"  Documents ingested : {len(docs)}")
    for doc_hash, meta in docs.items():
        print(
            f"    - {Path(meta.get('source', '?')).name}  "
            f"[{meta.get('status')}]  {meta.get('pages')} pages, "
            f"{meta.get('visuals', 0)} visuals  ({doc_hash[:10]})"
        )
    print(f"  Chunks indexed     : {len(run_state.get('indexed_chunks', []))}")

    if _port_open(config.QDRANT_URL, 6333):
        print(f"  Qdrant points      : {vector_store.count()}")
    else:
        print("  Qdrant points      : (offline)")

    if _port_open(config.NEO4J_URI, 7687):
        try:
            stats = graph_store.stats()
            print(f"  Graph entities     : {stats['entities']}")
            print(f"  Graph relationships: {stats['relationships']}")
            if stats["unlinked_entities"]:
                print(
                    f"  !! {stats['unlinked_entities']} entit(ies) have no chunk_ids "
                    "- the graph->vector link is incomplete."
                )
        except Exception as exc:
            print(f"  Graph              : (error: {exc})")
        finally:
            graph_store.close()
    else:
        print("  Graph              : (offline)")

    dead = state.dead_letter_count()
    print(f"  Dead-letter entries: {dead}" + ("  <- inspect data/processed/dead_letter.json" if dead else ""))

    loaded = ollama_client.loaded_models()
    print(f"  Models in RAM      : {', '.join(loaded) if loaded else 'none'}")
    print()


def cmd_reset(yes: bool) -> None:
    from src import graph_store, vector_store

    if not yes:
        print("This deletes the Qdrant collection, every Neo4j Entity, and all")
        print("local processing state. The PDFs in data/raw/ are left alone.")
        if input("Type 'reset' to confirm: ").strip().lower() != "reset":
            print("Aborted.")
            return

    if _port_open(config.QDRANT_URL, 6333):
        vector_store.wipe()
        print("  Qdrant collection dropped.")
    if _port_open(config.NEO4J_URI, 7687):
        try:
            graph_store.wipe()
            print("  Neo4j entities deleted.")
        finally:
            graph_store.close()

    for path in (config.STATE_FILE, config.DOCUMENT_CACHE, config.DEAD_LETTER_FILE):
        if path.exists():
            path.unlink()
            print(f"  Removed {path.name}.")
    print("Reset complete.\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description="Local multimodal GraphRAG pipeline (Apple Silicon, 16 GB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", metavar="PDF_OR_DIR", nargs="?", const="",
                       help="Everything: ingest -> build-index (-> --ask). "
                            "Omit the path to process every PDF in data/raw/")
    group.add_argument("--ingest", metavar="PDF", help="Pipeline 1: vision extraction")
    group.add_argument("--build-index", action="store_true",
                       help="Pipeline 2: chunk, embed, extract triples")
    group.add_argument("--chat", metavar="QUERY", nargs="?", const="",
                       help="Pipeline 3: ask a question (omit the query for a REPL)")
    group.add_argument("--status", action="store_true", help="Show corpus statistics")
    group.add_argument("--reset", action="store_true", help="Wipe all indexed data")

    parser.add_argument("--force", action="store_true",
                        help="--ingest: re-process a PDF already marked complete")
    parser.add_argument("--rebuild", action="store_true",
                        help="--build-index: re-index every chunk, ignoring state")
    parser.add_argument("--show-context", action="store_true",
                        help="--chat: print the fused context before the answer")
    parser.add_argument("--ask", metavar="QUERY",
                        help="--run: ask this question once indexing finishes")
    parser.add_argument("--no-db-start", action="store_true",
                        help="--run: do not auto-start docker compose")
    parser.add_argument("--yes", action="store_true", help="--reset: skip confirmation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    config.ensure_dirs()

    log = logging.getLogger("main")

    try:
        if args.status:
            cmd_status()
            return 0

        if args.reset:
            cmd_reset(args.yes)
            return 0

        if args.run is not None:
            # Only Ollama is checked here; the orchestrator brings the databases
            # up itself and gates on them before stage 2.
            _preflight(need_db=False)
            from src import orchestrator

            report = orchestrator.run_all(
                args.run or None,
                ask=args.ask,
                force=args.force,
                rebuild=args.rebuild,
                auto_start_db=not args.no_db_start,
                show_context=args.show_context,
            )
            print(report.render())
            if report.ok and not args.ask:
                print(
                    '\n  Ask it something:\n'
                    '    python src/main.py --chat "your question"\n'
                )
            return 0 if report.ok else 1

        if args.ingest:
            _preflight(need_db=False)
            from src import pipeline_1_vision

            result = pipeline_1_vision.ingest(Path(args.ingest), force=args.force)
            log.info("Ingest result: %s", result)

        elif args.build_index:
            _preflight(need_db=True)
            from src import pipeline_2_graph

            result = pipeline_2_graph.build_index(rebuild=args.rebuild)
            log.info("Index result: %s", result)

        elif args.chat is not None:
            _preflight(need_db=True)
            from src import pipeline_3_chat

            if args.chat:
                pipeline_3_chat.chat(args.chat, show_context=args.show_context)
            else:
                pipeline_3_chat.repl(show_context=args.show_context)

        log.info("Peak RSS this process: %.2f GB", _peak_rss_mb() / 1024)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted. Progress up to the last flush is saved.", file=sys.stderr)
        ollama_client.unload_all()
        return 130
    except Exception as exc:
        log.error("%s", exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
