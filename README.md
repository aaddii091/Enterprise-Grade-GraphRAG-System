# Enterprise-Grade GraphRAG System

A fully local, multimodal GraphRAG pipeline — PDFs in, grounded answers out — built
to run on an **Apple M1 Pro with 16 GB unified memory**.

Charts and tables are read by a vision model rather than discarded, facts are
stored as a Neo4j knowledge graph *and* as Qdrant vectors, and retrieval fuses
both. No data leaves the machine.

## The constraint that shapes everything

16 GB of unified memory cannot hold an 11B vision model, an 8B text model, an
embedding model and two databases at once. So the system is **three separate
commands**, never a single chained run. Each loads one model, does its work, and
verifiably evicts it from RAM before returning.

```
┌─ pipeline 1 ──────┐   ┌─ pipeline 2 ──────┐   ┌─ pipeline 3 ──────┐
│ llama3.2-vision   │   │ llama3.1 + bge    │   │ llama3.1 + bge    │
│ PDF → text+charts │ → │ → Qdrant + Neo4j  │ → │ hybrid retrieval  │
│ unload()          │   │ unload()          │   │ unload()          │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

Eviction is **verified**, not assumed: `ollama_client.unload()` posts
`keep_alive: 0` and then polls `/api/ps` until the model is actually gone.

## Setup

**1 — Shrink the Docker VM.** Docker Desktop → Settings → Resources → Memory →
**4 GB** → Apply & Restart. The containers cap at 1.5 g + 2 g; the remaining
~11 GB is what keeps the vision model out of swap. This step is not optional on a
16 GB machine.

**2 — Ollama.**
```bash
brew install ollama && brew services start ollama
ollama pull llama3.2-vision   # ~7.9 GB
ollama pull llama3.1          # ~4.9 GB
```

**3 — Python 3.12.** (The system default here is 3.14, where the torch /
llama-index wheels are unreliable.)
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**4 — Databases.**
```bash
cp .env.example .env
docker compose up -d
docker stats --no-stream     # LIMIT must read 1.5GiB / 2GiB, not 4GiB
```
If the limits read 4 GiB, `mem_limit` did not apply and a long run will
destabilise the machine.

## Use

One command, all three stages:

```bash
python src/main.py --run data/raw/report.pdf
python src/main.py --run                                  # every PDF in data/raw/
python src/main.py --run report.pdf --ask "Q3 revenue?"   # and answer at the end
```

`--run` starts the databases if they are down, then spawns each stage as its own
**subprocess**. That is deliberate: process exit is the only guarantee the OS
actually takes the memory back, and between stages it gates on `/api/ps` being
empty before launching the next. Chaining the stages as in-process function calls
would let a failed unload put two models in RAM at once.

Or drive the stages by hand:

```bash
python src/main.py --ingest data/raw/report.pdf   # 1: vision extraction
python src/main.py --build-index                  # 2: embed + extract triples
python src/main.py --chat "What was the Q3 revenue trend?"
python src/main.py --chat                         # interactive REPL
python src/main.py --status                       # corpus + RAM statistics
python src/main.py --reset                        # wipe indexed data
```

Useful flags: `--show-context` (print the fused context before the answer),
`--force` (re-ingest a completed PDF), `--rebuild` (re-index every chunk),
`-v` (debug logging).

Dashboards: Neo4j `localhost:7474` (`neo4j` / `password123`), Qdrant
`localhost:6333/dashboard`.

## How retrieval works

A question is answered from two sources fused into one context block:

- **Vector** — the query is embedded with `bge-small-en-v1.5` and the top-3
  passages come back from Qdrant.
- **Graph** — `llama3.1` names the entities in the question, they are normalised
  to match keys, and Neo4j returns paths up to 2 hops out.

The two halves are joined by `chunk_id`, which lives both in the Qdrant payload
and in a `chunk_ids` list on every `Entity` node. That lets a graph hit pull back
the exact passage it was extracted from — which is what makes a question like
*"What was the Q3 revenue trend for the company that acquired Firm X?"*
answerable: the hop is graph work, the revenue figure came out of a chart.

## Design notes

Things that are deliberate, and will look wrong if you skip the reasoning:

- **Memory caps are declared twice** in `docker-compose.yml` (`mem_limit` *and*
  `deploy.resources.limits.memory`). The `deploy` key is Swarm-mode; plain
  `docker compose up` may ignore it and leave containers uncapped.
- **Neo4j's page cache is pinned** to 512m. Left alone it sizes itself from the
  *host's* 16 GB rather than its own 2 g cap, and gets OOM-killed.
- **`num_ctx` is always passed explicitly.** Ollama defaults to 2048, which
  silently truncates 768-token chunks and makes extraction look randomly broken.
- **`canonical_key()` lives in exactly one place** (`src/text_utils.py`). Pipeline
  2 writes `Entity.key` with it and pipeline 3 looks entities up by it. If those
  diverge, graph retrieval returns zero paths and the system degrades to plain
  vector RAG *without raising anything*.
- **Relationship types go through `apoc.merge.relationship`.** Cypher cannot
  parameterise a relationship type, so LLM-authored relations are sanitised in
  `sanitize_relation()` before they reach the database.
- **Image bboxes are clamped to the page.** pdfplumber routinely reports regions
  extending past the media box; unclamped, `within_bbox` raises and kills the run.
- **`llama-index-graph-stores-neo4j` is intentionally absent.** The graph is
  driven by the raw `neo4j` driver; LlamaIndex owns embeddings and Qdrant only.

## Resumability

Ingestion is idempotent. PDFs are hashed, chunk IDs are deterministic UUID5s over
their content, and `data/processed/state.json` records what is done — so a crash
on page 40 of 50 never re-pays the vision cost for pages 1–39, and re-running any
command is free. Non-fatal failures land in `data/processed/dead_letter.json`
rather than aborting the run; `--status` reports the count.

## Layout

```
├── docker-compose.yml       # Qdrant + Neo4j, hard memory caps
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                 # input PDFs
│   └── processed/           # document_cache.json, state.json, dead_letter.json
└── src/
    ├── config.py            # every tunable; the only reader of os.environ
    ├── ollama_client.py     # generate + VERIFIED unload + health
    ├── text_utils.py        # hashing, chunk IDs, entity canonicalisation
    ├── state.py             # resumable run state, atomic writes, dead letters
    ├── graph_store.py       # Neo4j: schema, triple upsert, path queries
    ├── vector_store.py      # Qdrant + embeddings + chunking
    ├── pipeline_1_vision.py # PDF → text + chart/table descriptions
    ├── pipeline_2_graph.py  # chunks → Qdrant vectors + Neo4j triples
    ├── pipeline_3_chat.py   # hybrid retrieval → grounded answer
    └── main.py              # CLI, preflight checks, peak-RSS reporting
```

## Known limits

- **Scanned PDFs** have no text layer; those pages end up vision-only. An OCR
  pass would be needed for a scan-heavy corpus.
- **Triple quality is the accuracy ceiling.** `format="json"` guarantees valid
  syntax, not sensible content — expect noise on dense prose. `dead_letter.json`
  makes this measurable; the fix is a few-shot prompt, not a bigger model (there
  isn't RAM for one).
- **Ollama's unload is best-effort.** If a model refuses to leave within 30 s we
  warn and continue; that run may swap.
