# Architecture, explained from scratch

This document is for someone who has **not** worked with AI systems, databases, or
Python. It explains what this project is, why each piece exists, and what every
file in it does. No prior knowledge is assumed. Technical terms are introduced one
at a time, in plain English, the first time they appear.

If you only read one section, read [The thirty-second version](#the-thirty-second-version).

---

## Table of contents

1. [The thirty-second version](#the-thirty-second-version)
2. [The problem this solves](#the-problem-this-solves)
3. [Five ideas you need, in plain English](#five-ideas-you-need-in-plain-english)
4. [The one constraint that shapes everything](#the-one-constraint-that-shapes-everything)
5. [The system as three separate jobs](#the-system-as-three-separate-jobs)
6. [The cast: every file, and what it does](#the-cast-every-file-and-what-it-does)
7. [Following one question all the way through](#following-one-question-all-the-way-through)
8. [Where everything is stored](#where-everything-is-stored)
9. [What happens when things go wrong](#what-happens-when-things-go-wrong)
10. [How you actually use it](#how-you-actually-use-it)
11. [Honest limitations](#honest-limitations)
12. [Glossary](#glossary)

---

## The thirty-second version

You give this system a PDF — say, a twenty-one-page government report full of
tables and charts. It reads the whole thing, including the pictures, and files
everything away. Afterwards you can ask it questions in ordinary English, and it
answers using only what was actually in that document, telling you which page each
fact came from.

The two things that make it unusual:

- **It reads the pictures.** Most systems of this kind throw away charts and tables,
  because they are images, not text. This one shows each chart to an AI model that
  can see, gets back a written description of it, and files that description as if
  it had been printed text all along. A number that only ever existed inside a bar
  chart is still findable afterwards.

- **It runs entirely on one laptop.** Nothing is uploaded anywhere. No cloud service,
  no API key, no data leaving the machine. Everything — the AI models, the databases —
  runs locally on an Apple MacBook with 16 GB of memory. That single fact is
  responsible for most of the design decisions in this repository, and the rest of
  this document keeps coming back to it.

---

## The problem this solves

Imagine you have a stack of dense PDF reports and a question: *"What was the trend
in tax revenue, and what explains it?"*

Three bad options exist today.

**Option one: search for keywords.** You type "tax revenue" and get every page
containing those two words — including the twenty pages where they appear in
passing. Worse, if the report phrases it as "fiscal receipts," keyword search finds
nothing at all. It matches letters, not meaning.

**Option two: paste it into a chatbot.** This works until the document is too long
to fit, which is almost immediately. And a chatbot that *doesn't* know the answer
will often invent a plausible-sounding one rather than admit it. This is called
**hallucination**, and in a document where the whole point is the specific numbers,
a confidently invented number is worse than no answer.

**Option three: send it to a cloud AI service.** Fast and capable — but the document
leaves your machine. For financial, legal, medical, or internal corporate documents,
that is frequently not allowed.

This project is the fourth option. It reads the documents thoroughly once, stores
what it learned in two complementary ways, and then answers questions by looking
things up and quoting from what it found — on your own hardware, with the page
numbers attached so you can check it.

---

## Five ideas you need, in plain English

You do not need to know how any of these work internally. You need to know what
each one *is for*.

### 1. A large language model (LLM)

An AI that reads text and writes text. Here, it plays three distinct roles:

- it looks at a chart and writes a description of it,
- it reads a paragraph and pulls out the facts as short statements,
- it reads a question plus some supporting material and writes the answer.

Think of it as an extremely well-read assistant who is fast, tireless, and — this
matters — occasionally confidently wrong. Much of this system's design is about
constraining that assistant so it can only answer from material you have put in
front of it.

The models used here are **`qwen2.5vl:3b`** (the one that can see pictures) and
**`llama3.1`** (the one that handles text). They are files on disk, run by a local
program called **Ollama**, which is covered below.

### 2. An embedding, and why "meaning search" beats keyword search

An **embedding** turns a passage of text into a long list of numbers — 384 of them
here — that represents its *meaning*. Passages about similar things end up with
similar numbers, even when they share no words.

The useful analogy is a map. Imagine every paragraph ever written being assigned a
coordinate, so that paragraphs about tax policy all cluster in one region and
paragraphs about rainfall cluster somewhere far away. Now your question gets its
own coordinate on the same map, and you simply fetch whatever is nearest.

That is why asking about "tax revenue" can retrieve a passage that only ever said
"fiscal receipts" — they sit near each other on the map, because they *mean* nearly
the same thing. Keyword search cannot do this. The model that draws this map here
is called **`bge-small-en-v1.5`**.

### 3. A vector database

The place those coordinates live, and the thing that answers "what is nearest to
this point?" quickly, across tens of thousands of passages. Here it is a program
called **Qdrant**, running in the background on the laptop.

### 4. A knowledge graph

A completely different way of storing the same material. Instead of passages, you
store **facts as connections**.

Picture a detective's corkboard. Pins are named things — a company, a person, a
year, a figure. Strings between pins are stated relationships:

```
   [Germany] ──REPORTED_REVENUE_OF──> [DM 847 billion]
       │
       └──HAS_TAX_TYPE──> [Corporation tax] ──DECLINED_IN──> [1996]
```

A corkboard answers a question that a pile of paragraphs cannot: *"what connects to
what?"* If a question requires chaining two facts together — fact A is on page 3,
fact B is on page 19, and neither page alone answers you — the corkboard finds the
route between them. This is the "Graph" in **GraphRAG**. The program holding the
corkboard is **Neo4j**.

### 5. RAG — the technique the whole project is named after

**RAG** stands for *Retrieval-Augmented Generation*, which is a clumsy name for a
simple and sensible idea:

> Don't ask the AI what it knows. Go and find the relevant material first, hand it
> over, and tell the AI to answer from *that* and nothing else.

It is the difference between an open-book exam and a closed-book one. In a
closed-book exam, a student who doesn't know the answer guesses — that is
hallucination. In an open-book exam, they look it up and cite the page.

**GraphRAG** is RAG where the "going and finding" uses both the meaning-map *and*
the corkboard, and combines what both turn up. That is this project.

---

## The one constraint that shapes everything

The laptop has **16 GB of memory**. The components this system needs are large:

| Component | Roughly how much memory |
|---|---|
| The vision model, which reads charts | 3–8 GB depending on which one |
| The text model, `llama3.1` | ~5 GB |
| The embedding model | ~0.13 GB |
| Qdrant and Neo4j, the two databases | 1.5 GB + 2 GB, capped |
| macOS itself, and everything else you have open | several GB |

Add it up and it does not fit. If you try to run them all simultaneously, the Mac
starts **swapping** — shuffling memory out to disk to make room — and everything
slows to a crawl, sometimes by a factor of ten or more.

So the system is built around a rule: **only one AI model is in memory at any
moment.** Each stage loads its model, does its work, and then *verifiably* evicts
it before the next stage starts.

The word "verifiably" is doing real work in that sentence. Asking the model-runner
to unload a model is a *request*, not a guarantee — it may comply late or not at
all. So after asking, the system repeatedly checks what is actually resident in
memory until the model has genuinely disappeared, waits five more seconds for macOS
to reclaim the space, and only then proceeds. If it never disappears within thirty
seconds, the system says so out loud rather than quietly carrying on and bogging
down.

This constraint explains almost every unusual choice in this codebase. Whenever
something looks over-engineered, the reason is nearly always "sixteen gigabytes."

---

## The system as three separate jobs

The work is split into three stages, run as three separate commands. They are not
chained together inside one program, and that separation is deliberate — the reason
is in [Stage boundaries](#stage-boundaries-why-separate-programs) below.

```
   STAGE 1: READ                STAGE 2: FILE               STAGE 3: ANSWER
   ─────────────                ─────────────               ───────────────
                                                            
   PDF                          page text                   your question
    │                            │                           │
    ├─ pull out the text         ├─ cut into chunks          ├─ find similar passages
    ├─ find every chart/table    ├─ draw the meaning-map     ├─ walk the corkboard
    ├─ describe each one         ├─ pin facts to corkboard   ├─ combine both
    └─ staple descriptions       └─ store both               └─ write a cited answer
       back into the page           
                                                            
   uses: the vision model       uses: the text model        uses: the text model
                                      + embedding model            + embedding model
    ↓                            ↓                           ↓
   a JSON file of page text     Qdrant + Neo4j              an answer on your screen
```

### Stage 1 — Read the document, including the pictures

A PDF is opened and gone through page by page. The ordinary text is pulled out
directly. Then the page is scanned for anything that is a *picture* rather than
text — a chart, a table, a diagram.

Finding those regions is harder than it sounds, because PDFs are untidy. The
detection works in four passes: collect declared images; ask the PDF library to
find tables; discard any region sitting almost entirely inside another so nothing
is processed twice; then apply a cheap but effective test called **digit density** —
what percentage of the characters in this region are digits? A numeric table runs
around 25% or higher. Ordinary prose runs below 5%. Anything under 15% is dropped
as a false positive. And if nothing survives the filter but the page *as a whole* is
more than 20% digits, the system gives up on fine-grained detection and sends the
entire page to the vision model, on the reasoning that a page that numeric is
almost certainly a table it failed to outline properly.

Each surviving region is cropped out as a picture and sent to the vision model with
this instruction:

> *"Analyze this visual element. If it's a chart, describe the trends and specific
> numerical data points. If it's a table, format it as Markdown."*

The description that comes back is stapled into the page's text, right where the
picture was, under a visible marker:

```
[VISUAL ELEMENT 1 - table]
| Year | Tax revenue | Change |
|------|-------------|--------|
| 1995 | DM 814.5bn  | +2.1%  |
...
```

From that point onward, the chart *is text*. Everything downstream treats it like
any other paragraph — which is precisely the point.

Progress is written to disk every five pages. If the run crashes on page 40 of 50,
the first 39 pages do not have to be re-read, and the vision model is the slowest,
most expensive part of the whole system, so that matters.

### Stage 2 — File everything away, twice

The page text from Stage 1 gets filed in two different places, for the two different
kinds of lookup described earlier.

First it is cut into **chunks** of roughly 768 tokens (a token is about three-quarters
of a word), with a 96-token overlap between consecutive chunks. The overlap exists so
a fact that happens to straddle a boundary isn't sliced in half and lost from both
sides.

Each chunk then gets a permanent ID, and goes to both stores:

**Into Qdrant (the meaning-map):** the chunk is turned into its 384 numbers and
stored, along with a note of which file and page it came from.

**Into Neo4j (the corkboard):** the text model reads the chunk and extracts facts as
three-part statements — *subject, relationship, object*. The instruction it is given
is strict:

> *"Extract only facts stated in the text. Invent nothing."*

Subjects and objects must be specific named things, never pronouns and never whole
sentences. Relationships must be short, like `ACQUIRED` or `REPORTED_REVENUE_OF`.
If the model returns something malformed, it is handed its own bad output back once
and asked to fix it; if that fails too, the chunk's facts are skipped and the
failure logged.

Two cleanup steps matter here:

**Names are normalised.** "Apple Inc.", "apple  inc", and "The Apple Inc" must all
end up as *one* pin on the corkboard, not three. So every name is run through a
standardising function that strips accents, punctuation, leading "the/a/an", and
trailing corporate suffixes (`Inc`, `Ltd`, `GmbH`, `AG`, `SA`, and so on). All three
become `apple`. There is a careful detail: dots are removed *before* general
punctuation, specifically so "Nestlé S.A." and "Nestle SA" converge on the same pin
rather than splitting into two.

**Relationships are sanitised.** The relationship name comes from an AI model, and
it is inserted into a database command. Unvalidated text going into a database
command is the classic security hole, so the text is forcibly stripped down to
capital letters, digits, and underscores before it gets anywhere near the database.

#### The one detail worth understanding: the chunk ID

Every chunk gets an ID computed from its own contents — the document's fingerprint,
the page number, and a fingerprint of the chunk's text. Same text, same ID, always,
on any machine, forever.

This single design choice buys four separate things:

1. **No duplicates on re-runs.** Re-filing the same chunk overwrites the existing
   entry rather than creating a second copy, because it lands under the same ID.
2. **Resumability.** The system keeps a list of IDs already filed and skips them.
3. **A ledger of what's done**, independent of whether the databases are up.
4. **The join between the two stores.** This is the big one. That same chunk ID is
   written onto the Qdrant entry *and* onto every corkboard pin and every string
   derived from that chunk.

That fourth point is what makes the whole system more than the sum of its halves.
Because every pin on the corkboard carries the IDs of the passages it came from,
you can find a fact by following connections and then **walk back to the exact
paragraph that stated it** — and show it to the user as a citation. Facts are never
floating free of their source.

### Stage 3 — Answer a question

You ask something in ordinary English. Four things happen.

**A — Search the meaning-map.** Your question is turned into its own 384 numbers and
the three nearest passages come back from Qdrant.

**B — Walk the corkboard.** Separately, the text model is asked to name the entities
in your question. Those names are run through the *same* normalising function from
Stage 2 — this is essential, and is explained below — and Neo4j returns the
connection paths radiating out from them, up to two hops away. Then the reverse trip:
using the chunk IDs written on those pins, the system fetches the original passages
behind the connections it found, and adds up to two of them to the pile.

That reverse trip is what makes a layered question answerable. Take: *"What was the
revenue trend for the body that collects corporation tax?"* The meaning-map alone
would probably miss it, because the passage with the revenue figure never mentions
the phrase you used. But the corkboard knows which body collects corporation tax,
and the chunk IDs on that pin lead straight to the passage with the number.

**C — Combine into one briefing.** Both sets of findings are assembled into a single
block of text: a numbered list of passages, each labelled with its filename and page
number, followed by the connection paths rendered as readable lines like
`Germany -[REPORTED_REVENUE_OF]-> DM 847bn`. If the combined block is too long for
the model to read, it is trimmed, with room deliberately reserved for your question
and the answer.

**D — Write the answer under strict orders.** The whole briefing goes to the text
model with this instruction:

> *"You answer strictly from the supplied context. If the context does not contain
> the answer, say so plainly and name what is missing — never fill the gap from
> general knowledge. Cite sources as (filename p.N) when the context provides them."*

This is the anti-hallucination contract, and it is the point of the entire
architecture. Everything before this step exists to make that instruction possible
to follow.

### Stage boundaries: why separate programs

The three stages are run as separate operating-system processes, rather than three
function calls inside one program. There are two reasons, both about memory.

The first: if an unload silently fails, chaining stages in one program would start
Stage 2 while Stage 1's model is still occupying several gigabytes. Separate
processes make that impossible.

The second is subtler. Python is reluctant to hand memory back to the operating
system once it has claimed it. Over a long run, the AI libraries, the embedding
model, and the PDF buffers accumulate into a footprint that never shrinks, even
after the data is no longer needed. **Ending the process is the only reliable way to
get that memory back.** So each stage ends, genuinely, before the next begins.

---

## The cast: every file, and what it does

Thirteen files of code, plus three configuration files. They fall into four groups.

```
                    main.py          ← you type commands here
                       │
                       ├──────────────────────┐
                       │                      │
                 orchestrator.py        (or go straight to one stage)
                 "run all three"              │
                       │                      │
         ┌─────────────┼──────────────────────┤
         │             │                      │
   pipeline_1     pipeline_2            pipeline_3
   _vision.py     _graph.py             _chat.py
   READ           FILE                  ANSWER
         │             │                      │
         └─────────────┴──────────┬───────────┘
                                  │
     ┌──────────┬─────────────┬───┴──────┬────────────┐
     │          │             │          │            │
 ollama_   vector_store   graph_store  state.py   text_utils.py
 client.py  (Qdrant)       (Neo4j)     (progress)  (naming/IDs)
     │          │             │          │            │
     └──────────┴─────────────┴──────────┴────────────┘
                              │
                          config.py      ← every setting, in one place
```

Read it top to bottom: your command enters at the top, the work happens in the
middle, and the bottom row is shared equipment everything borrows.

---

### Group 1 — The front door

#### `src/main.py` — the control panel

**What it is:** the program you actually run. It reads the command you typed, checks
the equipment is plugged in, and hands the job to exactly one of the stages.

There are six main commands, and you use exactly one at a time:

| Command | What it does for you |
|---|---|
| `--run` | Everything: read, file, and optionally answer a question at the end |
| `--ingest <pdf>` | Stage 1 only — read one PDF |
| `--build-index` | Stage 2 only — file away whatever has been read |
| `--chat "question"` | Ask one question. Omit the question for a back-and-forth session |
| `--status` | Report what has been processed, how much is stored, what went wrong |
| `--reset` | Wipe everything that was filed. Your original PDFs are **not** touched |

Plus a few modifiers: `--force` (re-read a PDF already marked done), `--rebuild`
(re-file everything from scratch), `--show-context` (print the briefing material
before the answer, so you can see exactly what the model was working from), and `-v`
(verbose, for debugging).

**One behaviour worth calling out.** Before anything expensive starts, it checks
every piece of equipment — is the AI runner alive, are both databases reachable? —
and if several are down, it tells you about **all of them at once** rather than
failing on the first, making you fix it, and then failing on the second.

It also loads nothing it doesn't need: `--status` doesn't load the heavyweight AI
libraries at all, so asking "what's the state of things?" is instant rather than
taking twenty seconds.

**Analogy:** the control panel on a machine, with six big buttons that can't be
pressed at once, a few small switches that modify them, and a habit of walking round
the back to check every cable before it starts the motor.

#### `src/orchestrator.py` — the shift supervisor

**What it is:** what runs when you type `--run`. It performs the three stages in
order, as separate programs, with memory checks between them.

The sequence:

1. Work out which PDFs to process — one file, a folder, or everything in `data/raw/`.
2. **Read each PDF**, as its own separate program run. If one PDF fails, it logs the
   failure and *continues with the rest* — one corrupt file costs you one document,
   not the whole batch. After each one, check memory is clear.
3. If *every* PDF failed, stop here and say so. There is nothing to file.
4. **Now** start the databases, if they aren't already running. Not before —
   deliberately. The vision model may be using many gigabytes, and adding a database
   on top of that on a 16 GB machine guarantees swapping. Stage 1 doesn't need the
   databases, so they stay off until Stage 1 is finished. If the databases can't be
   started, stop: this *is* fatal, because filing is impossible without them.
5. **File everything.** If this fails, stop — there is no point asking questions of
   an empty index.
6. **Answer your question**, if you supplied one.

At the end it prints a table showing each stage, whether it succeeded, and how long
it took. A partially successful run is always legible rather than silently odd.

A nice touch: on macOS, Docker (which runs the databases) is an application, not a
background service, so the command-line tool cannot start it. The orchestrator
detects this and launches the app itself — because discovering the problem *after* a
forty-minute document read would waste all of it.

**Analogy:** a kitchen with one oven and one cook. Three dishes, but the oven fits
one at a time — so each dish is cooked completely, taken out, *the oven is checked to
be actually empty*, and only then does the next go in. The dishwasher stays off until
the biggest dish is out, because the kitchen's electrics can't run both.

---

### Group 2 — The three stages

#### `src/pipeline_1_vision.py` — read the document, including the pictures

**What it is:** Stage 1, described above. Opens the PDF, pulls out text, finds and
crops every chart and table, has the vision model describe each one, staples the
descriptions back into the page.

Some specifics worth knowing:

- Cropped pictures are kept **in memory only** and never written to disk — they are
  created, sent, and discarded.
- Regions smaller than about half an inch are dropped: those are horizontal rules,
  bullet points, and logos, and paying a vision model to describe a bullet point is
  pure waste.
- At most eight visuals per page, so one pathological page can't stall an entire run.
- The word-separation sensitivity is deliberately tightened from the PDF library's
  default, because at the default setting tightly-spaced PDFs come out as
  `MonthlyReport` and `August1997` — fused words that wreck both the meaning-map and
  the fact extraction.
- Page regions are clamped to the page boundary. PDFs routinely claim an image
  extends past the edge of the paper; cropping to that claim crashes the library and
  kills the run.
- If one crop or one description fails, it is logged and skipped. **One bad picture
  must not kill a two-hundred-page run.**
- The vision model is unloaded on the way out *no matter how the stage ends*,
  including after a crash. A comment in the code calls this non-negotiable, and on a
  16 GB machine it is.

**Analogy:** a research assistant going through a printed report with scissors. Every
chart they find, they snip out and hand to a colleague who's good at reading graphs;
the colleague dictates back "this shows revenue rising from 2.1 to 3.4 billion," and
the assistant staples that dictation onto the page where the picture was. From then
on the report can be *read* rather than *looked at*. They photocopy their progress
every five pages in case of interruption, and they send the expensive colleague home
the moment the last page is done.

#### `src/pipeline_2_graph.py` — file everything away, twice

**What it is:** Stage 2, described above. Chunks the text, embeds each chunk into
Qdrant, extracts facts into Neo4j, and ties the two together by chunk ID.

One design decision here deserves highlighting, because it looks like an
inconsistency until you see the reasoning. The two filing operations fail
differently:

- If the **meaning-map** filing fails, the chunk is abandoned entirely. An
  un-embedded chunk is invisible to search; there is no point pretending otherwise.
- If the **fact extraction** fails, the chunk still counts as successfully filed.
  Why? Because it is already findable by meaning. Only the bonus — the corkboard
  connections — was lost. Marking the whole chunk as failed would force a re-run of
  work that genuinely succeeded.

Both failures are written to a log file either way, so nothing is lost silently.

**Analogy:** a librarian who reads each page twice. The first read is for *feel* —
they file it in a meaning-based catalogue, so later you can find "pages like this
question" even when no word matches. The second read is for *facts* — they jot index
cards saying "Apple → ACQUIRED → Beats" and pin string between the names on a
corkboard. Every card and every string has the shelf number of its source page
written on it. And if the fact-reading fails on some page, that page still stays in
the meaning catalogue, because half a filing beats none.

#### `src/pipeline_3_chat.py` — answer the question

**What it is:** Stage 3, described above. Searches both stores, combines the results,
and has the model write a cited answer from them.

The standout behaviour here is **graceful degradation**. Every retrieval step can
fail without taking the answer down with it:

- Meaning-search unreachable → carry on with zero passages.
- Corkboard unreachable → carry on with zero connections.
- The model fails to name the entities in your question → fall back to simply
  grabbing the capitalised words from your question. The code's comment: *"Better a
  weak graph probe than none at all."*

The system answers with whatever it managed to gather, and the instruction it gives
the model — *say so plainly if the answer isn't here* — means degraded retrieval
produces an honest "I don't have that" rather than a fabrication.

It comes in two flavours: one question and done, or an interactive session where you
ask repeatedly. In the interactive version the model deliberately stays loaded
between questions, because reloading five gigabytes before each question would make
conversation unbearable. It is evicted once, when you leave.

**Analogy:** asking a research librarian a question and getting two helpers at once.
One skims the stacks for paragraphs that *sound like* your question. The other walks
to the corkboard and traces string from the names you mentioned — "you asked about
corporation tax; that's tied to this collecting body, which is tied to this 1996
figure" — then fetches the exact pages those strings came from. Both sets of findings
get stapled into one folder, the folder is trimmed to fit on the desk, and it goes to
a writer under one rule: *answer only from this folder, and if the answer isn't in
here, say so and say what's missing.*

---

### Group 3 — Shared equipment

These five files do no work of their own. They are the tools the stages borrow.

#### `src/ollama_client.py` — the phone line to the AI models

**What it is:** the single place in the project that talks to **Ollama**, the local
program that actually runs the AI models. Everything that needs a model goes through
here: ask a question, check which models are installed, check which are currently
loaded in memory, and — most importantly — unload one and *confirm* it is gone.

Its error handling is unusually deliberate, and distinguishes three cases:

- **The model refused the request outright** (wrong architecture, not enough memory).
  Raise the error immediately, do not retry. An architecture mismatch will not fix
  itself on a second attempt, and it reads the model-runner's actual explanation
  rather than a useless generic "500 Server Error."
- **The request timed out.** Also no retry — retrying replays the identical input and
  will time out identically.
- **The connection itself glitched.** *This* is worth retrying, up to twice, waiting
  a little longer each time.

The unload function is the memory-management heart of the project. It asks for the
model to be evicted, then **polls until the model has actually disappeared from the
list of what's resident** — up to thirty seconds — rather than sleeping and hoping.
Then it waits five more seconds for macOS to genuinely reclaim the space. It reports
truthfully whether eviction was confirmed.

**Analogy:** a single shared phone line to one very large, very forgetful expert who
takes up the whole room. You can ask a question, check whether they're in the
building, and — crucially — escort them out and *stand at the door watching until
they're actually gone*, because the next expert cannot fit while they're still inside.

#### `src/vector_store.py` — the meaning-map keeper

**What it is:** everything to do with Qdrant and embeddings. Turning text into its
384 numbers, storing them, searching for nearest neighbours, counting what's stored,
and wiping it all for `--reset`.

It also does the chunking — cutting page text into 768-token pieces with 96-token
overlap.

A small but telling detail: the embedding model is loaded **lazily**, only at the
moment it is first needed. That is specifically so `--status` and `--reset` don't pay
a model-loading cost for operations that never embed anything.

Two directions of lookup are supported: *"find me passages similar to this"* (used
for your question) and *"fetch me these exact passages by ID"* (used for the reverse
trip from the corkboard back to source text). Both return results in the same shape,
which is what lets Stage 3 simply pile them together.

**Analogy:** a library where every paragraph has been given a coordinate based on its
meaning. Ask a question, the question gets its own coordinate, and the library hands
you whatever paragraphs sit nearest — even when they share no words with what you
asked.

#### `src/graph_store.py` — the corkboard keeper

**What it is:** everything to do with Neo4j. Creating the structure, writing facts,
walking connections at query time, counting, and wiping.

The structure is simple on purpose. There is exactly **one kind of pin**, called an
`Entity`, with three things written on it: its standardised key (used for matching),
its display name (used for showing humans), and the list of chunk IDs it came from.
The **strings between pins are whatever the model named them** — `ACQUIRED`,
`HEADQUARTERED_IN`, `REPORTED_REVENUE_OF` — after sanitising. Each string also carries
its own list of source chunk IDs.

Two guardrails:

- A **uniqueness rule** on the entity key, enforced by the database itself. Without
  it, a second filing run would duplicate every single pin on the board.
- The traversal depth is **clamped between one and three hops** before it is used.
  The depth has to be written into the query as literal text rather than passed as a
  parameter — the query language doesn't permit a parameter there — and that clamp is
  what makes it safe.

This file also supplies the two numbers `--status` reports about the graph, including
a warning if any pin has *no* source chunk IDs attached, which would mean the
fact-to-source link is broken for that pin.

**Analogy:** the detective's corkboard itself. Pins are named things, strings are
stated relationships, and on the back of every pin and every string is the page
number it came from — so you can always walk back to the original source.

#### `src/state.py` — the site logbook

**What it is:** the memory of what has already been done, so a crash doesn't cost you
the work you already paid for. It tracks which documents have been read and how far,
which chunks have been filed, and keeps a separate log of things that went wrong
without stopping the run.

Its most important property is **crash-safe writing**. Rather than writing directly
over a file — which leaves it corrupt if the power cuts mid-write — it writes to a
temporary file first and then swaps it into place in a single, indivisible operation.
Either the old version or the new version exists; never a half-written mixture.

The failure log is called a **dead letter** file, borrowing a term from postal
services: mail that couldn't be delivered but was kept rather than thrown away. Each
entry records which stage failed, when, on which page, and the exact error. Running
`--status` tells you how many there are.

**Analogy:** a builder's site logbook. Before starting a day's work you check which
floors are already finished and skip them. Each entry is written on a loose sheet and
only stapled into the book once complete, so a dropped pen never ruins the record.
Anything that went wrong but didn't stop work goes on a separate snag list.

#### `src/text_utils.py` — the naming desk

**What it is:** the smallest file, and arguably the most important. It does three
things: fingerprint files and text, generate chunk IDs, and standardise names.

This file depends on *nothing* else in the project — not even the settings file. That
is deliberate: it sits at the very bottom of the stack, so everything can safely
depend on it.

**The project's single most safety-critical rule lives here.** Stage 2 uses the name-
standardising function to decide what to call each pin on the corkboard. Stage 3 uses
*the same function* to look pins up. If those two ever drifted apart — if one stripped
"Ltd" and the other didn't — then lookups would stop matching, the corkboard would
return nothing, and the system would quietly fall back to plain meaning-search
**without raising a single error**. It would just get subtly worse, silently, forever.
Keeping that function in exactly one place is what prevents it.

**Analogy:** the desk where a library catalogues donations. "The Apple Inc.", "apple
inc", and "Apple" all get filed under one card, so the shelf doesn't end up with three
entries for the same company. And every page gets a barcode generated from its own
contents, so scanning the same page twice never creates a duplicate record.

---

### Group 4 — Settings and setup

#### `src/config.py` — one place for every adjustable number

**What it is:** every setting in the project, in a single file. Addresses, model
names, chunk sizes, timeouts, thresholds. It is the only file that reads your
personal settings file.

A sample of what lives here and why it matters:

| Setting | Default | Why it is what it is |
|---|---|---|
| Vision model | `qwen2.5vl:3b` | The one that can see pictures |
| Text model | `llama3.1` | Fact extraction and answer writing |
| Embedding model | `bge-small-en-v1.5` | Small on purpose, so it can coexist with the text model |
| Context size | 8192 | How much text the model sees per call — see below |
| Chunk size / overlap | 768 / 96 | Passage granularity; overlap stops facts being cut in half |
| Passages per question | 3 | How many meaning-matches come back |
| Graph hops | 2 | How far from a named entity to walk the corkboard |
| Crop resolution | 150 DPI | Sharp enough to read a chart, small enough not to blow up memory |
| Minimum digit density | 15% | The test separating real numeric tables from multi-column prose |
| Unload timeout | 30s | How long to wait for a model to actually leave memory |
| Text timeout | 180s | A chunk takes ~50s; past 3 minutes the model is stuck, not slow |

That context-size setting illustrates the genre of problem this file exists to
prevent. Left unset, the model runner defaults to a much smaller value that would
**silently cut chunks short** — producing not an error but randomly degraded fact
extraction, which would be nearly impossible to diagnose. So it is always stated
explicitly.

**Analogy:** the fuse box and address book of the house, combined. Every appliance's
rating and every phone number on one board by the door — so changing something means
editing one board, not hunting through every room.

#### `.env` and `.env.example` — your personal settings

`.env.example` is a template with safe defaults, which you copy to `.env` during
setup. `.env` is **never committed to version control**, because it is where
passwords would go.

One honest note: the template covers most settings but not all. The timeout,
crop-filter, and graph-traversal values can all be set in `.env` and will be
respected — they are simply not listed in the template, so you would not discover
them without reading `config.py`.

#### `docker-compose.yml` — starting the two databases

**What it is:** the recipe for starting Qdrant and Neo4j, each in its own isolated
container, with hard memory ceilings.

Three things in it look odd and are not:

1. **The memory cap is declared twice for each database**, in two different formats.
   Depending on your Docker version and mode, one or the other may be ignored. Both
   are written so that *something* applies. The setup instructions have you verify it
   actually took effect — if the limits read 4 GB instead of 1.5 and 2, the cap didn't
   apply and a long run will destabilise the machine.

2. **Neo4j's internal cache is pinned to a fixed size.** Left alone, Neo4j sizes its
   cache from the *host machine's* 16 GB rather than from its own 2 GB container
   allowance, exceeds that allowance, and gets killed.

3. **A plugin called APOC is explicitly installed and permitted.** This is not
   optional decoration. The query language cannot accept a relationship type as a
   parameter, and APOC provides the only way to create a relationship whose type is
   decided at runtime. Since the relationship types here come from an AI model, they
   are necessarily decided at runtime. Without APOC, the corkboard cannot be built
   at all.

Note that the AI runner, Ollama, is deliberately **not** containerised — it runs
directly on the Mac, so the models get the host's full memory rather than the
container system's much smaller allowance.

#### `requirements.txt` — the shopping list

The Python libraries needed: the RAG framework, the PDF reader and its rendering
backend, the two database clients, and a few small utilities.

The interesting entry is one that **isn't there**. A ready-made package exists for
connecting this RAG framework to Neo4j, and it is deliberately not used. The graph is
driven directly instead, so that name standardisation, the uniqueness rule, and the
chunk-ID link back to source text all stay under this project's control rather than
being handled invisibly by someone else's library. The framework is used for
embeddings and Qdrant only.

**Analogy:** a shopping list that also says "do NOT buy the pre-made sauce" — because
the whole point of the dish is that the sauce is made by hand.

#### `src/__init__.py` — the nameplate

Three lines. It marks the folder as a proper Python package so the files inside can
refer to each other, and records the version number. It carries no logic.

**Analogy:** the nameplate on an office door. No furniture, but without it the post
never finds the room.

---

## Following one question all the way through

Here is the complete journey, using the document actually sitting in this repository:
a twenty-one-page Deutsche Bundesbank monthly report on German tax revenue from
August 1997.

**Setting up, once:**

1. You run `--ingest` on the PDF. It is fingerprinted so the system knows this exact
   file. Twenty-one pages are read. On most pages the text comes out directly; on the
   numeric ones, tables are detected, cropped, and described by the vision model.
   Progress is saved every five pages.
2. You run `--build-index`. The page text is cut into chunks. **Twenty-nine chunks**
   result. Each is embedded into Qdrant and its facts pinned onto the Neo4j corkboard.
   Every chunk, pin, and string carries the chunk's permanent ID.

**Asking, any time afterwards:**

3. You ask: *"What happened to corporation tax revenue, and why?"*
4. **Meaning-search:** your question is embedded and the three closest passages come
   back — possibly including one whose text came out of a *table image* in step 1,
   which plain text extraction would never have found.
5. **Corkboard walk:** `llama3.1` picks out "corporation tax" as the entity. It is
   standardised to the same key used when filing. Neo4j returns connection paths up
   to two hops out. Using the chunk IDs on those pins, up to two more source passages
   are pulled back.
6. **Combine:** everything becomes one briefing — passages labelled
   `[1] (1997-08-tax-revenue-data.pdf p.7 score=0.834)`, followed by connection paths.
7. **Answer:** the model writes a reply using only that briefing, citing
   `(1997-08-tax-revenue-data.pdf p.7)`. If the briefing doesn't contain the answer,
   it says so and names what is missing — it does not reach for what it half-remembers
   about German tax policy.
8. The model is unloaded. Memory is free.

---

## Where everything is stored

Nothing goes to any server. Four storage locations, all on your machine:

| Where | What's in it | Survives `--reset`? |
|---|---|---|
| `data/raw/` | Your original PDFs | **Yes** — never touched |
| `data/processed/document_cache.json` | Page text plus the vision descriptions | No |
| `data/processed/state.json` | What's been done; which chunks are filed | No |
| `data/processed/dead_letter.json` | Things that failed without stopping the run | No |
| Qdrant (in a container) | The meaning-map: 384 numbers per chunk | No |
| Neo4j (in a container) | The corkboard: entities and connections | No |

`--reset` requires you to type the word `reset` to confirm, unless you explicitly
pass `--yes`.

A real example of what the state file looks like after processing the Bundesbank
report: one document, 21 pages, 1 visual element, status `complete`, and a list of
29 chunk IDs.

---

## What happens when things go wrong

This is where the design shows its intent most clearly. The guiding principle:

> **A failure that doesn't have to stop the run, doesn't — but it is never hidden.**

| What fails | What happens |
|---|---|
| One chart can't be cropped or described | Logged to the failure file, move to the next. One bad picture doesn't kill a 200-page run |
| The fact extraction returns gibberish | Handed back to the model once to fix. If still bad, logged and skipped — the chunk stays searchable by meaning |
| One PDF in a batch fails entirely | Logged, continue with the remaining PDFs |
| The connection to the AI runner glitches | Retried twice with increasing waits |
| The AI runner rejects the request outright | Fail immediately with its real explanation — retrying won't help |
| A request times out | Fail immediately — retrying replays identical input |
| The databases aren't running | Started automatically, with up to two minutes to come up |
| The databases can't be started at all | **Stop.** Filing is impossible; continuing would be pointless |
| Meaning-search is down at question time | Answer from corkboard results alone |
| Corkboard is down at question time | Answer from meaning-search alone |
| A model won't leave memory within 30s | **Warn out loud** and continue — that run may be slow |
| You press Ctrl-C | Models are unloaded, and you are told that progress to the last save is kept |

A worked example from this repository's own failure log: during the Bundesbank run,
pages 7 and 20 failed vision processing with an internal error from the model runner.
The other nineteen pages processed normally. Both failures are recorded with the page
number, the region, the timestamp, and the full error text, and `--status` reports the
count. The run completed; the document is usable; and the two gaps are documented
rather than invisible.

---

## How you actually use it

The whole thing, in one command:

```bash
python src/main.py --run data/raw/report.pdf
python src/main.py --run                                  # every PDF in data/raw/
python src/main.py --run report.pdf --ask "Q3 revenue?"   # and answer at the end
```

Or drive each stage by hand:

```bash
python src/main.py --ingest data/raw/report.pdf   # 1: read, including charts
python src/main.py --build-index                  # 2: file it away
python src/main.py --chat "What drove the revenue decline?"
python src/main.py --chat                         # interactive session
python src/main.py --status                       # what's been done
python src/main.py --reset                        # wipe the index (not the PDFs)
```

You can also look at the stored data directly in a browser: the corkboard at
`localhost:7474` and the meaning-map at `localhost:6333/dashboard`.

If you want to see the system's working rather than just its conclusion, add
`--show-context` — it prints the exact briefing material the model was given before
it prints the answer. For checking whether an answer is trustworthy, this is the most
useful switch available.

---

## Honest limitations

Four things this system does not do well, stated plainly.

**Scanned documents.** A PDF that is really a photograph of paper has no text layer
at all. Those pages become vision-only, and quality varies. Handling a scan-heavy
collection properly would need a text-recognition pass that isn't built yet.

**Fact quality is the ceiling on accuracy.** The corkboard is only as good as what the
text model extracted. Forcing the model to return well-formed output guarantees the
*shape* is valid — it does not guarantee the *content* is sensible. On dense prose,
expect some noise. The failure log makes this measurable rather than mysterious. The
fix would be better instructions with worked examples, not a larger model — there
isn't memory for a larger model.

**Eviction is best-effort.** If a model refuses to leave memory within thirty seconds,
the system warns and continues. That run may be slow.

**Documentation drift.** The README still names `llama3.2-vision` as the vision model
and does its memory arithmetic around an eleven-billion-parameter model. The code
actually defaults to `qwen2.5vl:3b`, because the former cannot be loaded at all by the
installed version of the model runner — a comment in `config.py` records exactly this.
Following the README's setup steps literally would download roughly 8 GB of a model
the system never loads. The argument the README makes is still directionally right;
the specific numbers are stale.

---

## Glossary

Every term introduced in this document, in one place.

**APOC** — an add-on for Neo4j. Required here because it is the only way to create a
connection whose type is decided while the program is running — necessary, since the
types come from an AI model.

**Chunk** — a passage of roughly 768 tokens that a page is cut into, so it fits
comfortably inside what a model can read at once.

**Chunk ID** — a permanent identifier computed from a chunk's own contents. The same
text always produces the same ID. It is what links the meaning-map and the corkboard
together.

**Container** — an isolated box a program runs inside, with its own strict memory
allowance. Both databases run in containers; the AI runner deliberately does not.

**Dead letter** — a record of something that failed but didn't stop the run. The term
is borrowed from postal services: undeliverable mail that is kept rather than
discarded.

**Digit density** — the percentage of characters in a region that are digits. A cheap,
effective way to tell a numeric table (25%+) from ordinary prose (under 5%).

**Docker** — the software that runs containers.

**Embedding** — a list of 384 numbers representing the *meaning* of a passage, so that
similar meanings end up with similar numbers.

**Entity** — a named thing on the corkboard: a company, a person, a year, a figure.

**GraphRAG** — RAG where the retrieval step uses both a meaning-map and a knowledge
graph, and combines what each turns up.

**Hallucination** — when an AI model produces a confident, plausible, wrong answer
rather than admitting it doesn't know. Preventing this is the point of the whole
architecture.

**Knowledge graph** — the corkboard: facts stored as named things connected by named
relationships.

**LLM (large language model)** — an AI that reads and writes text.

**Neo4j** — the database program holding the knowledge graph.

**Ollama** — the program that runs AI models locally on your machine.

**Qdrant** — the database program holding the embeddings.

**RAG (Retrieval-Augmented Generation)** — find the relevant material first, hand it
to the model, and require the answer to come from it. An open-book exam rather than a
closed-book one.

**Swapping** — when a computer runs out of memory and shuffles data out to disk to
cope. Everything becomes dramatically slower. Avoiding this is why only one AI model
is ever loaded at a time.

**Token** — roughly three-quarters of a word; the unit in which AI models measure
text.

**Triple** — a fact in three parts: subject, relationship, object. For example,
*Germany — REPORTED_REVENUE_OF — DM 847bn*.

**Vector database** — a database that answers "what is nearest to this point?" quickly
across very many points. Here, Qdrant.

---

*For the mechanical, per-function walkthrough — exact control flow, the data shapes at
every boundary, the database queries, and the full failure taxonomy — see
[`TECHNICAL.md`](TECHNICAL.md). For setup and daily use, see [`README.md`](README.md).*
