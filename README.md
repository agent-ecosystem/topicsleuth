# topicsleuth

**topicsleuth** is a Python library for topic detection and tracking
over a stream of documents. Feed it items (papers, posts, release notes,
discussion threads) and it links each one to an existing topic or opens
a new one, keeps an append-only record of the evidence behind every
topic, and measures how each topic moves over time: how fast it is
gathering evidence, how its sources and vendors are spreading, and
whether it is maturing from proposals into shipped products. It is the
memory behind a daily briefing, built so that "remember that thing from
two weeks ago?" has an answer.

It is not a Kafka or pub/sub tool, a topic-modeling library in the LDA
sense, or a trend-detection dashboard. The closest neighbors are the
topic detection and tracking literature and tools like TopicTracker;
what topicsleuth adds is an LLM-judged link-or-propose decision with a
second-pass reviewer, a replay harness with a labeled pair set for
tuning that decision, and curation tools for merging, splitting and
moving evidence once a topic store has grown.

## Status

Early. The matching, recording, curation and tuning code has run daily
against a live store for months; the public API is new and may still
move between minor versions, so pin a release. The tracked thing is a
**topic** (the topic detection and tracking literature's word for it; an
item is a story, proposing one is first-story detection, linking is
tracking) and the evidence record is an **observation**.

## How it fits

topicsleuth owns the write path to a topic store: the topics and
observations and the nearest-topic search over topic embeddings. A
**feeder** decides which items are worth tracking and scores them however
it likes; it then hands the store each admitted item with its embedding
and lets the library decide where it belongs.

```
feeder: gate, evaluate, classify          topicsleuth: match, record, measure
┌──────────────────────────────┐          ┌──────────────────────────────────┐
│ items ──▶ scores + embedding │ ───────▶ │ candidates (vector + tag)        │
│                              │          │ link-or-propose (LLM, reviewed)  │
│ persists its own eval        │          │ observations, topics, snapshots  │
│ writes its own summary       │ ◀─────── │ status transitions               │
└──────────────────────────────┘          └──────────────────────────────────┘
```

The rule the library enforces: the only code that writes topics and
observations lives here. Two feeders writing one store with diverging
matching logic is how stores fragment.

## Install

```bash
pip install "topicsleuth[all]"                  # once released: Mongo, Voyage, Anthropic backends
pip install -e ".[all,test]"                    # from a checkout
```

Python 3.12 or newer. The core package depends only on pyyaml and numpy;
each backend is an extra (`mongo`, `voyage`, `anthropic`) and imports its
SDK lazily.

## Interfaces

Everything that talks to a service is behind a protocol in
`topicsleuth.interfaces`, and the caller supplies the implementation:

| Protocol | Role | Default |
|---|---|---|
| `Store` | topics, observations, nearest-topic search, item embeddings | `MongoStore` (Atlas Vector Search); `MemoryStore` for tests |
| `Embedder` | text to vectors | `VoyageEmbedder` |
| `Judge` | the language model that decides, reviews, and rewrites | `AnthropicJudge` |
| `DecisionModel` | optional probability-only reviewer | `JevDecisionModel` |

They are `typing.Protocol`s: anything with the right methods qualifies.
Library code reads no environment variables; `topicsleuth.env` builds the
defaults from the environment for the CLI and for feeders happy with them.

## Use from a feeder

```python
from topicsleuth.backends.anthropic import AnthropicJudge
from topicsleuth.backends.mongo import MongoStore
from topicsleuth.backends.voyage import VoyageEmbedder
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import synthesize_items

settings = load_settings(overrides={"domain": "the coding agent ecosystem"})
store = MongoStore.from_uri(uri, database="news")
embedder = VoyageEmbedder(model="voyage-4", dimensions=settings.embedding.dimensions)
judge = AnthropicJudge(model="claude-opus-4-8")

# evals: any objects with .url, .embedding and .above_threshold
decisions = synthesize_items(items, evals, store, settings, judge, embedder)
result = record(items, evals, decisions, store, settings=settings, embedder=embedder)
```

`synthesize_items` finds candidate topics for each admitted item by
vector search and tag overlap, asks the judge to link, propose or skip, and
has every proposal reviewed by an independent second pass. `record`
creates the proposed topics, writes one observation per decision, saves
item embeddings, appends a snapshot to every topic touched and applies
status transitions (emerging, active, mature, fading, dormant). With
`date_from_item=True` it dates everything by the item rather than the run,
for historical backfills.

A feeder keeps its own scores: `record` does not write them.

### Atlas Vector Search index

`MongoStore` expects a vector index on the topics collection:

- **Index name:** `topic_embedding_index` (constructor argument)
- **Field path:** `embedding`
- **Dimensions:** 1024, matching `settings.embedding.dimensions`
- **Similarity:** cosine

## Settings

Settings are the decisions two feeders writing one store must agree on,
and ship with the package in `topicsleuth/config/settings.yaml`: the
`domain` the prompts are rendered with (a noun phrase: what the store
follows), the embedding dimensionality, candidate assembly (vector limit,
tag overlap, tags to ignore because every item carries them), and the
link-or-propose controls (reviewer on or off, `judge` or `decision`
backend, its threshold). `load_settings(path, overrides)` layers a YAML
file and then a dict over the defaults.

Model names, database names and index names are not settings; they belong
to the backend objects.

## CLI

`topicsleuth` curates and tunes a store with the default backends, built
from the environment: `MONGODB_URI`, `TOPICSLEUTH_DATABASE`, and
optionally `TOPICSLEUTH_JUDGE_MODEL`, `TOPICSLEUTH_EMBEDDING_MODEL` and
`TYPESAFE_API_KEY` (see `topicsleuth/env.py`). A `.env` in the working
directory is read. Every writing command takes `--dry-run`.

```bash
topicsleuth seed topics.yaml                  # bootstrap an empty store
topicsleuth split-topic <id>                  # split an oversized topic
topicsleuth merge-topics <src> <tgt>          # retire src into tgt, rewrite tgt
topicsleuth suggest-merges --threshold 0.9    # candidate pairs by cosine
topicsleuth add-topic topic.yaml --link <url> # add a topic to a populated store
topicsleuth move-observation <url> <topic_id> # move one item's evidence
topicsleuth reprocess-tail --max-obs 1 --older-than-days 60   # clear dead singletons; lists items to re-queue
topicsleuth backfill-embeddings               # seed missing, refresh strong
topicsleuth audit --label before-x            # snapshot the store to tuning_runs/
topicsleuth tune synthesize --leave-one-out --replay <report.json>   # replay without writing
```

## Tuning

`tune synthesize` replays the link-or-propose decision against items the
store has already recorded, without writing. Use `--leave-one-out`: a
replayed item has already been recorded, so without it the topic the item
founded is a candidate with the item itself as representative evidence,
and the replay leaks its own answer. Decisions are noisy at the boundary,
so compare distributions over two seeds rather than single items.

Reports land in `tuning_runs/`. Keep the ones you treat as baselines with
the feeder that owns the store, not in this repository: they describe the
store's content. `scripts/diff_tuning_runs.py` compares two reports and
`scripts/compare_labels.py` scores rater files against a relationship
label set you build for your own store (`scripts/label_pairs_*.py` produce
rater files from a model). `data/examples/` shows every file shape with
synthetic data.

## Development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[all,test]"
.venv/bin/python -m pytest
```

The tests are hermetic. Deterministic parts (strength derivation, gate
enforcement, JSON recovery, threshold crossings, move planning, audit
bucketing) are unit tested; the record path runs end to end against
`MemoryStore`. Anything that calls a model is exercised by replays against
a real store. `DESIGN.md` records the reasoning behind the interfaces.

Releases publish to PyPI by trusted publishing from a `v*` tag
(`.github/workflows/publish.yml`).

## License

MIT.
