# topicsleuth

**topicsleuth** is a Python library for topic detection and tracking over
a stream of documents. Feed it items (papers, posts, release notes,
discussion threads) and it links each one to an existing topic or opens a
new one, keeps an append-only record of the evidence behind every topic,
and measures how each topic moves over time: how fast it is gathering
evidence, how its sources are spreading, and whether it is maturing. It is
the memory behind a daily briefing, built so that "remember that thing
from two weeks ago?" has an answer.

It is not a Kafka or pub/sub tool, a topic-modeling library in the LDA
sense, or a trend-detection dashboard. The closest neighbors are the topic
detection and tracking literature and tools like TopicTracker. What
topicsleuth adds is an LLM-judged link-or-propose decision with an
independent second-pass reviewer, a replay harness and labeled pair set
for tuning that decision, and curation tools for merging, splitting and
moving evidence once a store has grown.

## Contents

- [The problem it solves](#the-problem-it-solves)
- [How it works](#how-it-works)
- [What the models do](#what-the-models-do)
- [Data model](#data-model)
- [Install](#install)
- [Quickstart](#quickstart)
- [Settings](#settings)
- [CLI](#cli)
- [Tuning](#tuning)
- [Testing your integration](#testing-your-integration)
- [Status and versioning](#status-and-versioning)

## The problem it solves

Anything that reads a feed and writes a digest faces the same gap. Each
day's output is fresh, and nothing ties it to what came before. A
deduplicator can tell you an item is a repeat; it cannot tell you that a
paper from February, a release in June and a tutorial this morning are the
same question moving from idea to product to practice.

topicsleuth gives that stream a memory with a specific shape. The unit is a
**topic**: not a subject area ("agent failures") but a question that can
accumulate evidence over time ("how do agents fail in ways static analysis
misses?"). Every item that bears on the question becomes an
**observation** on the topic, with a strength and a connection type. Over
weeks, a topic's observations are its history, and the library measures
that history: velocity, source diversity, and a lifecycle status.

Use it when you have a stream of documents in one domain and want to
answer, reliably and cheaply, "what is this item an instance of, and what
else do we know about that?"

## How it works

```
 your feeder                              topicsleuth
 ┌────────────────────────┐               ┌─────────────────────────────────────┐
 │ gather items           │               │ 1 candidates: nearest topics by     │
 │ decide which to admit  │   items +     │   embedding, plus tag overlap       │
 │ embed each item        │ ──embeddings─▶│ 2 decide: the judge links,          │
 │ keep your own scores   │               │   proposes or skips, against a      │
 │                        │               │   labeled rubric                    │
 │ write your own summary │ ◀─decisions── │ 3 review: every proposal gets an    │
 │ from the decisions     │               │   independent second pass           │
 └────────────────────────┘               │ 4 record: observation, topic,       │
                                          │   snapshot, status transition       │
                                          └─────────────────────────────────────┘
```

**1. Candidates.** The item's embedding is searched against topic
embeddings in the store; the nearest topics are the vector candidates.
Topics sharing a tag with the item join them. This is retrieval, and it
is the only place similarity is used as a number.

**2. Decision.** The judge (a language model) sees the item and the
candidates, each with its summary and a few representative observations,
and classifies every candidate on a four-way scale: `same_question`,
`subset_question`, `adjacent_question`, `surface_overlap`. Rules then
decide: any `same_question` or `subset_question` means link; otherwise
propose a new topic if the item's question is right-sized and distinct, or
skip. The rules are enforced in code as well as in the prompt, so a model
that argues itself into a new topic despite a same-question candidate is
overridden.

Why a judgment and not a similarity cutoff: on real stores the nearest
cosine is the same for items that belong to an existing topic and items
that should found a new one. Retrieval finds the neighborhood; it cannot
tell a facet from a new question.

**3. Review.** A wrong link is one slightly noisy observation, easily moved
later. A wrong proposal is a permanent near-duplicate that splits all
future evidence. The two mistakes are not symmetric, so every proposal
clears a second, independent call that sees only the closest candidates
and the proposed topic and may confirm it, demote it to an `extends` link,
or skip it. Optionally a decision-only model scores the candidates
instead.

**4. Record.** The decision becomes an observation with a strength derived
from three yes/no fit questions (never a number the model emits), the
item's embedding is saved, and every touched topic gets a snapshot
(evidence count, source diversity, seven-day velocity, recency) and a
status check.

**The lifecycle.** A topic is `emerging` until 5 observations, `active`
until 20, then `mature`. Velocity under 0.1 observations a day marks it
`fading`; a fading topic with no recent evidence goes `dormant`; new
evidence revives either. Curation can `retire` a topic with provenance.
On well-evidenced topics, weak links are flagged `minor` so one busy
topic does not absorb everything near it.

**Embeddings stay honest.** A topic's embedding starts as its title and
summary. As evidence accumulates, at 5, 15, 30, 50 and 100 observations,
it is refreshed as a 70/30 blend of the centroid of its observations'
item embeddings and a fresh embedding of its text, so retrieval tracks
what the topic actually contains rather than how it was first described.

## What the models do

Every service is behind a `typing.Protocol` in `topicsleuth.interfaces`,
and you supply the implementation. Defaults ship behind extras.

| Protocol | Used for | Default | Calls per item |
|---|---|---|---|
| `Embedder` | item and topic embeddings; the centroid refresh | `VoyageEmbedder` | 0 to 1 (proposed topics only; your feeder embeds items) |
| `Judge` | the link-or-propose decision, the reviewer, and the title and summary rewrites that merge and split make | `AnthropicJudge` | 1, plus 1 per proposal |
| `DecisionModel` | optional: a probability-only reviewer instead of the judge's second pass | `JevDecisionModel` | 0, or 1 per shown candidate per proposal |
| `Store` | topics, observations, snapshots, nearest-topic search, item embeddings | `MongoStore` (Atlas Vector Search); `MemoryStore` for tests | |

Cost is dominated by the judge. A decision call carries the item, the
candidates with representative observations, and the rubric, typically
10 to 13k input tokens; a review adds about 4k. Budget for roughly one
strong-model call per item plus a fraction for proposals, and nothing
for items your feeder does not admit.

## Data model

Documents are plain dicts; the store owns their persistence.

**Item** (yours; the library reads it): `url`, `title`, `summary`,
optional `content_raw`, `source_name`, `source_type`, `tags`, optional
`evidence` (what kind of thing it is) and `date_published`. The store
keeps one field on it, `embedding`.

**Topic**: `topic_id` (a slug), `title`, `summary` (one sentence naming
the question), `tags`, `suggested_directions`, `embedding`, `status`,
`date_identified`, `last_updated`, `snapshots[]`, and on retirement
`retired_at` with `split_into` or `merged_into`.

**Observation**: `item_url`, `item_title`, `topic_id`, `date`, `strength`
(0 to 1), `connection_type` (`supports`, `extends`, `contradicts`,
`shifts`), `reasoning_trace`, `source_name`, `source_type`, `tags`,
`evidence`, `lanes`, `minor`, and `moved_from` when curation moved it.
Observations are append-only; curation re-points or deletes them but
never edits the judgment.

**Snapshot** (one per topic per run that touched it): `date`,
`evidence_count`, `source_diversity`, `source_type_diversity`, `velocity`,
`recency_weight`.

## Install

```bash
pip install "topicsleuth[all]"        # MongoDB Atlas, Voyage and Anthropic backends
pip install topicsleuth               # core only: bring your own backends
```

Python 3.12 or newer. The core depends on pyyaml and numpy; `mongo`,
`voyage` and `anthropic` are separate extras and import their SDKs lazily.

`MongoStore` needs an Atlas Vector Search index on the topics collection:
index name as passed to the constructor (default `topic_embedding_index`),
path `embedding`, dimensions matching your embedder, similarity cosine.

## Quickstart

A feeder owns three decisions the library does not make: which items to
admit, how to embed them, and what to do with the decisions afterwards.
Here is one, end to end.

```python
from topicsleuth.backends.anthropic import AnthropicJudge
from topicsleuth.backends.mongo import MongoStore
from topicsleuth.backends.voyage import VoyageEmbedder
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import synthesize_items

settings = load_settings(overrides={"domain": "municipal climate policy"})
store = MongoStore.from_uri(MONGODB_URI, database="briefing")
embedder = VoyageEmbedder(model="voyage-4", dimensions=settings.embedding.dimensions)
judge = AnthropicJudge(model="claude-opus-4-8")

# 1. Your items, however you gather them.
items = fetch_new_items()                      # list[dict] with url, title, summary, ...

# 2. Your admission decision and your embeddings. The library asks only for
#    .url, .embedding and .above_threshold on each eval.
class Eval:
    def __init__(self, url, embedding, admitted):
        self.url, self.embedding, self.above_threshold = url, embedding, admitted

texts = [f"{i['title']}\n\n{i.get('summary', '')}" for i in items]
vectors = embedder.embed_documents(texts)
evals = [Eval(i["url"], v, my_relevance_gate(i)) for i, v in zip(items, vectors)]

# 3. Match and record.
decisions = synthesize_items(items, evals, store, settings, judge, embedder)
result = record(items, evals, decisions, store, settings=settings, embedder=embedder)

# 4. Whatever you do with the outcome is yours: a digest, a dashboard, a
#    consumed marker on the item so it is not processed twice.
for obs in result.observations_created:
    print(obs["item_title"], "->", obs["topic_id"], obs["connection_type"], obs["strength"])
for topic in result.topics_created:
    print("new topic:", topic["title"])
for t in result.transitions:
    print(t["topic_id"], t["from_status"], "->", t["to_status"])
```

`synthesize_items` processes admitted items in order; a topic proposed for
an earlier item in the batch is a candidate for later ones, so a burst of
items on one new question yields one topic. `record` writes topics first,
then observations, then snapshots, and returns what it did. It does not
write your scores or your markers; persist those yourself.

For a historical backfill, pass `date_from_item=True` to `record`:
observations and snapshots are dated by the item, not the run, and a
topic's founding date follows its earliest evidence. Process oldest first
so snapshots accumulate in sequence.

A feeder that is happy with the defaults can build them from the
environment with `topicsleuth.env.backends_from_env(settings)`, which is
what the CLI does.

## Settings

Settings hold the decisions every feeder writing one store must share.
They ship as packaged defaults in `topicsleuth/config/settings.yaml`, and
`load_settings(path, overrides)` layers a YAML file and then a dict over
them. Model names, database names and index names are not settings; they
belong to the backend objects you construct.

```yaml
domain: the field this store follows   # rendered into the prompts; set it

embedding:
  dimensions: 1024                     # what the store's vector index was built for

candidates:
  tag_overlap: true                    # tag-matching topics join the vector candidates
  ignored_tags: []                     # tags every item carries (a feed's routing tag), which
                                       # would otherwise match every topic that copied them
  vector_limit: 30                     # nearest topics shown per item
  vector_num_candidates: 200           # approximate-search breadth, for stores that take it

synthesize:
  confirm_proposals: true              # the independent second pass on every proposal
  confirm_candidates: 5                # how many nearest candidates the reviewer sees
  confirm_backend: judge               # judge | decision
  confirm_link_threshold: 0.7          # decision backend: p(belongs) that demotes a proposal
```

Set `domain` to a noun phrase for what the store follows. The prompts say
"a topic store that follows {domain}", and that one phrase is all the
domain knowledge they carry.

## CLI

`topicsleuth` curates and tunes a store with the default backends, built
from the environment: `MONGODB_URI`, `TOPICSLEUTH_DATABASE`, and optionally
`TOPICSLEUTH_JUDGE_MODEL`, `TOPICSLEUTH_EMBEDDING_MODEL` and
`TYPESAFE_API_KEY` (see `topicsleuth/env.py`). A `.env` in the working
directory is read. Every writing command takes `--dry-run`.

```bash
topicsleuth seed topics.yaml                  # bootstrap an empty store from a topics: list
topicsleuth audit --label before-x            # snapshot the store to tuning_runs/
topicsleuth suggest-merges --threshold 0.9    # near-duplicate pairs by cosine
topicsleuth merge-topics <src> <tgt>          # retire src into tgt; the judge rewrites tgt
topicsleuth split-topic <id>                  # the judge proposes sub-topics and reassigns
topicsleuth add-topic topic.yaml --link <url> # add a topic by hand, moving named evidence onto it
topicsleuth move-observation <url> <topic_id> # move one item's evidence; empties retire
topicsleuth reprocess-tail --max-obs 1 --older-than-days 60   # clear dead singletons, list items to re-queue
topicsleuth backfill-embeddings               # seed missing topic embeddings, refresh strong ones
topicsleuth clean --hours 28                  # undo a bad run's store output
topicsleuth tune synthesize --leave-one-out --replay <report.json>   # replay the decision, see Tuning
```

Curation never asks the judge to re-decide an observation: merging
re-points evidence and rewrites the surviving topic's description;
splitting asks the judge to propose sub-topics and reassign; add and move
keep strength and connection type and annotate the trace with where the
evidence came from. Every retired topic keeps `merged_into` or
`split_into`, so links to it can be resolved forever.

## Tuning

The decision is a judgment, and judgments drift. A prompt edit, a model
upgrade, a change to the candidate limit or the reviewer threshold can
move the proposal rate by ten points without any error being raised, and
a proposal rate that is too high fragments the store silently: velocity
spreads across near-duplicate topics and nothing surfaces. Treat the
decision the way you would treat a classifier: baseline it, change one
thing, measure.

**Replay.** `tune synthesize` re-runs the decision on items the store has
already recorded, without writing, and produces a report: the decision
distribution, strength buckets, connection types, and per-call
diagnostics (candidate counts, which channel the linked topic came from,
its vector rank, token usage, reviewer outcome).

```bash
topicsleuth tune synthesize --leave-one-out --sample-query '{"my_feed": true}' --n 30 --seed 42 --label baseline
topicsleuth tune synthesize --leave-one-out --replay tuning_runs/synthesize-baseline-n30-seed42.json --label after-change
python scripts/diff_tuning_runs.py tuning_runs/synthesize-baseline-n30-seed42.json tuning_runs/synthesize-after-change-replay.json
```

Three rules make a replay mean something:

- **Always `--leave-one-out`.** A recorded item's own topic is in the
  store, often with the item as its only evidence. Without hiding it, the
  replay sees its historical answer and reports a proposal rate far below
  the live one.
- **Replay the same items.** `--replay` reuses a prior report's items so
  the diff is item by item. `--sample-query` draws a fresh seeded sample
  (store-specific syntax; a MongoDB filter for `MongoStore`).
- **Two seeds, not one.** About a third of items sit on the link-or-propose
  boundary and flip between runs at default temperature. Compare
  distributions over two seeds; do not read single-item changes as
  regressions.

**Baselines belong with your feeder.** Reports describe your store's
content, so commit the ones you treat as reference in the repository that
owns the store, not here. `data/examples/replay.example.json` shows the
shape.

**Labeled pairs.** When the question is whether the rubric itself is right,
not whether a change moved the numbers, build a relationship label set:
item-candidate pairs labeled `same_question`, `subset_question`,
`adjacent_question` or `surface_overlap`, blind to what the model decided.
Stratify toward the hard region (singletons against their nearest
topic, proposals against their closest candidate). `scripts/label_pairs_*.py`
produce rater files from a model and `scripts/compare_labels.py` reports
agreement, Cohen's kappa and confusion matrices between raters and
against the first pass. `data/examples/` shows every file shape.

**Curate what tuning cannot fix.** No prompt declines to make a topic out
of a genuinely one-off item. Run `audit` periodically, `suggest-merges`
after model changes, and `reprocess-tail` to clear singletons old enough
to be dead so their items are re-homed against a fuller store.

## Testing your integration

`topicsleuth.backends.fake` ships a hashing embedder, a scripted judge and
a scripted decision model. With `MemoryStore` they run the entire decision
and record path offline, so your feeder's tests need no model and no
database:

```python
from topicsleuth.backends.fake import HashEmbedder, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore

store, embedder = MemoryStore(), HashEmbedder(dimensions=8)
judge = ScriptedJudge([{"decision": "link_existing", "topic_id": "my-topic", "connection_type": "supports",
                        "candidate_classifications": [{"topic_id": "my-topic", "label": "same_question"}],
                        "fit_questions": {"q1_scope_expansion": False, "q2_representative_citation": True,
                                          "q3_distinguishing_evidence": True}, "reasoning_trace": "test"}])
```

The CLI selects the same fakes with `TOPICSLEUTH_STORE=memory`,
`TOPICSLEUTH_JUDGE=scripted`, `TOPICSLEUTH_EMBEDDER=hash` and
`TOPICSLEUTH_DECISION=scripted`. Any `Store` you write can run the
library's own contract test (`tests/test_store_contract.py`); the Mongo
half runs when `TOPICSLEUTH_TEST_MONGODB_URI` names a cluster.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[all,test]"
.venv/bin/python -m pytest
```

## Status and versioning

Early. The matching, recording, curation and tuning code has run daily
against a live store for months; the public API is new and may still move
between minor versions, so pin a release. `DESIGN.md` records the
reasoning behind the interfaces and the decision. Releases publish to
PyPI from a `v*` tag.

The tracked thing is a **topic** because that is the topic detection and
tracking literature's word for it: an item is a story, proposing one is
first-story detection, linking is tracking. The evidence record is an
**observation**.

## License

MIT.
