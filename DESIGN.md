# topicsleuth design notes

## What the library is, and is not

topicsleuth owns the write path to a topic store: matching an item to a
topic or opening a new one, recording the observation, and measuring how
topics move. A **feeder** decides which items are worth tracking, scores
them however it likes, keeps those scores, and hands the store each
admitted item with its embedding.

The rule the library enforces: the only code that writes topics and
observations is in this package. Two feeders writing one store with
diverging matching logic is how stores fragment.

What the library deliberately does not own:

- **Items.** A feeder's item collection is the feeder's. The store touches
  one field on an item (its embedding, written at record time and read back
  for the centroid refresh) and can fetch items by URL for replays. Consumed
  markers, admission scores and routing tags never appear in library code.
- **Services.** Which database, embedding model, language model and decision
  model to use are objects the caller constructs. See "Interfaces".
- **The environment.** Library code reads no environment variables and
  never exits the process. The CLI and `topicsleuth.env` are the only
  places that do.

## Interfaces

Four `typing.Protocol`s in `topicsleuth/interfaces.py`. Structural: any
object with the right methods qualifies, no base class.

| Protocol | What it is | Default implementation |
|---|---|---|
| `Store` | topics and observations, nearest-topic search, item embeddings | `backends.mongo.MongoStore` (Atlas Vector Search); `backends.memory.MemoryStore` for tests |
| `Embedder` | text to vectors, with `dimensions` | `backends.voyage.VoyageEmbedder` |
| `Judge` | a language model asked for text: the decision, its reviewer, merge and split rewrites | `backends.anthropic.AnthropicJudge` |
| `DecisionModel` | scores one item against one candidate topic, a relationship label and a link probability, no text; optional, for the `decision` reviewer | `backends.jev.JevDecisionModel` |

Each default imports its SDK lazily and ships behind an extra
(`topicsleuth[mongo]`, `[voyage]`, `[anthropic]`, or `[all]`), so the core
package depends on pyyaml and numpy only.

Why this shape and not a plugin registry: there is one implementation per
seam today and perhaps two tomorrow. Constructor injection is the idiom for
that (httpx transports, requests adapters, Django storage backends). A
registry earns its keep when third parties ship implementations the core
has never heard of; nothing here is at that stage.

Why `MemoryStore` is a real implementation and not a test double: it is the
reference semantics for the protocol, exact cosine where a vector index
approximates, and it lets the suite run record and synthesize end to end
without a database. A store contract test runs the same assertions against
every implementation. The fakes for the other seams (`backends.fake`) are
shipped for the same reason: a consumer's tests should not need a model.

## Settings

Settings are the decisions two feeders writing one store must agree on:
the domain and the rubric prose the prompts are rendered with, the
embedding dimensionality the index was built for, how candidates are
assembled (vector limit, tag overlap, ignored routing tags), and the
link-or-propose controls (whether proposals are reviewed, by which backend,
at what threshold). They ship as packaged defaults and a feeder layers a
YAML file or a dict on top. The CLI reads that file from
`TOPICSLEUTH_SETTINGS`.

Model names are not settings. They are properties of the Judge and
Embedder a caller constructs, and the replay reports record whichever were
used. Database names and index names belong to the `MongoStore`
constructor.

## The decision

The matching decision is an LLM judgment with a labeled rubric, not a
similarity threshold, because no cosine cutoff separates link from propose
on the store it was tuned on: the right topic was in the vector top-30 for
364 of 370 linked observations, and the nearest cosine was the same (median
0.77) for items that founded a topic and items that linked to one. Every proposal is then
reviewed by an independent second pass, because a wrong link is one noisy
observation and a wrong proposal is a permanent singleton.

Everything a model is told lives in `topicsleuth/rubric.py`: what a
right-sized topic is, the relationship labels and their criteria, the
decision rules, the fit questions, the connection types, and the prompts
assembled from those pieces. A definition appears once and every prompt
that needs it renders the same text; the decision model gets the same
criteria as data. The strength maps live beside the questions they score.

The rubric has two kinds of text, and the line between them is the
library's customization boundary. The structure (labels, rules, fit
questions, connection types, response formats) is fixed in code because
the parser, the strength map and the store depend on it; changing it is a
fork. The prose that describes a field (the kinds of source, how each
kind bears on a question, what a right-sized question looks like, the
counter-examples) is `RubricConfig`, set per store under `rubric:` in
settings next to `domain`, because a judge that reasons about papers and
releases is wrong for a feed of filings and minutes. Whole-prompt
overrides were rejected: they would let a consumer break the response
format silently and would undo the single source. The prompts frame a
topic as the question it tracks, not a subject area, and read each item
by its kind so that two kinds of source on the same question land on the
same topic.

The strength rubric is three yes/no questions mapped to a score
downstream, never a number the model emits.

## Writing safely

Two feeders writing one store, and one feeder retrying after a failure,
are the same problem: a write must mean the same thing however many times
it lands. The rules the write path follows:

- **Identity.** A topic is its `topic_id`; an observation is its
  `(item_url, topic_id)`. Stores enforce both atomically (a unique index
  in Mongo) and raise `DuplicateTopic` / `DuplicateObservation`, which
  `record` treats as "already recorded": no second observation, no
  failure, reported under `already_recorded`. A proposal whose id exists
  with the same title is the same proposal landing twice and is reused; a
  different title is a collision and the item fails. Synthesis picks a
  slug no existing or batch topic uses, so collisions are rare.
- **Failures are per item and write nothing.** A judge call that fails or
  answers outside the contract, a proposal whose embedding fails, a topic
  that cannot be created, a destination that no longer exists or is
  retired, a store write that fails: each becomes an `ItemFailure` with a
  stage, in `RecordResult.failures`, and nothing is written for that item,
  including its embedding. `synthesize_items` carries failures as
  decisions with `decision="failed"` so a batch is always fully accounted
  for. A feeder acknowledges only the items outside `failed_urls()`; the
  rest are retried, and idempotency makes the retry safe.
- **The judge's answer is validated before it is written.** Decision and
  connection type must be known values, a link must name a shown
  candidate and carry the fit answers (or come from the gate or the
  reviewer), a proposal must have a title, summary and slug, and
  classifications of topics the judge was not shown are dropped before
  the gate reads them. The write boundary re-checks that the destination
  exists and is not retired.
- **History reads as history.** A snapshot counts only evidence dated at
  or before its date, snapshots are kept in date order by the store, and a
  batch dated before a topic's latest snapshot is inserted without
  changing the topic's current status. Topics founded after the batch's
  date are not status-checked by it.
- **Removal, not undo.** Every topic, observation and snapshot carries
  `recorded_at` (ingestion time, stamped by the store) and optionally the
  `run_id` the feeder passed. `delete_since` removes by those, keeps a
  topic that still holds evidence from outside the window, and restores
  nothing it did not write. Run-scoped undo of status and embedding
  changes would need a journal; this is deliberately the smaller promise.

## Tuning

`tune synthesize` replays the decision against recorded items without
writing. Every replay is leave-one-out: each item's own singleton topic
and its own title are hidden so the replay cannot see its historical
answer. That is not a flag, because a replay that can see the answer
measures nothing. A store's baselines and its
relationship label set describe that store's content, so they live with
the feeder that owns it; this repository ships the tools and synthetic
examples of the file shapes under `data/examples/`.

## Naming

The tracked thing is a **topic**: the topic detection and tracking
literature's word, where an item is a story, proposing one is first-story
detection and linking is tracking. The evidence record is an
**observation**. Reports and label sets written before the name settled
say `theme_id`; the scripts read either.
