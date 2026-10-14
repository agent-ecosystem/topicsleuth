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
| `DecisionModel` | structured questions in, probabilities out; optional, for the `decision` reviewer | `backends.jev.JevDecisionModel` |

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
without a database.

## Settings

Settings are the decisions two feeders writing one store must agree on:
the embedding dimensionality the index was built for, how candidates are
assembled (vector limit, tag overlap, ignored routing tags), and the
link-or-propose controls (whether proposals are reviewed, by which backend,
at what threshold). They ship as packaged defaults and a feeder layers a
YAML file or a dict on top.

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

The prompts are rendered with the domain the store follows
(`settings.domain`). They frame a topic as the question it tracks, not a
subject area, and read each item by its kind (a paper proposes or measures, a release
ships, a tutorial shows use) so that a release and a paper on the same
question land on the same topic.

The strength rubric is three yes/no questions mapped to a score
downstream, never a number the model emits.

## Tuning

`tune synthesize` replays the decision against recorded items without
writing, with `--leave-one-out` hiding each item's own singleton topic so
the replay cannot see its historical answer. A store's baselines and its
relationship label set describe that store's content, so they live with
the feeder that owns it; this repository ships the tools and synthetic
examples of the file shapes under `data/examples/`.

## Naming

The tracked thing is a **topic**: the topic detection and tracking
literature's word, where an item is a story, proposing one is first-story
detection and linking is tracking. The evidence record is an
**observation**. Reports and label sets written before the name settled
say `theme_id`; the scripts read either.
