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
package depends on pyyaml, numpy and python-dotenv only.

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
assembled (vector limit, tag overlap, ignored routing tags), the
link-or-propose controls (whether proposals are reviewed, by which backend,
at what threshold, and the floor under a document's secondary links), and
the write-stage floor that marks weak observations on well-evidenced
topics as minor. They ship as packaged defaults and a feeder layers a
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

- **Identity.** A topic is its `topic_id`. An observation is its
  `(item_url, topic_id)`: an item may be evidence for several topics, once
  each. A link decision carries every topic the judge rated `same_question`
  (or, failing that, `subset_question`), each with its own fit answers,
  strength and connection type, and `record` writes one observation per
  link. Stores enforce both identities atomically (unique indexes in Mongo)
  on insert and update, and raise `DuplicateTopic` / `DuplicateObservation`.
  A proposal whose id exists with the same title is reused; a different
  title is a collision and the item fails. Synthesis picks a slug no
  existing or batch topic uses.
- **The stored observations are the decision.** `synthesize_items` does
  not call the judge for an item the store holds, and `record` reports it
  under `already_recorded` with the stored outcome and applies none of
  this batch's links for it, so a retry cannot choose different topics or
  open a second one, and two feeders deciding differently about one
  document leave the first writer's observations. The loser of that race
  removes the empty topic it opened. Every observation carries the whole
  decision it was part of, under `decision_links`, so the intended link
  set is durable from the first write: a retry reads it back from any one
  observation and writes the links that are missing, and an item is not
  clear of `failed_urls()` until every intended link is in the store.
  Whether a later batch may add links to a recorded item is a separate
  policy, and the answer today is no.
- **The plan is read against curation's history.** Curation does not
  rewrite `decision_links`. Every observation keeps a cumulative
  `removed_from` list: move, merge, fold, split and reprocess append the
  topic the item was taken off, reprocess writes the deleted topic onto
  the item's other observations, and a fold carries the folded
  observation's history and plan onto the survivor, so a link only it
  carried, approved and not yet written, is still intended, and one
  removed on purpose is still not. A topic in that history is never
  revived, however many steps ago it was removed. A planned topic the
  item is not yet on is followed through merges to its live successor,
  and dropped if that successor is already present or the topic was
  deleted or retired by a split. A topic that is only marked `deleting`
  is blocked, not removed: the link stays intended and recording fails
  the item with that reason until the mark is resolved, so a decision
  never looks complete because part of it was temporarily unwritable. An
  observation found on a retired merge source (it landed as the topic was
  merged, too late for the merge's own sweep) stands for the live
  successor: before treating the item as recorded, a run moves it there,
  folded into the item's observation on the successor if it has one. If
  the successor is itself leased for deletion, the observation waits
  where it is as the durable record of the decision and the item fails as
  blocked; if there is no live successor, the observation is withdrawn
  and the item fails, to be judged again. Where an observation sits is
  never taken as proof that its decision is complete.
- **Failures are per item, and a retry finishes the work.** A judge call
  that fails or answers outside the contract, a proposal whose embedding
  fails, a topic that cannot be created, a destination that does not
  exist or is retired, a store write that fails at any stage (the
  observation, a repair of a stored one, the item embedding, the snapshot,
  the status, the refresh): each becomes an `ItemFailure` with its stage
  in `RecordResult.failures`, and the other items in the batch go on.
  `synthesize_items` carries failures as decisions with
  `decision="failed"` so a batch is always fully accounted for. Nothing is
  written before the observation for an item that fails earlier; a failure
  after it leaves the observation in place, and the retry, finding it,
  resumes what is missing rather than inserting again. A feeder
  acknowledges only the items outside `failed_urls()`.
- **The judge's answer is validated before it is written.** Decision and
  connection type must be known values, a link must name a shown
  candidate and carry the fit answers (or come from the gate or the
  reviewer), a proposal must have a title, summary and slug, and
  classifications of topics the judge was not shown are dropped before
  the gate reads them. The write boundary re-checks that the destination
  exists and is not retired.
- **History reads as history.** Snapshots are event-time history, not a
  record of what was known when. A snapshot counts only evidence dated at
  or before its date, snapshots are kept in date order by the store, and a
  batch dated before a topic's latest snapshot is inserted without
  changing the topic's current status; every later snapshot is then
  recomputed as of its own date, keeping its `recorded_at` and `run_id`,
  so the series and the latest count the judge is shown include the
  backfill. Topics founded after the batch's date are not status-checked
  by it.
- **Maintenance holds a lease, and recording checks after it writes.**
  Merge, split, move, reprocess and clean may run while a feeder records.
  Merge and split retire their source (with `merged_into` or
  `split_into`) and carry along anything that landed on it before the
  mark. Clean and reprocess take a deletion lease on a topic before
  touching it: the topic is marked `deleting` with its owner, an expiry
  (`DELETING_LEASE_SECONDS`, fifteen minutes) and the status to restore,
  and the store owns the lease. Acquiring, renewing and releasing are
  each one conditional store operation (`acquire_deletion_lease`,
  `release_deletion_lease`), so two callers racing for a topic are
  ordered by the store and one loses. Every destructive write an owner
  makes names its lease: `delete_topic` is one conditional delete, and
  `delete_observations_for_topic` runs, in Mongo, in a transaction that
  writes the topic under the lease filter, so a lease that changed hands
  before the delete commits means nothing is deleted. A live lease is
  another operation at work and is left entirely alone, document,
  observations and snapshots, by clean, by reprocess and by recovery;
  only an expired one may be taken over, and the takeover is the same
  atomic acquisition. `recover_interrupted_deletions` takes over expired
  leases, restores a marked topic that still holds evidence and deletes
  one that does not, and the clean and reprocess commands run it first.
  Overlapping maintenance on one topic is refused, not serialized: the
  later operation reports the topic as busy. Clean leases each topic it
  will write to, an old topic with new evidence or snapshots included,
  just before those writes, makes them all under that lease (in Mongo,
  one transaction per topic), keeps a topic that still holds evidence
  from outside the window and restores its status, and deletes or gives
  back the topic under the same lease, writing nothing to it after.
  Recording refuses a retired or leased destination before the insert and
  re-reads it after. If the topic was merged meanwhile the observation
  follows the merge, folded into the item's observation on the successor
  if it has one; if it was retired by a split, leased or deleted, the
  observation is withdrawn and the item fails, to be re-judged next run.
  An insert whose post-check beat the mark is counted by the maintenance
  side, because the mark precedes the count; one that did not is refused
  by the post-check. Curation never writes onto a leased topic: move,
  merge and split refuse a `deleting` destination, and
  `update_observation` makes the liveness check and the write one
  operation (`DestinationNotLive`), so a lease that lands between a
  caller's check and its write still wins. A fold, an item already on the
  target of a move or merge, is one store operation too
  (`fold_observation`): the destination's liveness, the survivor's update
  and the source's deletion commit together or not at all, and the store
  computes what the survivor keeps (the union of the two plans, the two
  histories) from both observations as they are when the write commits,
  so two folds into one survivor each keep what the other added. The
  reference store serializes every operation under one lock, which is
  what its atomicity means across threads, and a store adapter is held to
  the same: the contract tests exercise the guarded operations against a
  lease that has already changed hands. An adapter needs conditional
  writes and, where one operation touches two documents, a transaction;
  nothing else.
- **Derived state retries until it lands.** The snapshot, the later
  snapshots a backfill recomputes, the status, the item embedding and the
  topic's embedding refresh are each derived from the store and safe to
  repeat. A failure in any of them fails the items that caused it, and the
  retry, finding the observation, redoes whichever is missing. What is
  missing is read from the store, never from what the failed call
  remembered:
  - the snapshot is owed until one of the topic postdates the
    observation's arrival there, by `recorded_at` or, when a repair moved
    the observation onto the topic, by `relocated_at`, stamped in the same
    write as the move;
  - a threshold refresh is owed whenever a threshold lies between the
    count the embedding was last refreshed at (`embedding_refreshed_count`,
    written with the embedding) and the count now; a count that fell
    below that mark owes a refresh at the next threshold from zero;
  - an item's embedding is owed to its topic as soon as its observation is
    in the store, and its save is attempted whatever else failed for the
    item; an input a refresh could not load is owed until a refresh that
    included it has been written. Every refresh records the inputs it was
    computed from under `embedding_inputs`; that list, not the time the
    refresh finished, is its claim to completeness, and a relocated
    observation is answered only by a refresh whose inputs include it;
  - `embedding_refresh_pending` and `embedding_refresh_missing` record a
    refresh that failed, was deferred for want of an embedder, or
    published without inputs it wanted. Registering them is one store
    operation (`defer_topic_refresh`) that merges with what other batches
    registered.

  The refresh runs after the batch's item embeddings are saved, computes
  from the inputs it can load, and publishes through one compare-and-swap
  store operation (`publish_topic_embedding`) on `embedding_revision`,
  which every embedding write and every registration bumps: the write
  lands only if nothing was published or registered since the refresh
  read the topic, and it sets the obligation, to exactly what the refresh
  could not load, in the same write. A refresh that loses that race
  recomputes from the current state; one that loses it three times
  reports a conflict rather than publish, and the items that asked for it
  fail and retry. So an older refresh can neither overwrite a newer result
  nor clear work registered after it read, a partial result never claims
  completeness, and the writer that saves an omitted input finds the
  refresh still owed.
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
**observation**.
