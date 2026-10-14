# topicsleuth

You follow a field. Documents arrive every day: papers, release notes,
posts, discussion threads. topicsleuth remembers them for you. It works out
which ongoing question each new document belongs to, keeps the evidence
for every question in one place, and tells you which questions are
gathering momentum, settling, or going quiet. It is the memory behind a
daily briefing, so that "what happened to that thing from two weeks ago?"
has an answer.

If that is the problem you have, the next section gets you running. The
rest of this page explains what the library does and how, so you can
decide whether it fits before you commit to it.

## Try it in five minutes

```bash
pip install "topicsleuth[all]"
```

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

items = fetch_new_documents()        # your dicts: url, title, summary, source_type ...

class Eval:                          # what the library needs per document
    def __init__(self, url, embedding, admitted):
        self.url, self.embedding, self.above_threshold = url, embedding, admitted

vectors = embedder.embed_documents([f"{d['title']}\n\n{d.get('summary', '')}" for d in items])
evals = [Eval(d["url"], v, True) for d, v in zip(items, vectors)]

decisions = synthesize_items(items, evals, store, settings, judge, embedder)
result = record(items, evals, decisions, store, settings=settings, embedder=embedder)

for obs in result.observations_created:
    print(f"{obs['item_title']} -> {obs['topic_id']} ({obs['connection_type']})")
```

You need Python 3.12, a MongoDB Atlas cluster with a vector search index
on the topics collection (path `embedding`, cosine, dimensions matching
your embedding model, named `topic_embedding_index`), and API keys for
Voyage and Anthropic in your environment. Other databases and models work
too; see "Bring your own models and database" below.

## Keep track of a field across days, not just today

Anyone who reads a feed and writes a summary ends up with the same thing:
a good account of today, and no connection to yesterday. The paper from
February, the product launch in June and the tutorial this morning are
one story, an idea becoming a product becoming practice. A daily summary
cannot tell that story because it has no memory, and a duplicate detector
cannot help because none of those three documents is a duplicate of the
others.

What is missing is a notebook with one page per question. Not "security",
which is a subject, but a question the field keeps working on, such as
"do coding assistants make developers write less secure code?" A question
is the right size for a page: specific enough that a document clearly
belongs on it or does not, broad enough to gather evidence for months.

topicsleuth keeps that notebook. It calls a page a **topic**, and an entry
on a page an **observation**: which document, how strongly it bears on the
question, and in what way, whether it supports what the page already says,
extends it, contradicts it, or marks a shift. When a document arrives that
raises a question with no page yet, it opens one. And because the entries
on each page are dated, it can measure what a summary never could: how
fast a question is gathering evidence, from how many different sources,
and whether it is just emerging, active, mature, fading, or dormant.

## Know whether it fits before you build on it

topicsleuth earns its place when three things are true. You have a steady
stream of documents about one field. You want continuity across days, not
only a digest of each day. And you are willing to pay for one call to a
strong language model per document you choose to track, because the
judgment of where a document belongs is made by a model, and that is where
the cost goes. You will also need a database with vector search; MongoDB
Atlas works out of the box, and anything else needs a small adapter.

It is the wrong tool if you want to discover clusters in a corpus you
already have (that is topic modeling, and this is not that), if you want a
dashboard or alerts (this produces the record a dashboard would read), or
if you were searching for something to do with Kafka topics.

## How a document finds its topic

Follow one document through the library: a changelog entry announcing that
a coding tool now runs commands in a sandbox.

**First, the library looks nearby.** Every topic and every document has an
embedding, a vector that places its meaning so that similar things sit
close together. The library pulls the topics nearest to the document,
around thirty, and adds any that share a tag with it. This is quick and
cheap, and it only finds the neighborhood; it does not decide anything.

**Then a language model decides.** Being near a topic is not the same as
belonging to it. A document can sit right beside a topic and be about a
different question. So the library shows a language model, which it calls
the **judge**, the document and the candidate topics, each with its
one-sentence summary and a few of the entries already on it, and asks a
precise question about each candidate: is the document's question the same
as this one, a narrower case of it, a neighbor of it, or does it only share
vocabulary? If any candidate is the same question or a narrower case, the
document goes there. If none is, the judge may propose a new topic, or
decide the document is interesting but not the start of anything.

The sandbox changelog lands on "are coding tools isolating what agents
execute by default?", even though every other entry on that page is a
paper and this is a release note. That is the point of the whole design:
a topic gathers evidence of every kind, and the question is what they
share.

**New topics get a second opinion.** Putting a document on a slightly
wrong page costs one noisy entry, easily moved later. Opening a page that
should have been part of an existing one is worse: from then on, evidence
splits between the two and nothing repairs it on its own. Because the two
mistakes are not equal, every proposed topic goes to a second, independent
model call that sees only the closest existing topics and the proposal,
and can confirm it, fold the document into an existing topic instead, or
drop it.

**Finally, it is written down.** The entry is recorded with a strength
that comes from three yes-or-no questions the judge answers about fit,
never a number the model invents. The page's statistics are refreshed,
and if the topic has crossed a threshold, its status changes.

That is the whole loop. Everything else in the library is setup, tuning,
or tidying.

## Keep the judge honest

The decision at the heart of the library is a judgment, and judgments
drift. Edit the prompt, upgrade the model, change how many candidates the
judge sees, and the share of documents that open new topics can move by
ten points with no error anywhere. Too many new topics is the quiet
failure: evidence spreads across near-duplicate pages, each looks slow,
and nothing surfaces in your briefing.

So treat the judge the way you would treat any classifier: take a
baseline, change one thing, measure. The library makes this cheap by
replaying the decision on documents the store already holds, without
writing anything:

```bash
# A baseline on 30 documents already in the store.
topicsleuth tune synthesize --leave-one-out --sample-query '{"source": "my-feed"}' --n 30 --seed 42 --label baseline

# After a change, the very same documents, then the diff.
topicsleuth tune synthesize --leave-one-out --replay tuning_runs/synthesize-baseline-n30-seed42.json --label after
python scripts/diff_tuning_runs.py tuning_runs/synthesize-baseline-n30-seed42.json tuning_runs/synthesize-after-replay.json
```

The report tells you how many documents linked, proposed or skipped, how
strong the links were, and for each call where the chosen topic came from
and how large the prompt was. The diff tells you which documents changed
their minds.

Three habits make the numbers mean something. Always use
`--leave-one-out`, because a document already in the store would otherwise
be shown the very topic it founded and the replay would flatter itself.
Replay the same documents, so the comparison is one to one. And run two
seeds before believing a change, because a fair share of documents sit
right on the boundary between "link" and "new topic" and flip on their own
from run to run.

When the question is whether the judge's rubric is right at all, rather
than whether a change moved it, build a small labeled set: pairs of a
document and a candidate topic, labeled by a person or an independent
model without seeing what the judge said. The `scripts/` folder can
produce such labels from a model and compare raters against each other
and against the judge. Keep your baselines and label sets with the code
that owns your store, since they describe its content; `data/examples/`
shows the shape of every file.

## Keep the store tidy

No judge declines to open a page for a document that is genuinely a
one-off, and a notebook that has grown for months needs editing. The
`topicsleuth` command does that against a store configured through the
environment (`MONGODB_URI`, `TOPICSLEUTH_DATABASE`, and optionally the
model names; a `.env` file in the working directory is read). Every
command that writes takes `--dry-run` and shows the plan first.

```bash
topicsleuth audit                               # how big the store is and how evidence is spread
topicsleuth suggest-merges                      # pairs of topics that look like the same question
topicsleuth merge-topics <source> <target>      # fold one into the other; the judge rewrites the survivor
topicsleuth split-topic <id>                    # the judge proposes sub-topics and reassigns the entries
topicsleuth add-topic topic.yaml --link <url>   # open a page by hand and move named entries onto it
topicsleuth move-observation <url> <topic_id>   # move one document's entry to another page
topicsleuth reprocess-tail --max-obs 1 --older-than-days 60   # clear pages that never got a second entry
topicsleuth seed topics.yaml                    # start an empty store from a list of topics
topicsleuth clean --hours 28                    # undo a bad run
```

Nothing here re-judges a document. Merging and moving keep each entry's
strength and connection type and note where it came from, and a retired
topic keeps a pointer to what replaced it, so old links still resolve.

## Bring your own models and database

topicsleuth never fetches documents and never decides which ones matter.
That is the job of your own program, which this page calls the **feeder**:
it gathers documents, chooses which to track, embeds them, and hands them
over, then does what it likes with the outcome. The library owns the
notebook, and it is the only code that writes to it, so two feeders
sharing one store cannot disagree about where things belong.

Every service the library talks to is an object you construct and pass
in: the embedding model, the language model that judges, the database,
and optionally a decision-only model that reviews proposals in place of
the judge's second call. Each is a small `typing.Protocol` in
`topicsleuth.interfaces`, with the methods listed there; anything that has
them works, with no base class to inherit. Defaults for Voyage, Anthropic,
MongoDB Atlas and TypeSafe Jev are included as extras (`voyage`,
`anthropic`, `mongo`), and the core package itself depends only on pyyaml
and numpy.

A few decisions have to be shared by everyone writing to one store, and
those live in settings rather than code: the domain phrase the prompts are
written for, the embedding dimensionality the index was built with, how
candidates are assembled, and whether and how proposals are reviewed. The
defaults ship with the package and `load_settings(path, overrides)` layers
yours on top:

```yaml
domain: the field this store follows   # the only domain knowledge the prompts carry
embedding:
  dimensions: 1024                     # match your embedding model and your index
candidates:
  tag_overlap: true                    # topics sharing a tag with the document join the candidates
  ignored_tags: []                     # a tag every document carries would match everything; list it here
  vector_limit: 30                     # how many nearest topics the judge sees
synthesize:
  confirm_proposals: true              # the second opinion on every new topic
  confirm_candidates: 5                # how many nearest topics the reviewer sees
  confirm_backend: judge               # judge | decision
  confirm_link_threshold: 0.7          # decision backend: how sure it must be to fold a proposal in
```

Loading history rather than today's documents? Pass `date_from_item=True`
to `record` so entries are dated by publication rather than by the run,
and load oldest first.

## Test without spending anything

The package ships stand-ins for every service: an embedder that hashes
text to a stable vector, a judge that returns scripted answers, a decision
model driven by a table, and an in-memory store. Together they run the
whole loop with no network, so a feeder's tests are ordinary unit tests.

```python
from topicsleuth.backends.fake import HashEmbedder, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore
```

The `topicsleuth` command uses the same stand-ins when
`TOPICSLEUTH_STORE=memory`, `TOPICSLEUTH_JUDGE=scripted` and
`TOPICSLEUTH_EMBEDDER=hash` are set, which is how its own tests run. If
you write a store adapter, the library's store contract test will run
against it.

## Where things stand

Early. The matching, recording, curation and tuning code has run daily
against a live store for months, but the public interface is new and may
move between minor versions, so pin a release. Releases are published to
PyPI from version tags, and `DESIGN.md` explains the reasoning behind the
interfaces and the decision for anyone who wants to go deeper.

The name follows the topic detection and tracking literature, where the
tracked thing is a topic, a document is a story, noticing a new topic is
first-story detection, and attaching a document to one is tracking.

MIT licensed.
