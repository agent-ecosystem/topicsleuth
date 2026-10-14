# topicsleuth

Topic detection and tracking for a stream of documents. Give it the papers,
posts, release notes and discussions you follow, and it keeps track of the
questions they keep coming back to: which document belongs to which
question, what evidence has piled up on each, and which questions are
heating up, settling, or going quiet.

## Why this exists

Suppose you follow a field. Every day new documents arrive: a paper, a
changelog, a blog post, a long discussion thread. You can read them, or
have something summarize them, and either way you end up with a digest of
today. Tomorrow you get another digest of tomorrow.

What you lose is the thread between days. The paper from February, the
product release in June and the tutorial this morning are all the same
story: an idea becoming a product becoming practice. A daily digest cannot
see that, because it has no memory. Tools that remove duplicates do not
help; they can tell you a document is a repeat, not that it is a new
chapter in something you have been watching.

topicsleuth is that memory. It was built to sit behind a daily briefing so
that when a reader thinks "wait, what happened to that thing from two weeks
ago?", there is an answer.

## What it does

Think of a notebook with one page per **topic**. A topic is not a subject
like "security"; it is a question the field keeps working on, such as "do
coding assistants make developers write less secure code?" Questions are
the right size for a page: specific enough that documents clearly belong
or do not, broad enough to collect evidence for months.

When a new document arrives, topicsleuth finds the page it belongs on, or
decides the document raises a question that has no page yet and opens one.
Either way it writes an entry on the page: an **observation** that records
which document, how strongly it bears on the question, and how (does it
support what the page already says, extend it, contradict it, or mark a
shift?).

Over time the pages tell a story the daily digest never could. The library
measures it: how many observations a topic has, how fast they are arriving,
how many different sources they come from, and whether the topic is just
emerging, actively gathering evidence, mature, fading, or dormant.

You also get tools for keeping the notebook tidy once it has grown:
merging two pages that turned out to be the same question, splitting a
page that grew too broad, moving an entry that landed on the wrong page,
and clearing out pages that never got a second entry.

## Is it for you?

It is a good fit if:

- You have a steady stream of documents about one field, and you want
  continuity across days, not just summaries of each day.
- You are willing to pay for one call to a strong language model per
  document you choose to track. The judgment of where a document belongs
  is made by a model, and that is where the cost goes.
- You have, or can set up, a database with vector search. MongoDB Atlas
  is supported out of the box; anything else needs a small adapter.

It is not:

- A topic-modeling library in the statistical sense. It does not discover
  clusters in a corpus; it tracks named questions as documents arrive.
- A trend dashboard or an alerting tool. It produces the record a
  dashboard would read.
- Anything to do with Kafka or message-queue topics.

## How it works

Here is the journey of one document, a changelog entry announcing that a
coding tool now runs commands in a sandbox.

**1. The document is compared to existing topics by meaning.** Every topic
and every document gets an embedding, a vector that places its meaning in
a space where similar things sit close together. The library pulls the
handful of topics nearest to the document, say thirty, plus any that share
a tag with it. This step is fast and cheap; its job is only to find the
neighborhood.

**2. A language model decides where it belongs.** Nearness is not
belonging. A document can sit right next to a topic and still be about a
different question. So the library shows a language model, called the
**judge**, the document and the candidate topics, each with its
one-sentence summary and a few of the observations already on it, and asks
a precise question about each candidate: is the document's question the
same as this topic's, a narrower case of it, a neighbor of it, or does it
merely share vocabulary? If any candidate is the same question or a
narrower case, the document is linked there. If none is, the judge may
propose a new topic, or decide the document is interesting but not the
start of anything.

The sandbox changelog lands on "are coding tools isolating what agents
execute by default?" because that is the question it bears on, even though
every other observation on that page is a paper and this one is a release
note. That is the point: a topic gathers evidence of every kind.

**3. New topics get a second opinion.** Linking a document to a slightly
wrong topic costs one noisy entry that can be moved later. Opening a topic
that should have been part of an existing one is worse: it splits all
future evidence between two pages, and nothing fixes that automatically.
Because the mistakes are not equal, every proposed topic goes through a
second, independent model call that sees only the closest existing topics
and the proposal, and can confirm it, fold the document into an existing
topic instead, or drop it.

**4. The result is written down.** The observation is recorded with its
strength, which comes from three yes-or-no questions the judge answers
about fit rather than a number it invents. The topic's page gets a fresh
snapshot of its statistics, and if it has crossed a threshold, its status
changes.

That is the whole loop. Everything else in the library is either setup
(choosing the models and the database), tuning (checking that the judge's
decisions are still good), or curation (tidying the notebook).

## What you bring and what the library brings

topicsleuth does not fetch documents or decide which ones matter. That is
the job of your own program, which the documentation calls the **feeder**:
it gathers documents from wherever they come from, decides which are
worth tracking, turns each into an embedding, and hands them over. Then it
does whatever it wants with the outcome: a digest, a dashboard, a
notification.

The library takes it from there. It owns the notebook: it is the only code
that writes topics and observations, so that if two programs feed the same
store, they cannot disagree about where things belong.

You also choose the services. Which embedding model, which language model,
which database, and optionally which decision model reviews proposals are
all objects you construct and pass in. Defaults for Voyage, Anthropic,
MongoDB Atlas and TypeSafe Jev are included; anything with the same few
methods works.

## Install

```bash
pip install "topicsleuth[all]"        # with the MongoDB Atlas, Voyage and Anthropic defaults
pip install topicsleuth               # core only, if you bring your own
```

Python 3.12 or newer. For MongoDB Atlas, create a vector search index on
the topics collection: path `embedding`, dimensions matching your
embedding model, similarity cosine, named `topic_embedding_index` unless
you pass another name.

## A first feeder

```python
from topicsleuth.backends.anthropic import AnthropicJudge
from topicsleuth.backends.mongo import MongoStore
from topicsleuth.backends.voyage import VoyageEmbedder
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import synthesize_items

# Tell the prompts what the store follows; it is the only domain knowledge they carry.
settings = load_settings(overrides={"domain": "municipal climate policy"})

store = MongoStore.from_uri(MONGODB_URI, database="briefing")
embedder = VoyageEmbedder(model="voyage-4", dimensions=settings.embedding.dimensions)
judge = AnthropicJudge(model="claude-opus-4-8")

# Your documents: dicts with at least url and title, usually summary and source_type too.
items = fetch_new_documents()

# Your call on which to track, and your embeddings. The library needs three
# things per document: its url, its embedding, and whether you admitted it.
class Eval:
    def __init__(self, url, embedding, admitted):
        self.url, self.embedding, self.above_threshold = url, embedding, admitted

vectors = embedder.embed_documents([f"{d['title']}\n\n{d.get('summary', '')}" for d in items])
evals = [Eval(d["url"], v, worth_tracking(d)) for d, v in zip(items, vectors)]

# The loop from "How it works".
decisions = synthesize_items(items, evals, store, settings, judge, embedder)
result = record(items, evals, decisions, store, settings=settings, embedder=embedder)

for obs in result.observations_created:
    print(f"{obs['item_title']} -> {obs['topic_id']} ({obs['connection_type']}, {obs['strength']})")
for topic in result.topics_created:
    print("new topic:", topic["title"])
```

Documents are processed in order, and a topic proposed for one document is
a candidate for the next, so ten documents about one new question produce
one topic, not ten. The library writes nothing about your own scores or
bookkeeping; keep those where you like.

If you are loading history rather than today's documents, pass
`date_from_item=True` to `record` so observations are dated by when the
document was published, not when you ran the import, and load oldest
first.

## Settings

A few decisions have to be shared by everyone writing to one store, so
they live in a settings file rather than in code. The defaults ship with
the package; `load_settings(path, overrides)` lets you layer your own on
top.

```yaml
domain: the field this store follows   # a noun phrase; set it

embedding:
  dimensions: 1024                     # must match your embedding model and your index

candidates:
  tag_overlap: true                    # topics sharing a tag with the document join the candidates
  ignored_tags: []                     # any tag every document carries, which would match everything
  vector_limit: 30                     # how many nearest topics the judge sees

synthesize:
  confirm_proposals: true              # the second opinion on every new topic
  confirm_candidates: 5                # how many nearest topics the reviewer sees
  confirm_backend: judge               # judge | decision (a probability-only model instead)
  confirm_link_threshold: 0.7          # decision backend: how sure it must be to fold a proposal in
```

Model names and database names are not settings. They are properties of
the objects you construct.

## Keeping the store tidy

The `topicsleuth` command works on a store using the default services,
configured through environment variables (`MONGODB_URI`,
`TOPICSLEUTH_DATABASE`, and optionally `TOPICSLEUTH_JUDGE_MODEL`,
`TOPICSLEUTH_EMBEDDING_MODEL` and `TYPESAFE_API_KEY`; a `.env` file in the
working directory is read). Every command that writes accepts `--dry-run`
and shows you the plan first.

```bash
topicsleuth audit                               # how big is the store, how is evidence distributed
topicsleuth suggest-merges                      # pairs of topics that look like the same question
topicsleuth merge-topics <source> <target>      # fold one into the other; the judge rewrites the survivor
topicsleuth split-topic <id>                    # the judge proposes sub-topics and reassigns the evidence
topicsleuth add-topic topic.yaml --link <url>   # open a page by hand and move named evidence onto it
topicsleuth move-observation <url> <topic_id>   # move one document's entry to another page
topicsleuth reprocess-tail --max-obs 1 --older-than-days 60   # clear pages that never got a second entry
topicsleuth seed topics.yaml                    # start an empty store from a list of topics
topicsleuth clean --hours 28                    # undo a bad run
```

Nothing here re-judges a document. Merging and moving keep every
observation's strength and connection type and note where it came from.
A retired topic keeps a pointer to what replaced it, so old links still
resolve.

## Tuning: making sure the judge stays good

The decision at the heart of the library is a judgment, and judgments
drift. Change the prompt, upgrade the model, show the judge more or fewer
candidates, and the share of documents that open new topics can move by
ten points with no error anywhere. Too many new topics is the quiet
failure: evidence spreads across near-duplicate pages, each looks slow,
and nothing surfaces.

So treat the judge the way you would treat any classifier. Take a
baseline, change one thing, measure. The library makes that cheap:

```bash
# Replay the decision on 30 documents the store already holds, without writing anything.
topicsleuth tune synthesize --leave-one-out --sample-query '{"source": "my-feed"}' --n 30 --seed 42 --label baseline

# After a change, replay the very same documents and compare.
topicsleuth tune synthesize --leave-one-out --replay tuning_runs/synthesize-baseline-n30-seed42.json --label after
python scripts/diff_tuning_runs.py tuning_runs/synthesize-baseline-n30-seed42.json tuning_runs/synthesize-after-replay.json
```

The report shows how many documents linked, proposed or skipped, how
strong the links were, and for each call where the chosen topic came
from and how big the prompt was. The diff shows which documents changed
their minds.

Three habits make the numbers trustworthy. Always pass `--leave-one-out`,
because a document that is already in the store would otherwise see the
topic it founded and the replay would flatter itself. Replay the same
documents, so the diff is one to one. And use two different seeds before
believing a change, because a fair share of documents sit right on the
boundary and flip from run to run on their own.

When the question is whether the judge's rubric is right at all, rather
than whether a change moved it, build a small labeled set: pairs of a
document and a candidate topic, labeled by a person or an independent
model as the same question, a narrower case, a neighbor, or surface
overlap, without looking at what the judge said. The `scripts/` folder has
tools to produce such labels from a model and to compare raters. Keep
your baselines and label sets in the repository that owns your store;
they describe its content. `data/examples/` shows the shape of every
file.

## Testing a feeder without a model

The package ships stand-ins for every service: an embedder that hashes
text to a stable vector, a judge that returns scripted answers, a
decision model driven by a table, and an in-memory store. Together they
run the whole loop with no network, so a feeder's tests can be ordinary
unit tests.

```python
from topicsleuth.backends.fake import HashEmbedder, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore
```

The `topicsleuth` command uses the same stand-ins when
`TOPICSLEUTH_STORE=memory`, `TOPICSLEUTH_JUDGE=scripted` and
`TOPICSLEUTH_EMBEDDER=hash` are set, which is how its own tests run.

## Status

Early. The matching, recording, curation and tuning code has run daily
against a live store for months, but the public interface is new and may
move between minor versions, so pin a release. Releases are published to
PyPI from version tags. `DESIGN.md` explains the choices behind the
interfaces and the decision for anyone who wants to go deeper.

The name follows the topic detection and tracking literature, where the
tracked thing is a topic, a document is a story, noticing a new topic is
first-story detection and attaching a document to one is tracking.

## License

MIT.
