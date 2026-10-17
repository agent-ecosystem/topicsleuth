# topicsleuth

[![Tests](https://github.com/agent-ecosystem/topicsleuth/actions/workflows/test.yml/badge.svg)](https://github.com/agent-ecosystem/topicsleuth/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/topicsleuth)](https://pypi.org/project/topicsleuth/)
[![Python](https://img.shields.io/pypi/pyversions/topicsleuth)](https://pypi.org/project/topicsleuth/)
[![License: MIT](https://img.shields.io/pypi/l/topicsleuth)](https://github.com/agent-ecosystem/topicsleuth/blob/main/LICENSE)

You're interested in something specific. New content arrives every day:
papers, release notes, posts, discussion threads. topicsleuth remembers
them for you. It works out which ongoing question each new document
belongs to, keeps the content related to every question in one place,
and tells you which questions are gathering momentum, settling, or going
quiet. It is the memory behind the topics you care about, so
"what happened to that thing from two weeks ago?" has an answer.

If that is the problem you have, the next section gets you running. The
rest of this page explains what the library does and how, so you can
decide whether it fits before you commit to it.

## Try it in five minutes

To use topicsleuth with the default backends, you need:

- Python 3.12 or higher
- A MongoDB Atlas cluster
- A vector search index on the `topics` collection with:
  - path `embedding`
  - cosine similarity
  - number of dimensions matching your embedding model
  - index name `topic_embedding_index` (or set `TOPICSLEUTH_VECTOR_INDEX`)
- API keys for:
  - Voyage embeddings
  - Anthropic

Other databases and models work too; see "Bring your own models and
database" below.

### Install topicsleuth

```bash
pip install "topicsleuth[all]"
```

### Tell it what you're interested in

The judge works from prompts that describe the field it is working in.
The defaults describe a feed of papers, release notes and posts about
a technical field. If that is not the content you're interested in,
write a settings file before the first run, so the first topics it
opens are the right shape. Two things go in it: the domain, as a
noun phrase, and a `rubric` block with the prose the judge reads about
your field.

```yaml
domain: municipal housing policy
rubric:
  source_kinds: council minutes, local news, planning filings, advocacy reports
  reading_by_kind: a filing bears on a question by proposing a change, a minute by recording what was decided, a news story by reporting an outcome, an advocacy report by arguing a position
  topic_size: 10-50 observations over a year
  question_examples:
    - question: Are cities loosening parking minimums near transit?
      subject: parking
    - question: Is inclusionary zoning producing units where it is adopted?
      subject: zoning
  subset_example: The candidate asks whether cities are loosening parking minimums near transit; the item is one city's ordinance doing so. The item is a subset, not a different question.
  counter_examples:
    - A council agenda that lists every item of the month, with candidates all rated `adjacent_question`: skip, not a "city council activity" meta-topic.
```

Every key is optional, and the full list with what each one does is on
`RubricConfig` in `topicsleuth/rubric.py`. Only the wording about your
field is yours to set. The labels the judge applies, the rules it follows
and the questions it answers are fixed, because the code depends on them.
`topicsleuth show-prompts` prints the prompts exactly as the judge will
see them, which is the quickest way to check that the result reads well.

### Use the library

To use the default clients, import from `backends`. Otherwise,
import `record`, `settings`, and `synthesize` to make and store
observations related to your topics:

```python
from datetime import date

from topicsleuth.backends.anthropic import AnthropicJudge
from topicsleuth.backends.mongo import MongoStore
from topicsleuth.backends.voyage import VoyageEmbedder
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import synthesize_items

settings = load_settings("settings.yaml")
store = MongoStore.from_uri(MONGODB_URI, database="briefing")
embedder = VoyageEmbedder(model="voyage-4", dimensions=settings.embedding.dimensions)
judge = AnthropicJudge(model="claude-opus-4-8")

items = fetch_new_documents()        # your dicts: url, title, summary, source_type, tags ...

class Eval:                          # what the library needs per document
    def __init__(self, url, embedding, admitted):
        self.url, self.embedding, self.above_threshold = url, embedding, admitted

vectors = embedder.embed_documents([f"{d['title']}\n\n{d.get('summary', '')}" for d in items])
evals = [Eval(d["url"], v, True) for d, v in zip(items, vectors)]

decisions = synthesize_items(items, evals, store, settings, judge, embedder)
result = record(items, evals, decisions, store, settings=settings, embedder=embedder,
                run_id=f"my-feed-{date.today()}")   # so `clean --run-id` can remove this run

for obs in result.observations_created:
    print(f"{obs['item_title']} -> {obs['topic_id']} ({obs['connection_type']})")
for obs in result.already_recorded:    # held from an earlier run; not judged or written again
    print(f"{obs['item_title']} was already on {obs['topic_id']}")
for failure in result.failures:        # retry these next run; do not mark them done
    print(f"{failure.item_url} failed at {failure.stage}: {failure.error}")
```

Recording is safe to retry. A document is evidence for a topic once: the
store holds one observation per document per topic, so running the same
batch twice, or two programs admitting the same document, writes it once,
and a document the store already holds is not judged again; the result
lists it under `already_recorded`. A document that could not be
finished is listed in `result.failures` with the stage it stopped at.
Keep it in your queue and hand it over again next run: every stage is
idempotent, so the retry completes what the failed run left undone (a
snapshot, a status, an embedding) without repeating what it did. Two
failures repeat for the same input and are worth parking instead: an
`error` beginning `JudgeTruncated:` (the judge ran past its output cap;
raise the cap it names) or `JudgeRefused:` (the provider's classifier
declined the prompt). Pass a
`run_id` and everything the run writes is stamped with it, which is what
lets `clean --run-id` remove one run and nothing else.

## Keep track of content across days, not just today

Anyone who reads a feed and writes a summary ends up with the same thing:
a good account of today, and no connection to yesterday. The paper from
February, the product launch in June and the tutorial this morning are
one story; an idea becomes a product which begets a practice. Your daily
summary can't tell that story because it has no memory.

Think of topicsleuth as the tool that helps you keep a notebook with one
page per question. A subject like "security" is too broad. Instead, the
topic is a question the field keeps working on, such as "do coding
assistants make developers write less secure code?" A question is the
right size for a page: specific enough that a document clearly belongs
on it or not, but also broad enough to gather evidence for months.

When you let topicsleuth keep that notebook, each piece of content that
bears on a question becomes an **observation** on that topic's page. A
document that bears on two questions, say a paper that is both about
evaluation design and about how a coding harness runs commands, gets an
entry on both pages, each judged on its own. An observation records which
document it was, how strongly it bears on the question, and a label that
says how it connects to what's already in the topic. The label is there so you can see at a glance when you open a topic
which entries agree with each other and which ones pushed the question
somewhere new. An observation can relate to its topic in one of four ways:

- **supports**: adds evidence consistent with what the topic already says.
  Most observations are this.
- **extends**: broadens the question by bringing in a facet, method, or
  kind of source the page had not covered.
- **contradicts**: offers evidence against the topic's claim.
- **shifts**: marks a change in the topic's trajectory, such as the first
  vendor to turn a practice on by default.

When a piece of content arrives that raises a question with no page yet,
it starts a new one. And because the entries on each page are dated, it can
measure what a summary never could: how fast a question is gathering
evidence, from how many different sources, and where the question is in
its life. Every topic carries a status that moves as evidence arrives or
stops arriving:

- **emerging**: a new page with fewer than five observations. It may be
  the start of something, or a one-off: one whose last evidence is more
  than thirty days old goes dormant directly, since it never had a pace
  to fade from.
- **active**: five or more observations. The question has drawn repeated
  evidence.
- **mature**: twenty or more observations, a well-established thread. From
  here on, a weak match (strength below 0.7) is still filed but marked
  minor, so a mature topic does not drift on tangential links. Both numbers
  are settings, under `record`.
- **fading**: an active or mature topic that has gone a week with no new
  evidence. Fresh evidence sends it back to active.
- **dormant**: a fading or emerging topic whose last evidence is more than
  thirty days old. Fresh evidence sends it back to emerging, to earn its
  place again.

Topics you merge or split are marked **retired** and keep a pointer to
what replaced them, so old links still resolve.

## Know whether it fits before you build on it

You might find topicsleuth useful when:

- You have a steady stream of content about one area of interest.
- You want continuity across days, not only a digest of each day.
- You are willing to pay for at least one call to a strong language model
  per document you choose to track. The judgment of where a document belongs
  is made by a model, and that is where the cost goes.
  
You will also need a database with vector search; MongoDB Atlas works out
of the box, and anything else needs an adapter (see "Bring your own
models and database" for what one has to provide).

It is the wrong tool if you want to discover clusters in a corpus you
already have (that is topic modeling), if you want a dashboard or alerts
(this produces the record a dashboard would read), or if you were
searching for something to do with Kafka topics.

## How a piece of content finds its topic

Let's follow one document through the library: a changelog entry announcing
that a coding tool now runs commands in a sandbox.

**First, the library looks nearby.** Every topic and every document has an
embedding, a vector that places its meaning so that similar things sit
close together. The library pulls the topics nearest to the document,
around thirty, and adds any that share a tag with it. This is quick and
cheap, and it only finds the neighborhood; it does not decide anything.

Tags are the one piece of your own vocabulary the library uses. A
document arrives with whatever tags your program gave it, free-form
strings such as `security` or `benchmarks`, and the judge sees them with
the document. When the judge opens a topic it gives the page tags of its
own, a merge or a split rewrites them, and topics you seed or add by hand
carry the tags you wrote. From then on the tags are a second way to find
the neighborhood: a topic that shares a tag with the document joins the
candidates even when its embedding sits far away. Each entry also keeps a
copy of its document's tags, so you can see later how a page's evidence
breaks down. A tag every document carries, such as the name of the feed,
would match every page; list it under `ignored_tags` in settings and the
library drops it from matching and from the tags of new topics.

**Then a language model decides.** Being near a topic is not the same as
belonging to it. A document can sit right beside a topic and be about a
different question. So the library shows a language model, which it calls
the **judge**, the document and the similar topics, each with its
one-sentence summary and a few of the entries already on it. The lib asks
the judge if the document's question is the same as the existing topic,
a narrower case of it, a neighbor of it, or does it only share vocabulary?
The document goes on the page of every similar topic that is the same
question or a narrower case, as an observation on each, scored for each.
If there is none, the judge may propose a new topic, or decide the
document is interesting but not the start of anything.

The sandbox changelog lands on "are coding tools isolating what agents
execute by default?", even though every other entry on that page is a
paper and this is a release note. That is the point of the whole design:
a topic gathers evidence of every kind related to the shared question.

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

## Keep the judge honest

The decision at the heart of the library is a judgment, and judgments
drift. Edit the prompt, upgrade the model, change how many candidates the
judge sees, and the topic assignment results can change without any
noise letting you know something is different.

In real-world use, changing the judge model from one Opus version to a
newer one resulted in far more new topics being created versus observations
linking to existing topics. Nothing in a daily briefing hints that 
evidence is spreading across an increasing percentage of one-item topics
until you notice the topic library has grown by hundreds.

Treat the judge the way you would treat any classifier: take a baseline,
change one thing, and measure. The library provides a tool for this; the
ability to replay the decision on documents the store already holds,
without writing anything:

```bash
# A baseline on 30 documents already in the store, drawn twice.
topicsleuth tune synthesize --sample-query '{"source": "my-feed"}' --n 30 --seed 42 --label baseline
topicsleuth tune synthesize --sample-query '{"source": "my-feed"}' --n 30 --seed 7 --label baseline

# Each baseline writes tuning_runs/synthesize-<label>-n<n>-seed<seed>.json,
# so the two runs above land in two files.

# After a change, use `--replay` with the baseline report, then the diff.
topicsleuth tune synthesize --replay tuning_runs/synthesize-baseline-n30-seed42.json --label after-42
python scripts/diff_tuning_runs.py tuning_runs/synthesize-baseline-n30-seed42.json tuning_runs/synthesize-after-42-replay.json
```

Generate a baseline report over a sample of documents that you can
use to to check how the judge assesses documents before and after
a change. Run at least two sample sets before assessing whether a
change has had the desired result. A fair share of documents often
sit right on the boundary between "link" and "new topic" and can
flip on their own from run to run. Two samples of thirty can tell
you whether a shift is real or just those documents.

The topicsleuth `tune synthesize` command can take a `--seed` flag
that accepts a random integer to create a sample of `--n` documents
from your data store. Any two integers can create random seeds, but
keep using the *same* two integers so every experiment is judged on
the same two samples. On every later run, `--replay` the report, so
the comparison is always one to one even after the store has grown.

Drawing a sample needs a store that can sample. The default store is
MongoDB, which supports sampling. If you're using another store,
write the first report by hand, as a list of document URLs in the shape
of `data/examples/replay.example.json`, and `--replay` from it.

The report tells you how many documents linked, proposed or skipped, how
strong the links were, and for each call, where the chosen topic came from
and how large the prompt was. The diff tells you which documents changed
their minds. The `tune` command reads your settings file through
`TOPICSLEUTH_SETTINGS`, so a change to the rubric block is replayed by
editing the file.

### What you can change to tune library performance

Five things move the judge, listed here in order from the largest
effect to the smallest. All of them live in the settings file. The
first three can be overridden for a single replay from the command line,
so you can try a value before you keep it.

- **The model.** The judge is an object you construct, and `--model`
  swaps it for one replay. Expect the boundary to move: a different model
  draws the line between "link" and "new topic" somewhere else, which is
  why a model change gets the full two-seed treatment.
- **Which topics the judge sees.** `candidates.vector_limit` is how many
  nearest topics go in the prompt, `candidates.tag_overlap` adds every
  topic sharing a tag with the document, and `candidates.ignored_tags`
  keeps a routing tag from matching everything. A judge cannot link to a
  topic it was not shown. Flags: `--limit`, `--no-tag-overlap`,
  `--ignore-tags`.
- **The second opinion.** `synthesize.confirm_proposals` turns the
  reviewer on, `confirm_candidates` is how many nearest topics it sees,
  and `confirm_backend` picks the judge's second call or a decision model
  with `confirm_link_threshold` as its bar. These change how many
  proposals survive without touching the rubric. Flags: `--no-confirm`,
  `--confirm-backend`, `--confirm-threshold`. In the same group,
  `secondary_link_floor` drops a document's second or later link when its
  strength is below the floor, and `--secondary-link-floor` tries one.
- **The prose about your field.** `domain` and the `rubric` block from
  "Tell it what you're interested in". Counter-examples are the working lever: a
  replay shows you the decisions the judge gets wrong, and one
  well-chosen counter-example states that case and its right answer.
- **The rubric itself.** The labels, rules and questions in
  `topicsleuth/rubric.py`. Change them in a fork, replay both seeds, and
  expect every baseline you have to need re-taking.

The problems below are the ones a store develops in practice. Each names
the report line that shows it and the lever that moves it.

### Too many new topics

Evidence spreads across near-duplicate pages, each looks slow, and
`suggest-merges` keeps finding pairs. In the report, the `propose_new`
share is high and `confirmation outcomes` shows most proposals confirmed.
Read the proposed topics and the diff first. If the right page existed
and the judge was shown it, the fix is in the prose: write the case up
as a counter-example in your `rubric` block. If the reviewer is waving
proposals through, give it more to compare against with
`confirm_candidates`, or switch `confirm_backend` to `decision` and lower
`confirm_link_threshold` a step at a time. If the right page was not
among the candidates at all, that is the next problem.

### The right topic was never shown

The report's `linked topic vector rank` says where each chosen topic sat
in the nearest-neighbor list. When links cluster near the bottom of the
list, or the diff shows documents finding a better page only after you
raised `--limit`, the judge is being starved of candidates. Raise
`vector_limit` in steps of ten and watch `mean input tokens`, which is
the cost of each step. If `linked topic channel` shows the tag channel
doing real work, check that `ignored_tags` is not dropping a tag that
matters. The reverse also happens: a tag every document carries turns the
channel into "every topic", and the fix is to list it.

### One topic swallows everything

A page with hundreds of entries whose summary could cover half the feed.
In the report, `gate overrides` counts the times the judge wanted a new
topic and the rules forced a link, and the strength buckets lean toward
the low bands. The rules are doing what they are for, so the fix is in
the store and the prose. Split the page with `split-topic`, and make
`topic_size` and the `question_examples` say what narrow looks like in
your field. A counter-example of the form "an item that only mentions X,
judged against the X page, rated `surface_overlap`: do not link" closes
the door.

### The judge reasons about the wrong kind of document

The reasoning traces talk about papers and releases when your feed is
filings and minutes. The judge is working from the default prose. Set
the `rubric` block, check it with `show-prompts`, and take a fresh
baseline, because every comparison from before was made by a different
judge.

### Weak secondary links

A document that bears on two questions is filed on both pages, and the
second link is often the weaker one: the judge said the question is the
same but answered the fit questions "no". On a well-evidenced topic that
entry is marked minor, which is correct, but a page can fill with minor
entries from documents whose main home is elsewhere. The report's
`LINKS` line shows how many links were secondary, their strength
buckets, and how many a floor dropped. Set `synthesize.secondary_link_floor`
a step at a time (0.5 drops the "all three no" and "not citable" bands)
and replay both seeds; a replayed report shows what each value would
have dropped without a new model call. The first link is never dropped.

### It costs too much

`mean input tokens per call` in the report is the bill. The candidate
list is most of it, so `vector_limit` is the first thing to lower and
`confirm_candidates` the second. A cheaper model is worth a replay to see
what it gives up, and `--model` tries it without committing to it.

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
environment (`MONGODB_URI`, `TOPICSLEUTH_DATABASE`, `TOPICSLEUTH_SETTINGS`
for your settings file, and optionally the model names; a `.env` file in
the working directory is read). A feeder that keeps its store in
non-default collections, say because two feeders share a database and
its `items` collection but curate their topics apart, names them with
`TOPICSLEUTH_TOPICS_COLLECTION`, `TOPICSLEUTH_OBSERVATIONS_COLLECTION`
and `TOPICSLEUTH_ITEMS_COLLECTION` (and the index on that topics
collection with `TOPICSLEUTH_VECTOR_INDEX`), so the command and the
pipeline run against the same store. Every
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
topicsleuth backfill-embeddings                 # embed topics that have none; refresh well-evidenced ones
topicsleuth clean --hours 28                    # remove what the last run wrote
```

`reprocess-tail` clears the pages but does not re-judge their documents:
it writes their URLs to a file (under `tuning_runs/` by default, or
`--urls-out`), and your feeder hands those documents over again on its
next run so they find a home among the topics that exist now.
`backfill-embeddings` is for after an import or a change of embedding
model: it embeds every topic that has no vector and recomputes the
vector of every topic with at least five entries from the entries'
own embeddings.

Nothing here re-judges a document, except that a split asks the judge to
re-assess each entry's strength and connection type against its narrower
sub-topic. Merging and moving keep both. All three keep each entry's
identity, date and metadata and note where it came from, and a retired
topic keeps a pointer to what replaced it, so old links still resolve. A
document that was evidence on both sides of a merge or move keeps the
target's entry, with the other folded into it and noted. A
split is checked in full before anything is written: every entry assigned
exactly once, to a sub-topic that exists, with a known connection type.

These commands may run while a feeder is writing. A document caught
between a command and the feeder fails that run and is retried on the
next, and a merge or split carries along anything that arrived on its
source while it ran. If a command dies part-way through deleting a
topic, the topic stays marked for up to fifteen minutes, during which
recording and the other commands skip it and report it as busy; the next
`clean` or `reprocess-tail` then finishes or undoes the interrupted
deletion on its own.

`clean` removes the observations, topics and snapshots written in a
window, by when they were written rather than when their evidence
happened, and with `--run-id` only what one run wrote (pass `run_id` to
`record` and the run stamps everything it writes). It is not a full undo.
A topic written in the window that still holds older evidence is kept,
and a status, embedding or description a run changed is not restored.
A topic another command is working on is left alone and reported as
busy; run `clean` again once that command has finished.

## Bring your own content

topicsleuth never fetches documents and never decides which ones matter.
That is the job of your own program, which this page calls the **feeder**:
it gathers documents, chooses which to track, embeds them, and hands them
over, then does what it likes with the outcome. The library owns the
notebook, and it is the only code that writes to it, so two feeders
sharing one store cannot disagree about where things belong.

## Bring your own models and database

Every service the library talks to is an object you construct and pass
in: the embedding model, the language model that judges, the database,
and optionally a decision-only model that reviews proposals in place of
the judge's second call. Each is a `typing.Protocol` in
`topicsleuth.interfaces`, with the methods listed there; anything that has
them works, with no base class to inherit. The model protocols are a
method or two. The store protocol is larger, and a few of its operations
must be atomic: a conditional update for the lease a maintenance command
takes on a topic, for publishing a topic's embedding and for registering
a refresh it still owes, and a transaction wherever one operation writes
two documents (deleting a leased topic's entries, moving or folding an
entry onto a topic that must still be live). The docstrings on the
protocol say which, and the store contract test in the source tree is the
specification: an adapter that passes it will behave under retries and
concurrent writers the way the defaults do. Three of its reads
(`list_topics`, `nearest_topics` and `get_observations_for_topic`) take a
`fields` keyword naming what the caller will read, and a store returns
exactly those fields plus the document's identity; that is how the
synthesize and record paths avoid transferring every topic's embedding and
every observation's reasoning trace, which they never look at. A store
with no native projection filters what it fetched. Defaults for Voyage, Anthropic,
MongoDB Atlas and TypeSafe Jev are included as extras (`voyage`,
`anthropic`, `mongo`), and the core package itself depends only on pyyaml,
numpy and python-dotenv.

A few decisions have to be shared by everyone writing to one store, and
those live in settings rather than code: the domain and the prose the
prompts are written with, the embedding dimensionality the index was
built with, how candidates are assembled, and whether and how proposals
are reviewed. The defaults ship with the package and
`load_settings(path, overrides)` layers yours on top:

```yaml
domain: the field this store follows   # a noun phrase; every prompt is rendered with it
rubric: {}                             # the prose about your field; see "Tell it what you're interested in"
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
  secondary_link_floor: 0.0            # drop a document's second or later link below this strength
judge:
  first_pass_max_tokens: 8192          # output cap per judge call; a reply that hits one fails as JudgeTruncated,
                                       # one the model declines to give as JudgeRefused: both repeat, so park, do not retry
  confirm_max_tokens: 2048             # (the first pass grows with vector_limit; raise the cap it names)
  merge_max_tokens: 4096
  split_propose_max_tokens: 4096
  split_reassign_max_tokens: 8192
record:
  minor_floor_after: 20                # from this many observations on a topic...
  minor_floor_strength: 0.7            # ...a new one below this strength is marked minor
  carry_fields: [evidence, lanes]      # item fields copied onto its observations when present
                                       # (tags is always copied: the library acts on it)
```

Loading history rather than today's documents? Pass `date_from_item=True`
to `record` so entries are dated by publication rather than by the run,
and load oldest first. A batch dated before a topic's latest snapshot is
inserted into its history: the snapshot counts only evidence up to its
date, and the topic's current status is left to the newer evidence. Pass
`backfill=True` as well and every entry the batch writes is stamped
`backfill`, so you can tell re-homed history from evidence that arrived
fresh.

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
`TOPICSLEUTH_STORE=memory`, `TOPICSLEUTH_JUDGE=scripted`,
`TOPICSLEUTH_EMBEDDER=hash` and `TOPICSLEUTH_DECISION=scripted` are set,
which is how its own tests run. If
you write a store adapter, the store contract test in the source tree's
`tests/` folder (not in the wheel) runs against it.

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
