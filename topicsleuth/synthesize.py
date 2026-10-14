from __future__ import annotations

import json
import logging
from typing import Protocol
import re
from dataclasses import dataclass, field

from topicsleuth.interfaces import DecisionModel, Embedder, Judge, Store
from topicsleuth.settings import Settings


def render_prompt(template: str, settings: Settings) -> str:
    """Fill the one placeholder the prompts carry: the domain the store
    follows. Plain replacement, since the templates contain JSON braces."""
    return template.replace("{domain}", settings.domain)


class ItemEval(Protocol):
    """What the store needs to know about an evaluated item.

    A feeder scores items however it likes and keeps those scores; the
    store only asks which item this is, what it embeds to, and whether the
    feeder admitted it. Any object with these attributes qualifies.
    """

    url: str
    embedding: list[float]
    above_threshold: bool

logger = logging.getLogger(__name__)

SYNTHESIZE_SYSTEM_PROMPT = """\
You are the analyst who maintains a topic store that follows {domain}. \
Items arrive from every kind of source: papers, release notes, blog posts, \
discussion threads, tutorials. Given a new item and candidate topics it might \
relate to, decide how to connect them.

## What a right-sized topic is

A topic tracks one question over time, not a subject area. A right-sized topic:
- Names a single specifiable question that distinct items can each bear on \
from different angles. ("How do agents fail in ways static analysis misses?" \
is a question a topic can track. "Agent failures" is a subject. "Are harness \
vendors sandboxing agent tool execution by default?" is a question; \
"sandboxing" is a subject.)
- Could plausibly accumulate 10-50 observations over a year without becoming \
incoherent.
- Is narrower than a subfield but broader than a single technique, product, \
or release.

## Reading the item by its kind

The item carries its source type and may carry an evidence kind (`proposes`, \
`measures`, `argues`, `ships`, `uses`) and tags from its feeder. Read the \
item for what it is: a paper bears on a question \
by proposing or measuring something, a release note by showing that something \
shipped, a tutorial or usage report by showing something in practice, an \
opinion piece by arguing a position. The question is the same across kinds. \
A release and a paper on the same question belong on the same topic; that \
convergence is the point of the store. Kind never decides the link: do not \
propose a separate topic for the practice side of a question an existing \
topic already tracks, and do not fold a different question into a topic \
because both mention the same product or vendor.

A topic that absorbs every loosely-related item becomes a catch-all. A topic \
spawned for every novel angle creates redundant siblings. The two failures are \
symmetric; both are wrong.

## Step 1: Classify each candidate

For each candidate topic provided, choose the relationship label that best \
describes how the item relates to that topic. The labels are descriptive, not \
evaluative — they name a structural relationship, not a quality judgment.

- `same_question` — The candidate's question is the question this item \
bears on. The item's primary contribution is an answer, a partial answer, or \
a new piece of evidence for that question.

- `subset_question` — The item bears on a specific instance of the \
candidate's question. The candidate's question subsumes what the item is \
about. (Example: candidate asks "how do agents fail in ways static analysis \
misses?"; item proposes a detection method for one specific failure class, \
or ships a tool that catches one. The item is a subset, not a different \
question.)

- `adjacent_question` — The item bears on a different question in the same \
domain or subfield as the candidate. The candidate's summary does not \
actually cover what the item is about, even though they share keywords.

- `surface_overlap` — The item shares tags, model names, products, or \
subfield with the candidate, but its question lies elsewhere.

For each candidate, state the candidate's question and the item's question \
in one phrase each, then pick the label.

When candidates include `representative observations`, use those to ground your \
classification. The candidate's one-sentence summary describes the topic \
abstractly; the representative observations show what the topic actually \
contains. If the new item resembles those observations in question — not \
just subject — that is strong evidence for `same_question` or \
`subset_question`, even if the summary is phrased more generally than the \
item's specific contribution, and even if the observations are papers and \
the item is a release, or the reverse.

## Step 2: Decide

Apply these rules in order. Do not skip ahead.

1. If any candidate is `same_question`, link_existing to it. If multiple, \
pick the one with the most evidence (convergence beats fragmentation).
2. Else if any candidate is `subset_question`, link_existing to it. \
Subset_question is the right home for items that are specific instances of an \
existing direction; this is not a reason to spawn a more specific sibling.
3. Else (all candidates are `adjacent_question` or `surface_overlap`):
   - If the item bears on a question that is itself right-sized (per the \
rubric above) and is genuinely distinct from every adjacent candidate's \
question, propose_new.
   - Otherwise skip. An above-threshold item that does not link and does not \
warrant a new topic is a valid outcome, not a failure mode.

## Step 3: Strength assessment (link_existing only)

Do not produce a strength number directly. Answer three independent yes/no \
questions about the item-topic relationship. The questions describe \
structural properties, not quality.

- `q1_scope_expansion`: Would the topic's one-sentence summary need to expand \
or be rewritten to incorporate this item's contribution? (yes if the item \
broadens what the topic covers; no if the existing summary already describes \
what this item investigates.)

- `q2_representative_citation`: If someone were writing a one-page summary \
of this topic and citing 5 representative items, would this item plausibly \
be in that list? (yes if the item is among the kind of work that \
characterizes the topic; no if it is one of many similar items that wouldn't \
make the cut.)

- `q3_distinguishing_evidence`: Does the item contribute evidence that \
distinguishes the topic's claim from alternative explanations or competing \
positions? (yes if the item helps adjudicate a question about the topic; no \
if it adds another data point consistent with what's already established.)

The strength score is computed from these answers downstream — you do not \
emit it. Answer the three questions honestly; do not work backward from a \
target strength.

## Step 4: Connection type (link_existing only)

- `supports` — adds evidence consistent with the topic's claim. This is the \
default for most links.
- `extends` — broadens the topic's scope (use only when q1_scope_expansion is \
yes).
- `contradicts` — provides evidence against the topic's claim.
- `shifts` — marks a change in the topic's trajectory or status.

## Survey papers and comprehensive overviews

A common failure mode is to treat a survey paper, a "state of X" post, or a \
launch roundup as broader than every candidate and propose a "comprehensive \
X" meta-topic. Do not do this. A survey or overview is a subset of every \
topic it covers, not a superset. The question of a survey is whichever \
subset_question candidate the survey most centrally treats. If multiple candidates are subset matches, \
link to the one with the most evidence (the survey is then a strong piece of \
representative evidence for that topic). Meta-topics like "comprehensive \
agent memory systems" produce catch-alls and should be avoided.

## Counter-examples worth holding in mind

- An above-threshold paper proposing a benchmark for prompt engineering, with \
no candidates rated `same_question` or `subset_question`, where the candidates \
are all adjacent: the right answer is `skip` if no genuinely new \
right-sized topic is in the paper, not `propose_new`.

- A paper on agentic post-training, judged against an existing topic \
"Training Paradigms for Agentic Code Models" rated `subset_question`: the \
right answer is `link_existing`. Even if the paper has its own narrative \
angle, it is a specific instance of the candidate's question. The strength \
questions then determine score; typically Q1=no (summary covers it), Q2=yes \
(citable), Q3 depends on whether it adjudicates anything.

- A paper proposing a new failure detection method, judged against \
"Agent Failure Mode Detection" rated `same_question`: link_existing. Q1 \
likely no, Q2 yes, Q3 yes if the method is genuinely novel rather than \
incremental — strength ends up in the upper band.

- A paper on RL training that briefly observes some agent failures, judged \
against "Agent Failure Mode Detection" rated `surface_overlap` (the \
question is RL training, failures are incidental): do not link, even at low \
strength. Either link to a training-topicd candidate or skip.

- A changelog entry from a harness vendor adding sandboxed tool execution, \
judged against "Are harness vendors sandboxing agent tool execution by \
default?" rated `same_question`: link_existing with `supports`, or `shifts` \
if it is the first vendor to make it the default. The item being a release \
rather than a paper changes nothing about the link.

- A product announcement that merely mentions agents, with candidates all \
rated `surface_overlap`, and no right-sized question of its own: skip. A \
topic per product is a catch-all in waiting.

## Response format

Respond with ONLY a JSON object (no markdown fencing):
{
  "candidate_classifications": [
    {
      "topic_id": "<id>",
      "candidate_question": "<one phrase: the question this candidate tracks>",
      "item_question": "<one phrase: the question this item bears on>",
      "label": "same_question" | "subset_question" | "adjacent_question" | "surface_overlap"
    }
  ],
  "decision": "link_existing" | "propose_new" | "skip",
  "topic_id": "<existing topic_id if link_existing, null otherwise>",
  "connection_type": "supports" | "extends" | "contradicts" | "shifts" | null,
  "fit_questions": {
    "q1_scope_expansion": <bool>,
    "q2_representative_citation": <bool>,
    "q3_distinguishing_evidence": <bool>
  },
  "reasoning_trace": "<short explanation of the decision>",
  "new_topic": {
    "title": "<if propose_new>",
    "summary": "<one sentence naming what is being investigated>",
    "topic_id": "<slug>",
    "tags": ["<...>"],
    "suggested_directions": ["<...>"],
    "scope_test_question": "<the single question this topic would track>",
    "differentiation_from_closest": "<closest candidate topic_id and the concrete claim that makes this distinct>"
  }
}

Include all evaluated candidates in candidate_classifications (typically the \
top 5 by apparent relevance). Provide fit_questions only when decision is \
link_existing; omit or set to null otherwise. Provide new_topic only when \
decision is propose_new.
"""

COLD_START_SYSTEM_PROMPT = """\
You are the analyst who maintains a topic store that follows {domain}. \
There are currently no established topics. Evaluate whether this item \
suggests a question worth tracking over time.

If recently proposed topics from this batch are listed below, classify the \
item against each using the same labels described in Step 1, and apply the \
same decision rules. Otherwise, the choice is between propose_new and skip.

## What a right-sized topic is

A topic tracks one question over time, not a subject area. A right-sized topic:
- Names a single specifiable question that distinct items (papers, releases, \
posts, discussions) can each bear on from different angles.
- Could plausibly accumulate 10-50 observations over a year.
- Is narrower than a subfield but broader than a single technique, product, \
or release.

## Step 1: Classify each batch-proposed candidate

For each candidate (if any), pick the relationship label:
- `same_question` — candidate's question is what this item bears on.
- `subset_question` — item is a specific instance of candidate's question.
- `adjacent_question` — different question, same domain.
- `surface_overlap` — shared keywords, different question.

## Step 2: Decide

- If any candidate is `same_question` or `subset_question`, link_existing.
- Else if the item bears on a right-sized question genuinely distinct from \
any adjacent candidate, propose_new.
- Else skip.

## Step 3: Strength (only if link_existing)

Answer three yes/no questions:
- `q1_scope_expansion`: Would the candidate's summary need to expand to \
incorporate this item?
- `q2_representative_citation`: Would this item plausibly be in a 5-item \
representative citation list for the candidate's topic?
- `q3_distinguishing_evidence`: Does the item adjudicate the candidate's \
claim against alternatives, beyond just adding a data point?

The strength score is computed from these downstream.

## Response format

Respond with ONLY a JSON object (no markdown fencing):
{
  "candidate_classifications": [
    { "topic_id": "<id>", "candidate_question": "<one phrase>", "item_question": "<one phrase>",
      "label": "same_question" | "subset_question" | "adjacent_question" | "surface_overlap" }
  ],
  "decision": "propose_new" | "link_existing" | "skip",
  "topic_id": "<existing topic_id if link_existing, null otherwise>",
  "connection_type": "supports" | "extends" | "contradicts" | "shifts" | null,
  "fit_questions": {
    "q1_scope_expansion": <bool>,
    "q2_representative_citation": <bool>,
    "q3_distinguishing_evidence": <bool>
  },
  "reasoning_trace": "<short explanation>",
  "new_topic": {
    "title": "<if propose_new>",
    "summary": "<one sentence naming what is being investigated>",
    "topic_id": "<slug>",
    "tags": ["<...>"],
    "suggested_directions": ["<...>"],
    "scope_test_question": "<the question this topic would track>",
    "differentiation_from_closest": "<closest candidate topic_id and concrete distinguishing claim, or 'none' if no candidates>"
  }
}
"""


CONFIRM_SYSTEM_PROMPT = """\
You are a second, independent reviewer for a topic store that follows \
{domain}. Items are papers, release notes, posts, discussion threads and \
tutorials; a topic tracks one question across all of them. A first-pass analyst decided that a new item does not fit any \
existing topic and proposed a new topic for it. Your job is to check \
that call. You see the item, the closest existing topics, and the proposed \
topic.

## Why this check exists

The two possible mistakes are not symmetric. Linking an item to an existing \
topic that only partly covers it costs one slightly noisy observation, and the \
topic's summary can later be broadened. Creating a topic that should have been \
a facet of an existing one creates a permanent near-duplicate that splits \
future evidence. Two thirds of newly created topics never receive a second \
observation. So the bar for a new topic is higher than "the item's question \
differs from every candidate's question".

## The test

A new topic is warranted only if BOTH hold:

1. No candidate is the same *question* viewed from a different facet. Ask, \
for the closest candidate: if that topic's one-sentence summary were \
broadened by a clause, would it naturally cover this item? If yes, the item \
is an extension of that topic, not a new question. Different method, \
different artifact type, different source kind (a release where the topic \
holds papers, or the reverse), different subdomain, or a narrower instance \
are all facets, not new questions.

2. The proposed topic is right-sized on its own: it names one specifiable \
question that other, different items are likely to bear on, and it could \
plausibly accumulate 10-50 observations over a year. A topic that restates \
this one item's contribution, or that is one product's changelog under \
another name, fails this test.

If (1) fails: `link_existing` to that candidate with connection_type \
`extends`. If (1) holds but (2) fails: `skip`. Skip is a normal outcome for \
an above-threshold item that is interesting but not a direction. If both \
hold: `confirm_new`.

You do not assess link strength. A link made here is by construction an \
edge extension of the topic and is scored as such downstream.

## Response format

Respond with ONLY a JSON object (no markdown fencing):
{
  "verdict": "confirm_new" | "link_existing" | "skip",
  "topic_id": "<candidate topic_id if link_existing, else null>",
  "connection_type": "extends" | "supports" | null,
  "closest_candidate": "<topic_id of the closest candidate regardless of verdict>",
  "broadened_summary": "<if link_existing: the candidate summary with the clause that covers this item; else null>",
  "reasoning": "<two or three sentences>"
}
"""


@dataclass
class SynthesisDecision:
    item_url: str
    item_title: str
    decision: str  # link_existing, propose_new, skip
    topic_id: str | None
    connection_type: str | None
    strength: float
    reasoning_trace: str
    new_topic: dict | None = None
    # Non-persisted retrieval/prompt facts for the tuning harness: candidate
    # counts per channel, vector scores, token usage, classification labels.
    diagnostics: dict | None = None


def _find_tag_overlap_topics(
    item: dict, all_topics: list[dict], ignored_tags: list[str] | None = None
) -> list[dict]:
    item_tags = set(item.get("tags", [])) - set(ignored_tags or [])
    matches = []
    for topic in all_topics:
        topic_tags = set(topic.get("tags", []))
        overlap = item_tags & topic_tags
        if overlap:
            matches.append(topic)
    return matches


def _merge_candidates(
    vector_results: list[dict], tag_results: list[dict]
) -> list[dict]:
    seen = set()
    merged = []
    for topic in vector_results:
        tid = topic["topic_id"]
        if tid not in seen:
            seen.add(tid)
            merged.append(topic)
    for topic in tag_results:
        tid = topic["topic_id"]
        if tid not in seen:
            seen.add(tid)
            merged.append(topic)
    return merged


def _get_evidence_count(topic: dict) -> int:
    """Extract evidence count from the topic's latest snapshot."""
    snapshots = topic.get("snapshots", [])
    if snapshots:
        return snapshots[-1].get("evidence_count", 0)
    return 0


OBSERVATION_SAMPLE_SIZE = 3

# How many of the current batch's proposed topics to show as candidates for a
# later item in the same batch. They are not in the vector index yet, so they
# are ranked here by cosine against the item embedding.
BATCH_CANDIDATE_LIMIT = 5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _select_batch_candidates(
    item_embedding: list[float] | None, proposed: list[dict], k: int
) -> list[dict]:
    """Top-k batch-proposed topics by cosine to the item, each annotated with
    a vector_score so the prompt shows them like vector-search results.
    Proposals without an embedding (or an item without one) are returned
    unranked and uncapped, which matches the previous behavior."""
    if not proposed:
        return []
    if not item_embedding or any(not t.get("embedding") for t in proposed):
        return list(proposed)
    scored = []
    for t in proposed:
        c = dict(t)
        c["vector_score"] = _cosine(item_embedding, t["embedding"])
        scored.append(c)
    scored.sort(key=lambda t: t["vector_score"], reverse=True)
    return scored[:k]


def _representative_observation_titles(
    store: Store,
    topic_id: str,
    n: int = OBSERVATION_SAMPLE_SIZE,
    exclude_item_url: str | None = None,
) -> list[str]:
    """Return up to n observation titles that best characterize the topic.

    Picks highest-strength non-minor observations, breaking ties by recency.
    Returns an empty list for newly-proposed topics (no observations yet) or
    on store errors. exclude_item_url hides one item's own observation
    (used by the tuning harness so a replayed item cannot see itself).
    """
    try:
        observations = store.get_observations_for_topic(topic_id)
    except Exception:
        return []
    if exclude_item_url:
        observations = [
            o for o in observations if o.get("item_url") != exclude_item_url
        ]
    if not observations:
        return []

    def sort_key(obs: dict) -> tuple:
        strength = obs.get("strength") or 0.0
        is_minor = bool(obs.get("minor"))
        date = obs.get("date")
        # Sort: non-minor first, then highest strength, then most recent.
        date_score = date.timestamp() if date else 0.0
        return (not is_minor, strength, date_score)

    ranked = sorted(observations, key=sort_key, reverse=True)
    titles = []
    for obs in ranked:
        title = obs.get("item_title")
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= n:
            break
    return titles


def _format_candidates(
    candidates: list[dict],
    store: Store | None = None,
    exclude_item_url: str | None = None,
) -> str:
    if not candidates:
        return "No existing topics found."
    lines = []
    for c in candidates:
        score_str = ""
        if "vector_score" in c:
            score_str = f" (vector similarity: {c['vector_score']:.3f})"
        evidence_count = _get_evidence_count(c)

        sample_block = ""
        if store is not None:
            samples = _representative_observation_titles(
                store, c["topic_id"], exclude_item_url=exclude_item_url
            )
            if samples:
                sample_lines = "\n".join(f"    * {t}" for t in samples)
                sample_block = f"\n  representative observations:\n{sample_lines}"

        lines.append(
            f"- topic_id: {c['topic_id']}\n"
            f"  title: {c['title']}\n"
            f"  summary: {c.get('summary', 'N/A')}\n"
            f"  status: {c.get('status', 'unknown')}"
            f"  evidence_count: {evidence_count}{score_str}"
            f"{sample_block}"
        )
    return "\n".join(lines)


def _format_proposed_topics(proposed: list[dict]) -> str:
    if not proposed:
        return "None yet."
    lines = []
    for t in proposed:
        lines.append(
            f"- topic_id: {t['topic_id']}\n"
            f"  title: {t['title']}\n"
            f"  summary: {t['summary']}"
        )
    return "\n".join(lines)


def _format_item(item: dict) -> str:
    parts = [f"Title: {item.get('title', 'N/A')}"]
    if item.get("summary"):
        parts.append(f"Summary: {item['summary']}")
    kind = []
    if item.get("source_type"):
        kind.append(f"Source type: {item['source_type']}")
    if item.get("evidence"):
        kind.append(f"Evidence kind: {item['evidence']}")
    if kind:
        parts.append("\n".join(kind))
    if item.get("tags"):
        parts.append(f"Tags: {', '.join(item['tags'])}")
    if item.get("content_raw"):
        parts.append(f"Content excerpt:\n{item['content_raw'][:4000]}")
    return "\n\n".join(parts)


def _sanitize_topic_id(topic_id: str) -> str:
    """Normalize a topic_id to a clean slug: lowercase, hyphens, no periods or special chars."""
    slug = topic_id.lower().strip()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _try_load_json(text: str) -> dict | None:
    """Best-effort json.loads. strict=False allows literal control chars in
    strings, which is the most common LLM emission glitch."""
    try:
        return json.loads(text, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_json_text(text: str) -> str:
    """Apply common LLM-emission fixes that don't require structural parsing."""
    # Smart quotes -> straight quotes. Mangles legit smart quotes in content,
    # which is acceptable; we'd rather parse than preserve typography.
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # Python literals -> JSON literals (only outside string content; we accept
    # the small risk of mangling the words inside strings since these are
    # uncommon inside our reasoning traces).
    text = re.sub(r":\s*True\b", ": true", text)
    text = re.sub(r":\s*False\b", ": false", text)
    text = re.sub(r":\s*None\b", ": null", text)
    # Trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _extract_json_object(text: str) -> str | None:
    """Find the outermost balanced { ... } via a brace-matching state machine.
    Handles nested objects and quoted strings; ignores braces inside strings.
    """
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _robust_json_parse(raw_text: str) -> dict:
    """Parse model JSON with progressive fallbacks. Raises JSONDecodeError
    only if every stage fails."""
    text = _strip_markdown_fence(raw_text)

    # 1. Direct parse with strict=False.
    parsed = _try_load_json(text)
    if parsed is not None:
        return parsed

    # 2. Normalize (smart quotes, Python literals, trailing commas), retry.
    normalized = _normalize_json_text(text)
    parsed = _try_load_json(normalized)
    if parsed is not None:
        return parsed

    # 3. Extract the outermost balanced object via brace matching.
    extracted = _extract_json_object(text)
    if extracted:
        parsed = _try_load_json(extracted)
        if parsed is not None:
            return parsed
        parsed = _try_load_json(_normalize_json_text(extracted))
        if parsed is not None:
            return parsed

    # 4. Same brace-matching against the normalized text.
    extracted_norm = _extract_json_object(normalized)
    if extracted_norm:
        parsed = _try_load_json(extracted_norm)
        if parsed is not None:
            return parsed

    # Out of options. Surface the original error for debugging.
    snippet = raw_text[:300].replace("\n", "\\n")
    logger.error("Could not parse model JSON. First 300 chars: %s", snippet)
    raise json.JSONDecodeError(
        "Could not parse model response after defensive repairs",
        raw_text,
        0,
    )


GATE_OVERRIDE_DEFAULT_STRENGTH = 0.65


def _enforce_link_gate(parsed: dict, all_candidates: list[dict]) -> dict:
    """Mechanical override of propose_new when candidates include strong fits.

    The synthesis prompt instructs the model to link rather than propose when
    any candidate is rated `same_question` or `subset_question`, but the model
    sometimes overrides that instruction with its own judgment (e.g. for
    survey papers that subsume multiple subset candidates). The reliable fix
    is to enforce the gate in code: if propose_new is paired with one or more
    same/subset classifications, force a link to the highest-evidence-count
    match, prefer same_question over subset_question.

    Strength is set to the default since the model didn't produce
    fit_questions for the propose_new path. The reasoning_trace is annotated
    so the override is visible in the audit log.
    """
    if parsed.get("decision") != "propose_new":
        return parsed
    classifications = parsed.get("candidate_classifications") or []
    if not isinstance(classifications, list):
        return parsed

    same_matches = [
        c for c in classifications
        if isinstance(c, dict) and c.get("label") == "same_question"
    ]
    subset_matches = [
        c for c in classifications
        if isinstance(c, dict) and c.get("label") == "subset_question"
    ]
    pool = same_matches if same_matches else subset_matches
    if not pool:
        return parsed

    candidate_lookup = {c["topic_id"]: c for c in all_candidates if c.get("topic_id")}

    def evidence_for(topic_id: str | None) -> int:
        if not topic_id:
            return 0
        candidate = candidate_lookup.get(topic_id)
        if not candidate:
            return 0
        return _get_evidence_count(candidate)

    pool_sorted = sorted(
        pool,
        key=lambda c: evidence_for(c.get("topic_id")),
        reverse=True,
    )
    chosen = pool_sorted[0]
    target_id = chosen.get("topic_id")
    if not target_id:
        return parsed

    note = (
        f"\n\nGATE OVERRIDE: model proposed a new topic but rated "
        f"{len(pool)} candidate(s) as "
        f"{'same_question' if same_matches else 'subset_question'}. "
        f"Forced link to '{target_id}' "
        f"(evidence_count={evidence_for(target_id)}). "
        f"Strength defaulted to {GATE_OVERRIDE_DEFAULT_STRENGTH} since "
        f"fit_questions were not produced under the propose_new path."
    )
    parsed["decision"] = "link_existing"
    parsed["topic_id"] = target_id
    parsed["connection_type"] = "supports"
    parsed["strength"] = GATE_OVERRIDE_DEFAULT_STRENGTH
    parsed["new_topic"] = None
    parsed["reasoning_trace"] = (parsed.get("reasoning_trace", "") or "") + note
    parsed["gate_override"] = True

    logger.info(
        "Gate override: propose_new -> link_existing %s (had %d same/subset matches)",
        target_id, len(pool),
    )
    return parsed


def _format_proposed_topic(new_topic: dict) -> str:
    lines = [
        f"title: {new_topic.get('title', '')}",
        f"summary: {new_topic.get('summary', '')}",
    ]
    if new_topic.get("scope_test_question"):
        lines.append(f"scope_test_question: {new_topic['scope_test_question']}")
    if new_topic.get("differentiation_from_closest"):
        lines.append(
            f"first_pass_differentiation: {new_topic['differentiation_from_closest']}"
        )
    return "\n".join(lines)


def _confirmation_candidates(
    parsed: dict,
    vector_candidates: list[dict],
    all_candidates: list[dict],
    n: int,
) -> list[dict]:
    """Top-n vector candidates plus anything the first pass classified."""
    by_id = {c["topic_id"]: c for c in all_candidates if c.get("topic_id")}
    chosen: list[dict] = []
    seen: set[str] = set()
    for c in vector_candidates[:n]:
        if c["topic_id"] not in seen:
            seen.add(c["topic_id"])
            chosen.append(c)
    for cls in parsed.get("candidate_classifications") or []:
        if not isinstance(cls, dict):
            continue
        tid = cls.get("topic_id")
        if tid and tid not in seen and tid in by_id:
            seen.add(tid)
            chosen.append(by_id[tid])
    return chosen


# Strength assigned to every link made by the confirmation pass. The reviewer
# is not asked the fit questions: having just argued that the item belongs,
# it answered yes to all three nearly every time (0.95) in tuning runs, which
# is agreement with itself, not calibration. What these links are is known
# by construction: an item the first pass judged distinct enough to want its
# own topic, pulled back as an edge extension. That is real scope evidence,
# so it stays above the 0.7 minor floor, but it should not outrank a topic's
# core observations (0.80/0.95) when representative examples are chosen.
CONFIRM_LINK_STRENGTH = 0.70


def _apply_confirmation(parsed: dict, confirm: dict, candidate_ids: set[str]) -> dict:
    """Fold the confirmation verdict into the first-pass decision.

    confirm_new keeps propose_new. link_existing to a shown candidate becomes
    an `extends` link at CONFIRM_LINK_STRENGTH. skip becomes skip. A
    malformed verdict, or a link to a topic that was not shown, is treated
    as confirm_new so a bad second call can never invent a link; the
    reasoning trace records what happened.
    """
    verdict = confirm.get("verdict")
    reasoning = (confirm.get("reasoning") or "").strip()
    closest = confirm.get("closest_candidate")
    note_head = "\n\nCONFIRMATION PASS: "

    if verdict == "link_existing":
        target = _sanitize_topic_id(confirm.get("topic_id") or "")
        if target in candidate_ids:
            parsed["decision"] = "link_existing"
            parsed["topic_id"] = target
            parsed["connection_type"] = confirm.get("connection_type") or "extends"
            parsed["strength"] = CONFIRM_LINK_STRENGTH
            parsed["fit_questions"] = None
            parsed["new_topic"] = None
            note = (
                f"first pass proposed a new topic; independent reviewer judged "
                f"the item an extension of '{target}'. Strength fixed at "
                f"{CONFIRM_LINK_STRENGTH} (edge extension; not reviewer-scored). "
                f"{reasoning}"
            )
            if confirm.get("broadened_summary"):
                note += f"\nBroadened summary: {confirm['broadened_summary']}"
            parsed["confirmation"] = "overridden_to_link"
        else:
            note = (
                f"reviewer chose link_existing to '{target}', which was not "
                f"among the shown candidates; keeping propose_new. {reasoning}"
            )
            parsed["confirmation"] = "confirmed_new"
    elif verdict == "skip":
        parsed["decision"] = "skip"
        parsed["topic_id"] = None
        parsed["connection_type"] = None
        parsed["strength"] = 0.0
        parsed["new_topic"] = None
        note = (
            f"first pass proposed a new topic; independent reviewer judged it "
            f"not right-sized (closest: {closest}). {reasoning}"
        )
        parsed["confirmation"] = "overridden_to_skip"
    else:
        note = f"independent reviewer confirmed the new topic (closest: {closest}). {reasoning}"
        parsed["confirmation"] = "confirmed_new"

    parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + note_head + note
    logger.info(
        "  confirmation=%s verdict=%s topic=%s closest=%s",
        parsed["confirmation"], verdict, parsed.get("topic_id"), closest,
    )
    return parsed


def _apply_jev_confirmation(parsed: dict, scores: list[dict], threshold: float) -> dict:
    """Fold per-candidate Jev scores into the first-pass decision.

    scores: [{topic_id, link_probability, label, confidence}] for the shown
    candidates. The candidate with the highest link probability wins; at or
    above threshold the proposal becomes an `extends` link to it at
    CONFIRM_LINK_STRENGTH, otherwise the proposal stands. No skip path: Jev
    is not asked whether the proposed topic is right-sized.
    """
    usable = [s for s in scores if isinstance(s.get("link_probability"), (int, float)) and s.get("topic_id")]
    note_head = "\n\nCONFIRMATION PASS (jev): "
    if not usable:
        parsed["confirmation"] = "confirmed_new"
        parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + note_head + "no candidate scores available; keeping propose_new."
        logger.info("  confirmation=confirmed_new (jev) no candidate scores available")
        return parsed
    best = max(usable, key=lambda s: s["link_probability"])
    ranked = ", ".join(
        f"{s['topic_id']}={s['link_probability']:.2f}/{(s.get('label') or '?')[:8]}"
        for s in sorted(usable, key=lambda s: -s["link_probability"])
    )
    if best["link_probability"] >= threshold:
        parsed["decision"] = "link_existing"
        parsed["topic_id"] = best["topic_id"]
        parsed["connection_type"] = "extends"
        parsed["strength"] = CONFIRM_LINK_STRENGTH
        parsed["fit_questions"] = None
        parsed["new_topic"] = None
        parsed["confirmation"] = "overridden_to_link"
        note = (
            f"first pass proposed a new topic; Jev scored p(belongs)={best['link_probability']:.2f} "
            f"(relationship {best.get('label')}, confidence {best.get('confidence')}) for "
            f"'{best['topic_id']}', above the {threshold} threshold. Strength fixed at "
            f"{CONFIRM_LINK_STRENGTH} (edge extension). All scores: {ranked}"
        )
    else:
        parsed["confirmation"] = "confirmed_new"
        note = f"no candidate reached p(belongs) {threshold}; best {best['topic_id']}={best['link_probability']:.2f}. All scores: {ranked}"
    parsed["jev_scores"] = usable
    parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + note_head + note
    logger.info(
        "  confirmation=%s (jev) best=%s p=%.2f threshold=%.2f",
        parsed["confirmation"], best["topic_id"], best["link_probability"], threshold,
    )
    return parsed


def _confirm_proposal_decision(
    parsed: dict,
    item: dict,
    vector_candidates: list[dict],
    all_candidates: list[dict],
    store: Store,
    settings: Settings,
    decision_model: DecisionModel | None,
    exclude_url: str | None,
) -> dict:
    """Decision-model review of a propose_new decision: score each shown
    candidate as an item-topic pair and demote on a confident match."""
    from topicsleuth.backends import jev

    if decision_model is None:
        logger.warning("confirm_backend=decision but no decision model was supplied; keeping propose_new")
        return _apply_jev_confirmation(parsed, [], settings.synthesize.confirm_link_threshold)
    shown = _confirmation_candidates(
        parsed, vector_candidates, all_candidates, settings.synthesize.confirm_candidates
    )
    scores: list[dict] = []
    for c in shown:
        reps = _representative_observation_titles(store, c["topic_id"], exclude_item_url=exclude_url)
        topic = dict(c)
        topic["evidence_count"] = _get_evidence_count(c)
        try:
            sc = jev.score_pair(decision_model, item, topic, reps)
        except Exception as e:
            logger.warning("decision model call failed for %s: %s", c["topic_id"], e)
            continue
        sc["topic_id"] = c["topic_id"]
        scores.append(sc)
    return _apply_jev_confirmation(parsed, scores, settings.synthesize.confirm_link_threshold)


def _confirm_proposal(
    parsed: dict,
    item_text: str,
    vector_candidates: list[dict],
    all_candidates: list[dict],
    store: Store,
    settings: Settings,
    judge: Judge,
    exclude_url: str | None,
):
    """Run the second-pass review of a propose_new decision. Returns the
    updated parsed dict and the reviewer's completion (for diagnostics)."""
    shown = _confirmation_candidates(
        parsed, vector_candidates, all_candidates, settings.synthesize.confirm_candidates
    )
    candidates_text = _format_candidates(shown, store=store, exclude_item_url=exclude_url)
    user_content = (
        f"## Item\n{item_text}\n\n"
        f"## Closest existing topics\n{candidates_text}\n\n"
        f"## Proposed new topic (first pass)\n"
        f"{_format_proposed_topic(parsed.get('new_topic') or {})}"
    )
    response = judge.complete(render_prompt(CONFIRM_SYSTEM_PROMPT, settings), user_content, max_tokens=1024)
    try:
        confirm = _robust_json_parse(response.text)
    except json.JSONDecodeError:
        logger.warning("Confirmation pass returned unparseable JSON; keeping propose_new")
        confirm = {"verdict": "confirm_new", "reasoning": "(unparseable reviewer response)"}
    parsed = _apply_confirmation(parsed, confirm, {c["topic_id"] for c in shown})
    return parsed, response


def _derive_strength(fit_questions: dict | None) -> float | None:
    """Map the three binary fit questions to a strength score.

    Q2 (representative_citation) is the load-bearing question; the others
    refine within bands. Returns None if fit_questions are missing or
    malformed, in which case callers should fall back.

    Mapping:
    - All 3 yes -> 0.95 (scope-expanding, citable, distinguishing)
    - Q2 yes + (Q1 or Q3) yes -> 0.80 (citable plus one strong property)
    - Q2 yes only -> 0.65 (citable but neither expanding nor distinguishing)
    - Q2 no + (Q1 or Q3 or both) yes -> 0.50 (not citable but some property holds)
    - All 3 no -> 0.30 (link is forced by the gate but the item is marginal)
    """
    if not isinstance(fit_questions, dict):
        return None
    try:
        q1 = bool(fit_questions["q1_scope_expansion"])
        q2 = bool(fit_questions["q2_representative_citation"])
        q3 = bool(fit_questions["q3_distinguishing_evidence"])
    except (KeyError, TypeError):
        return None

    yes_count = sum([q1, q2, q3])
    if yes_count == 3:
        return 0.95
    if q2 and yes_count == 2:
        return 0.80
    if q2 and yes_count == 1:
        return 0.65
    if not q2 and yes_count >= 1:
        return 0.50
    return 0.30


def _parse_decision(raw_text: str) -> dict:
    parsed = _robust_json_parse(raw_text)

    # Sanitize topic_ids
    if parsed.get("topic_id"):
        parsed["topic_id"] = _sanitize_topic_id(parsed["topic_id"])
    if parsed.get("new_topic") and parsed["new_topic"].get("topic_id"):
        parsed["new_topic"]["topic_id"] = _sanitize_topic_id(parsed["new_topic"]["topic_id"])

    # Derive strength from fit_questions for link_existing decisions. The
    # model is instructed not to emit strength directly, but if it does (legacy
    # behavior), the derived value wins when fit_questions are present.
    if parsed.get("decision") == "link_existing":
        derived = _derive_strength(parsed.get("fit_questions"))
        if derived is not None:
            parsed["strength"] = derived
        elif parsed.get("strength") is None:
            parsed["strength"] = 0.0
    elif parsed.get("decision") == "propose_new":
        # The founding observation links the proposing item to a brand-new
        # topic. By construction the item is the central evidence motivating
        # the topic: Q2 (citable for representative set) is yes; Q1 (scope
        # expanding) is yes since the topic summary was written to describe
        # this item's contribution; Q3 (distinguishing) is yes since
        # differentiation_from_closest articulates what makes this item
        # distinct from existing topics. We don't ask the model for explicit
        # fit_questions here (the prompt's strength assessment is keyed to
        # link_existing only), so we default to 0.85 -- one notch below the
        # all-three-yes derived strength of 0.95, leaving room for later
        # items that explicitly score the new topic.
        parsed["strength"] = 0.85
    else:
        parsed.setdefault("strength", 0.0)

    # Fold the structured audit fields into reasoning_trace so they persist
    # alongside the explanation without expanding the observation schema.
    extras = []
    classifications = parsed.get("candidate_classifications")
    if classifications:
        lines = ["Candidate classifications:"]
        for c in classifications:
            tid = c.get("topic_id", "?")
            label = c.get("label", "?")
            cq = c.get("candidate_question", "")
            iq = c.get("item_question", "")
            lines.append(f"- {tid} [{label}]")
            if cq:
                lines.append(f"    candidate_question: {cq}")
            if iq:
                lines.append(f"    item_question: {iq}")
        extras.append("\n".join(lines))

    fit_questions = parsed.get("fit_questions")
    if isinstance(fit_questions, dict) and parsed.get("decision") == "link_existing":
        lines = ["Fit questions:"]
        for k in ("q1_scope_expansion", "q2_representative_citation",
                 "q3_distinguishing_evidence"):
            v = fit_questions.get(k)
            lines.append(f"  {k}: {v}")
        lines.append(f"  derived_strength: {parsed.get('strength')}")
        extras.append("\n".join(lines))

    new_topic = parsed.get("new_topic") or {}
    scope_q = new_topic.get("scope_test_question")
    diff = new_topic.get("differentiation_from_closest")
    if scope_q:
        extras.append(f"Scope test question: {scope_q}")
    if diff:
        extras.append(f"Differentiation from closest: {diff}")

    if extras:
        base = parsed.get("reasoning_trace", "") or ""
        parsed["reasoning_trace"] = (base + "\n\n" + "\n\n".join(extras)).strip()

    return parsed


def synthesize_item(
    item: dict,
    eval_result: ItemEval,
    store: Store,
    settings: Settings,
    judge: Judge,
    proposed_topics: list[dict],
    leave_one_out: bool = False,
    decision_model: DecisionModel | None = None,
) -> SynthesisDecision:
    all_topics = [
        t for t in store.get_all_topics() if t.get("status") != "retired"
    ]

    # Replay mode (tune.py): the item has already been recorded, so the topic
    # it founded, and its own title among representative observations, would
    # leak the historical answer back into the prompt. Hide both: drop topics
    # whose only observation is this item, and never show this item's title.
    hidden_topic_ids: set[str] = set()
    exclude_url: str | None = None
    if leave_one_out:
        exclude_url = item["url"]
        for obs in store.get_observations_for_item(exclude_url):
            if store.count_observations_for_topic(obs["topic_id"]) <= 1:
                hidden_topic_ids.add(obs["topic_id"])
        all_topics = [t for t in all_topics if t["topic_id"] not in hidden_topic_ids]

    is_cold_start = len(all_topics) == 0 and len(proposed_topics) == 0

    # Find candidates via vector search (filter retired post-query)
    vector_candidates = [
        t for t in store.nearest_topics(eval_result.embedding, settings.candidates.vector_limit)
        if t.get("status") != "retired" and t["topic_id"] not in hidden_topic_ids
    ]
    if not vector_candidates and all_topics:
        # Atlas answers a missing or misnamed vector index with an empty
        # result, not an error; left unnoticed, every item would propose a
        # new topic and the store would fragment silently.
        logger.warning(
            "nearest_topics returned nothing although %d topics exist; "
            "check the store's vector index", len(all_topics),
        )

    # Find candidates via tag overlap
    if settings.candidates.tag_overlap:
        tag_candidates = _find_tag_overlap_topics(
            item, all_topics, ignored_tags=settings.candidates.ignored_tags
        )
    else:
        tag_candidates = []

    # Merge and deduplicate
    candidates = _merge_candidates(vector_candidates, tag_candidates)

    # Include this batch's proposed topics as candidates, ranked by similarity
    # to the item and capped, so a long batch does not append every earlier
    # proposal to every later prompt.
    batch_candidates = _select_batch_candidates(
        eval_result.embedding, proposed_topics, k=BATCH_CANDIDATE_LIMIT
    )
    all_candidates = candidates + batch_candidates

    # Build prompt
    if is_cold_start and not proposed_topics:
        system = render_prompt(COLD_START_SYSTEM_PROMPT, settings)
    elif not all_topics and proposed_topics:
        system = render_prompt(COLD_START_SYSTEM_PROMPT, settings)
    else:
        system = render_prompt(SYNTHESIZE_SYSTEM_PROMPT, settings)

    item_text = _format_item(item)
    candidates_text = _format_candidates(
        all_candidates, store=store, exclude_item_url=exclude_url
    )

    if proposed_topics and not all_topics:
        user_content = (
            f"## New Item\n{item_text}\n\n"
            f"## Recently Proposed Topics (this batch)\n"
            f"{_format_proposed_topics(proposed_topics)}\n\n"
            f"## Existing Topics\n{candidates_text}"
        )
    else:
        user_content = (
            f"## New Item\n{item_text}\n\n"
            f"## Candidate Topics\n{candidates_text}"
        )

    response = judge.complete(system, user_content, max_tokens=2048)
    parsed = _parse_decision(response.text)
    parsed = _enforce_link_gate(parsed, all_candidates)

    confirm_response = None
    if (
        parsed.get("decision") == "propose_new"
        and settings.synthesize.confirm_proposals
        and all_candidates
    ):
        if settings.synthesize.confirm_backend == "decision":
            parsed = _confirm_proposal_decision(
                parsed, item, vector_candidates, all_candidates,
                store, settings, decision_model, exclude_url,
            )
        else:
            parsed, confirm_response = _confirm_proposal(
                parsed, item_text, vector_candidates, all_candidates,
                store, settings, judge, exclude_url,
            )

    diagnostics = _build_diagnostics(
        parsed, vector_candidates, tag_candidates, candidates, proposed_topics,
        response, confirm_response,
    )

    return SynthesisDecision(
        item_url=item["url"],
        item_title=item.get("title", ""),
        decision=parsed["decision"],
        topic_id=parsed.get("topic_id"),
        connection_type=parsed.get("connection_type"),
        strength=parsed.get("strength", 0.0),
        reasoning_trace=parsed.get("reasoning_trace", ""),
        new_topic=parsed.get("new_topic") if parsed["decision"] == "propose_new" else None,
        diagnostics=diagnostics,
    )


def _build_diagnostics(
    parsed: dict,
    vector_candidates: list[dict],
    tag_candidates: list[dict],
    merged_candidates: list[dict],
    proposed_topics: list[dict],
    response,
    confirm_response=None,
) -> dict:
    """Collect retrieval and prompt facts about one synthesis call.

    Purely observational: nothing here feeds back into the decision. Used by
    tune.py to answer questions like "was the linked topic in the vector
    top-k or did it come from the tag channel?" and "how big was the prompt?".
    """
    vector_scores = [
        round(float(t["vector_score"]), 4)
        for t in vector_candidates if "vector_score" in t
    ]
    vector_rank = {t["topic_id"]: i + 1 for i, t in enumerate(vector_candidates)}
    vector_score_by_id = {
        t["topic_id"]: round(float(t["vector_score"]), 4)
        for t in vector_candidates if "vector_score" in t
    }
    batch_ids = {t.get("topic_id") for t in proposed_topics}

    classifications = parsed.get("candidate_classifications") or []
    labels: list[dict] = []
    if isinstance(classifications, list):
        for c in classifications:
            if not isinstance(c, dict):
                continue
            tid = c.get("topic_id")
            labels.append({
                "topic_id": tid,
                "label": c.get("label"),
                "vector_rank": vector_rank.get(tid),
                "vector_score": vector_score_by_id.get(tid),
                "from_batch": tid in batch_ids,
            })

    chosen = parsed.get("topic_id") if parsed.get("decision") == "link_existing" else None
    if chosen in vector_rank:
        chosen_channel = "vector"
    elif chosen in batch_ids:
        chosen_channel = "batch"
    elif chosen is not None:
        chosen_channel = "tag_only"
    else:
        chosen_channel = None

    return {
        "confirmation": parsed.get("confirmation"),
        "jev_scores": parsed.get("jev_scores"),
        "confirm_input_tokens": getattr(confirm_response, "input_tokens", None),
        "confirm_output_tokens": getattr(confirm_response, "output_tokens", None),
        "vector_candidates": len(vector_candidates),
        "tag_candidates": len(tag_candidates),
        "tag_only_candidates": len(merged_candidates) - len(vector_candidates),
        "batch_candidates": len(proposed_topics),
        "total_candidates": len(merged_candidates) + len(proposed_topics),
        "vector_top1": vector_scores[0] if vector_scores else None,
        "vector_top5": vector_scores[:5],
        "vector_min": vector_scores[-1] if vector_scores else None,
        "chosen_channel": chosen_channel,
        "chosen_vector_rank": vector_rank.get(chosen) if chosen else None,
        "chosen_vector_score": vector_score_by_id.get(chosen) if chosen else None,
        "classified": labels,
        "gate_override": bool(parsed.get("gate_override")),
        "input_tokens": getattr(response, "input_tokens", None),
        "output_tokens": getattr(response, "output_tokens", None),
    }


def synthesize_items(
    items: list[dict],
    eval_results: list[ItemEval],
    store: Store,
    settings: Settings,
    judge: Judge,
    embedder: Embedder,
    decision_model: DecisionModel | None = None,
) -> list[SynthesisDecision]:

    # Build lookup from url to eval result
    eval_by_url = {er.url: er for er in eval_results}

    # Filter to above-threshold items
    above_threshold = [
        (item, eval_by_url[item["url"]])
        for item in items
        if item["url"] in eval_by_url and eval_by_url[item["url"]].above_threshold
    ]

    if not above_threshold:
        logger.info("No items above threshold, skipping synthesis.")
        return []

    proposed_topics: list[dict] = []
    decisions: list[SynthesisDecision] = []

    for i, (item, eval_result) in enumerate(above_threshold, 1):
        logger.info(
            "Synthesizing item %d/%d: %s",
            i,
            len(above_threshold),
            item.get("title", "")[:60],
        )
        try:
            decision = synthesize_item(
                item, eval_result, store, settings, judge, proposed_topics,
                decision_model=decision_model,
            )
            decisions.append(decision)

            # Track proposed topics so subsequent items can link to them.
            # The embedding is set on decision.new_topic directly so record.py
            # persists it; proposed_topics references the same dict.
            if decision.decision == "propose_new" and decision.new_topic:
                topic_text = (
                    f"{decision.new_topic['title']}\n\n"
                    f"{decision.new_topic['summary']}"
                )
                decision.new_topic["embedding"] = embedder.embed_documents([topic_text])[0]
                proposed_topics.append(decision.new_topic)

            logger.info("  decision=%s topic=%s", decision.decision, decision.topic_id)

        except Exception:
            logger.error(
                "Failed to synthesize item: %s", item.get("url"), exc_info=True
            )

    return decisions
