from __future__ import annotations

import json
import logging
from typing import Protocol
import re
from dataclasses import asdict, dataclass, field

from topicsleuth.interfaces import DecisionModel, Embedder, Judge, Store
from topicsleuth.settings import Settings
from topicsleuth.rubric import (
    CONFIRM_LINK_STRENGTH,
    CONNECTION_TYPES,
    DECISIONS,
    DEFAULT_CONNECTION_TYPE,
    GATE_OVERRIDE_DEFAULT_STRENGTH,
    REVIEWER_CONNECTION_TYPE,
    FIT_QUESTIONS,
    derive_strength,
    cold_start_system_prompt,
    confirm_system_prompt,
    synthesize_system_prompt,
)


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

class DecisionError(ValueError):
    """The judge's answer parsed but does not satisfy the response contract:
    an unknown decision, a link to a topic it was not shown, a proposal with
    no title. The item is reported as failed, never written."""


@dataclass
class Link:
    """One topic an item is evidence for, with the judge's assessment of
    that relationship. A link decision carries one or more."""
    topic_id: str
    connection_type: str
    strength: float
    fit_questions: dict | None = None


@dataclass
class SynthesisDecision:
    item_url: str
    item_title: str
    decision: str  # link_existing, propose_new, skip, or failed
    # The first link, for readers that want one topic; ``links`` has them all.
    topic_id: str | None
    connection_type: str | None
    strength: float
    reasoning_trace: str
    links: list[Link] = field(default_factory=list)
    new_topic: dict | None = None

    def __post_init__(self):
        # One invariant, whichever way the decision was built: for a link
        # decision, ``links`` holds every topic and the three scalar fields
        # mirror its first entry.
        if self.decision == "link_existing":
            if not self.links and self.topic_id:
                self.links = [Link(topic_id=self.topic_id,
                                   connection_type=self.connection_type or DEFAULT_CONNECTION_TYPE,
                                   strength=self.strength)]
            if self.links:
                first = self.links[0]
                self.topic_id, self.connection_type, self.strength = first.topic_id, first.connection_type, first.strength
    # Non-persisted retrieval/prompt facts for the tuning harness: candidate
    # counts per channel, vector scores, token usage, classification labels.
    diagnostics: dict | None = None
    # Set when decision is "failed": what went wrong, so the caller can
    # retry the item instead of acknowledging it.
    error: str | None = None


def intended_links(observations: list[dict], store: Store) -> list[Link]:
    """The decision an item's stored observations still call for, resolved
    against what curation has done since.

    Each observation records the whole decision under ``decision_links``
    (an observation without the field is its own intent); the union of
    those copies is the plan. A plan is not revised by curation, so it is
    read against curation's own marks:

    - a topic in any of the item's observations' ``removed_from`` history
      was deliberately taken off the item (by a move, merge, fold, split or
      reprocess, however many steps ago); it is not unfinished work,
      whatever the plan says;
    - a planned topic the item is not yet on is followed through merges
      (``merged_into``) to its live successor, and dropped if that successor
      is already present, or if the topic was deleted or retired by a
      split. A topic that is only marked ``deleting`` is blocked, not
      removed: the link is kept, and recording fails the item until the
      mark is resolved, so an incomplete decision never looks complete;
    - an observation the item holds on a retired merge source (it landed
      as the topic was merged, and its relocation did not finish) stands
      for the live successor, where recording puts it before trusting it.

    The result is every link the item has, plus every planned link that is
    still writable and was not removed on purpose, in first-seen order."""
    present: dict[str, Link] = {}
    planned: dict[str, Link] = {}
    removed: set[str] = set()
    for o in observations:
        tid = o.get("topic_id")
        if tid and tid not in present:
            present[tid] = Link(topic_id=tid, connection_type=o.get("connection_type") or DEFAULT_CONNECTION_TYPE,
                                strength=o.get("strength") if o.get("strength") is not None else 0.0,
                                fit_questions=o.get("fit_questions"))
        removed.update(o.get("removed_from") or [])
        for l in o.get("decision_links") or []:
            pid = l.get("topic_id")
            if pid and pid not in planned:
                planned[pid] = Link(topic_id=pid, connection_type=l.get("connection_type") or DEFAULT_CONNECTION_TYPE,
                                    strength=l.get("strength") if l.get("strength") is not None else 0.0,
                                    fit_questions=l.get("fit_questions"))

    links: dict[str, Link] = {}
    for pid, link in present.items():
        topic = store.get_topic(pid)
        target = pid
        if topic and topic.get("status") == "retired" and topic.get("merged_into"):
            target = _live_successor(topic["merged_into"], store) or pid
        if target not in links:
            links[target] = Link(topic_id=target, connection_type=link.connection_type,
                                 strength=link.strength, fit_questions=link.fit_questions)
    for pid, link in planned.items():
        if pid in links or pid in removed:
            continue
        target = _live_successor(pid, store)
        if target is None or target in links or target in removed:
            continue
        links[target] = Link(topic_id=target, connection_type=link.connection_type,
                             strength=link.strength, fit_questions=link.fit_questions)
    return list(links.values())


def _live_successor(topic_id: str, store: Store) -> str | None:
    """Follow merges to the topic that now holds what ``topic_id`` tracked.
    None if it was deleted or retired by a split. A topic marked
    ``deleting`` is returned as it is: temporarily blocked, not gone."""
    seen: set[str] = set()
    while topic_id and topic_id not in seen:
        seen.add(topic_id)
        topic = store.get_topic(topic_id)
        if topic is None:
            return None
        if topic.get("status") != "retired":
            return topic_id
        topic_id = topic.get("merged_into")
    return None


def failed_decision(item: dict, error: BaseException | str) -> SynthesisDecision:
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return SynthesisDecision(
        item_url=item["url"], item_title=item.get("title", ""), decision="failed",
        topic_id=None, connection_type=None, strength=0.0, reasoning_trace="", error=text,
    )


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

# What synthesize reads of a topic and of a topic's observations, declared
# to the store (the ``fields`` of ``Store.list_topics``, ``nearest_topics``
# and ``get_observations_for_topic``) so a store that pays per byte ships
# no embeddings or reasoning traces that nothing here looks at. A topic's
# head is what the prompt shows of it plus what candidate selection
# filters on; an observation's title row is what ranks and names it.
TOPIC_HEAD_FIELDS = ("topic_id", "title", "summary", "status", "tags", "snapshots")
OBSERVATION_TITLE_FIELDS = ("item_url", "item_title", "strength", "minor", "date")

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
        observations = store.get_observations_for_topic(topic_id, fields=OBSERVATION_TITLE_FIELDS)
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


def _drop_unshown_classifications(parsed: dict, shown_ids: set[str]) -> dict:
    """Keep only classifications of topics the judge was actually shown. A
    fabricated same_question entry would otherwise let the gate force a link
    to a topic that does not exist."""
    classifications = parsed.get("candidate_classifications")
    if not isinstance(classifications, list):
        parsed["candidate_classifications"] = []
        return parsed
    kept, dropped = [], []
    for c in classifications:
        if isinstance(c, dict) and _sanitize_topic_id(str(c.get("topic_id") or "")) in shown_ids:
            c["topic_id"] = _sanitize_topic_id(str(c["topic_id"]))
            kept.append(c)
        else:
            dropped.append(c.get("topic_id") if isinstance(c, dict) else c)
    if dropped:
        logger.warning("Judge classified topics it was not shown; ignored: %s", dropped)
        parsed["reasoning_trace"] = (
            (parsed.get("reasoning_trace") or "")
            + f"\n\nIGNORED: classifications of unshown topics {dropped}"
        )
    parsed["candidate_classifications"] = kept
    return parsed


def _unique_topic_id(topic_id: str, store: Store, proposed_topics: list[dict]) -> str:
    """A slug no existing or batch-proposed topic already uses."""
    taken = {t["topic_id"] for t in proposed_topics}
    candidate, n = topic_id, 1
    while candidate in taken or store.get_topic(candidate) is not None:
        n += 1
        candidate = f"{topic_id}-{n}"
    return candidate


def _validate_decision(parsed: dict, shown_ids: set[str], store: Store, proposed_topics: list[dict],
                       settings: Settings | None = None) -> dict:
    """Enforce the response contract on a parsed decision. Raises
    DecisionError for anything that must not be written; normalizes the
    rest (connection type, proposal fields, slug uniqueness)."""
    decision = parsed.get("decision")
    if decision not in DECISIONS:
        raise DecisionError(f"unknown decision {decision!r}")

    if decision == "link_existing":
        links = parsed.get("links") or []
        if not links:
            raise DecisionError("link_existing with no links")
        fixed = parsed.get("gate_override") or parsed.get("confirmation") == "overridden_to_link"
        seen: set[str] = set()
        kept = []
        for link in links:
            topic_id = link.get("topic_id")
            if not topic_id or topic_id not in shown_ids:
                raise DecisionError(f"link to {topic_id!r}, which was not among the shown candidates")
            if topic_id in seen:
                continue
            seen.add(topic_id)
            if link.get("strength") is None and not fixed:
                raise DecisionError(f"link to {topic_id!r} without the three fit questions answered")
            strength = link.get("strength")
            if not isinstance(strength, (int, float)) or strength != strength or not 0.0 <= strength <= 1.0:
                raise DecisionError(f"strength {strength!r} is not a number in [0, 1]")
            ctype = link.get("connection_type")
            if ctype is None:
                link["connection_type"] = DEFAULT_CONNECTION_TYPE
            elif ctype not in CONNECTION_TYPES:
                raise DecisionError(f"unknown connection_type {ctype!r}")
            kept.append(link)
        # The first link always stands. A later one below the secondary
        # floor is dropped: the judge said the item bears on that question
        # too, but not strongly enough for this store to file it.
        floor = settings.synthesize.secondary_link_floor if settings else 0.0
        dropped = [l for l in kept[1:] if l["strength"] < floor]
        if dropped:
            kept = [kept[0]] + [l for l in kept[1:] if l["strength"] >= floor]
            parsed["dropped_links"] = dropped
            parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + (
                "\n\nSECONDARY LINKS DROPPED (below floor "
                f"{floor}): " + ", ".join(f"{l['topic_id']} at {l['strength']}" for l in dropped)
            )
        parsed["links"] = kept
        parsed["strength"] = kept[0]["strength"]
        parsed["new_topic"] = None

    elif decision == "propose_new":
        new_topic = parsed.get("new_topic")
        if not isinstance(new_topic, dict):
            raise DecisionError("propose_new without a new_topic")
        for key in ("title", "summary"):
            if not isinstance(new_topic.get(key), str) or not new_topic[key].strip():
                raise DecisionError(f"proposed topic has no {key}")
        slug = _sanitize_topic_id(str(new_topic.get("topic_id") or ""))
        if not slug:
            raise DecisionError("proposed topic has no usable topic_id")
        unique = _unique_topic_id(slug, store, proposed_topics)
        if unique != slug:
            logger.info("Proposed topic_id %s is taken; using %s", slug, unique)
            parsed["reasoning_trace"] = (
                (parsed.get("reasoning_trace") or "") + f"\n\nTOPIC ID: '{slug}' is taken; created as '{unique}'"
            )
        new_topic["topic_id"] = unique
        for key in ("tags", "suggested_directions"):
            value = new_topic.get(key)
            new_topic[key] = [str(v) for v in value] if isinstance(value, list) else []
        parsed["links"] = []

    else:  # skip
        parsed["links"] = []
        parsed["new_topic"] = None
        parsed["strength"] = 0.0

    return parsed


def _enforce_link_gate(parsed: dict, all_candidates: list[dict]) -> dict:
    """Mechanical override of propose_new when candidates include strong fits.

    The synthesis prompt instructs the model to link rather than propose when
    any candidate is rated `same_question` or `subset_question`, but the model
    sometimes overrides that instruction with its own judgment (e.g. for
    survey papers that subsume multiple subset candidates). The reliable fix
    is to enforce the gate in code: if propose_new is paired with same or
    subset classifications, force links to every same_question candidate, or
    failing that every subset_question candidate, most evidence first.

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
    targets: list[str] = []
    for c in pool_sorted:
        tid = c.get("topic_id")
        if tid and tid not in targets:
            targets.append(tid)
    if not targets:
        return parsed

    note = (
        f"\n\nGATE OVERRIDE: model proposed a new topic but rated "
        f"{len(pool)} candidate(s) as "
        f"{'same_question' if same_matches else 'subset_question'}. "
        f"Forced links to {targets} "
        f"(evidence_count {[evidence_for(t) for t in targets]}). "
        f"Strength defaulted to {GATE_OVERRIDE_DEFAULT_STRENGTH} since "
        f"fit_questions were not produced under the propose_new path."
    )
    parsed["decision"] = "link_existing"
    parsed["links"] = [
        {"topic_id": t, "connection_type": DEFAULT_CONNECTION_TYPE,
         "fit_questions": None, "strength": GATE_OVERRIDE_DEFAULT_STRENGTH}
        for t in targets
    ]
    parsed["strength"] = GATE_OVERRIDE_DEFAULT_STRENGTH
    parsed["new_topic"] = None
    parsed["reasoning_trace"] = (parsed.get("reasoning_trace", "") or "") + note
    parsed["gate_override"] = True

    logger.info(
        "Gate override: propose_new -> link_existing %s (had %d same/subset matches)",
        targets, len(pool),
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
            parsed["links"] = [{
                "topic_id": target,
                "connection_type": confirm.get("connection_type") or REVIEWER_CONNECTION_TYPE,
                "fit_questions": None, "strength": CONFIRM_LINK_STRENGTH,
            }]
            parsed["strength"] = CONFIRM_LINK_STRENGTH
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
        parsed["links"] = []
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


def _apply_decision_confirmation(parsed: dict, scores: list[dict], threshold: float) -> dict:
    """Fold per-candidate decision-model scores into the first-pass decision.

    scores: [{topic_id, link_probability, label, confidence, ...}] for the
    shown candidates (a PairScore per candidate, as a dict). The candidate
    with the highest link probability wins; at or above threshold the
    proposal becomes an `extends` link to it at CONFIRM_LINK_STRENGTH,
    otherwise the proposal stands. No skip path: a decision model is not
    asked whether the proposed topic is right-sized.
    """
    usable = [s for s in scores if isinstance(s.get("link_probability"), (int, float)) and s.get("topic_id")]
    note_head = "\n\nCONFIRMATION PASS (decision model): "
    if not usable:
        parsed["confirmation"] = "confirmed_new"
        parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + note_head + "no candidate scores available; keeping propose_new."
        logger.info("  confirmation=confirmed_new (decision model) no candidate scores available")
        return parsed
    best = max(usable, key=lambda s: s["link_probability"])
    ranked = ", ".join(
        f"{s['topic_id']}={s['link_probability']:.2f}/{(s.get('label') or '?')[:8]}"
        for s in sorted(usable, key=lambda s: -s["link_probability"])
    )
    if best["link_probability"] >= threshold:
        parsed["decision"] = "link_existing"
        parsed["links"] = [{"topic_id": best["topic_id"], "connection_type": REVIEWER_CONNECTION_TYPE,
                            "fit_questions": None, "strength": CONFIRM_LINK_STRENGTH}]
        parsed["strength"] = CONFIRM_LINK_STRENGTH
        parsed["new_topic"] = None
        parsed["confirmation"] = "overridden_to_link"
        note = (
            f"first pass proposed a new topic; the decision model scored p(belongs)={best['link_probability']:.2f} "
            f"(relationship {best.get('label')}, confidence {best.get('confidence')}) for "
            f"'{best['topic_id']}', above the {threshold} threshold. Strength fixed at "
            f"{CONFIRM_LINK_STRENGTH} (edge extension). All scores: {ranked}"
        )
    else:
        parsed["confirmation"] = "confirmed_new"
        note = f"no candidate reached p(belongs) {threshold}; best {best['topic_id']}={best['link_probability']:.2f}. All scores: {ranked}"
    parsed["decision_scores"] = usable
    parsed["reasoning_trace"] = (parsed.get("reasoning_trace") or "") + note_head + note
    logger.info(
        "  confirmation=%s (decision model) best=%s p=%.2f threshold=%.2f",
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
    if decision_model is None:
        logger.warning("confirm_backend=decision but no decision model was supplied; keeping propose_new")
        return _apply_decision_confirmation(parsed, [], settings.synthesize.confirm_link_threshold)
    shown = _confirmation_candidates(
        parsed, vector_candidates, all_candidates, settings.synthesize.confirm_candidates
    )
    scores: list[dict] = []
    for c in shown:
        reps = _representative_observation_titles(store, c["topic_id"], exclude_item_url=exclude_url)
        topic = dict(c)
        topic["evidence_count"] = _get_evidence_count(c)
        try:
            score = decision_model.score_pair(item, topic, reps)
        except Exception as e:
            logger.warning("decision model call failed for %s: %s", c["topic_id"], e)
            continue
        scores.append({"topic_id": c["topic_id"], **asdict(score)})
    return _apply_decision_confirmation(parsed, scores, settings.synthesize.confirm_link_threshold)


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
    response = judge.complete(
        confirm_system_prompt(settings), user_content, max_tokens=settings.judge.confirm_max_tokens,
    )
    # The reviewer is advisory: a reply it could not finish, or declined to
    # give, like one it garbled, leaves the first-pass proposal standing.
    if response.refused:
        logger.warning("Confirmation pass was refused by the judge; keeping propose_new")
        confirm = {"verdict": "confirm_new", "reasoning": "(reviewer refused to answer)"}
    elif response.truncated:
        logger.warning("Confirmation pass was truncated at its output cap; keeping propose_new")
        confirm = {"verdict": "confirm_new", "reasoning": "(truncated reviewer response)"}
    else:
        try:
            confirm = _robust_json_parse(response.text)
        except json.JSONDecodeError:
            logger.warning("Confirmation pass returned unparseable JSON; keeping propose_new")
            confirm = {"verdict": "confirm_new", "reasoning": "(unparseable reviewer response)"}
    parsed = _apply_confirmation(parsed, confirm, {c["topic_id"] for c in shown})
    return parsed, response


def _parse_decision(raw_text: str) -> dict:
    parsed = _robust_json_parse(raw_text)

    if parsed.get("new_topic") and parsed["new_topic"].get("topic_id"):
        parsed["new_topic"]["topic_id"] = _sanitize_topic_id(parsed["new_topic"]["topic_id"])

    # A link decision names every topic the item is evidence for, each with
    # its own fit answers; strength is derived per link, never emitted.
    if parsed.get("decision") == "link_existing":
        links = []
        for raw in parsed.get("links") if isinstance(parsed.get("links"), list) else []:
            if not isinstance(raw, dict):
                continue
            fit = raw.get("fit_questions") if isinstance(raw.get("fit_questions"), dict) else None
            links.append({
                "topic_id": _sanitize_topic_id(str(raw.get("topic_id") or "")),
                "connection_type": raw.get("connection_type"),
                "fit_questions": fit,
                "strength": derive_strength(fit),
            })
        parsed["links"] = links
        parsed["strength"] = links[0]["strength"] if links and links[0]["strength"] is not None else 0.0
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

    if parsed.get("decision") == "link_existing":
        for link in parsed.get("links") or []:
            if not isinstance(link.get("fit_questions"), dict):
                continue
            lines = [f"Fit questions for {link['topic_id']}:"]
            for k in FIT_QUESTIONS:
                lines.append(f"  {k}: {link['fit_questions'].get(k)}")
            lines.append(f"  derived_strength: {link['strength']}")
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
    topics: list[dict] | None = None,
) -> SynthesisDecision:
    """Decide one item. ``topics`` is the store's live topic list, as
    ``list_topics(include_retired=False, fields=TOPIC_HEAD_FIELDS)`` returns
    it, for a caller that already holds it: ``synthesize_items`` lists once
    per batch, since nothing within a batch changes it but the proposals,
    which travel separately. Without it the list is read here."""
    if topics is None:
        topics = store.list_topics(include_retired=False, fields=TOPIC_HEAD_FIELDS)
    all_topics = list(topics)

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
        t for t in store.nearest_topics(eval_result.embedding, settings.candidates.vector_limit, fields=TOPIC_HEAD_FIELDS)
        if t.get("status") not in ("retired", "deleting") and t["topic_id"] not in hidden_topic_ids
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
        system = cold_start_system_prompt(settings)
    elif not all_topics and proposed_topics:
        system = cold_start_system_prompt(settings)
    else:
        system = synthesize_system_prompt(settings)

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

    response = judge.complete(
        system, user_content, max_tokens=settings.judge.first_pass_max_tokens,
    ).require_complete("first-pass decision", "settings judge.first_pass_max_tokens")
    parsed = _parse_decision(response.text)
    shown_ids = {c["topic_id"] for c in all_candidates if c.get("topic_id")}
    parsed = _drop_unshown_classifications(parsed, shown_ids)
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

    parsed = _validate_decision(parsed, shown_ids, store, proposed_topics, settings)

    diagnostics = _build_diagnostics(
        parsed, vector_candidates, tag_candidates, candidates, proposed_topics,
        response, confirm_response,
    )

    links = [Link(**link) for link in parsed.get("links") or []]
    return SynthesisDecision(
        item_url=item["url"],
        item_title=item.get("title", ""),
        decision=parsed["decision"],
        topic_id=links[0].topic_id if links else None,
        connection_type=links[0].connection_type if links else None,
        strength=parsed.get("strength", 0.0),
        reasoning_trace=parsed.get("reasoning_trace", ""),
        links=links,
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

    links = parsed.get("links") or []
    chosen = links[0]["topic_id"] if parsed.get("decision") == "link_existing" and links else None
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
        "decision_scores": parsed.get("decision_scores"),
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
        "dropped_links": [{"topic_id": l["topic_id"], "strength": l["strength"]} for l in parsed.get("dropped_links") or []],
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

    # The live topic list is read once for the batch: record runs after
    # synthesize, so nothing within the batch adds to it but this batch's
    # own proposals, and those are passed on their own. It is read inside
    # the per-item try, so a store failure fails that item, as any read
    # did, and the next item reads again.
    topics: list[dict] | None = None

    for i, (item, eval_result) in enumerate(above_threshold, 1):
        logger.info(
            "Synthesizing item %d/%d: %s",
            i,
            len(above_threshold),
            item.get("title", "")[:60],
        )
        try:
            # An item the store already holds was decided by an earlier run (a
            # retry after a late failure, or another feeder). The stored
            # observation is that decision; the judge is not asked again, so a
            # retry cannot pick a different topic or open a second one. The
            # check reads the store, so it fails the item, not the batch.
            existing = store.get_observations_for_item(item["url"])
            if existing:
                links = intended_links(existing, store)
                logger.info("  already recorded on %s; not re-judged", [l.topic_id for l in links])
                decisions.append(SynthesisDecision(
                    item_url=item["url"], item_title=item.get("title", ""), decision="link_existing",
                    topic_id=links[0].topic_id, connection_type=links[0].connection_type, strength=links[0].strength,
                    reasoning_trace=f"ALREADY RECORDED on {[l.topic_id for l in links]} at {existing[0].get('recorded_at')}; the judge was not called.",
                    links=links,
                ))
                continue
            if topics is None:
                topics = store.list_topics(include_retired=False, fields=TOPIC_HEAD_FIELDS)
            decision = synthesize_item(
                item, eval_result, store, settings, judge, proposed_topics,
                decision_model=decision_model, topics=topics,
            )
        except Exception as e:
            # A failed item is still a decision, with decision="failed", so
            # the caller can see it, keep it in the queue, and retry it.
            logger.error("Failed to synthesize item: %s", item.get("url"), exc_info=True)
            decisions.append(failed_decision(item, e))
            continue

        # A proposal is not a proposal until its embedding exists: later
        # items are matched against it and record persists it. The
        # embedding is set on decision.new_topic directly; proposed_topics
        # references the same dict.
        if decision.decision == "propose_new" and decision.new_topic:
            topic_text = f"{decision.new_topic['title']}\n\n{decision.new_topic['summary']}"
            try:
                decision.new_topic["embedding"] = embedder.embed_documents([topic_text])[0]
            except Exception as e:
                logger.error("Failed to embed proposed topic for %s", item.get("url"), exc_info=True)
                decisions.append(failed_decision(item, f"embed_proposal: {type(e).__name__}: {e}"))
                continue
            proposed_topics.append(decision.new_topic)

        decisions.append(decision)
        logger.info("  decision=%s topic=%s", decision.decision, decision.topic_id)

    return decisions
