from __future__ import annotations

import json
import logging
from typing import Protocol
import re
from dataclasses import asdict, dataclass

from topicsleuth.interfaces import DecisionModel, Embedder, Judge, Store
from topicsleuth.settings import Settings
from topicsleuth.rubric import (
    CONFIRM_LINK_STRENGTH,
    DEFAULT_CONNECTION_TYPE,
    GATE_OVERRIDE_DEFAULT_STRENGTH,
    REVIEWER_CONNECTION_TYPE,
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
    parsed["connection_type"] = DEFAULT_CONNECTION_TYPE
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
            parsed["connection_type"] = confirm.get("connection_type") or REVIEWER_CONNECTION_TYPE
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
        parsed["topic_id"] = best["topic_id"]
        parsed["connection_type"] = REVIEWER_CONNECTION_TYPE
        parsed["strength"] = CONFIRM_LINK_STRENGTH
        parsed["fit_questions"] = None
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
    response = judge.complete(confirm_system_prompt(settings), user_content, max_tokens=1024)
    try:
        confirm = _robust_json_parse(response.text)
    except json.JSONDecodeError:
        logger.warning("Confirmation pass returned unparseable JSON; keeping propose_new")
        confirm = {"verdict": "confirm_new", "reasoning": "(unparseable reviewer response)"}
    parsed = _apply_confirmation(parsed, confirm, {c["topic_id"] for c in shown})
    return parsed, response


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
        derived = derive_strength(parsed.get("fit_questions"))
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
    all_topics = store.list_topics(include_retired=False)

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
