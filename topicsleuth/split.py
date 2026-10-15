from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from topicsleuth.interfaces import Embedder, Judge, Store
from topicsleuth.rubric import (
    CONNECTION_TYPES,
    DEFAULT_CONNECTION_TYPE,
    derive_strength,
    split_propose_system_prompt,
    split_reassign_system_prompt,
)
from topicsleuth.settings import Settings
from topicsleuth.history import removed_history
from topicsleuth.synthesize import _cosine, _sanitize_topic_id

logger = logging.getLogger(__name__)

# Maximum observations per reassignment batch
REASSIGN_BATCH_SIZE = 20


class SplitPlanError(ValueError):
    """The judge's plan cannot be executed as given: a sub-topic id that is
    missing, duplicated or already taken, or an observation that is left
    out, assigned twice, or assigned to a sub-topic that does not exist.
    Nothing is written when this is raised."""


def split_topic(
    topic_id: str,
    store: Store,
    settings: Settings,
    judge: Judge,
    embedder: Embedder,
    dry_run: bool = False,
) -> dict:
    """Split an oversized topic into more specific sub-topics.

    Two judge phases: propose sub-topics (one call), then assign each
    observation to one of them (in batches). The plan is validated in
    full before anything is written (see ``validate_split_plan``), and a
    dry run returns the plan with its problems instead of raising.

    Execution moves the existing observations rather than replacing them:
    each keeps its identity, date, tags and the rest of its metadata, gets
    its new topic, the strength and connection type the judge assessed
    against the narrower topic, and ``moved_from`` pointing at the parent.
    The parent is retired with ``split_into``. Writes are per document, in
    the order sub-topics, observations, parent; if one fails midway, the
    parent is still live with whatever was not yet moved, the sub-topics
    exist, and ``move-observation`` finishes the job by hand.
    """
    # 1. Fetch topic and observations
    topic = store.get_topic(topic_id)
    if not topic:
        raise ValueError(f"Topic not found: {topic_id}")
    if topic.get("status") in ("retired", "deleting"):
        raise ValueError(f"Topic is {topic['status']}: {topic_id}")

    observations = store.get_observations_for_topic(topic_id)
    if not observations:
        raise ValueError(f"No observations for topic: {topic_id}")

    logger.info(
        "Splitting topic '%s' (%d observations)", topic["title"], len(observations)
    )

    # 2. Phase 1: Propose sub-topics
    sub_topics = _propose_sub_topics(topic, observations, judge, settings)
    for st in sub_topics:
        st["topic_id"] = _sanitize_topic_id(str(st.get("topic_id") or ""))
    logger.info("Proposed %d sub-topics", len(sub_topics))
    for st in sub_topics:
        logger.info("  - %s: %s", st["topic_id"], st.get("title"))

    # 3. Phase 2: Assign observations in batches
    reassignments = _reassign_observations(topic, observations, sub_topics, judge, settings)
    for r in reassignments:
        if isinstance(r, dict):
            r["new_topic_id"] = _sanitize_topic_id(str(r.get("new_topic_id") or ""))

    # 4. Validate the whole plan before writing anything
    problems = validate_split_plan(topic_id, observations, sub_topics, reassignments, store)

    if dry_run:
        return {
            "original_topic": topic["title"],
            "observation_count": len(observations),
            "sub_topics": sub_topics,
            "reassignments": reassignments,
            "problems": problems,
        }
    if problems:
        raise SplitPlanError("; ".join(problems))

    now = datetime.now(timezone.utc)

    # 5. Create sub-topics with embeddings
    sub_embeddings: dict[str, list[float]] = {}
    for st in sub_topics:
        topic_text = f"{st['title']}\n\n{st['summary']}"
        embedding = embedder.embed_documents([topic_text])[0]
        sub_embeddings[st["topic_id"]] = embedding
        store.create_topic({
            "topic_id": st["topic_id"],
            "title": st["title"],
            "summary": st["summary"],
            "tags": st.get("tags", []) if isinstance(st.get("tags"), list) else [],
            "suggested_directions": st.get("suggested_directions", []) if isinstance(st.get("suggested_directions"), list) else [],
            "embedding": embedding,
        })
        logger.info("Created sub-topic: %s", st["topic_id"])

    # 6. Move each observation onto its sub-topic, keeping it whole
    obs_by_url = {obs["item_url"]: obs for obs in observations}
    moved = 0
    for r in reassignments:
        original = obs_by_url[r["item_url"]]
        derived = derive_strength(r.get("fit_questions"))
        note = (
            f"\n\nMOVED by split-topic on {now.date().isoformat()}: from "
            f"'{topic_id}' to '{r['new_topic_id']}'. {r.get('reasoning', '')}".rstrip()
        )
        updates = {
            "topic_id": r["new_topic_id"],
            "strength": derived if derived is not None else original.get("strength", 0.5),
            "connection_type": r.get("new_connection_type") or original.get("connection_type", DEFAULT_CONNECTION_TYPE),
            "fit_questions": r.get("fit_questions"),
            "reasoning_trace": (original.get("reasoning_trace") or "") + note,
            "moved_from": topic_id,
            "removed_from": removed_history(original, topic_id),
        }
        store.update_observation(original["_id"], updates)
        moved += 1

    # 7. Retire the original topic with provenance
    store.update_topic(topic_id, {
        "status": "retired",
        "retired_at": now,
        "split_into": [st["topic_id"] for st in sub_topics],
    })
    logger.info("Retired topic: %s", topic_id)

    # Anything a concurrent run recorded on the parent since the plan was
    # made was not in the judge's plan. It goes to the nearest sub-topic by
    # embedding (or the largest, if the item has no embedding), with a note.
    late = [o for o in store.get_observations_for_topic(topic_id) if o["item_url"] not in obs_by_url]
    if late:
        logger.warning("%d observation(s) arrived on %s during the split; assigning by nearest sub-topic", len(late), topic_id)
        item_vectors = store.get_item_embeddings([o["item_url"] for o in late])
        counts = {sid: store.count_observations_for_topic(sid) for sid in sub_embeddings}
        for o in late:
            vec = item_vectors.get(o["item_url"])
            if vec:
                target = max(sub_embeddings, key=lambda sid: _cosine(vec, sub_embeddings[sid]))
                how = "nearest sub-topic by embedding"
            else:
                target = max(counts, key=counts.get)
                how = "largest sub-topic (item has no embedding)"
            store.update_observation(o["_id"], {
                "topic_id": target, "moved_from": topic_id, "removed_from": removed_history(o, topic_id),
                "reasoning_trace": (o.get("reasoning_trace") or "")
                + f"\n\nMOVED by split-topic on {now.date().isoformat()}: arrived during the split; {how}: '{target}'.",
            })
            counts[target] += 1
            moved += 1

    # 8. Snapshots for the new sub-topics
    from topicsleuth.record import _compute_snapshot

    for st in sub_topics:
        store.append_snapshot(st["topic_id"], _compute_snapshot(store, st["topic_id"], now))

    return {
        "original_topic": topic["title"],
        "original_topic_id": topic_id,
        "observation_count": len(observations),
        "sub_topics_created": [st["topic_id"] for st in sub_topics],
        "observations_reassigned": moved,
    }


def validate_split_plan(
    topic_id: str,
    observations: list[dict],
    sub_topics: list[dict],
    reassignments: list[dict],
    store: Store,
) -> list[str]:
    """Every problem with a split plan, as sentences. Empty means the plan
    can be executed: at least two well-formed sub-topics with distinct,
    unused ids, and every observation assigned exactly once to one of
    them."""
    problems: list[str] = []

    if not isinstance(sub_topics, list) or len(sub_topics) < 2:
        problems.append(f"need at least 2 sub-topics, got {len(sub_topics) if isinstance(sub_topics, list) else 'none'}")
        return problems
    ids: list[str] = []
    for st in sub_topics:
        if not isinstance(st, dict):
            problems.append(f"sub-topic is not an object: {st!r}")
            continue
        sid = st.get("topic_id") or ""
        if not sid:
            problems.append(f"sub-topic {st.get('title')!r} has no usable topic_id")
            continue
        for key in ("title", "summary"):
            if not isinstance(st.get(key), str) or not st[key].strip():
                problems.append(f"sub-topic {sid!r} has no {key}")
        if sid == topic_id:
            problems.append(f"sub-topic {sid!r} reuses the parent's id")
        elif store.get_topic(sid) is not None:
            problems.append(f"sub-topic id {sid!r} is already taken")
        if sid in ids:
            problems.append(f"sub-topic id {sid!r} is proposed twice")
        ids.append(sid)
    valid_ids = set(ids)

    expected = {o["item_url"] for o in observations}
    seen: dict[str, int] = {}
    for r in reassignments:
        if not isinstance(r, dict):
            problems.append(f"assignment is not an object: {r!r}")
            continue
        url = r.get("item_url")
        if url not in expected:
            problems.append(f"assignment names an observation the topic does not hold: {url!r}")
            continue
        seen[url] = seen.get(url, 0) + 1
        if r.get("new_topic_id") not in valid_ids:
            problems.append(f"{url!r} is assigned to unknown sub-topic {r.get('new_topic_id')!r}")
        ctype = r.get("new_connection_type")
        if ctype is not None and ctype not in CONNECTION_TYPES:
            problems.append(f"{url!r} has unknown connection type {ctype!r}")
    for url, n in seen.items():
        if n > 1:
            problems.append(f"{url!r} is assigned {n} times")
    missing = sorted(expected - set(seen))
    if missing:
        problems.append(f"{len(missing)} observation(s) not assigned: {missing}")
    return problems


def _propose_sub_topics(
    topic: dict,
    observations: list[dict],
    judge: Judge,
    settings: Settings,
) -> list[dict]:
    """Phase 1: Ask the judge to propose sub-topics based on observation titles."""
    # Use titles only to keep the prompt compact
    obs_summary = "\n".join(
        f"- {obs.get('item_title', 'N/A')} (strength: {obs.get('strength', 'N/A')}, "
        f"type: {obs.get('connection_type', 'N/A')})"
        for obs in observations
    )

    user_content = (
        f"## Topic Being Split\n"
        f"topic_id: {topic['topic_id']}\n"
        f"title: {topic['title']}\n"
        f"summary: {topic.get('summary', 'N/A')}\n"
        f"tags: {', '.join(topic.get('tags', []))}\n\n"
        f"## Observations ({len(observations)} total)\n"
        f"{obs_summary}"
    )

    completion = judge.complete(split_propose_system_prompt(settings), user_content, max_tokens=2048)
    parsed = _parse_json_response(completion.text)
    return parsed["sub_topics"]


def _reassign_observations(
    topic: dict,
    observations: list[dict],
    sub_topics: list[dict],
    judge: Judge,
    settings: Settings,
) -> list[dict]:
    """Phase 2: Reassign observations to sub-topics in batches. The judge
    answers the fit questions per observation; strength is derived here."""
    # Format sub-topics for the prompt
    sub_topic_text = "\n".join(
        f"- topic_id: {st['topic_id']}\n"
        f"  title: {st['title']}\n"
        f"  summary: {st['summary']}"
        for st in sub_topics
    )

    all_reassignments = []

    # Process in batches
    for batch_start in range(0, len(observations), REASSIGN_BATCH_SIZE):
        batch = observations[batch_start : batch_start + REASSIGN_BATCH_SIZE]
        batch_num = batch_start // REASSIGN_BATCH_SIZE + 1
        total_batches = (len(observations) + REASSIGN_BATCH_SIZE - 1) // REASSIGN_BATCH_SIZE
        logger.info(
            "Reassigning batch %d/%d (%d observations)",
            batch_num, total_batches, len(batch),
        )

        obs_lines = []
        for obs in batch:
            obs_lines.append(
                f"- item_url: {obs['item_url']}\n"
                f"  item_title: {obs.get('item_title', 'N/A')}\n"
                f"  strength: {obs.get('strength', 'N/A')}\n"
                f"  connection_type: {obs.get('connection_type', 'N/A')}\n"
                f"  reasoning: {obs.get('reasoning_trace', 'N/A')[:200]}"
            )

        user_content = (
            f"## Original Topic\n"
            f"{topic['title']}: {topic.get('summary', 'N/A')}\n\n"
            f"## Sub-Topics to Assign To\n{sub_topic_text}\n\n"
            f"## Observations to Reassign ({len(batch)} items)\n"
            + "\n\n".join(obs_lines)
        )

        completion = judge.complete(split_reassign_system_prompt(settings), user_content, max_tokens=4096)
        batch_result = _parse_json_response(completion.text)
        # Response is a JSON array
        if isinstance(batch_result, list):
            all_reassignments.extend(batch_result)
        else:
            # In case the LLM wraps it in an object
            all_reassignments.extend(batch_result.get("reassignments", []))

    return all_reassignments


def _parse_json_response(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text or text[0] not in ("{", "["):
        match = re.search(r"[\[{][\s\S]*[\]}]", text)
        if match:
            text = match.group(0)
    return json.loads(text)
