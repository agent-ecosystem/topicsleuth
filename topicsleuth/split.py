from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from topicsleuth.interfaces import Embedder, Judge, Store
from topicsleuth.rubric import (
    DEFAULT_CONNECTION_TYPE,
    derive_strength,
    split_propose_system_prompt,
    split_reassign_system_prompt,
)
from topicsleuth.settings import Settings
from topicsleuth.synthesize import _sanitize_topic_id

logger = logging.getLogger(__name__)

# Maximum observations per reassignment batch
REASSIGN_BATCH_SIZE = 20


def split_topic(
    topic_id: str,
    store: Store,
    settings: Settings,
    judge: Judge,
    embedder: Embedder,
    dry_run: bool = False,
) -> dict:
    """Split an oversized topic into more specific sub-topics.

    Uses two LLM phases:
    1. Propose sub-topics (single call)
    2. Reassign observations in batches

    Returns a summary dict with sub_topics created and observations reassigned.
    """
    # 1. Fetch topic and observations
    topic = store.get_topic(topic_id)
    if not topic:
        raise ValueError(f"Topic not found: {topic_id}")

    observations = store.get_observations_for_topic(topic_id)
    if not observations:
        raise ValueError(f"No observations for topic: {topic_id}")

    logger.info(
        "Splitting topic '%s' (%d observations)", topic["title"], len(observations)
    )

    # 2. Phase 1: Propose sub-topics
    sub_topics = _propose_sub_topics(topic, observations, judge, settings)
    logger.info("Proposed %d sub-topics", len(sub_topics))
    for st in sub_topics:
        logger.info("  - %s: %s", st["topic_id"], st["title"])

    # 3. Phase 2: Reassign observations in batches
    reassignments = _reassign_observations(topic, observations, sub_topics, judge, settings)

    # Validate all observations are reassigned
    obs_urls = {obs["item_url"] for obs in observations}
    reassigned_urls = {r["item_url"] for r in reassignments}
    missing = obs_urls - reassigned_urls
    if missing:
        logger.warning(
            "%d observations not reassigned, will be dropped: %s",
            len(missing),
            missing,
        )

    if dry_run:
        return {
            "original_topic": topic["title"],
            "observation_count": len(observations),
            "sub_topics": sub_topics,
            "reassignments": reassignments,
            "missing_reassignments": list(missing),
        }

    now = datetime.now(timezone.utc)

    # 4. Create sub-topics with embeddings
    for st in sub_topics:
        st["topic_id"] = _sanitize_topic_id(st["topic_id"])
        topic_text = f"{st['title']}\n\n{st['summary']}"
        embedding = embedder.embed_documents([topic_text])[0]
        topic_doc = {
            "topic_id": st["topic_id"],
            "title": st["title"],
            "summary": st["summary"],
            "tags": st.get("tags", []),
            "suggested_directions": st.get("suggested_directions", []),
            "embedding": embedding,
        }
        store.create_topic(topic_doc)
        logger.info("Created sub-topic: %s", st["topic_id"])

    # 5. Build lookup for original observations (for source metadata)
    obs_by_url = {}
    for obs in observations:
        obs_by_url[obs["item_url"]] = obs

    # 6. Create new observations for reassignments
    created = 0
    for r in reassignments:
        original_obs = obs_by_url.get(r["item_url"])
        if not original_obs:
            logger.warning("No original observation for %s, skipping", r["item_url"])
            continue

        new_topic_id = _sanitize_topic_id(r["new_topic_id"])
        derived = derive_strength(r.get("fit_questions"))
        obs_doc = {
            "item_id": original_obs.get("item_id"),
            "item_url": r["item_url"],
            "item_title": original_obs.get("item_title", ""),
            "topic_id": new_topic_id,
            "date": now,
            "strength": derived if derived is not None else original_obs.get("strength", 0.5),
            "connection_type": r.get(
                "new_connection_type",
                original_obs.get("connection_type", DEFAULT_CONNECTION_TYPE),
            ),
            "fit_questions": r.get("fit_questions"),
            "reasoning_trace": r.get("reasoning", ""),
            "source_name": original_obs.get("source_name", ""),
            "source_type": original_obs.get("source_type", ""),
            "entities_mentioned": original_obs.get("entities_mentioned", []),
        }
        store.create_observation(obs_doc)
        created += 1

    # 7. Delete old observations for the original topic
    deleted = store.delete_observations_for_topic(topic_id)
    logger.info("Deleted %d old observations for %s", deleted, topic_id)

    # 8. Retire the original topic with provenance
    store.update_topic(topic_id, {
        "status": "retired",
        "retired_at": now,
        "split_into": [_sanitize_topic_id(st["topic_id"]) for st in sub_topics],
    })
    logger.info("Retired topic: %s", topic_id)

    # 9. Compute snapshots for new sub-topics
    from topicsleuth.record import _compute_snapshot

    for st in sub_topics:
        tid = _sanitize_topic_id(st["topic_id"])
        snapshot = _compute_snapshot(store, tid, now)
        store.append_snapshot(tid, snapshot)

    return {
        "original_topic": topic["title"],
        "original_topic_id": topic_id,
        "observation_count": len(observations),
        "sub_topics_created": [st["topic_id"] for st in sub_topics],
        "observations_reassigned": created,
        "observations_deleted": deleted,
    }


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
