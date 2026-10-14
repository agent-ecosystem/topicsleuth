from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from topicsleuth.interfaces import Embedder, Judge, Store
from topicsleuth.settings import Settings
from topicsleuth.synthesize import _sanitize_topic_id

logger = logging.getLogger(__name__)

PROPOSE_SYSTEM_PROMPT = """\
You are the analyst who maintains a topic store. A topic has grown too broad and \
needs to be split into more specific sub-topics. You will be given the topic and a summary \
of its observations. Propose 2-4 more specific sub-topics that collectively cover \
the same space at a finer granularity.

Each sub-topic should be specific enough to attract 5-15 observations over time, not \
so broad that it becomes a catch-all. Think of sub-topics as distinct \
questions or hypotheses within the broader space.

You must respond with ONLY a JSON object (no markdown fencing) containing:
{
  "sub_topics": [
    {
      "topic_id": "<slug, lowercase with hyphens>",
      "title": "<concise title>",
      "summary": "<2-3 sentence description of the specific question tracked>",
      "tags": ["<relevant tags>"],
      "suggested_directions": ["<specific open questions>"]
    }
  ]
}
"""

REASSIGN_SYSTEM_PROMPT = """\
You are the analyst who maintains a topic store. Observations are being reassigned from a broad \
parent topic to more specific sub-topics. For each observation, decide which sub-topic \
it best fits and recalibrate the strength score for the narrower topic.

You must respond with ONLY a JSON array (no markdown fencing) containing one entry \
per observation:
[
  {
    "item_url": "<url of the original item>",
    "new_topic_id": "<topic_id of the best-fitting sub-topic>",
    "new_strength": <float 0-1>,
    "new_connection_type": "supports" | "extends" | "contradicts" | "shifts",
    "reasoning": "<brief explanation>"
  }
]

Strength calibration:
- 0.9-1.0: Primary evidence for the sub-topic's core claim.
- 0.7-0.8: Clear connection, meaningfully extends or supports the sub-topic.
- 0.5-0.6: Moderate connection, relevant but requires interpretation.
- 0.3-0.4: Weak signal, tangentially related.

Every observation must appear exactly once. Do not drop any.
"""

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
    sub_topics = _propose_sub_topics(topic, observations, judge)
    logger.info("Proposed %d sub-topics", len(sub_topics))
    for st in sub_topics:
        logger.info("  - %s: %s", st["topic_id"], st["title"])

    # 3. Phase 2: Reassign observations in batches
    reassignments = _reassign_observations(topic, observations, sub_topics, judge)

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
        obs_doc = {
            "item_id": original_obs.get("item_id"),
            "item_url": r["item_url"],
            "item_title": original_obs.get("item_title", ""),
            "topic_id": new_topic_id,
            "date": now,
            "strength": r.get("new_strength", original_obs.get("strength", 0.5)),
            "connection_type": r.get(
                "new_connection_type",
                original_obs.get("connection_type", "supports"),
            ),
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
) -> list[dict]:
    """Phase 1: Ask the LLM to propose sub-topics based on observation titles."""
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

    completion = judge.complete(PROPOSE_SYSTEM_PROMPT, user_content, max_tokens=2048)
    parsed = _parse_json_response(completion.text)
    return parsed["sub_topics"]


def _reassign_observations(
    topic: dict,
    observations: list[dict],
    sub_topics: list[dict],
    judge: Judge,
) -> list[dict]:
    """Phase 2: Reassign observations to sub-topics in batches."""
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

        completion = judge.complete(REASSIGN_SYSTEM_PROMPT, user_content, max_tokens=4096)
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
