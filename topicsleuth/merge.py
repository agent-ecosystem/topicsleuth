"""Topic merging: collapse two redundant topics into one.

Mirrors split.py in shape. Where split breaks one broad topic into multiple
specific ones, merge collapses two near-duplicate topics into a single
surviving topic. The source topic is retired with provenance metadata
(merged_into); its observations have their topic_id updated to point at
the target. The target's title, summary, tags, and suggested_directions
are rewritten by the LLM to reflect the unified scope, and its embedding
is recomputed.

Observation strengths and connection_types are preserved as-is. The merged
topic should have roughly the same scope as the union of inputs, so
strengths shouldn't need recalibration.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from topicsleuth.interfaces import Embedder, Judge, Store
from topicsleuth.settings import Settings

logger = logging.getLogger(__name__)

MERGE_SYSTEM_PROMPT = """\
You are the analyst who maintains a topic store. Two topics have been \
identified as near-duplicates and are being merged into one surviving topic. \
You will see both topics (titles, summaries, representative observations) \
and your job is to write a unified title, summary, tag set, and suggested \
directions for the merged topic.

The merged topic should:
- Track a single specifiable question that both source topics were \
following from slightly different angles.
- Be no broader than the union of the two source topics. Do not pull in \
adjacent territory.
- Have a one-sentence summary that names what is being investigated, not \
just the subject area.
- Have a tag set that is the union of both source tag sets, deduplicated.
- Have suggested directions that capture the most useful open questions \
from both source topics, deduplicated.

You must respond with ONLY a JSON object (no markdown fencing):
{
  "title": "<concise title for the merged topic>",
  "summary": "<one sentence naming what is being investigated>",
  "tags": ["<deduplicated union of both source tag sets>"],
  "suggested_directions": ["<deduplicated useful directions from both>"],
  "rationale": "<one or two sentences explaining what the merged topic captures and what was preserved>"
}

The topic_id of the surviving topic is fixed by the caller; do not propose \
a new topic_id.
"""


def suggest_merges(
    store: Store,
    threshold: float = 0.85,
    min_tag_overlap: int = 0,
) -> list[dict]:
    """Find candidate topic pairs for merging by cosine similarity.

    Returns a list of suggestions sorted by similarity descending. Each
    suggestion includes both topic_ids, both titles, both observation
    counts, the cosine similarity, the tag overlap count, and a recommended
    merge direction (smaller-evidence topic retired into larger-evidence
    topic).

    Skips retired topics. Filters to pairs with cosine >= threshold and
    tag overlap >= min_tag_overlap (default 0 = no tag filter).
    """
    import numpy as np

    topics = [
        t for t in store.list_topics(include_retired=False)
        if t.get("embedding")
        and isinstance(t.get("embedding"), list)
    ]
    if len(topics) < 2:
        return []

    # Normalize all embeddings once (cosine = dot product on normalized).
    vectors = np.array(
        [t["embedding"] for t in topics], dtype=np.float32
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    # Pairwise cosine via matrix multiply.
    sim = vectors @ vectors.T
    suggestions: list[dict] = []
    n = len(topics)
    obs_counts = [
        store.count_observations_for_topic(t["topic_id"]) for t in topics
    ]
    tag_sets = [set(t.get("tags") or []) for t in topics]

    for i in range(n):
        for j in range(i + 1, n):
            cosine = float(sim[i, j])
            if cosine < threshold:
                continue
            tag_overlap = len(tag_sets[i] & tag_sets[j])
            if tag_overlap < min_tag_overlap:
                continue
            # Direction: retire whichever has fewer observations.
            if obs_counts[i] <= obs_counts[j]:
                src_idx, tgt_idx = i, j
            else:
                src_idx, tgt_idx = j, i
            suggestions.append({
                "cosine": cosine,
                "tag_overlap": tag_overlap,
                "source_topic_id": topics[src_idx]["topic_id"],
                "source_title": topics[src_idx].get("title", ""),
                "source_obs": obs_counts[src_idx],
                "target_topic_id": topics[tgt_idx]["topic_id"],
                "target_title": topics[tgt_idx].get("title", ""),
                "target_obs": obs_counts[tgt_idx],
            })

    suggestions.sort(key=lambda s: s["cosine"], reverse=True)
    return suggestions


def merge_topics(
    source_topic_id: str,
    target_topic_id: str,
    store: Store,
    settings: Settings,
    judge: Judge,
    embedder: Embedder,
    dry_run: bool = False,
) -> dict:
    """Merge a source topic into a target topic.

    The source topic is retired (status=retired, merged_into=<target_id>).
    Its observations have their topic_id updated to the target. The target's
    title, summary, tags, and suggested_directions are rewritten by the LLM,
    and its embedding is recomputed from the new title+summary.

    Returns a summary dict. In dry_run mode, no writes occur and the dict
    contains the proposed merged description for review.
    """
    if source_topic_id == target_topic_id:
        raise ValueError("source and target topic_id must differ")

    source = store.get_topic(source_topic_id)
    if not source:
        raise ValueError(f"source topic not found: {source_topic_id}")
    target = store.get_topic(target_topic_id)
    if not target:
        raise ValueError(f"target topic not found: {target_topic_id}")

    if source.get("status") == "retired":
        raise ValueError(
            f"source topic is already retired: {source_topic_id} "
            f"(merged_into={source.get('merged_into')}, "
            f"split_into={source.get('split_into')})"
        )
    if target.get("status") == "retired":
        raise ValueError(
            f"target topic is retired and cannot be merged into: {target_topic_id}"
        )

    source_obs = store.get_observations_for_topic(source_topic_id)
    target_obs = store.get_observations_for_topic(target_topic_id)
    logger.info(
        "Merging '%s' (%d obs) into '%s' (%d obs)",
        source["title"], len(source_obs), target["title"], len(target_obs),
    )

    merged_description = _propose_merged_description(
        source, target, source_obs, target_obs, judge
    )

    if dry_run:
        return {
            "dry_run": True,
            "source_topic_id": source_topic_id,
            "source_title": source["title"],
            "source_summary": source.get("summary"),
            "source_observation_count": len(source_obs),
            "target_topic_id": target_topic_id,
            "target_title_before": target["title"],
            "target_summary_before": target.get("summary"),
            "target_observation_count_before": len(target_obs),
            "target_observation_count_after": len(source_obs) + len(target_obs),
            "merged_title": merged_description["title"],
            "merged_summary": merged_description["summary"],
            "merged_tags": merged_description.get("tags", []),
            "merged_suggested_directions": merged_description.get(
                "suggested_directions", []
            ),
            "rationale": merged_description.get("rationale", ""),
        }

    now = datetime.now(timezone.utc)

    new_title = merged_description["title"]
    new_summary = merged_description["summary"]
    new_embedding = embedder.embed_documents([f"{new_title}\n\n{new_summary}"])[0]

    store.update_topic(target_topic_id, {
        "title": new_title,
        "summary": new_summary,
        "tags": merged_description.get("tags", target.get("tags", [])),
        "suggested_directions": merged_description.get(
            "suggested_directions", target.get("suggested_directions", [])
        ),
        "embedding": new_embedding,
    })
    logger.info("Updated target topic: %s", target_topic_id)

    reassigned = store.reassign_observations(
        source_topic_id, target_topic_id
    )
    logger.info(
        "Reassigned %d observations from %s to %s",
        reassigned, source_topic_id, target_topic_id,
    )

    store.update_topic(source_topic_id, {
        "status": "retired",
        "retired_at": now,
        "merged_into": target_topic_id,
    })
    logger.info("Retired source topic: %s", source_topic_id)

    from topicsleuth.record import _compute_snapshot

    snapshot = _compute_snapshot(store, target_topic_id, now)
    store.append_snapshot(target_topic_id, snapshot)

    return {
        "dry_run": False,
        "source_topic_id": source_topic_id,
        "target_topic_id": target_topic_id,
        "merged_title": new_title,
        "merged_summary": new_summary,
        "observations_reassigned": reassigned,
        "target_observation_count_after": len(source_obs) + len(target_obs),
    }


def _representative_observations(observations: list[dict], n: int = 5) -> list[dict]:
    """Pick highest-strength non-minor observations as a representative sample."""
    def sort_key(obs):
        return (
            not bool(obs.get("minor")),
            obs.get("strength") or 0.0,
            obs.get("date").timestamp() if obs.get("date") else 0.0,
        )
    return sorted(observations, key=sort_key, reverse=True)[:n]


def _format_topic_for_merge(topic: dict, observations: list[dict], label: str) -> str:
    reps = _representative_observations(observations)
    rep_lines = "\n".join(
        f"    * {o.get('item_title','')}" for o in reps if o.get("item_title")
    ) or "    (no observations yet)"
    return (
        f"## {label}\n"
        f"topic_id: {topic['topic_id']}\n"
        f"title: {topic['title']}\n"
        f"summary: {topic.get('summary','')}\n"
        f"tags: {', '.join(topic.get('tags', []))}\n"
        f"suggested_directions:\n"
        + "\n".join(f"  - {d}" for d in topic.get("suggested_directions", []) or [])
        + f"\nobservation_count: {len(observations)}\n"
        f"representative_observations:\n{rep_lines}"
    )


def _propose_merged_description(
    source: dict,
    target: dict,
    source_obs: list[dict],
    target_obs: list[dict],
    judge: Judge,
) -> dict:
    user_content = (
        _format_topic_for_merge(source, source_obs, "Source Topic (will be retired)")
        + "\n\n"
        + _format_topic_for_merge(target, target_obs, "Target Topic (will absorb)")
    )

    completion = judge.complete(MERGE_SYSTEM_PROMPT, user_content, max_tokens=2048)
    return _parse_json_response(completion.text)


def _parse_json_response(raw_text: str):
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if not text or text[0] not in ("{", "["):
        match = re.search(r"[\[{][\s\S]*[\]}]", text)
        if match:
            text = match.group(0)
    return json.loads(text, strict=False)
