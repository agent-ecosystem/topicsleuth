"""Add a single topic to a populated corpus.

`seed` only runs on an empty topics collection and `split-topic` retires a
parent to extract children, so until now there was no supported way to add a
topic the pipeline had not proposed on its own. This takes a topic definition
(the seed_topics.yaml entry shape, plus an optional topic_id and an optional
list of item URLs whose observations should move onto the new topic as its
founding evidence), generates the embedding so vector search can match
against it, inserts the document, and records snapshots on every topic touched.

Moved observations keep their strength and connection type; their reasoning
trace gains a line saying where they came from, so the audit trail survives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from topicsleuth.interfaces import Embedder, Store
from topicsleuth.synthesize import _sanitize_topic_id

logger = logging.getLogger(__name__)


@dataclass
class TopicDefinition:
    title: str
    summary: str
    topic_id: str
    tags: list[str] = field(default_factory=list)
    suggested_directions: list[str] = field(default_factory=list)
    link: list[str] = field(default_factory=list)  # item URLs to move onto the topic


def load_topic_definition(path: Path) -> TopicDefinition:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "topics" in raw:
        raise ValueError(
            "add-topic takes one topic definition, not a seed_topics.yaml file "
            "with a `topics:` list"
        )
    title = (raw.get("title") or "").strip()
    summary = (raw.get("summary") or "").strip()
    if not title or not summary:
        raise ValueError("topic definition needs a title and a summary")
    topic_id = _sanitize_topic_id(raw.get("topic_id") or title)
    if not topic_id:
        raise ValueError("could not derive a topic_id from the title")
    return TopicDefinition(
        title=title,
        summary=summary,
        topic_id=topic_id,
        tags=[str(t) for t in raw.get("tags") or []],
        suggested_directions=[str(d) for d in raw.get("suggested_directions") or []],
        link=[str(u) for u in raw.get("link") or []],
    )


def plan_moves(store: Store, urls: list[str], target_topic_id: str) -> tuple[list[dict], list[str]]:
    """Find the observations for the given item URLs. Returns (observations,
    missing_urls). An item with no observation cannot be moved; it would have
    to go back through the pipeline."""
    found: list[dict] = []
    missing: list[str] = []
    for url in urls:
        obs = store.get_observations_for_item(url)
        obs = [o for o in obs if o.get("topic_id") != target_topic_id]
        if not obs:
            missing.append(url)
        found.extend(obs)
    return found, missing


def add_topic(
    definition: TopicDefinition,
    store: Store,
    embedder: Embedder,
    dry_run: bool = False,
) -> dict:
    if store.get_topic(definition.topic_id):
        raise ValueError(f"topic already exists: {definition.topic_id}")

    moves, missing = plan_moves(store, definition.link, definition.topic_id)
    source_counts: dict[str, int] = {}
    for o in moves:
        source_counts[o["topic_id"]] = source_counts.get(o["topic_id"], 0) + 1

    report = {
        "dry_run": dry_run,
        "topic_id": definition.topic_id,
        "title": definition.title,
        "observations_to_move": [
            {"item_url": o["item_url"], "item_title": o.get("item_title", ""),
             "from_topic": o["topic_id"], "strength": o.get("strength"),
             "connection_type": o.get("connection_type")}
            for o in moves
        ],
        "urls_without_observation": missing,
        "source_topics_affected": source_counts,
    }
    if dry_run:
        return report

    embedding = embedder.embed_documents([f"{definition.title}\n\n{definition.summary}"])[0]
    now = datetime.now(timezone.utc)
    store.create_topic({
        "topic_id": definition.topic_id,
        "title": definition.title,
        "summary": definition.summary,
        "tags": definition.tags,
        "suggested_directions": definition.suggested_directions,
        "embedding": embedding,
        "added_manually": True,
    })
    logger.info("Created topic %s", definition.topic_id)

    retired = move_observations(
        store, moves, definition.topic_id, now, reason="add-topic", retire_empty_sources=False
    )
    report["observations_moved"] = len(moves)
    report["sources_retired"] = retired
    return report


def move_observations(
    store: Store,
    observations: list[dict],
    target_topic_id: str,
    now: datetime,
    reason: str,
    retire_empty_sources: bool = True,
) -> list[str]:
    """Re-point observations at target_topic_id, annotate their trace, and
    re-snapshot every topic touched. With retire_empty_sources, a source
    topic left with no observations is retired with merged_into=target so
    the provenance survives (a topic with no evidence is not a topic).
    Returns the ids of retired source topics."""
    from topicsleuth.record import _compute_snapshot

    sources: set[str] = set()
    for o in observations:
        src = o["topic_id"]
        if src == target_topic_id:
            continue
        sources.add(src)
        # An item may already be evidence on the target (it bore on both
        # questions). One observation per item per topic: the target's
        # survives and the moving one is folded into it, noted in the trace.
        holder = next((h for h in store.get_observations_for_item(o["item_url"])
                       if h.get("topic_id") == target_topic_id), None)
        if holder is not None:
            note = (
                f"\n\nFOLDED by {reason} on {now.date().isoformat()}: this item's "
                f"observation on '{src}' (strength {o.get('strength')}, {o.get('connection_type')}) "
                f"was merged into this one."
            )
            store.update_observation(holder["_id"], {
                "reasoning_trace": (holder.get("reasoning_trace") or "") + note,
                # Curation took this item off ``src``: a retry must not read
                # the item's plan as still calling for it.
                "folded_from": sorted(set(holder.get("folded_from") or []) | {src}),
            })
            store.delete_observation(o["_id"])
            logger.info("Folded duplicate observation of %s from %s into %s", o["item_url"], src, target_topic_id)
            continue
        note = (
            f"\n\nMOVED by {reason} on {now.date().isoformat()}: from "
            f"'{src}' to '{target_topic_id}'."
        )
        store.update_observation(o["_id"], {
            "topic_id": target_topic_id,
            "reasoning_trace": (o.get("reasoning_trace") or "") + note,
            "moved_from": src,
        })

    retired: list[str] = []
    for tid in [target_topic_id, *sorted(sources)]:
        store.append_snapshot(tid, _compute_snapshot(store, tid, now))
        if (
            retire_empty_sources
            and tid != target_topic_id
            and store.count_observations_for_topic(tid) == 0
        ):
            topic = store.get_topic(tid)
            if topic and topic.get("status") != "retired":
                store.update_topic(tid, {
                    "status": "retired", "retired_at": now, "merged_into": target_topic_id,
                })
                retired.append(tid)
                logger.info("Retired emptied topic %s (merged_into %s)", tid, target_topic_id)
    return retired


def move_observation(
    item_url: str,
    target_topic_id: str,
    store: Store,
    dry_run: bool = False,
) -> dict:
    """Move one item's observation(s) onto an existing topic."""
    target = store.get_topic(target_topic_id)
    if not target:
        raise ValueError(f"target topic not found: {target_topic_id}")
    if target.get("status") == "retired":
        raise ValueError(f"target topic is retired: {target_topic_id}")
    moves, missing = plan_moves(store, [item_url], target_topic_id)
    if not moves:
        raise ValueError(
            f"no observation to move for {item_url}"
            + (" (already on the target topic or never recorded)" if missing else "")
        )
    will_empty = [
        o["topic_id"] for o in moves
        if store.count_observations_for_topic(o["topic_id"]) == 1
    ]
    report = {
        "dry_run": dry_run,
        "target_topic_id": target_topic_id,
        "moves": [
            {"item_title": o.get("item_title", ""), "from_topic": o["topic_id"],
             "strength": o.get("strength"), "connection_type": o.get("connection_type")}
            for o in moves
        ],
        "sources_that_would_empty": sorted(set(will_empty)),
    }
    if dry_run:
        return report
    report["sources_retired"] = move_observations(
        store, moves, target_topic_id, datetime.now(timezone.utc), reason="move-observation"
    )
    return report
