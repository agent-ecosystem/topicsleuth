from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from topicsleuth.embeddings import (
    crossed_refresh_threshold,
    refresh_topic_embedding,
)
from topicsleuth.interfaces import Embedder, Store
from topicsleuth.settings import Settings, load_settings
from topicsleuth.synthesize import ItemEval, SynthesisDecision

logger = logging.getLogger(__name__)

# Thresholds for status transitions based on observation count and velocity
STATUS_THRESHOLDS = {
    "emerging_to_active": 5,     # observations needed to become active
    "active_to_mature": 20,      # observations needed to become mature
    "fading_velocity": 0.1,      # velocity below this marks fading
    "dormant_days": 30,          # days without evidence marks dormant
}

# Strength floors for well-evidenced topics to prevent over-accumulation
STRENGTH_FLOORS = {
    20: 0.7,   # 20+ observations: require strength >= 0.7
}


def _topical_tags(tags: list[str], settings: Settings) -> list[str]:
    """Drop the feeder's routing tags (settings.candidates.ignored_tags) that
    the model copies from the item onto a proposed topic. They carry no
    topical signal and, left in place, make every future item tag-match the
    topic."""
    ignored = set(settings.candidates.ignored_tags)
    return [t for t in tags if t not in ignored]


def _as_utc(dt: datetime) -> datetime:
    """A naive datetime on a feeder's item is taken as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _item_date(item: dict | None) -> datetime | None:
    """When an item happened: its publish date, else when it was gathered.
    Items are the feeder's documents, so this accepts datetimes or ISO
    strings, naive or aware, which is how feeders write them."""
    for key in ("date_published", "date_gathered"):
        value = (item or {}).get(key)
        if isinstance(value, datetime):
            return _as_utc(value)
        if isinstance(value, str) and value:
            try:
                return _as_utc(datetime.fromisoformat(value))
            except ValueError:
                continue
    return None


def _strength_floor(evidence_count: int) -> float | None:
    """Return the minimum strength required for a topic with this many observations."""
    for threshold, floor in sorted(STRENGTH_FLOORS.items(), reverse=True):
        if evidence_count >= threshold:
            return floor
    return None


@dataclass
class RecordResult:
    observations_created: list[dict] = field(default_factory=list)
    topics_created: list[dict] = field(default_factory=list)
    topics_updated: list[str] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)


def _compute_snapshot(store: Store, topic_id: str, now: datetime) -> dict:
    observations = store.get_observations_for_topic(topic_id)
    evidence_count = len(observations)

    source_names = set()
    source_types = set()
    for obs in observations:
        if obs.get("source_name"):
            source_names.add(obs["source_name"])
        if obs.get("source_type"):
            source_types.add(obs["source_type"])

    # Velocity: observations in the last 7 days / 7. Dates from the store
    # are timezone-aware (the Store contract), as is ``now``.
    week_ago = now - timedelta(days=7)
    recent_count = sum(
        1 for obs in observations if obs.get("date") and obs["date"] >= week_ago
    )
    velocity = recent_count / 7.0

    # Recency weight: inverse of days since latest observation
    if observations:
        latest = max(obs.get("date") or now for obs in observations)
        days_since = (now - latest).total_seconds() / 86400.0
        recency_weight = 1.0 / (1.0 + days_since)
    else:
        days_since = None
        recency_weight = 0.0

    return {
        "date": now,
        "evidence_count": evidence_count,
        "source_diversity": len(source_names),
        "source_type_diversity": len(source_types),
        "velocity": velocity,
        "recency_weight": recency_weight,
        "days_since_evidence": days_since,
    }


def _detect_transition(
    topic: dict, snapshot: dict
) -> dict | None:
    current_status = topic.get("status", "emerging")
    evidence_count = snapshot["evidence_count"]
    velocity = snapshot["velocity"]

    new_status = current_status

    if current_status == "emerging":
        if evidence_count >= STATUS_THRESHOLDS["emerging_to_active"]:
            new_status = "active"
    elif current_status == "active":
        if evidence_count >= STATUS_THRESHOLDS["active_to_mature"]:
            new_status = "mature"
        elif velocity < STATUS_THRESHOLDS["fading_velocity"]:
            new_status = "fading"
    elif current_status == "mature":
        if velocity < STATUS_THRESHOLDS["fading_velocity"]:
            new_status = "fading"
    elif current_status == "fading":
        days_since = snapshot.get("days_since_evidence")
        if velocity >= STATUS_THRESHOLDS["fading_velocity"]:
            new_status = "active"
        elif days_since is None or days_since >= STATUS_THRESHOLDS["dormant_days"]:
            new_status = "dormant"
    elif current_status == "dormant":
        if velocity > 0:
            new_status = "emerging"

    if new_status != current_status:
        return {
            "topic_id": topic["topic_id"],
            "title": topic["title"],
            "from_status": current_status,
            "to_status": new_status,
        }
    return None


def record(
    items: list[dict],
    eval_results: list[ItemEval],
    decisions: list[SynthesisDecision],
    store: Store,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    backfill: bool = False,
    date_from_item: bool = False,
) -> RecordResult:
    """Persist a batch: new topics, their observations, item embeddings,
    then a snapshot and status check on every topic touched, and a status
    check on every topic that was not, so a question that stops drawing
    evidence fades and goes dormant on its own.

    The feeder's own scores for an item are not written here; a feeder
    persists its evaluation under its own key. With backfill=True every
    observation is stamped backfill: true, so a consumer can keep re-homed
    items out of "what is new today" while still counting them on topic
    pages.

    With date_from_item=True, observations are dated when their item was
    published (or gathered) rather than when this ran, a proposed topic is
    dated by its founding item, snapshots are taken as of the latest item
    in the batch for that topic, and a topic's date_identified moves back
    if an observation predates it. That is what makes a historical backfill
    read as history: process items oldest first so snapshots accumulate in
    sequence."""
    settings = settings or load_settings()
    now = datetime.now(timezone.utc)
    result = RecordResult()

    affected_topics: set[str] = set()
    # Latest observation date per topic in this batch; the snapshot's "now".
    batch_dates: dict[str, datetime] = {}

    def item_for(url: str) -> dict | None:
        return next((i for i in items if i["url"] == url), None)

    def date_for(item: dict | None) -> datetime:
        if date_from_item:
            return _item_date(item) or now
        return now

    # 1. Create new topics from propose_new decisions
    for decision in decisions:
        if decision.decision == "propose_new" and decision.new_topic:
            founded = date_for(item_for(decision.item_url))
            topic_doc = {
                "topic_id": decision.new_topic["topic_id"],
                "title": decision.new_topic["title"],
                "summary": decision.new_topic["summary"],
                "tags": _topical_tags(
                    decision.new_topic.get("tags", []), settings
                ),
                "suggested_directions": decision.new_topic.get(
                    "suggested_directions", []
                ),
                "embedding": decision.new_topic.get("embedding", []),
                "date_identified": founded,
                "last_updated": founded,
            }
            try:
                store.create_topic(topic_doc)
                result.topics_created.append(topic_doc)
                logger.info("Created topic: %s", topic_doc["topic_id"])
            except Exception:
                logger.error(
                    "Failed to create topic: %s",
                    topic_doc["topic_id"],
                    exc_info=True,
                )

    # 2. Create observations for link_existing and propose_new decisions
    for decision in decisions:
        if decision.decision == "skip":
            continue

        topic_id = decision.topic_id
        if decision.decision == "propose_new" and decision.new_topic:
            topic_id = decision.new_topic["topic_id"]

        if not topic_id:
            continue

        # Flag weak observations for well-evidenced topics as minor
        is_minor = False
        if decision.decision == "link_existing":
            evidence_count = store.count_observations_for_topic(topic_id)
            strength = decision.strength if decision.strength is not None else 0.7
            required = _strength_floor(evidence_count)
            if required and strength < required:
                is_minor = True
                logger.info(
                    "Marking minor observation (%.2f < %.2f floor) for "
                    "well-evidenced topic %s (%d obs): %s",
                    strength, required, topic_id, evidence_count,
                    decision.item_title[:50],
                )

        # Look up source info from the item
        item = item_for(decision.item_url)
        obs_date = date_for(item)

        obs_doc = {
            "item_id": item["_id"] if item and "_id" in item else None,
            "item_url": decision.item_url,
            "item_title": decision.item_title,
            "topic_id": topic_id,
            "date": obs_date,
            "strength": decision.strength if decision.strength is not None else 0.7,
            "connection_type": decision.connection_type or "supports",
            "reasoning_trace": decision.reasoning_trace,
            "source_name": item.get("source_name", "") if item else "",
            "source_type": item.get("source_type", "") if item else "",
            # Copied from the item so a topic's lane mix and evidence mix
            # over time are computable without joining back to items.
            "tags": list(item.get("tags") or []) if item else [],
            "evidence": item.get("evidence") if item else None,
            "lanes": item.get("lanes") if item else None,
            "entities_mentioned": [],
            "minor": is_minor,
        }
        if backfill:
            obs_doc["backfill"] = True

        try:
            store.create_observation(obs_doc)
            result.observations_created.append(obs_doc)
            affected_topics.add(topic_id)
            if topic_id not in batch_dates or obs_date > batch_dates[topic_id]:
                batch_dates[topic_id] = obs_date
            logger.info(
                "Created observation: %s -> %s",
                decision.item_title[:40],
                topic_id,
            )
        except Exception:
            logger.error(
                "Failed to create observation for %s",
                decision.item_url,
                exc_info=True,
            )

    # 3. Save embeddings on item documents
    for eval_result in eval_results:
        try:
            store.save_item_embedding(eval_result.url, eval_result.embedding)
        except Exception:
            logger.error(
                "Failed to save embedding for %s", eval_result.url, exc_info=True
            )

    # 4. Compute and append snapshots to affected topics, detect transitions,
    #    refresh embeddings on evidence-count threshold crossings.
    for topic_id in affected_topics:
        topic = store.get_topic(topic_id)
        if not topic:
            continue

        prev_snapshots = topic.get("snapshots") or []
        prev_count = (
            prev_snapshots[-1].get("evidence_count", 0) if prev_snapshots else 0
        )

        as_of = batch_dates.get(topic_id, now)
        snapshot = _compute_snapshot(store, topic_id, as_of)
        store.append_snapshot(topic_id, snapshot)
        result.topics_updated.append(topic_id)

        if date_from_item:
            # Vector search does not filter candidates by date, so an old
            # item can link to a topic founded later; the topic then dates
            # from its earliest evidence.
            earliest = min(
                (o["date"] for o in result.observations_created if o["topic_id"] == topic_id),
                default=None,
            )
            identified = topic.get("date_identified")
            if earliest and identified and earliest < identified:
                store.update_topic(topic_id, {"date_identified": earliest})

        if crossed_refresh_threshold(prev_count, snapshot["evidence_count"]):
            if embedder is None:
                logger.warning(
                    "Topic %s crossed a refresh threshold but no embedder was "
                    "given; skipping the embedding refresh", topic_id,
                )
            else:
                try:
                    refresh_topic_embedding(topic_id, store, settings, embedder)
                except Exception:
                    logger.warning(
                        "Failed to refresh embedding for topic %s", topic_id,
                        exc_info=True,
                    )

        transition = _detect_transition(topic, snapshot)
        if transition:
            _apply_transition(store, result, transition)

    # 5. Status check on the quiet topics. A topic that received nothing
    #    this run can only fade or go dormant, and nothing else would ever
    #    look at it. The check is as of this run's "now", or, when loading
    #    history, the latest date in the batch, so history reads as history.
    #    A snapshot is written only when the status changes, so a long quiet
    #    spell does not pile up identical snapshots.
    as_of = max(batch_dates.values(), default=None) if date_from_item else now
    if as_of is not None:
        for topic in store.list_topics(include_retired=False):
            topic_id = topic["topic_id"]
            if topic_id in affected_topics:
                continue
            snapshot = _compute_snapshot(store, topic_id, as_of)
            transition = _detect_transition(topic, snapshot)
            if transition:
                store.append_snapshot(topic_id, snapshot)
                result.topics_updated.append(topic_id)
                _apply_transition(store, result, transition)

    return result


def _apply_transition(store: Store, result: RecordResult, transition: dict) -> None:
    store.update_topic(transition["topic_id"], {"status": transition["to_status"]})
    result.transitions.append(transition)
    logger.info(
        "Topic transition: %s %s -> %s",
        transition["topic_id"],
        transition["from_status"],
        transition["to_status"],
    )
