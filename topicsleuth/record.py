from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from topicsleuth.embeddings import (
    refresh_owed,
    refresh_topic_embedding,
)
from topicsleuth.interfaces import DestinationNotLive, DuplicateObservation, DuplicateTopic, Embedder, ItemFailure, Store
from topicsleuth.rubric import DEFAULT_CONNECTION_TYPE
from topicsleuth.settings import Settings, load_settings
from topicsleuth.history import removed_history
from topicsleuth.synthesize import ItemEval, Link, SynthesisDecision, _live_successor, intended_links

logger = logging.getLogger(__name__)

# Thresholds for status transitions based on observation count and velocity
STATUS_THRESHOLDS = {
    "emerging_to_active": 5,     # observations needed to become active
    "active_to_mature": 20,      # observations needed to become mature
    "fading_velocity": 0.1,      # velocity below this marks fading
    "dormant_days": 30,          # days without evidence marks dormant
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


def _strength_floor(evidence_count: int, settings: Settings) -> float | None:
    """The strength below which a new observation on a topic with this many
    observations is marked minor, or None while the topic is small
    (settings.record)."""
    if evidence_count >= settings.record.minor_floor_after:
        return settings.record.minor_floor_strength
    return None


@dataclass
class RecordResult:
    """What a batch did. ``failures`` lists every item that did not finish,
    with the stage it stopped at; a feeder acknowledges only the items
    outside ``failed_urls()`` and retries the rest. An item whose
    observation already existed (a retried batch, or two feeders admitting
    the same document) is ``already_recorded``, not a failure and not a
    second observation."""
    observations_created: list[dict] = field(default_factory=list)
    already_recorded: list[dict] = field(default_factory=list)
    topics_created: list[dict] = field(default_factory=list)
    topics_updated: list[str] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    failures: list[ItemFailure] = field(default_factory=list)

    def failed_urls(self) -> set[str]:
        return {f.item_url for f in self.failures}


def _compute_snapshot(store: Store, topic_id: str, now: datetime) -> dict:
    """The topic as of ``now``: only evidence dated at or before it counts,
    so a snapshot taken for a historical date reads as history even when
    the store already holds newer observations."""
    observations = [
        o for o in store.get_observations_for_topic(topic_id)
        if not o.get("date") or o["date"] <= now
    ]
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
    run_id: str | None = None,
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
    sequence. A batch dated before a topic's latest snapshot is inserted
    into its history and does not change the topic's current status.

    Recording is idempotent: an observation is identified by its item and
    topic, and recording a decision whose observation exists reports it as
    already recorded instead of inserting it again. A failed decision, a
    topic that could not be created, or a destination that no longer
    exists is reported in ``failures`` and nothing is written for that
    item. ``run_id``, if given, is stamped on everything written so a run
    can be removed by ``delete_since``."""
    settings = settings or load_settings()
    now = datetime.now(timezone.utc)
    result = RecordResult()

    affected_topics: set[str] = set()
    # Latest observation date per topic in this batch; the snapshot's "now".
    batch_dates: dict[str, datetime] = {}
    # Proposals that could not be created, so no observation lands on them.
    failed_topics: set[str] = set()

    def fail(url: str, stage: str, error: BaseException | str) -> None:
        text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        result.failures.append(ItemFailure(item_url=url, stage=stage, error=text))

    for decision in decisions:
        if decision.decision == "failed":
            fail(decision.item_url, "synthesize", decision.error or "failed")

    # Items the store already holds. The stored observation is the durable
    # decision for that item (a retried batch, or a document another feeder
    # admitted first); this batch's decision for it is not applied, and a
    # proposal for it does not open a topic. What a failed earlier run left
    # unfinished after the insert (snapshot, status, embedding) is resumed.
    already: dict[str, list[dict]] = {}
    for decision in decisions:
        if decision.decision in ("skip", "failed"):
            continue
        existing = store.get_observations_for_item(decision.item_url)
        if existing:
            already[decision.item_url] = existing

    # Topics whose status was re-checked from an existing snapshot while
    # resuming; the quiet-topic pass leaves them alone.
    resumed_status: set[str] = set()
    # Topics found with a refresh still owed; done after the item
    # embeddings are saved, so the refresh sees them.
    pending_refresh: set[str] = set()

    def resume(stored: dict) -> None:
        """Pick up an observation the store already held. One its topic has
        not snapshotted since it arrived there, whether it was recorded
        (``recorded_at``) or moved onto the topic by a repair
        (``relocated_at``, which the move stamps: the topic's count changed
        then, whatever older evidence the observation is), gets the
        derived work of a new observation."""
        result.already_recorded.append(stored)
        topic_id = stored.get("topic_id")
        topic = store.get_topic(topic_id) if topic_id else None
        if not topic:
            return
        snapshots = topic.get("snapshots") or []
        stamps = [s.get("recorded_at") for s in snapshots if s.get("recorded_at")]
        arrived = max((t for t in (stored.get("recorded_at"), stored.get("relocated_at")) if t), default=None)
        snapshotted = arrived and stamps and max(stamps) >= arrived
        # An observation moved onto the topic owes it a refresh until a
        # refresh that included its embedding has been written: every
        # refresh records its inputs, and that list, not a timestamp, is
        # what answers the move. The obligation lives on the observation,
        # so it survives whatever failed after the move; no snapshot and no
        # refresh that left the item out stands in for it.
        if stored.get("relocated_at") and not stored.get("minor") \
                and stored.get("item_url") not in (topic.get("embedding_inputs") or []):
            pending_refresh.add(topic_id)
        # A threshold refresh is owed whenever the count has crossed one
        # since the embedding was last refreshed, read from the store:
        # whatever run inserted the evidence, and whatever it got to.
        if refresh_owed(topic, store.count_observations_for_topic(topic_id)):
            pending_refresh.add(topic_id)
        if not snapshotted:
            logger.info("Observation %s arrived on %s without a snapshot since; finishing it", stored.get("item_url"), topic_id)
            affected_topics.add(topic_id)
            d = stored.get("date") or now
            if topic_id not in batch_dates or d > batch_dates[topic_id]:
                batch_dates[topic_id] = d
        elif topic_id not in affected_topics and topic_id not in resumed_status:
            # Snapshotted, but the status change it implied may not have
            # landed: apply the status the latest snapshot calls for.
            resumed_status.add(topic_id)
            transition = _detect_transition(topic, snapshots[-1])
            if transition:
                try:
                    _apply_transition(store, result, transition)
                except Exception as e:
                    logger.error("Failed to update status of %s", topic_id, exc_info=True)
                    fail(stored.get("item_url", ""), "status", e)
            # The two derived stages that can fail after the snapshot: the
            # rebuild of later snapshots when this was history, and the
            # embedding refresh. Both are recomputed from the store, so
            # repeating them is safe; a failure fails the item again.
            latest = max((s["date"] for s in snapshots if s.get("date")), default=None)
            if stored.get("date") and latest is not None and stored["date"] < latest:
                try:
                    _rebuild_later_snapshots(store, topic_id, stored["date"])
                except Exception as e:
                    logger.error("Failed to rebuild later snapshots of %s", topic_id, exc_info=True)
                    fail(stored.get("item_url", ""), "snapshot_rebuild", e)
            if topic.get("embedding_refresh_pending"):
                # Not here: the item embeddings this batch saves come first.
                pending_refresh.add(topic_id)

    def item_for(url: str) -> dict | None:
        return next((i for i in items if i["url"] == url), None)

    def date_for(item: dict | None) -> datetime:
        if date_from_item:
            return _item_date(item) or now
        return now

    # 1. Create new topics from propose_new decisions
    for decision in decisions:
        if decision.decision == "propose_new" and decision.new_topic and decision.item_url not in already:
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
            if run_id:
                topic_doc["run_id"] = run_id
            try:
                store.create_topic(topic_doc)
                result.topics_created.append(topic_doc)
                logger.info("Created topic: %s", topic_doc["topic_id"])
            except DuplicateTopic:
                existing = store.get_topic(topic_doc["topic_id"])
                if existing and existing.get("title") == topic_doc["title"]:
                    # The same proposal landing twice: a retried batch, or
                    # two feeders proposing it. The topic exists; its
                    # observation is handled below like any other.
                    logger.info("Topic %s already exists with this title; reusing it", topic_doc["topic_id"])
                else:
                    failed_topics.add(topic_doc["topic_id"])
                    fail(decision.item_url, "create_topic",
                         f"topic_id {topic_doc['topic_id']!r} already names a different topic")
            except Exception as e:
                logger.error("Failed to create topic: %s", topic_doc["topic_id"], exc_info=True)
                failed_topics.add(topic_doc["topic_id"])
                fail(decision.item_url, "create_topic", e)

    # 2. Create observations: one per link, or one on the proposed topic
    for decision in decisions:
        if decision.decision in ("skip", "failed"):
            continue

        item = item_for(decision.item_url)
        obs_date = date_for(item)

        if decision.item_url in already:
            # The stored observations carry the decision that was made for
            # this item, including every link it intended. This batch's
            # decision for it, whatever it says, is not applied; what the
            # earlier run left unwritten of its own intent is written now.
            # First, an observation left on a topic that was merged as it
            # landed is put where the merge would have put it; where it is
            # is not taken as proof that the decision is complete.
            try:
                stored, blocked = _rehome_stranded(store, already[decision.item_url], decision.item_url, now, fail)
            except Exception as e:
                # The store failed mid-repair. Whatever it had committed is
                # self-describing (the move carries its stamp), so a fresh
                # run picks the repair up; this item fails, the rest go on.
                logger.error("Failed to repair the stored observations of %s", decision.item_url, exc_info=True)
                fail(decision.item_url, "relocate", e)
                continue
            for o in stored:
                resume(o)
            if blocked:
                continue
            present = {o.get("topic_id") for o in stored}
            intended = intended_links(stored, store)
            missing = [l for l in intended if l.topic_id not in present]
            if missing:
                logger.info("Item %s was recorded on %s of %s intended topics; finishing it",
                            decision.item_url, len(present), len(intended))
                earlier = SynthesisDecision(
                    item_url=decision.item_url, item_title=stored[0].get("item_title", decision.item_title),
                    decision="link_existing", topic_id=intended[0].topic_id, connection_type=intended[0].connection_type,
                    strength=intended[0].strength, reasoning_trace=stored[0].get("reasoning_trace", ""), links=intended,
                )
                for link in missing:
                    _record_link(
                        store, result, earlier, link, item, stored[0].get("date") or obs_date, now,
                        bool(stored[0].get("backfill")), run_id,
                        failed_topics, affected_topics, batch_dates, resume, fail, settings,
                    )
            continue

        if decision.decision == "propose_new" and decision.new_topic:
            links = [Link(topic_id=decision.new_topic["topic_id"], connection_type=DEFAULT_CONNECTION_TYPE,
                          strength=decision.strength if decision.strength is not None else 0.7)]
        else:
            links = list(decision.links)
        if decision.decision == "propose_new":
            decision = SynthesisDecision(**{**decision.__dict__, "links": links})

        for link in links:
            _record_link(
                store, result, decision, link, item, obs_date, now, backfill, run_id,
                failed_topics, affected_topics, batch_dates, resume, fail, settings,
            )

    # 3. Save embeddings on item documents. An item whose observation is in
    #    the store owes its embedding to that topic's centroid, whatever
    #    else failed for it; an item with no observation written is left
    #    untouched.
    persisted = {o["item_url"] for o in result.observations_created + result.already_recorded}
    for eval_result in eval_results:
        if eval_result.url not in persisted:
            continue
        try:
            store.save_item_embedding(eval_result.url, eval_result.embedding)
        except Exception as e:
            logger.error("Failed to save embedding for %s", eval_result.url, exc_info=True)
            fail(eval_result.url, "save_embedding", e)

    # Inputs owed and not in the store, however that came about: a refresh
    # that would read them is deferred until a retry has saved them.
    embed_failed = persisted - set(store.get_item_embeddings(sorted(persisted))) if persisted else set()

    # 4. Compute and append snapshots to affected topics, detect transitions,
    #    refresh embeddings on evidence-count threshold crossings.
    for topic_id in affected_topics:
        topic = store.get_topic(topic_id)
        if not topic:
            continue

        urls_here = [o["item_url"] for o in result.observations_created + result.already_recorded
                     if o.get("topic_id") == topic_id]

        as_of = batch_dates.get(topic_id, now)
        snapshot = _compute_snapshot(store, topic_id, as_of)
        if run_id:
            snapshot["run_id"] = run_id
        try:
            store.append_snapshot(topic_id, snapshot)
        except Exception as e:
            logger.error("Failed to snapshot topic %s", topic_id, exc_info=True)
            for url in urls_here:
                fail(url, "snapshot", e)
            continue
        result.topics_updated.append(topic_id)

        # A batch dated before the topic's latest snapshot is history being
        # filled in: it gets its snapshot, every later snapshot is recomputed
        # to include it, and the topic's current status, set by newer
        # evidence, stays.
        latest_snapshot = max(
            (s["date"] for s in topic.get("snapshots") or [] if s.get("date")), default=None,
        )
        is_history = latest_snapshot is not None and as_of < latest_snapshot
        if is_history:
            try:
                _rebuild_later_snapshots(store, topic_id, as_of)
            except Exception as e:
                logger.error("Failed to rebuild later snapshots of %s", topic_id, exc_info=True)
                for url in urls_here:
                    fail(url, "snapshot_rebuild", e)

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

        new_count = store.count_observations_for_topic(topic_id)
        if refresh_owed(topic, new_count) or topic.get("embedding_refresh_pending") or topic_id in pending_refresh:
            _refresh_or_defer(store, topic_id, settings, embedder, urls_here, fail, embed_failed)
        pending_refresh.discard(topic_id)

        transition = None if is_history else _detect_transition(topic, snapshot)
        if transition:
            try:
                _apply_transition(store, result, transition)
            except Exception as e:
                logger.error("Failed to update status of %s", topic_id, exc_info=True)
                for url in urls_here:
                    fail(url, "status", e)

    # 4b. Refreshes still owed on topics this batch only resumed.
    for topic_id in pending_refresh:
        if store.get_topic(topic_id):
            urls_here = [o["item_url"] for o in result.already_recorded if o.get("topic_id") == topic_id]
            _refresh_or_defer(store, topic_id, settings, embedder, urls_here, fail, embed_failed)

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
            if topic_id in affected_topics or topic_id in resumed_status:
                continue
            identified = topic.get("date_identified")
            if identified and identified > as_of:
                continue  # did not exist yet as of this batch
            latest_snapshot = max(
                (s["date"] for s in topic.get("snapshots") or [] if s.get("date")), default=None,
            )
            if latest_snapshot is not None and as_of < latest_snapshot:
                continue  # history being filled in; its status is current
            snapshot = _compute_snapshot(store, topic_id, as_of)
            transition = _detect_transition(topic, snapshot)
            if transition:
                if run_id:
                    snapshot["run_id"] = run_id
                store.append_snapshot(topic_id, snapshot)
                result.topics_updated.append(topic_id)
                _apply_transition(store, result, transition)

    return result


def _refresh_or_defer(store: Store, topic_id: str, settings: Settings, embedder: Embedder | None,
                      urls: list[str], fail, embed_failed: set[str] = frozenset()) -> None:
    """Refresh a topic's embedding now, or leave the obligation registered
    on the topic (``embedding_refresh_pending``) so the next run that
    touches it tries again. The registration is a convenience, not the
    record: whether a threshold refresh is owed is read from the store
    (``refresh_owed``, against the count the embedding was last refreshed
    at), so an interruption or a failed registration loses nothing.

    A refresh computed from incomplete inputs never looks finished: the
    refresh publishes, in the same write as its embedding, every input it
    wanted and could not load, whether that input's writer failed to save
    it or is still saving it. Inputs this batch failed to save are
    registered first (``Store.defer_topic_refresh``, which merges them
    into ``embedding_refresh_missing`` without dropping what other batches
    registered), in case the refresh does not get as far as publishing.
    Registration bumps the same revision publication is conditional on,
    and publication sets the obligation only inside that compare-and-swap,
    so a refresh settles exactly the obligation it knew of, never one
    registered while it computed or after it published.

    A failed attempt, including a refresh that could not publish because
    other publications kept landing first (``RefreshConflict``), fails the
    items that caused it and leaves the obligation, so nothing is
    acknowledged on a result that did not commit."""
    unsaved_here = sorted(set(urls) & set(embed_failed))
    try:
        if unsaved_here:
            logger.warning("Topic %s needs an embedding refresh but %s have no saved embedding yet; registered", topic_id, unsaved_here)
            store.defer_topic_refresh(topic_id, unsaved_here)
        if embedder is None:
            logger.warning("Topic %s needs an embedding refresh but no embedder was given; deferred", topic_id)
            store.defer_topic_refresh(topic_id, [])
            return
        refreshed = refresh_topic_embedding(topic_id, store, settings, embedder)
        if not refreshed:
            store.defer_topic_refresh(topic_id, [])
    except Exception as e:
        # The refresh, or registering that it is owed, failed. The items
        # fail and retry; what is owed is read from the store next time
        # whether or not the registration below lands.
        logger.error("Failed to refresh embedding for topic %s; deferred", topic_id, exc_info=True)
        try:
            store.defer_topic_refresh(topic_id, [])
        except Exception:
            logger.error("Could not register the owed refresh of %s either", topic_id, exc_info=True)
        for url in urls:
            fail(url, "embedding_refresh", e)


def _rebuild_later_snapshots(store: Store, topic_id: str, after: datetime) -> None:
    """Snapshots are event-time history: when evidence is inserted before
    them, every later snapshot is recomputed as of its own date so the
    series, and the latest count the judge is shown, include it. Each keeps
    its recorded_at and run_id."""
    topic = store.get_topic(topic_id)
    if not topic:
        return
    measured = ("evidence_count", "source_diversity", "source_type_diversity",
                "velocity", "recency_weight", "days_since_evidence")
    rebuilt = []
    for s in topic.get("snapshots") or []:
        if s.get("date") and s["date"] > after:
            fresh = _compute_snapshot(store, topic_id, s["date"])
            s = {**s, **{k: fresh[k] for k in measured}}
        rebuilt.append(s)
    store.update_topic(topic_id, {"snapshots": rebuilt})


def _apply_transition(store: Store, result: RecordResult, transition: dict) -> None:
    store.update_topic(transition["topic_id"], {"status": transition["to_status"]})
    result.transitions.append(transition)
    logger.info(
        "Topic transition: %s %s -> %s",
        transition["topic_id"],
        transition["from_status"],
        transition["to_status"],
    )


def _rehome_stranded(store: Store, stored: list[dict], url: str, now: datetime, fail) -> tuple[list[dict], bool]:
    """Observations an item holds on retired merge sources: a write that
    landed on a topic as it was being merged, too late for the merge's own
    sweep, whose relocation did not finish. Each is moved to the live
    successor, folded into the item's observation there if it has one, as
    the merge would have done. A successor that is marked for deletion
    blocks the item: it fails, and the observation waits where it is as
    the durable record of the decision. One with no live successor at all
    is withdrawn and the item fails, to be judged again. A move stamps
    ``relocated_at`` on the observation, in the same write: the
    destination's count changed then, and the stamp is the whole record
    of what is owed for it. The snapshot is owed until one postdates the
    stamp, and the refresh until one that included the item's embedding
    has been written (``embedding_inputs`` on the topic), whichever run
    gets to do them; there is no second write to register either, so
    nothing that fails after the move can lose them.
    Returns the item's observations as they now stand, less any still
    stranded, and whether the item is blocked."""
    blocked = changed = False
    stranded: set = set()
    for o in stored:
        topic = store.get_topic(o.get("topic_id") or "")
        if not topic or topic.get("status") != "retired" or not topic.get("merged_into"):
            continue
        tid = topic["topic_id"]
        successor = _live_successor(topic["merged_into"], store)
        if successor is None:
            store.delete_observation(o["_id"])
            changed = True
            fail(url, "create_observation", f"observation on retired topic {tid!r} has no live successor; withdrawn to be judged again")
            continue
        day = now.date().isoformat()
        try:
            try:
                store.update_observation(o["_id"], {
                    "topic_id": successor, "moved_from": tid, "removed_from": removed_history(o, tid), "relocated_at": now,
                    "reasoning_trace": (o.get("reasoning_trace") or "")
                    + f"\n\nMOVED by record on {day}: '{tid}' had been merged into '{successor}' when this landed.",
                })
            except DuplicateObservation:
                _fold_into_holder(store, o, successor, url, f"this item's observation on '{tid}' (strength {o.get('strength')}, "
                                  f"{o.get('connection_type')}) landed as that topic was merged into this one.", now)
        except DestinationNotLive as e:
            blocked = True
            stranded.add(o["_id"])
            fail(url, "create_observation", f"observation on retired topic {tid!r} belongs on {successor!r}, which is not live ({e}); left for the next run")
            continue
        changed = True
        logger.warning("Observation for %s had been left on retired topic %s; moved to %s", url, tid, successor)
    current = store.get_observations_for_item(url) if changed else stored
    return [o for o in current if o["_id"] not in stranded], blocked


def _fold_into_holder(store: Store, obs: dict, topic_id: str, url: str, why: str, now: datetime) -> dict:
    """Fold ``obs`` into the item's observation on ``topic_id``. The store
    gives the survivor the union of the two plans and histories, read at
    the moment of the write, so a link only the folded observation carried
    is not lost with it. Returns the holder as it was read."""
    holder = next(h for h in store.get_observations_for_item(url) if h.get("topic_id") == topic_id)
    store.fold_observation(obs["_id"], holder["_id"], f"\n\nFOLDED by record on {now.date().isoformat()}: {why}")
    return holder


def _record_link(store, result, decision, link, item, obs_date, now, backfill, run_id,
                 failed_topics, affected_topics, batch_dates, resume, fail, settings) -> None:
    """Write one observation: the item as evidence for one topic."""
    topic_id = link.topic_id
    if not topic_id:
        return

    # The destination must exist at the moment of writing: a proposal that
    # failed to create, a batch-mate's proposal that failed, or a topic
    # retired by a concurrent merge are all failures, not orphans.
    if topic_id in failed_topics:
        if decision.decision == "link_existing":
            fail(decision.item_url, "create_topic", f"linked to proposal {topic_id!r}, which was not created")
        return
    destination = store.get_topic(topic_id)
    if destination is None:
        fail(decision.item_url, "create_observation", f"topic {topic_id!r} does not exist")
        return
    if destination.get("status") in ("retired", "deleting"):
        fail(decision.item_url, "create_observation", f"topic {topic_id!r} is {destination['status']}")
        return

    evidence_count = store.count_observations_for_topic(topic_id)

    # Flag weak observations for well-evidenced topics as minor
    strength = link.strength if link.strength is not None else 0.7
    is_minor = False
    if decision.decision == "link_existing":
        required = _strength_floor(evidence_count, settings)
        if required and strength < required:
            is_minor = True
            logger.info(
                "Marking minor observation (%.2f < %.2f floor) for "
                "well-evidenced topic %s (%d obs): %s",
                strength, required, topic_id, evidence_count,
                decision.item_title[:50],
            )

    obs_doc = {
        "item_id": item["_id"] if item and "_id" in item else None,
        "item_url": decision.item_url,
        "item_title": decision.item_title,
        "topic_id": topic_id,
        "date": obs_date,
        "strength": strength,
        "connection_type": link.connection_type or DEFAULT_CONNECTION_TYPE,
        "fit_questions": link.fit_questions,
        # The whole decision, so a retry can finish it from any one of its
        # observations: every topic the item was to be evidence for.
        "decision_links": [
            {"topic_id": l.topic_id, "connection_type": l.connection_type,
             "strength": l.strength, "fit_questions": l.fit_questions}
            for l in decision.links
        ],
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
    if run_id:
        obs_doc["run_id"] = run_id

    try:
        obs_id = store.create_observation(obs_doc)
    except DuplicateObservation:
        # Another feeder got here between our read and our write. The
        # evidence is in the store once, which is the point; if we had
        # opened a topic for this item, it is empty and goes.
        logger.info("Observation already recorded by another writer: %s on %s", decision.item_url, topic_id)
        stored = [o for o in store.get_observations_for_item(decision.item_url) if o.get("topic_id") == topic_id]
        resume(stored[0] if stored else obs_doc)
        ours = {t["topic_id"]: t for t in result.topics_created}
        if decision.decision == "propose_new" and topic_id in ours and store.count_observations_for_topic(topic_id) == 0:
            store.delete_topic(topic_id)
            result.topics_created.remove(ours[topic_id])
            logger.info("Removed empty topic %s opened for an item another writer recorded", topic_id)
        return
    except Exception as e:
        logger.error("Failed to create observation for %s", decision.item_url, exc_info=True)
        fail(decision.item_url, "create_observation", e)
        return

    # Verify the destination after the write. Maintenance (merge, split,
    # reprocess, clean) can retire or delete a topic between the check
    # above and the insert; the observation then follows a merge, or is
    # withdrawn and the item fails, so nothing is stranded.
    landed = store.get_topic(topic_id)
    if landed is None or landed.get("status") in ("retired", "deleting"):
        target = _live_successor(landed["merged_into"], store) if landed and landed.get("merged_into") else None
        if target is not None:
            try:
                try:
                    store.update_observation(obs_id, {
                        "topic_id": target, "moved_from": topic_id, "removed_from": removed_history(obs_doc, topic_id),
                        "reasoning_trace": obs_doc["reasoning_trace"]
                        + f"\n\nMOVED by record on {now.date().isoformat()}: '{topic_id}' was merged into '{target}' during the write.",
                    })
                except DuplicateObservation:
                    # The item was already evidence on the merge target:
                    # ours is folded into that observation, plan included,
                    # so a link this decision intends and that one did not
                    # is kept.
                    holder = _fold_into_holder(store, {**obs_doc, "_id": obs_id}, target, decision.item_url,
                                               f"this item's observation on '{topic_id}' landed as that topic was merged into this one.", now)
                    resume(holder)
                    return
            except DestinationNotLive as e:
                # The successor is leased for deletion. The observation
                # stays where it landed, on the retired source, as the
                # durable record of the decision; the item fails, and the
                # next run re-homes it once the lease is resolved.
                fail(decision.item_url, "create_observation",
                     f"topic {topic_id!r} was merged into {target!r} during the write, and {target!r} is not live ({e}); "
                     f"the observation waits on {topic_id!r}")
                return
            logger.warning("Topic %s was merged during the write; observation for %s moved to %s", topic_id, decision.item_url, target)
            obs_doc = {**obs_doc, "topic_id": target, "moved_from": topic_id}
            topic_id = target
        else:
            store.delete_observation(obs_id)
            what = "deleted" if landed is None else ("retired" if landed.get("status") == "retired" else "marked for deletion")
            fail(decision.item_url, "create_observation", f"topic {topic_id!r} was {what} during the write")
            return
    if decision.decision == "propose_new":
        # A proposal is a claim that no topic fits. If another writer
        # recorded this item on some topic between our pre-check and our
        # insert, that claim is contested: withdraw, drop the empty topic,
        # and fail the item so the next run sees their decision and resumes.
        others = [o for o in store.get_observations_for_item(decision.item_url) if o.get("topic_id") != topic_id]
        if others:
            store.delete_observation(obs_id)
            ours = {t["topic_id"]: t for t in result.topics_created}
            if topic_id in ours and store.count_observations_for_topic(topic_id) == 0:
                store.delete_topic(topic_id)
                result.topics_created.remove(ours[topic_id])
            elsewhere = sorted(o["topic_id"] for o in others)
            fail(decision.item_url, "create_observation",
                 f"another writer recorded this item on {elsewhere} during the write; proposal withdrawn")
            return
    result.observations_created.append(obs_doc)
    affected_topics.add(topic_id)
    if topic_id not in batch_dates or obs_date > batch_dates[topic_id]:
        batch_dates[topic_id] = obs_date
    logger.info("Created observation: %s -> %s", decision.item_title[:40], topic_id)
