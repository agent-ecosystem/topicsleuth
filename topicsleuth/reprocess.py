"""Selective reprocess: clear the long tail of weak topics and let the
improved synthesis pipeline re-home their items against the now-fully-
populated topic universe.

Targets exactly the duplicate-prone tail (topics with <= max_obs
observations) without disturbing well-evidenced topics or their
trajectory data. The cleanup step deletes orphaned observations and
removes the empty weak topics, and reports the item URLs that lost their
home. Re-queuing those items is the feeder's job: the store does not know
how a feeder marks an item consumed.

Retired topics are skipped: their provenance metadata (split_into,
merged_into) is part of the audit trail and should not be touched.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from topicsleuth.interfaces import DELETING_LEASE_SECONDS, Store
from topicsleuth.history import removed_history

logger = logging.getLogger(__name__)


@dataclass
class ReprocessPlan:
    weak_topics: list[dict]
    item_urls: list[str]
    observation_count: int

    @property
    def topic_count(self) -> int:
        return len(self.weak_topics)

    @property
    def item_count(self) -> int:
        return len(self.item_urls)


def plan_reprocess(
    store: Store, max_obs: int = 2, older_than_days: int | None = None
) -> ReprocessPlan:
    """Identify the weak topics, their observations, and the items that
    will be reset. Pure read; no writes."""
    weak_topics = store.find_weak_topics(
        max_obs=max_obs, older_than_days=older_than_days
    )
    item_urls: set[str] = set()
    obs_count = 0
    for topic in weak_topics:
        observations = store.get_observations_for_topic(topic["topic_id"])
        obs_count += len(observations)
        for obs in observations:
            url = obs.get("item_url")
            if url:
                item_urls.add(url)
    return ReprocessPlan(
        weak_topics=weak_topics,
        item_urls=sorted(item_urls),
        observation_count=obs_count,
    )


def mark_deleting(store: Store, topic_id: str, by: str, owner: str,
                  lease_seconds: int = DELETING_LEASE_SECONDS, now: datetime | None = None) -> bool:
    """Take the deletion lease on a topic (``Store.acquire_deletion_lease``):
    mark it ``deleting`` with the owner, an expiry, and the prior status
    kept on the document. Returns False, and changes nothing, if another
    owner's lease is still live. An expired lease is taken over, keeping
    the original prior status. The store makes the test and the write one
    operation."""
    now = now or datetime.now(timezone.utc)
    return store.acquire_deletion_lease(topic_id, owner, by, now + timedelta(seconds=lease_seconds)) is not None


def recover_interrupted_deletions(store: Store, force: bool = False) -> dict:
    """Finish or undo deletions whose lease has expired: the runs that took
    them crashed or stalled. Any caller can run this; it needs nothing from
    those runs. A marked topic that still holds observations gets its prior
    status back; one that holds none is deleted. A live lease is another
    operation still at work and is left alone unless ``force`` is given.
    Returns the ids restored, deleted and left alone."""
    now = datetime.now(timezone.utc)
    owner = uuid.uuid4().hex
    restored: list[str] = []
    deleted: list[str] = []
    busy: list[str] = []
    for topic in store.list_topics(include_retired=True):
        if topic.get("status") != "deleting":
            continue
        topic_id = topic["topic_id"]
        lease = topic.get("deleting") or {}
        if not force and lease.get("expires_at") and lease["expires_at"] > now:
            busy.append(topic_id)
            continue
        if force and lease.get("expires_at") and lease["expires_at"] > now:
            store.update_topic(topic_id, {"deleting": {**lease, "expires_at": now}})
        # Take the lease over first, atomically, so the stalled owner, if it
        # wakes, finds its guarded writes refused; a competing recovery
        # loses the same race and moves on.
        taken = store.acquire_deletion_lease(topic_id, owner, "recover", now + timedelta(seconds=DELETING_LEASE_SECONDS))
        if taken is None:
            busy.append(topic_id)
            continue
        if store.count_observations_for_topic(topic_id):
            if store.release_deletion_lease(topic_id, owner):
                restored.append(topic_id)
                logger.warning("Restored topic %s to %s: an abandoned deletion had left it marked", topic_id, taken.get("prior_status"))
        else:
            store.delete_observations_for_topic(topic_id, deleting_owner=owner)
            if store.delete_topic(topic_id, deleting_owner=owner):
                deleted.append(topic_id)
                logger.warning("Finished an abandoned deletion of topic %s", topic_id)
    return {"restored": restored, "deleted": deleted, "busy": busy}


def execute_reprocess(store: Store, plan: ReprocessPlan) -> dict:
    """Apply the plan: delete observations on weak topics and the weak
    topics themselves. Returns counts plus ``item_urls``, the items the
    feeder should re-queue so its next run re-homes them.
    """
    deleted_obs = 0
    deleted_topics = 0
    skipped: list[str] = []
    owner = uuid.uuid4().hex

    for topic in plan.weak_topics:
        topic_id = topic["topic_id"]
        # Take the deletion lease before touching anything: an insert that
        # checks its destination from here on is refused, one that landed
        # earlier is deleted with the rest, and every destructive write is
        # guarded by the lease, so a run that lost it writes nothing more.
        if not mark_deleting(store, topic_id, by="reprocess", owner=owner):
            skipped.append(topic_id)
            logger.warning("Topic %s is being deleted by another operation; skipped", topic_id)
            continue
        # The deleted topic leaves the item's other observations' plans
        # pointing at it: record the removal there, so a retry never revives
        # it should the id be reused.
        for o in store.get_observations_for_topic(topic_id):
            for sibling in store.get_observations_for_item(o["item_url"]):
                if sibling.get("topic_id") != topic_id:
                    store.update_observation(sibling["_id"], {"removed_from": removed_history(sibling, topic_id)})
        deleted_obs += store.delete_observations_for_topic(topic_id, deleting_owner=owner)
        if store.delete_topic(topic_id, deleting_owner=owner):
            deleted_topics += 1
            logger.info("Deleted weak topic: %s", topic_id)
        else:
            skipped.append(topic_id)
            logger.warning("Lost the deletion lease on %s; left as is", topic_id)

    return {
        "topics_deleted": deleted_topics,
        "observations_deleted": deleted_obs,
        "topics_skipped": skipped,
        "item_urls": list(plan.item_urls),
    }
