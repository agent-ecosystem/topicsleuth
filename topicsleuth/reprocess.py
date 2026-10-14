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
from dataclasses import dataclass
from datetime import datetime, timezone

from topicsleuth.interfaces import Store

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


def mark_deleting(store: Store, topic_id: str, by: str) -> None:
    """Mark a topic as being deleted, keeping its prior status on the
    document. A topic already marked keeps its original prior status."""
    topic = store.get_topic(topic_id)
    if not topic or topic.get("status") == "deleting":
        return
    store.update_topic(topic_id, {
        "status": "deleting",
        "deleting": {"prior_status": topic.get("status", "emerging"),
                     "started_at": datetime.now(timezone.utc), "by": by},
    })


def recover_interrupted_deletions(store: Store) -> dict:
    """Finish or undo every deletion a crashed or failed maintenance run
    left half done. Any caller can run this; it needs nothing from the
    run that set the marks. A marked topic that still holds observations
    gets its prior status back; one that holds none is deleted. Returns
    the ids restored and deleted."""
    restored: list[str] = []
    deleted: list[str] = []
    for topic in store.list_topics(include_retired=True):
        if topic.get("status") != "deleting":
            continue
        topic_id = topic["topic_id"]
        if store.count_observations_for_topic(topic_id):
            prior = (topic.get("deleting") or {}).get("prior_status", "emerging")
            store.update_topic(topic_id, {"status": prior, "deleting": None})
            restored.append(topic_id)
            logger.warning("Restored topic %s to %s: an interrupted deletion had left it marked", topic_id, prior)
        else:
            store.delete_observations_for_topic(topic_id)
            store.delete_topic(topic_id)
            deleted.append(topic_id)
            logger.warning("Finished an interrupted deletion of topic %s", topic_id)
    return {"restored": restored, "deleted": deleted}


def execute_reprocess(store: Store, plan: ReprocessPlan) -> dict:
    """Apply the plan: delete observations on weak topics and the weak
    topics themselves. Returns counts plus ``item_urls``, the items the
    feeder should re-queue so its next run re-homes them.
    """
    deleted_obs = 0
    deleted_topics = 0

    for topic in plan.weak_topics:
        topic_id = topic["topic_id"]
        # Mark the topic as being deleted before touching anything: an
        # insert that checks its destination from here on is refused, and
        # one that landed earlier is deleted with the rest. The marker
        # keeps the prior status so an interrupted run can be recovered.
        mark_deleting(store, topic_id, by="reprocess")
        deleted_obs += store.delete_observations_for_topic(topic_id)
        if store.delete_topic(topic_id):
            deleted_topics += 1
            logger.info("Deleted weak topic: %s", topic_id)

    return {
        "topics_deleted": deleted_topics,
        "observations_deleted": deleted_obs,
        "item_urls": list(plan.item_urls),
    }
