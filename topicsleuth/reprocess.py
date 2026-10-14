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


def execute_reprocess(store: Store, plan: ReprocessPlan) -> dict:
    """Apply the plan: delete observations on weak topics and the weak
    topics themselves. Returns counts plus ``item_urls``, the items the
    feeder should re-queue so its next run re-homes them.
    """
    deleted_obs = 0
    deleted_topics = 0

    for topic in plan.weak_topics:
        topic_id = topic["topic_id"]
        deleted_obs += store.delete_observations_for_topic(topic_id)
        if store.delete_topic(topic_id):
            deleted_topics += 1
            logger.info("Deleted weak topic: %s", topic_id)

    return {
        "topics_deleted": deleted_topics,
        "observations_deleted": deleted_obs,
        "item_urls": list(plan.item_urls),
    }
