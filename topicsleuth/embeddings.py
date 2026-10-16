"""Topic embedding computation utilities.

A topic's initial embedding comes from its title+summary text. As the topic
accumulates observations, that seed embedding drifts away from the topic's
actual semantic territory: vector search starts to miss items whose phrasing
matches the accumulated evidence but not the seed sentence. The refresh
function here blends the centroid of observation-item embeddings with the
seed embedding to keep the topic grounded in both its identity and its
evidence.

Used by:
- record.py snapshot path, to refresh on evidence-count threshold crossings
- cli.py backfill-embeddings command, for one-shot bulk refresh
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from topicsleuth.interfaces import Embedder, Store
from topicsleuth.settings import Settings

logger = logging.getLogger(__name__)

# Evidence-count thresholds at which a topic's embedding should be refreshed.
# Crossing INTO any of these (prev < T <= new) triggers a refresh.
TOPIC_EMBEDDING_REFRESH_THRESHOLDS = [5, 15, 30, 50, 100]

# Blend weight for the centroid-vs-seed mix. 0.7 keeps the seed identity
# meaningfully present (preventing arbitrary drift) while letting the actual
# evidence dominate the embedding direction.
CENTROID_WEIGHT = 0.7


class RefreshConflict(RuntimeError):
    """A refresh could not publish: every time it computed, another
    embedding was published to the topic, or an obligation registered on
    it, first. The refresh is still owed."""


REFRESH_ATTEMPTS = 3


def refresh_topic_embedding(
    topic_id: str,
    store: Store,
    settings: Settings,
    embedder: Embedder,
) -> bool:
    """Recompute a topic's embedding as a 70/30 blend of (a) the centroid
    of its non-minor observations' item embeddings and (b) a fresh embedding
    of its title+summary text. Publishes the result with
    ``embedding_inputs``, the items whose embeddings went into it: a
    refresh's claim to completeness is exactly that list, not the time it
    finished.

    Publication is conditional (``Store.publish_topic_embedding``): the
    topic's ``embedding_revision`` is read before anything else, and the
    write lands only if no embedding has been published and no obligation
    registered since. A refresh that loses that race recomputes from the
    current state; one that loses it ``REFRESH_ATTEMPTS`` times raises
    ``RefreshConflict`` rather than publish a result computed from a state
    that is gone.

    A refresh computes from the inputs it can load and says so: every
    input it wanted (each non-minor observation's item) and could not
    load, whether its writer failed to save it or is still saving it, is
    written as the topic's remaining obligation in the same publication.
    A partial result therefore never claims completeness, and the writer
    that saves an omitted input finds the refresh still owed.

    Returns True if an update was performed; False if the topic was missing,
    had no usable observations, or no item embeddings could be loaded.
    """
    for _ in range(REFRESH_ATTEMPTS):
        topic = store.get_topic(topic_id)
        if not topic:
            logger.warning("refresh_topic_embedding: topic not found: %s", topic_id)
            return False
        revision = topic.get("embedding_revision", 0)
        computed = _compute_refresh(topic, store, settings, embedder)
        if computed is None:
            return False
        blended, included, wanted, count = computed
        omitted = sorted(set(wanted) - set(included))
        published = store.publish_topic_embedding(topic_id, {
            "embedding": blended.tolist(), "embedding_inputs": included,
            # The count the topic had when this refresh read it: what a
            # later run compares the count then against to see whether a
            # threshold has been crossed since, from the store alone.
            "embedding_refreshed_count": count,
            "embedding_refreshed_at": datetime.now(timezone.utc),
        }, expected_revision=revision, missing=omitted)
        if published:
            logger.info(
                "Refreshed embedding for %s: blended centroid of %d items with seed "
                "(%.0f/%.0f weight)%s",
                topic_id, len(included),
                100 * CENTROID_WEIGHT, 100 * (1 - CENTROID_WEIGHT),
                f"; still owed: {omitted}" if omitted else "",
            )
            return True
        logger.info("Topic %s had an embedding published while this refresh computed; recomputing", topic_id)
    raise RefreshConflict(f"topic {topic_id!r}: an embedding was published ahead of this refresh {REFRESH_ATTEMPTS} times")


# What a refresh reads of a topic's observations, declared to the store
# (the ``fields`` of ``Store.get_observations_for_topic``): which items
# count and which are minor.
REFRESH_FIELDS = ("item_url", "minor")


def _compute_refresh(topic, store, settings, embedder):
    """The blended embedding, the sorted item urls it was computed from,
    the urls it wanted (every non-minor observation's item) and the
    topic's observation count, or None if there is nothing to compute it
    from."""
    topic_id = topic["topic_id"]
    observations = store.get_observations_for_topic(topic_id, fields=REFRESH_FIELDS)
    non_minor = [o for o in observations if not o.get("minor")]
    if not non_minor:
        logger.debug(
            "refresh_topic_embedding: %s has no non-minor observations; skipping",
            topic_id,
        )
        return None

    item_urls = [o["item_url"] for o in non_minor if o.get("item_url")]
    if not item_urls:
        return None

    loaded = {
        url: e for url, e in store.get_item_embeddings(item_urls).items()
        if e and len(e) == settings.embedding.dimensions
    }
    if not loaded:
        logger.warning(
            "refresh_topic_embedding: %s has %d non-minor observations but no "
            "item embeddings could be loaded; skipping",
            topic_id, len(non_minor),
        )
        return None

    included = sorted(loaded)
    centroid = _normalized_mean([loaded[u] for u in included])
    seed = _seed_embedding(topic, embedder)

    blended = CENTROID_WEIGHT * centroid + (1.0 - CENTROID_WEIGHT) * seed
    return _l2_normalize(blended), included, item_urls, len(observations)


def _seed_embedding(topic: dict, embedder: Embedder) -> np.ndarray:
    """Embed the topic's title+summary text. Re-computed each refresh since
    the title/summary may have been updated by a merge."""
    topic_text = f"{topic.get('title','')}\n\n{topic.get('summary','')}"
    return _l2_normalize(np.array(embedder.embed_documents([topic_text])[0], dtype=np.float32))


def _normalized_mean(vectors: Iterable[list[float]]) -> np.ndarray:
    arr = np.array(list(vectors), dtype=np.float32)
    return _l2_normalize(arr.mean(axis=0))


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector
    return vector / norm


def refresh_owed(topic: dict, count: int) -> bool:
    """Whether the topic owes a threshold refresh: a threshold lies between
    the count its embedding was last refreshed at
    (``embedding_refreshed_count``, written with every refresh) and the
    count now. Read from the store alone, so an interruption anywhere
    between an observation's insert and its refresh is recovered by the
    next run that touches the topic. A count that fell below that mark
    (evidence removed) owes a refresh at the next threshold from zero."""
    last = topic.get("embedding_refreshed_count") or 0
    return crossed_refresh_threshold(last if count >= last else 0, count)


def crossed_refresh_threshold(prev_count: int, new_count: int) -> bool:
    """Return True if the evidence_count crossed INTO any of the refresh
    thresholds (prev < T <= new) between the previous and current snapshot.
    """
    if new_count <= prev_count:
        return False
    return any(
        prev_count < t <= new_count
        for t in TOPIC_EMBEDDING_REFRESH_THRESHOLDS
    )
