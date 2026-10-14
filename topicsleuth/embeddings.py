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


def refresh_topic_embedding(
    topic_id: str,
    store: Store,
    settings: Settings,
    embedder: Embedder,
) -> bool:
    """Recompute a topic's embedding as a 70/30 blend of (a) the centroid
    of its non-minor observations' item embeddings and (b) a fresh embedding
    of its title+summary text. Persists the result via store.update_topic.

    Returns True if an update was performed; False if the topic was missing,
    had no usable observations, or no item embeddings could be loaded.
    """
    topic = store.get_topic(topic_id)
    if not topic:
        logger.warning("refresh_topic_embedding: topic not found: %s", topic_id)
        return False

    observations = store.get_observations_for_topic(topic_id)
    non_minor = [o for o in observations if not o.get("minor")]
    if not non_minor:
        logger.debug(
            "refresh_topic_embedding: %s has no non-minor observations; skipping",
            topic_id,
        )
        return False

    item_urls = [o["item_url"] for o in non_minor if o.get("item_url")]
    if not item_urls:
        return False

    item_embeddings = [
        e for e in store.get_item_embeddings(item_urls).values()
        if e and len(e) == settings.embedding.dimensions
    ]
    if not item_embeddings:
        logger.warning(
            "refresh_topic_embedding: %s has %d non-minor observations but no "
            "item embeddings could be loaded; skipping",
            topic_id, len(non_minor),
        )
        return False

    centroid = _normalized_mean(item_embeddings)
    seed = _seed_embedding(topic, embedder)

    blended = CENTROID_WEIGHT * centroid + (1.0 - CENTROID_WEIGHT) * seed
    blended = _l2_normalize(blended)

    store.update_topic(topic_id, {"embedding": blended.tolist()})
    logger.info(
        "Refreshed embedding for %s: blended centroid of %d items with seed "
        "(%.0f/%.0f weight)",
        topic_id, len(item_embeddings),
        100 * CENTROID_WEIGHT, 100 * (1 - CENTROID_WEIGHT),
    )
    return True


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
