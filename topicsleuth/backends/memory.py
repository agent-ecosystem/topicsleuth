"""An in-memory Store: the reference implementation and the one tests use.

Holds plain dicts in lists. Nearest-topic search is exact cosine over every
topic with an embedding, which is what a vector index approximates.

Every public method runs under one re-entrant lock, so each is atomic
with respect to every other whichever threads call them. That is the
reference for what the contract asks of a real store: the check and the
write of a guarded operation cannot be interleaved with another caller's.
"""
from __future__ import annotations

import copy
import functools
import itertools
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from topicsleuth.history import fold_fields
from topicsleuth.interfaces import DELETING_LEASE_SECONDS, DestinationNotLive, DuplicateObservation, DuplicateTopic, StoreError

LIVE_STATUSES_EXCLUDED = ("retired", "deleting")


def _aware(value):
    """The Store contract: a naive datetime written to the store is UTC,
    and everything read back is timezone-aware."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, dict):
        return {k: _aware(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_aware(v) for v in value]
    return value


def _pick(doc: dict, fields: Sequence[str] | None, identity: str) -> dict:
    """A copy of the document with exactly ``fields`` and its identity,
    the Store contract for a narrowed read; the whole document without
    ``fields``."""
    if fields is None:
        return copy.deepcopy(doc)
    keep = {*fields, identity}
    return copy.deepcopy({k: v for k, v in doc.items() if k in keep})


def _atomic(method):
    """Run the method under the store's lock: one operation, start to end."""
    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class MemoryStore:
    def __init__(self) -> None:
        self.topics: dict[str, dict] = {}
        self.observations: dict[int, dict] = {}
        self.item_embeddings: dict[str, list[float]] = {}
        self.items: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self._lock = threading.RLock()

    # ── topics ──
    @_atomic
    def get_topic(self, topic_id: str) -> dict | None:
        t = self.topics.get(topic_id)
        return copy.deepcopy(t) if t else None

    @_atomic
    def list_topics(self, include_retired: bool = True, *, fields: Sequence[str] | None = None) -> list[dict]:
        return [_pick(t, fields, "topic_id") for t in self.topics.values()
                if include_retired or t.get("status") not in LIVE_STATUSES_EXCLUDED]

    @_atomic
    def create_topic(self, topic_doc: dict) -> str:
        if topic_doc["topic_id"] in self.topics:
            raise DuplicateTopic(f"duplicate topic_id: {topic_doc['topic_id']}")
        now = datetime.now(timezone.utc)
        doc = _aware(dict(topic_doc))
        doc.setdefault("date_identified", now)
        doc.setdefault("recorded_at", now)
        doc.setdefault("last_updated", now)
        doc.setdefault("status", "emerging")
        doc.setdefault("snapshots", [])
        doc.setdefault("entities", [])
        doc.setdefault("suggested_directions", [])
        self.topics[doc["topic_id"]] = doc
        return doc["topic_id"]

    @_atomic
    def update_topic(self, topic_id: str, updates: dict) -> None:
        t = self.topics[topic_id]
        t.update(_aware(updates))
        if "embedding" in updates:
            t["embedding_revision"] = t.get("embedding_revision", 0) + 1
        t["last_updated"] = datetime.now(timezone.utc)

    @_atomic
    def publish_topic_embedding(self, topic_id: str, fields: dict, *, expected_revision: int,
                                missing: list[str] = ()) -> bool:
        t = self.topics.get(topic_id)
        if t is None or t.get("embedding_revision", 0) != expected_revision:
            return False
        t.update(_aware(fields))
        t["embedding_refresh_pending"] = bool(missing)
        t["embedding_refresh_missing"] = sorted(set(missing))
        t["embedding_revision"] = expected_revision + 1
        t["last_updated"] = datetime.now(timezone.utc)
        return True

    @_atomic
    def defer_topic_refresh(self, topic_id: str, missing: list[str]) -> None:
        t = self.topics[topic_id]
        t["embedding_refresh_pending"] = True
        t["embedding_refresh_missing"] = sorted(set(t.get("embedding_refresh_missing") or []) | set(missing))
        t["embedding_revision"] = t.get("embedding_revision", 0) + 1
        t["last_updated"] = datetime.now(timezone.utc)

    # Each lease operation is one method under the lock, with no
    # observable intermediate state: the reference implementation of
    # atomicity.
    def _held_by(self, topic_id: str, deleting_owner: str | None) -> bool:
        if deleting_owner is None:
            return True
        t = self.topics.get(topic_id)
        return bool(t and t.get("status") == "deleting" and (t.get("deleting") or {}).get("owner") == deleting_owner)

    @_atomic
    def delete_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> bool:
        if not self._held_by(topic_id, deleting_owner):
            return False
        return self.topics.pop(topic_id, None) is not None

    @_atomic
    def acquire_deletion_lease(self, topic_id: str, owner: str, by: str, expires_at: datetime) -> dict | None:
        t = self.topics.get(topic_id)
        if not t:
            return None
        lease = t.get("deleting") or {}
        now = datetime.now(timezone.utc)
        if t.get("status") == "deleting":
            if lease.get("owner") != owner and lease.get("expires_at") and _aware(lease["expires_at"]) > now:
                return None
            prior = lease.get("prior_status", "emerging")
        else:
            prior = t.get("status", "emerging")
        t["deleting"] = {"prior_status": prior, "started_at": lease.get("started_at", now) if lease.get("owner") == owner else now,
                         "by": by, "owner": owner, "expires_at": _aware(expires_at)}
        t["status"] = "deleting"
        return copy.deepcopy(t["deleting"])

    @_atomic
    def release_deletion_lease(self, topic_id: str, owner: str) -> bool:
        if not self._held_by(topic_id, owner):
            return False
        t = self.topics[topic_id]
        t["status"] = (t.get("deleting") or {}).get("prior_status", "emerging")
        t.pop("deleting", None)
        return True

    @_atomic
    def append_snapshot(self, topic_id: str, snapshot: dict) -> None:
        t = self.topics[topic_id]
        snapshots = t.setdefault("snapshots", [])
        snap = _aware(dict(snapshot))
        snap.setdefault("recorded_at", datetime.now(timezone.utc))
        snapshots.append(snap)
        snapshots.sort(key=lambda s: s.get("date") or datetime.min.replace(tzinfo=timezone.utc))
        t["last_updated"] = datetime.now(timezone.utc)

    @_atomic
    def nearest_topics(self, embedding: list[float], limit: int, *, fields: Sequence[str] | None = None) -> list[dict]:
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored = []
        for t in self.topics.values():
            e = t.get("embedding")
            if not e or len(e) != len(embedding):
                continue
            v = np.asarray(e, dtype=np.float32)
            vn = float(np.linalg.norm(v)) or 1.0
            scored.append((float(q @ v / (qn * vn)), t))
        scored.sort(key=lambda s: -s[0])
        out = []
        for score, t in scored[:limit]:
            d = _pick(t, fields, "topic_id")
            d["vector_score"] = score
            out.append(d)
        return out

    @_atomic
    def seed_topics_if_empty(self, topic_docs: list[dict]) -> int:
        if self.topics:
            return 0
        for d in topic_docs:
            self.create_topic(d)
        return len(topic_docs)

    @_atomic
    def find_weak_topics(self, max_obs: int, older_than_days: int | None) -> list[dict]:
        cutoff = None
        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        weak = []
        for t in self.topics.values():
            if t.get("status") not in ("emerging", "active"):
                continue
            if cutoff and t.get("date_identified") and t["date_identified"] >= cutoff:
                continue
            if self.count_observations_for_topic(t["topic_id"]) <= max_obs:
                weak.append(copy.deepcopy(t))
        return weak

    # ── observations ──
    def _holder(self, item_url, topic_id, except_id=None):
        return next((k for k, o in self.observations.items()
                     if k != except_id and o.get("item_url") == item_url and o.get("topic_id") == topic_id), None)

    def _live(self, topic_id: str) -> bool:
        dest = self.topics.get(topic_id)
        return dest is not None and dest.get("status") not in LIVE_STATUSES_EXCLUDED

    @_atomic
    def create_observation(self, obs_doc: dict) -> Any:
        if self._holder(obs_doc.get("item_url"), obs_doc.get("topic_id")) is not None:
            raise DuplicateObservation(f"observation exists for {obs_doc.get('item_url')} on {obs_doc.get('topic_id')}")
        oid = next(self._ids)
        doc = _aware(dict(obs_doc))
        doc.setdefault("date", datetime.now(timezone.utc))
        doc.setdefault("recorded_at", datetime.now(timezone.utc))
        doc["_id"] = oid
        self.observations[oid] = doc
        return oid

    @_atomic
    def get_observations_for_topic(self, topic_id: str, *, fields: Sequence[str] | None = None) -> list[dict]:
        return [_pick(o, fields, "_id") for o in self.observations.values() if o.get("topic_id") == topic_id]

    @_atomic
    def count_observations_for_topic(self, topic_id: str) -> int:
        return sum(1 for o in self.observations.values() if o.get("topic_id") == topic_id)

    @_atomic
    def get_observations_for_item(self, item_url: str) -> list[dict]:
        return [copy.deepcopy(o) for o in self.observations.values() if o.get("item_url") == item_url]

    @_atomic
    def list_observations(self) -> list[dict]:
        return [copy.deepcopy(o) for o in self.observations.values()]

    @_atomic
    def update_observation(self, obs_id: Any, updates: dict) -> None:
        current = self.observations[obs_id]
        url = updates.get("item_url", current.get("item_url"))
        topic_id = updates.get("topic_id", current.get("topic_id"))
        if "topic_id" in updates and topic_id != current.get("topic_id") and not self._live(topic_id):
            dest = self.topics.get(topic_id)
            raise DestinationNotLive(f"topic {topic_id!r} is {'gone' if dest is None else dest['status']}")
        if ("item_url" in updates or "topic_id" in updates) and self._holder(url, topic_id, except_id=obs_id) is not None:
            raise DuplicateObservation(f"observation exists for {url} on {topic_id}")
        current.update(_aware(updates))

    @_atomic
    def fold_observation(self, obs_id: Any, into_obs_id: Any, note: str) -> None:
        survivor = self.observations.get(into_obs_id)
        if survivor is None:
            raise StoreError(f"observation {into_obs_id!r} not found")
        source = self.observations.get(obs_id)
        if source is None:
            return  # already folded, or gone: nothing to fold
        if not self._live(survivor.get("topic_id")):
            dest = self.topics.get(survivor.get("topic_id"))
            raise DestinationNotLive(f"topic {survivor.get('topic_id')!r} is {'gone' if dest is None else dest['status']}")
        survivor.update(fold_fields(survivor, source, note))
        del self.observations[obs_id]

    @_atomic
    def delete_observation(self, obs_id: Any) -> bool:
        return self.observations.pop(obs_id, None) is not None

    @_atomic
    def delete_observations_for_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> int:
        if not self._held_by(topic_id, deleting_owner):
            return 0
        doomed = [k for k, o in self.observations.items() if o.get("topic_id") == topic_id]
        for k in doomed:
            del self.observations[k]
        return len(doomed)

    # ── item embeddings ──
    @_atomic
    def save_item_embedding(self, item_url: str, embedding: list[float]) -> None:
        self.item_embeddings[item_url] = list(embedding)
        self.items.setdefault(item_url, {"url": item_url})["embedding"] = list(embedding)

    @_atomic
    def get_item_embeddings(self, item_urls: list[str]) -> dict[str, list[float]]:
        return {u: self.item_embeddings[u] for u in item_urls if u in self.item_embeddings}

    @_atomic
    def get_items(self, item_urls: list[str]) -> list[dict]:
        return [copy.deepcopy(self.items[u]) for u in item_urls if u in self.items]

    # ── maintenance ──
    @_atomic
    def delete_since(self, cutoff: datetime, run_id: str | None = None) -> dict:
        c = _aware(cutoff)

        def in_window(doc: dict) -> bool:
            stamp = doc.get("recorded_at")
            return bool(stamp and stamp >= c and (run_id is None or doc.get("run_id") == run_id))

        owner = uuid.uuid4().hex
        expires = datetime.now(timezone.utc) + timedelta(seconds=DELETING_LEASE_SECONDS)
        # Every topic the window touches: its own document, an observation
        # on it or a snapshot of it. An old topic with new evidence counts.
        affected = ({k for k, t in self.topics.items() if in_window(t) or any(in_window(s) for s in t.get("snapshots", []))}
                    | {o.get("topic_id") for o in self.observations.values() if in_window(o)})
        counts = {"observations_deleted": 0, "topics_deleted": 0, "topics_kept": 0, "topics_busy": 0, "snapshots_cleared": 0}
        for tid in sorted(affected, key=str):
            topic = self.topics.get(tid)
            doomed = [k for k, o in self.observations.items() if o.get("topic_id") == tid and in_window(o)]
            if topic is None:
                # Orphaned evidence: no topic, so nothing to lease.
                for k in doomed:
                    del self.observations[k]
                counts["observations_deleted"] += len(doomed)
                continue
            # Lease the topic for the whole of its cleanup, observations,
            # snapshots and then the document, and give it back or delete
            # it under that same lease. One another operation holds is
            # left entirely alone.
            if self.acquire_deletion_lease(tid, owner, "clean", expires) is None:
                counts["topics_busy"] += 1
                continue
            for k in doomed:
                del self.observations[k]
            counts["observations_deleted"] += len(doomed)
            before = len(topic.get("snapshots", []))
            topic["snapshots"] = [s for s in topic.get("snapshots", []) if not in_window(s)]
            if in_window(topic) and not self.count_observations_for_topic(tid):
                if self.delete_topic(tid, deleting_owner=owner):
                    counts["topics_deleted"] += 1
                continue
            # Snapshots cleared counts what was taken off a topic that stays.
            counts["snapshots_cleared"] += before - len(topic["snapshots"])
            self.release_deletion_lease(tid, owner)
            if in_window(topic):
                counts["topics_kept"] += 1
        return counts
