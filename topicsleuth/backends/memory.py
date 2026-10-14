"""An in-memory Store: the reference implementation and the one tests use.

Holds plain dicts in lists. Nearest-topic search is exact cosine over every
topic with an embedding, which is what a vector index approximates.
"""
from __future__ import annotations

import copy
import itertools
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from topicsleuth.interfaces import DuplicateObservation, DuplicateTopic


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


class MemoryStore:
    def __init__(self) -> None:
        self.topics: dict[str, dict] = {}
        self.observations: dict[int, dict] = {}
        self.item_embeddings: dict[str, list[float]] = {}
        self.items: dict[str, dict] = {}
        self._ids = itertools.count(1)

    # ── topics ──
    def get_topic(self, topic_id: str) -> dict | None:
        t = self.topics.get(topic_id)
        return copy.deepcopy(t) if t else None

    def list_topics(self, include_retired: bool = True) -> list[dict]:
        return [copy.deepcopy(t) for t in self.topics.values()
                if include_retired or t.get("status") not in ("retired", "deleting")]

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

    def update_topic(self, topic_id: str, updates: dict) -> None:
        t = self.topics[topic_id]
        t.update(_aware(updates))
        t["last_updated"] = datetime.now(timezone.utc)

    def delete_topic(self, topic_id: str) -> bool:
        return self.topics.pop(topic_id, None) is not None

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None:
        t = self.topics[topic_id]
        snapshots = t.setdefault("snapshots", [])
        snap = _aware(dict(snapshot))
        snap.setdefault("recorded_at", datetime.now(timezone.utc))
        snapshots.append(snap)
        snapshots.sort(key=lambda s: s.get("date") or datetime.min.replace(tzinfo=timezone.utc))
        t["last_updated"] = datetime.now(timezone.utc)

    def nearest_topics(self, embedding: list[float], limit: int) -> list[dict]:
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
            d = copy.deepcopy(t)
            d["vector_score"] = score
            out.append(d)
        return out

    def seed_topics_if_empty(self, topic_docs: list[dict]) -> int:
        if self.topics:
            return 0
        for d in topic_docs:
            self.create_topic(d)
        return len(topic_docs)

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

    def get_observations_for_topic(self, topic_id: str) -> list[dict]:
        return [copy.deepcopy(o) for o in self.observations.values() if o.get("topic_id") == topic_id]

    def count_observations_for_topic(self, topic_id: str) -> int:
        return sum(1 for o in self.observations.values() if o.get("topic_id") == topic_id)

    def get_observations_for_item(self, item_url: str) -> list[dict]:
        return [copy.deepcopy(o) for o in self.observations.values() if o.get("item_url") == item_url]

    def list_observations(self) -> list[dict]:
        return [copy.deepcopy(o) for o in self.observations.values()]

    def update_observation(self, obs_id: Any, updates: dict) -> None:
        current = self.observations[obs_id]
        url = updates.get("item_url", current.get("item_url"))
        topic_id = updates.get("topic_id", current.get("topic_id"))
        if ("item_url" in updates or "topic_id" in updates) and self._holder(url, topic_id, except_id=obs_id) is not None:
            raise DuplicateObservation(f"observation exists for {url} on {topic_id}")
        current.update(_aware(updates))

    def delete_observation(self, obs_id: Any) -> bool:
        return self.observations.pop(obs_id, None) is not None

    def delete_observations_for_topic(self, topic_id: str) -> int:
        doomed = [k for k, o in self.observations.items() if o.get("topic_id") == topic_id]
        for k in doomed:
            del self.observations[k]
        return len(doomed)

    # ── item embeddings ──
    def save_item_embedding(self, item_url: str, embedding: list[float]) -> None:
        self.item_embeddings[item_url] = list(embedding)
        self.items.setdefault(item_url, {"url": item_url})["embedding"] = list(embedding)

    def get_item_embeddings(self, item_urls: list[str]) -> dict[str, list[float]]:
        return {u: self.item_embeddings[u] for u in item_urls if u in self.item_embeddings}

    def get_items(self, item_urls: list[str]) -> list[dict]:
        return [copy.deepcopy(self.items[u]) for u in item_urls if u in self.items]

    # ── maintenance ──
    def delete_since(self, cutoff: datetime, run_id: str | None = None) -> dict:
        c = _aware(cutoff)

        def in_window(doc: dict) -> bool:
            stamp = doc.get("recorded_at")
            return bool(stamp and stamp >= c and (run_id is None or doc.get("run_id") == run_id))

        obs = [k for k, o in self.observations.items() if in_window(o)]
        for k in obs:
            del self.observations[k]
        deleted = kept = 0
        for k in [k for k, t in self.topics.items() if in_window(t)]:
            # Mark first, then count: an insert that lands after the mark
            # sees it and withdraws, one that landed before is counted. The
            # prior status is kept on the document, so an interrupted
            # cleanup can be finished by anyone, and a retry never mistakes
            # the mark itself for the status to restore.
            t = self.topics[k]
            if t.get("status") != "deleting":
                t["deleting"] = {"prior_status": t.get("status", "emerging"), "started_at": datetime.now(timezone.utc), "by": "clean"}
                t["status"] = "deleting"
            if self.count_observations_for_topic(k):
                t["status"] = (t.get("deleting") or {}).get("prior_status", "emerging")
                t.pop("deleting", None)
                kept += 1
            else:
                del self.topics[k]
                deleted += 1
        cleared = 0
        for t in self.topics.values():
            before = len(t.get("snapshots", []))
            t["snapshots"] = [s for s in t.get("snapshots", []) if not in_window(s)]
            cleared += before - len(t["snapshots"])
        return {"observations_deleted": len(obs), "topics_deleted": deleted,
                "topics_kept": kept, "snapshots_cleared": cleared}
