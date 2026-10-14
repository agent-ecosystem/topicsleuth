"""MongoDB Atlas as the Store. ``pip install "topicsleuth[mongo]"``.

Topics live in one collection with an Atlas Vector Search index over their
``embedding`` field; observations in another. Item embeddings are kept on
the item documents of a third collection, which a feeder usually owns, so
the only fields this store touches there are ``url`` and ``embedding``.

MongoDB stores datetimes without a zone and pymongo returns them naive by
default; this store reads every collection with a timezone-aware codec so
dates come back as UTC, as the Store contract requires, whichever client
it is handed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from topicsleuth.interfaces import DuplicateObservation, DuplicateTopic, StoreError

DEFAULT_COLLECTIONS = {"topics": "topics", "observations": "observations", "items": "items"}


class MongoStore:
    def __init__(
        self,
        client,
        database: str,
        *,
        collections: dict[str, str] | None = None,
        vector_index: str = "topic_embedding_index",
        num_candidates: int = 200,
    ):
        try:
            from bson.codec_options import CodecOptions
            from pymongo import ASCENDING
            from pymongo.errors import ConnectionFailure, OperationFailure
        except ImportError as e:  # pragma: no cover
            raise ImportError('MongoStore needs the pymongo package: pip install "topicsleuth[mongo]"') from e
        names = {**DEFAULT_COLLECTIONS, **(collections or {})}
        self.client = client
        self.db = client[database]
        utc = CodecOptions(tz_aware=True, tzinfo=timezone.utc)
        self.topics = self.db[names["topics"]].with_options(codec_options=utc)
        self.observations = self.db[names["observations"]].with_options(codec_options=utc)
        self.items = self.db[names["items"]].with_options(codec_options=utc)
        self.vector_index = vector_index
        self.num_candidates = num_candidates
        try:
            self.topics.create_index("topic_id", unique=True)
            # Observation identity: one per (item_url, topic_id). The unique
            # index is what makes a retried or doubly-fed batch safe.
            self.observations.create_index(
                [("item_url", ASCENDING), ("topic_id", ASCENDING)], unique=True, name="item_url_topic_id_unique",
            )
            self.observations.create_index([("item_id", ASCENDING), ("topic_id", ASCENDING)])
            self.observations.create_index("topic_id")
        except OperationFailure as e:
            raise StoreError(f"MongoDB refused the connection (check credentials): {e}") from e
        except ConnectionFailure as e:
            raise StoreError(f"Could not connect to MongoDB: {e}") from e

    @classmethod
    def from_uri(cls, uri: str, database: str, **kwargs) -> "MongoStore":
        from pymongo import MongoClient
        return cls(MongoClient(uri, serverSelectionTimeoutMS=5000), database, **kwargs)

    # ── topics ──
    def get_topic(self, topic_id: str) -> dict | None:
        return self.topics.find_one({"topic_id": topic_id})

    def list_topics(self, include_retired: bool = True) -> list[dict]:
        q = {} if include_retired else {"status": {"$ne": "retired"}}
        return list(self.topics.find(q))

    def create_topic(self, topic_doc: dict) -> str:
        now = datetime.now(timezone.utc)
        topic_doc.setdefault("date_identified", now)
        topic_doc.setdefault("recorded_at", now)
        topic_doc.setdefault("last_updated", now)
        topic_doc.setdefault("status", "emerging")
        topic_doc.setdefault("snapshots", [])
        topic_doc.setdefault("entities", [])
        topic_doc.setdefault("suggested_directions", [])
        from pymongo.errors import DuplicateKeyError
        try:
            self.topics.insert_one(topic_doc)
        except DuplicateKeyError as e:
            raise DuplicateTopic(f"duplicate topic_id: {topic_doc['topic_id']}") from e
        return topic_doc["topic_id"]

    def update_topic(self, topic_id: str, updates: dict) -> None:
        updates = {**updates, "last_updated": datetime.now(timezone.utc)}
        self.topics.update_one({"topic_id": topic_id}, {"$set": updates})

    def delete_topic(self, topic_id: str) -> bool:
        return self.topics.delete_one({"topic_id": topic_id}).deleted_count > 0

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None:
        snapshot = {"recorded_at": datetime.now(timezone.utc), **snapshot}
        self.topics.update_one(
            {"topic_id": topic_id},
            {"$push": {"snapshots": {"$each": [snapshot], "$sort": {"date": 1}}},
             "$set": {"last_updated": datetime.now(timezone.utc)}},
        )

    def nearest_topics(self, embedding: list[float], limit: int) -> list[dict]:
        from pymongo.errors import OperationFailure
        pipeline = [
            {"$vectorSearch": {
                "index": self.vector_index,
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": max(self.num_candidates, limit),
                "limit": limit,
            }},
            {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            return list(self.topics.aggregate(pipeline))
        except OperationFailure as e:
            raise StoreError(f"vector search failed (does index {self.vector_index} exist?): {e}") from e

    def seed_topics_if_empty(self, topic_docs: list[dict]) -> int:
        if self.topics.count_documents({}) > 0:
            return 0
        for doc in topic_docs:
            self.create_topic(doc)
        return len(topic_docs)

    def find_weak_topics(self, max_obs: int, older_than_days: int | None) -> list[dict]:
        query: dict = {"status": {"$in": ["emerging", "active"]}}
        if older_than_days is not None:
            query["date_identified"] = {"$lt": datetime.now(timezone.utc) - timedelta(days=older_than_days)}
        return [t for t in self.topics.find(query)
                if self.count_observations_for_topic(t["topic_id"]) <= max_obs]

    # ── observations ──
    def create_observation(self, obs_doc: dict) -> Any:
        from pymongo.errors import DuplicateKeyError
        obs_doc.setdefault("date", datetime.now(timezone.utc))
        obs_doc.setdefault("recorded_at", datetime.now(timezone.utc))
        try:
            return self.observations.insert_one(obs_doc).inserted_id
        except DuplicateKeyError as e:
            raise DuplicateObservation(
                f"observation exists for {obs_doc.get('item_url')} on {obs_doc.get('topic_id')}"
            ) from e

    def get_observations_for_topic(self, topic_id: str) -> list[dict]:
        return list(self.observations.find({"topic_id": topic_id}))

    def count_observations_for_topic(self, topic_id: str) -> int:
        return self.observations.count_documents({"topic_id": topic_id})

    def get_observations_for_item(self, item_url: str) -> list[dict]:
        return list(self.observations.find({"item_url": item_url}))

    def list_observations(self) -> list[dict]:
        return list(self.observations.find())

    def update_observation(self, obs_id: Any, updates: dict) -> None:
        self.observations.update_one({"_id": obs_id}, {"$set": updates})

    def delete_observations_for_topic(self, topic_id: str) -> int:
        return self.observations.delete_many({"topic_id": topic_id}).deleted_count

    # ── item embeddings ──
    def save_item_embedding(self, item_url: str, embedding: list[float]) -> None:
        self.items.update_one({"url": item_url}, {"$set": {"embedding": embedding}})

    def get_item_embeddings(self, item_urls: list[str]) -> dict[str, list[float]]:
        cursor = self.items.find(
            {"url": {"$in": item_urls}, "embedding": {"$exists": True, "$type": "array"}},
            {"url": 1, "embedding": 1},
        )
        return {i["url"]: i["embedding"] for i in cursor if i.get("embedding")}

    def get_items(self, item_urls: list[str]) -> list[dict]:
        return list(self.items.find({"url": {"$in": item_urls}}))

    def sample_items(self, query: dict, n: int, seed: int) -> list[dict]:
        """Mongo-only convenience for the replay harness: a seeded random
        sample of items matching ``query`` that carry an embedding."""
        import random
        pool = list(self.items.find({**query, "embedding": {"$exists": True}}))
        return random.Random(seed).sample(pool, min(n, len(pool)))

    # ── maintenance ──
    def delete_since(self, cutoff: datetime, run_id: str | None = None) -> dict:
        # pymongo encodes an aware datetime as UTC; a naive one is taken as UTC.
        window: dict = {"recorded_at": {"$gte": cutoff}}
        if run_id is not None:
            window["run_id"] = run_id
        obs = self.observations.delete_many(window).deleted_count
        deleted = kept = 0
        for t in self.topics.find(window, {"topic_id": 1}):
            if self.count_observations_for_topic(t["topic_id"]):
                kept += 1
            else:
                self.topics.delete_one({"topic_id": t["topic_id"]})
                deleted += 1
        snap_window = {"recorded_at": {"$gte": cutoff}}
        if run_id is not None:
            snap_window["run_id"] = run_id
        before = sum(len(t.get("snapshots") or []) for t in self.topics.find({}, {"snapshots": 1}))
        self.topics.update_many({}, {"$pull": {"snapshots": snap_window}})
        after = sum(len(t.get("snapshots") or []) for t in self.topics.find({}, {"snapshots": 1}))
        return {"observations_deleted": obs, "topics_deleted": deleted,
                "topics_kept": kept, "snapshots_cleared": before - after}
