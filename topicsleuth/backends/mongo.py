"""MongoDB Atlas as the Store. ``pip install "topicsleuth[mongo]"``.

Topics live in one collection with an Atlas Vector Search index over their
``embedding`` field; observations in another. Item embeddings are kept on
the item documents of a third collection, which a feeder usually owns, so
the only fields this store touches there are ``url`` and ``embedding``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from topicsleuth.interfaces import StoreError

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
            from pymongo import ASCENDING
            from pymongo.errors import ConnectionFailure, OperationFailure
        except ImportError as e:  # pragma: no cover
            raise ImportError('MongoStore needs the pymongo package: pip install "topicsleuth[mongo]"') from e
        names = {**DEFAULT_COLLECTIONS, **(collections or {})}
        self.client = client
        self.db = client[database]
        self.topics = self.db[names["topics"]]
        self.observations = self.db[names["observations"]]
        self.items = self.db[names["items"]]
        self.vector_index = vector_index
        self.num_candidates = num_candidates
        try:
            self.topics.create_index("topic_id", unique=True)
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
        topic_doc.setdefault("last_updated", now)
        topic_doc.setdefault("status", "emerging")
        topic_doc.setdefault("snapshots", [])
        topic_doc.setdefault("entities", [])
        topic_doc.setdefault("suggested_directions", [])
        self.topics.insert_one(topic_doc)
        return topic_doc["topic_id"]

    def update_topic(self, topic_id: str, updates: dict) -> None:
        updates = {**updates, "last_updated": datetime.now(timezone.utc)}
        self.topics.update_one({"topic_id": topic_id}, {"$set": updates})

    def delete_topic(self, topic_id: str) -> bool:
        return self.topics.delete_one({"topic_id": topic_id}).deleted_count > 0

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None:
        self.topics.update_one(
            {"topic_id": topic_id},
            {"$push": {"snapshots": snapshot}, "$set": {"last_updated": datetime.now(timezone.utc)}},
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
        obs_doc.setdefault("date", datetime.now(timezone.utc))
        return self.observations.insert_one(obs_doc).inserted_id

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

    def reassign_observations(self, source_topic_id: str, target_topic_id: str) -> int:
        return self.observations.update_many(
            {"topic_id": source_topic_id}, {"$set": {"topic_id": target_topic_id}}
        ).modified_count

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
    def delete_since(self, cutoff: datetime) -> dict:
        c = cutoff.replace(tzinfo=None) if cutoff.tzinfo else cutoff  # Mongo stores naive UTC
        obs = self.observations.delete_many({"date": {"$gte": c}}).deleted_count
        older = [t["topic_id"] for t in self.topics.find({"date_identified": {"$lt": c}}, {"topic_id": 1})]
        topics = self.topics.delete_many({"date_identified": {"$gte": c}}).deleted_count
        self.topics.update_many({"topic_id": {"$in": older}}, {"$pull": {"snapshots": {"date": {"$gte": c}}}})
        return {"observations_deleted": obs, "topics_deleted": topics, "snapshots_cleared_on": len(older)}
