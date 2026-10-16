"""MongoDB Atlas as the Store. ``pip install "topicsleuth[mongo]"``.

Topics live in one collection with an Atlas Vector Search index over their
``embedding`` field; observations in another. Item embeddings are kept on
the item documents of a third collection, which a feeder usually owns, so
the only fields this store touches there are ``url`` and ``embedding``.

MongoDB stores datetimes without a zone and pymongo returns them naive by
default; this store reads every collection with a timezone-aware codec so
dates come back as UTC, as the Store contract requires, whichever client
it is handed.

The guarded operations (a lease-holder's deletes, a move or fold onto a
topic that must be live, cleanup's writes on each topic) run as
multi-document transactions that first write the topic document under the
lease or liveness filter. A lease or retirement that landed after the
transaction's snapshot conflicts with that write or leaves it matching
nothing, and the operation does not commit. Transactions need a replica
set, which every Atlas tier is.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from topicsleuth.history import fold_fields
from topicsleuth.interfaces import DELETING_LEASE_SECONDS, DestinationNotLive, DuplicateObservation, DuplicateTopic, StoreError

DEFAULT_COLLECTIONS = {"topics": "topics", "observations": "observations", "items": "items"}
LIVE = {"$nin": ["retired", "deleting"]}


def _projection(fields: Sequence[str] | None, identity: str) -> dict | None:
    """The Store contract's ``fields`` as a pymongo projection: exactly
    those fields plus the identity, so ``_id`` comes back only when it is
    the identity (an observation's) or was asked for."""
    if fields is None:
        return None
    projection = {f: 1 for f in (*fields, identity)}
    projection.setdefault("_id", 0)
    return projection


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
            # Observation identity: one per item per topic. The unique index
            # is what makes a retried or doubly-fed batch safe.
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

    def list_topics(self, include_retired: bool = True, *, fields: Sequence[str] | None = None) -> list[dict]:
        q = {} if include_retired else {"status": LIVE}
        return list(self.topics.find(q, _projection(fields, "topic_id")))

    def _transaction(self, body):
        """Run ``body(session)`` as one transaction and return its result.
        A write conflict with another writer aborts the transaction and
        it is retried from the start, so the filters inside see the state
        that won."""
        with self.client.start_session() as session:
            return session.with_transaction(body)

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
        op: dict = {"$set": {**updates, "last_updated": datetime.now(timezone.utc)}}
        if "embedding" in updates:
            op["$inc"] = {"embedding_revision": 1}
        self.topics.update_one({"topic_id": topic_id}, op)

    def publish_topic_embedding(self, topic_id: str, fields: dict, *, expected_revision: int,
                                missing: list[str] = ()) -> bool:
        # One conditional update: the filter is the revision the caller
        # read, so a publication in between leaves this one matching
        # nothing and writing nothing.
        q: dict = {"topic_id": topic_id, "embedding_revision": expected_revision}
        if expected_revision == 0:
            q = {"topic_id": topic_id, "$or": [{"embedding_revision": 0}, {"embedding_revision": {"$exists": False}}]}
        r = self.topics.update_one(q, {"$set": {
            **fields, "embedding_refresh_pending": bool(missing), "embedding_refresh_missing": sorted(set(missing)),
            "embedding_revision": expected_revision + 1, "last_updated": datetime.now(timezone.utc),
        }})
        return r.matched_count > 0

    def defer_topic_refresh(self, topic_id: str, missing: list[str]) -> None:
        op: dict = {"$set": {"embedding_refresh_pending": True, "last_updated": datetime.now(timezone.utc)},
                    "$inc": {"embedding_revision": 1}}
        if missing:
            op["$addToSet"] = {"embedding_refresh_missing": {"$each": sorted(set(missing))}}
        self.topics.update_one({"topic_id": topic_id}, op)

    def delete_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> bool:
        q: dict = {"topic_id": topic_id}
        if deleting_owner is not None:
            q.update({"status": "deleting", "deleting.owner": deleting_owner})
        return self.topics.delete_one(q).deleted_count > 0

    # ── the deletion lease ──
    # Acquire and release are single conditional document updates, so two
    # callers racing for one topic are ordered by the server. The guarded
    # writes run in transactions that first write the topic document under
    # the lease filter: if the lease changed hands since the transaction's
    # snapshot, that write conflicts or matches nothing and nothing commits.

    def acquire_deletion_lease(self, topic_id: str, owner: str, by: str, expires_at: datetime) -> dict | None:
        from pymongo import ReturnDocument
        now = datetime.now(timezone.utc)
        doc = self.topics.find_one_and_update(
            {"topic_id": topic_id, "$or": [
                {"status": {"$ne": "deleting"}},
                {"deleting.owner": owner},
                {"deleting.expires_at": {"$lte": now}},
            ]},
            [{"$set": {
                "deleting": {
                    "prior_status": {"$cond": [{"$eq": ["$status", "deleting"]}, "$deleting.prior_status", "$status"]},
                    "started_at": {"$cond": [{"$eq": ["$deleting.owner", owner]}, "$deleting.started_at", now]},
                    "by": by, "owner": owner, "expires_at": expires_at,
                },
                "status": "deleting",
            }}],
            return_document=ReturnDocument.AFTER,
        )
        return doc["deleting"] if doc else None

    def release_deletion_lease(self, topic_id: str, owner: str) -> bool:
        r = self.topics.update_one(
            {"topic_id": topic_id, "status": "deleting", "deleting.owner": owner},
            [{"$set": {"status": "$deleting.prior_status"}}, {"$unset": "deleting"}],
        )
        return r.modified_count > 0

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None:
        snapshot = {"recorded_at": datetime.now(timezone.utc), **snapshot}
        self.topics.update_one(
            {"topic_id": topic_id},
            {"$push": {"snapshots": {"$each": [snapshot], "$sort": {"date": 1}}},
             "$set": {"last_updated": datetime.now(timezone.utc)}},
        )

    def nearest_topics(self, embedding: list[float], limit: int, *, fields: Sequence[str] | None = None) -> list[dict]:
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
        if fields is not None:
            pipeline.append({"$project": {**_projection(fields, "topic_id"), "vector_score": 1}})
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

    def get_observations_for_topic(self, topic_id: str, *, fields: Sequence[str] | None = None) -> list[dict]:
        return list(self.observations.find({"topic_id": topic_id}, _projection(fields, "_id")))

    def count_observations_for_topic(self, topic_id: str) -> int:
        return self.observations.count_documents({"topic_id": topic_id})

    def get_observations_for_item(self, item_url: str) -> list[dict]:
        return list(self.observations.find({"item_url": item_url}))

    def list_observations(self) -> list[dict]:
        return list(self.observations.find())

    def update_observation(self, obs_id: Any, updates: dict) -> None:
        from pymongo.errors import DuplicateKeyError
        if "topic_id" not in updates:
            try:
                self.observations.update_one({"_id": obs_id}, {"$set": updates})
            except DuplicateKeyError as e:
                raise DuplicateObservation(f"update would duplicate an observation: {updates}") from e
            return
        # Moving to another topic: the destination must be live when the
        # write commits. The transaction touches the destination document
        # under a liveness filter, so a lease or retirement that lands in
        # between conflicts or matches nothing, and the move does not commit.
        target = updates["topic_id"]

        def move(session):
            self._touch_live(target, session)
            self.observations.update_one({"_id": obs_id}, {"$set": updates}, session=session)
        try:
            self._transaction(move)
        except DuplicateKeyError as e:
            raise DuplicateObservation(f"update would duplicate an observation: {updates}") from e

    def _touch_live(self, topic_id: str, session) -> None:
        touched = self.topics.update_one(
            {"topic_id": topic_id, "status": LIVE},
            {"$set": {"last_updated": datetime.now(timezone.utc)}}, session=session,
        )
        if touched.matched_count == 0:
            raise DestinationNotLive(f"topic {topic_id!r} is not live")

    def fold_observation(self, obs_id: Any, into_obs_id: Any, note: str) -> None:
        # Both observations are read inside the transaction, so the union
        # is computed from what is there when the write commits; a fold
        # that got in first conflicts with this one, which is retried and
        # reads again.
        def fold(session):
            survivor = self.observations.find_one({"_id": into_obs_id}, session=session)
            if survivor is None:
                raise StoreError(f"observation {into_obs_id!r} not found")
            source = self.observations.find_one({"_id": obs_id}, session=session)
            if source is None:
                return  # already folded, or gone: nothing to fold
            self._touch_live(survivor["topic_id"], session)
            self.observations.update_one({"_id": into_obs_id}, {"$set": fold_fields(survivor, source, note)}, session=session)
            self.observations.delete_one({"_id": obs_id}, session=session)
        self._transaction(fold)

    def delete_observation(self, obs_id: Any) -> bool:
        return self.observations.delete_one({"_id": obs_id}).deleted_count > 0

    def delete_observations_for_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> int:
        if deleting_owner is None:
            return self.observations.delete_many({"topic_id": topic_id}).deleted_count

        def guarded(session):
            held = self.topics.update_one(
                {"topic_id": topic_id, "status": "deleting", "deleting.owner": deleting_owner},
                {"$set": {"deleting.last_write": datetime.now(timezone.utc)}}, session=session,
            )
            if held.matched_count == 0:
                return 0
            return self.observations.delete_many({"topic_id": topic_id}, session=session).deleted_count
        return self._transaction(guarded)

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
        from pymongo import ReturnDocument
        # pymongo encodes an aware datetime as UTC; a naive one is taken as UTC.
        c = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=timezone.utc)
        window: dict = {"recorded_at": {"$gte": cutoff}}
        if run_id is not None:
            window["run_id"] = run_id

        def in_window(doc: dict) -> bool:
            stamp = doc.get("recorded_at")
            return bool(stamp and stamp >= c and (run_id is None or doc.get("run_id") == run_id))

        owner = uuid.uuid4().hex
        expires = datetime.now(timezone.utc) + timedelta(seconds=DELETING_LEASE_SECONDS)
        # Every topic the window touches: its own document, an observation
        # on it or a snapshot of it. An old topic with new evidence counts.
        affected = ({t["topic_id"] for t in self.topics.find(
            {"$or": [window, {"snapshots": {"$elemMatch": window}}]}, {"topic_id": 1})}
            | {tid for tid in self.observations.distinct("topic_id", window) if tid})
        counts = {"observations_deleted": 0, "topics_deleted": 0, "topics_kept": 0, "topics_busy": 0, "snapshots_cleared": 0}

        def clean(session, tid: str) -> dict | None:
            # One transaction per topic, under the lease taken just before
            # it: snapshots and observations in the window go, then the
            # document is deleted if it is in the window and empty, else
            # given back. Nothing is written to it after that.
            before = self.topics.find_one_and_update(
                {"topic_id": tid, "status": "deleting", "deleting.owner": owner},
                {"$pull": {"snapshots": window}, "$set": {"deleting.last_write": datetime.now(timezone.utc)}},
                projection={"snapshots": 1, "recorded_at": 1, "run_id": 1},
                return_document=ReturnDocument.BEFORE, session=session,
            )
            if before is None:
                return None
            cleared = sum(1 for snap in before.get("snapshots") or [] if in_window(snap))
            deleted = self.observations.delete_many({**window, "topic_id": tid}, session=session).deleted_count
            held = {"topic_id": tid, "status": "deleting", "deleting.owner": owner}
            doomed = in_window(before) and self.observations.count_documents({"topic_id": tid}, session=session) == 0
            if doomed:
                self.topics.delete_one(held, session=session)
            else:
                self.topics.update_one(held, [{"$set": {"status": "$deleting.prior_status"}}, {"$unset": "deleting"}], session=session)
            # Snapshots cleared counts what was taken off a topic that stays.
            return {"observations": deleted, "snapshots": 0 if doomed else cleared, "deleted": doomed, "in_window": in_window(before)}

        for tid in sorted(affected, key=str):
            if self.acquire_deletion_lease(tid, owner, "clean", expires) is None:
                if self.get_topic(tid) is None:
                    # Orphaned evidence: no topic, so nothing to lease.
                    counts["observations_deleted"] += self.observations.delete_many({**window, "topic_id": tid}).deleted_count
                else:
                    counts["topics_busy"] += 1
                continue
            outcome = self._transaction(lambda session, tid=tid: clean(session, tid))
            if outcome is None:
                counts["topics_busy"] += 1
                continue
            counts["observations_deleted"] += outcome["observations"]
            counts["snapshots_cleared"] += outcome["snapshots"]
            if outcome["deleted"]:
                counts["topics_deleted"] += 1
            elif outcome["in_window"]:
                counts["topics_kept"] += 1
        return counts
