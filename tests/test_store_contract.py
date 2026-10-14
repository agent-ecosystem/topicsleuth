"""One contract, every Store. MemoryStore always runs; MongoStore runs when
TOPICSLEUTH_TEST_MONGODB_URI points at an Atlas cluster, on scratch
collections it creates and drops (the vector index is created for the
test and takes a minute to become queryable)."""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from topicsleuth.backends.memory import MemoryStore
from topicsleuth.interfaces import Store

DIM = 4


@pytest.fixture(scope="module")
def _mongo_resource():
    uri = os.environ.get("TOPICSLEUTH_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("TOPICSLEUTH_TEST_MONGODB_URI not set")
    pymongo = pytest.importorskip("pymongo")
    from pymongo.operations import SearchIndexModel
    from topicsleuth.backends.mongo import MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = os.environ.get("TOPICSLEUTH_TEST_DATABASE", "topicsleuth_test")
    tag = uuid.uuid4().hex[:8]
    names = {k: f"contract_{tag}_{k}" for k in ("topics", "observations", "items")}
    index = f"contract_{tag}_index"
    store = MongoStore(client, db, collections=names, vector_index=index, num_candidates=10)
    store.topics.create_search_index(SearchIndexModel(
        definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": DIM, "similarity": "cosine"}]},
        name=index, type="vectorSearch"))
    for _ in range(80):
        idx = {i["name"]: i for i in store.topics.list_search_indexes()}.get(index)
        if idx and idx.get("queryable"):
            break
        time.sleep(3)
    else:
        pytest.fail("vector index never became queryable")
    yield store
    for n in names.values():
        client[db].drop_collection(n)


@pytest.fixture(params=["memory", "mongo"])
def store(request):
    if request.param == "memory":
        return MemoryStore()
    st = request.getfixturevalue("_mongo_resource")
    for coll in (st.topics, st.observations, st.items):
        coll.delete_many({})
    return st


def _settle(store):
    """Atlas indexes vector writes asynchronously; give a fresh insert a
    moment before searching it."""
    if not isinstance(store, MemoryStore):
        time.sleep(3)


def _seed(store):
    store.create_topic({"topic_id": "a", "title": "A", "embedding": [1.0, 0.0, 0.0, 0.0], "tags": ["x"]})
    store.create_topic({"topic_id": "b", "title": "B", "embedding": [0.0, 1.0, 0.0, 0.0]})
    store.create_topic({"topic_id": "r", "title": "R", "embedding": [1.0, 0.1, 0.0, 0.0], "status": "retired"})


def test_is_a_store(store):
    assert isinstance(store, Store)


def test_topic_lifecycle(store):
    _seed(store)
    assert store.get_topic("missing") is None
    t = store.get_topic("a")
    assert t["status"] == "emerging" and t["snapshots"] == [] and t["entities"] == [] and "date_identified" in t
    assert {x["topic_id"] for x in store.list_topics()} == {"a", "b", "r"}
    assert {x["topic_id"] for x in store.list_topics(include_retired=False)} == {"a", "b"}
    store.update_topic("a", {"title": "A2", "status": "active"})
    assert store.get_topic("a")["title"] == "A2" and store.get_topic("a")["status"] == "active"
    store.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 1})
    assert store.get_topic("a")["snapshots"][-1]["evidence_count"] == 1
    assert store.delete_topic("b") is True and store.delete_topic("b") is False
    assert store.seed_topics_if_empty([{"topic_id": "z"}]) == 0


def test_nearest_topics_scores_by_cosine(store):
    _seed(store)
    _settle(store)
    # nearest_topics may be approximate; the two closest of three are stable
    out = store.nearest_topics([1.0, 0.0, 0.0, 0.0], limit=2)
    ids = [t["topic_id"] for t in out]
    assert ids[0] == "a" and set(ids) == {"a", "r"}
    assert all("vector_score" in t for t in out)
    assert out[0]["vector_score"] >= out[1]["vector_score"]


def test_observations(store):
    _seed(store)
    oid = store.create_observation({"topic_id": "a", "item_url": "u1", "item_title": "One"})
    store.create_observation({"topic_id": "a", "item_url": "u2"})
    assert store.count_observations_for_topic("a") == 2
    assert sorted(o["item_url"] for o in store.get_observations_for_topic("a")) == ["u1", "u2"]
    assert [o["item_title"] for o in store.get_observations_for_item("u1")] == ["One"]
    assert "date" in store.get_observations_for_item("u1")[0]
    store.update_observation(oid, {"strength": 0.9})
    assert store.get_observations_for_item("u1")[0]["strength"] == 0.9
    assert len(store.list_observations()) == 2
    assert store.reassign_observations("a", "b") == 2
    assert store.count_observations_for_topic("b") == 2 and store.count_observations_for_topic("a") == 0
    assert store.delete_observations_for_topic("b") == 2


def test_weak_topics(store):
    old = datetime.now(timezone.utc) - timedelta(days=90)
    store.create_topic({"topic_id": "old-single", "date_identified": old})
    store.create_topic({"topic_id": "young-single"})
    store.create_topic({"topic_id": "busy"})
    store.create_observation({"topic_id": "busy", "item_url": "u1"})
    store.create_observation({"topic_id": "busy", "item_url": "u2"})
    store.create_topic({"topic_id": "ret", "status": "retired", "date_identified": old})
    assert {t["topic_id"] for t in store.find_weak_topics(max_obs=1, older_than_days=None)} == {"old-single", "young-single"}
    assert [t["topic_id"] for t in store.find_weak_topics(max_obs=1, older_than_days=30)] == ["old-single"]


def test_item_embeddings_round_trip(store):
    # an item the feeder already wrote
    if hasattr(store, "items") and not isinstance(store, MemoryStore):
        store.items.insert_one({"url": "u1", "title": "one"})
    store.save_item_embedding("u1", [0.5] * DIM)
    assert store.get_item_embeddings(["u1", "nope"]) == {"u1": [0.5] * DIM}
    assert [i["url"] for i in store.get_items(["u1"])] == ["u1"]


def test_delete_since(store):
    then = datetime.now(timezone.utc) - timedelta(days=3)
    store.create_topic({"topic_id": "old", "date_identified": then})
    store.create_topic({"topic_id": "new"})
    store.create_observation({"topic_id": "old", "item_url": "u1", "date": then})
    store.create_observation({"topic_id": "old", "item_url": "u2"})
    store.append_snapshot("old", {"date": then, "evidence_count": 1})
    store.append_snapshot("old", {"date": datetime.now(timezone.utc), "evidence_count": 2})
    result = store.delete_since(datetime.now(timezone.utc) - timedelta(days=1))
    assert result["observations_deleted"] == 1 and result["topics_deleted"] == 1
    assert store.get_topic("new") is None and store.get_topic("old") is not None
    assert [s["evidence_count"] for s in store.get_topic("old")["snapshots"]] == [1]
    assert [o["item_url"] for o in store.list_observations()] == ["u1"]
