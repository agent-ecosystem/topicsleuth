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
    assert store.delete_observations_for_topic("a") == 2
    assert store.count_observations_for_topic("a") == 0


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
    store.create_topic({"topic_id": "old", "date_identified": then, "recorded_at": then})
    store.create_topic({"topic_id": "new"})
    store.create_topic({"topic_id": "kept"})
    store.create_observation({"topic_id": "old", "item_url": "u1", "date": then, "recorded_at": then})
    store.create_observation({"topic_id": "old", "item_url": "u2"})
    store.create_observation({"topic_id": "kept", "item_url": "u3", "recorded_at": then}, )
    store.create_observation({"topic_id": "old", "item_url": "u4", "run_id": "r2"})
    store.append_snapshot("old", {"date": then, "evidence_count": 1, "recorded_at": then})
    store.append_snapshot("old", {"date": datetime.now(timezone.utc), "evidence_count": 2})
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    scoped = store.delete_since(cutoff, run_id="r2")
    assert scoped["observations_deleted"] == 1 and scoped["topics_deleted"] == 0
    assert sorted(o["item_url"] for o in store.list_observations()) == ["u1", "u2", "u3"]
    result = store.delete_since(cutoff)
    assert result == {"observations_deleted": 1, "topics_deleted": 1, "topics_kept": 1, "topics_busy": 0, "snapshots_cleared": 1}
    assert store.get_topic("new") is None and store.get_topic("old") is not None
    assert store.get_topic("kept") is not None, "a topic recorded in the window but holding older evidence is kept"
    assert [s["evidence_count"] for s in store.get_topic("old")["snapshots"]] == [1]
    assert sorted(o["item_url"] for o in store.list_observations()) == ["u1", "u3"]


def test_an_item_is_evidence_for_a_topic_once(store):
    from topicsleuth.interfaces import DuplicateObservation, DuplicateTopic
    _seed(store)
    oid = store.create_observation({"topic_id": "a", "item_url": "u1"})
    with pytest.raises(DuplicateObservation):
        store.create_observation({"topic_id": "a", "item_url": "u1", "strength": 0.9})
    on_b = store.create_observation({"topic_id": "b", "item_url": "u1"})
    assert len(store.get_observations_for_item("u1")) == 2, "the same item on another topic is another observation"
    with pytest.raises(DuplicateObservation):
        store.update_observation(oid, {"topic_id": "b"})
    other = store.create_observation({"topic_id": "b", "item_url": "u2"})
    with pytest.raises(DuplicateObservation):
        store.update_observation(other, {"item_url": "u1"})
    store.update_observation(other, {"strength": 0.5}), "an update that keeps the pair is fine"
    assert store.delete_observation(on_b) and not store.delete_observation(on_b)
    assert sorted(o["item_url"] for o in store.list_observations()) == ["u1", "u2"]
    with pytest.raises(DuplicateTopic):
        store.create_topic({"topic_id": "a", "title": "again"})


def test_snapshots_are_kept_in_date_order(store):
    _seed(store)
    now = datetime.now(timezone.utc)
    store.append_snapshot("a", {"date": now, "evidence_count": 3})
    store.append_snapshot("a", {"date": now - timedelta(days=2), "evidence_count": 1})
    store.append_snapshot("a", {"date": now - timedelta(days=1), "evidence_count": 2})
    snaps = store.get_topic("a")["snapshots"]
    assert [s["evidence_count"] for s in snaps] == [1, 2, 3]
    assert all("recorded_at" in s for s in snaps)


def test_datetimes_come_back_timezone_aware(store):
    """The contract: a store returns aware UTC datetimes and reads a naive
    one as UTC, so the library never has to guess."""
    naive = datetime(2026, 3, 1, 12, 0)
    store.create_topic({"topic_id": "a", "title": "A", "date_identified": naive})
    store.create_observation({"topic_id": "a", "item_url": "u", "date": naive})
    store.create_observation({"topic_id": "a", "item_url": "v"})
    store.append_snapshot("a", {"date": naive, "evidence_count": 2})
    store.update_topic("a", {"retired_at": naive})
    t = store.get_topic("a")
    assert t["date_identified"] == naive.replace(tzinfo=timezone.utc)
    assert t["retired_at"] == naive.replace(tzinfo=timezone.utc)
    assert t["last_updated"].tzinfo is not None
    assert t["snapshots"][0]["date"] == naive.replace(tzinfo=timezone.utc)
    for o in store.get_observations_for_topic("a"):
        assert o["date"].tzinfo is not None
    assert store.list_topics()[0]["date_identified"].tzinfo is not None


def test_deletes_guarded_by_a_deletion_lease(store):
    """A maintenance operation's destructive writes name the lease it holds;
    without that lease on the topic, atomically, they do nothing."""
    _seed(store)
    store.create_observation({"topic_id": "a", "item_url": "u1"})
    assert store.delete_observations_for_topic("a", deleting_owner="nobody") == 0
    assert not store.delete_topic("a", deleting_owner="nobody")
    assert store.get_topic("a") is not None and store.count_observations_for_topic("a") == 1
    now = datetime.now(timezone.utc)
    store.update_topic("a", {"status": "deleting", "deleting": {"prior_status": "emerging", "owner": "me",
                                                                 "started_at": now, "expires_at": now + timedelta(minutes=15)}})
    assert store.delete_observations_for_topic("a", deleting_owner="other") == 0
    assert not store.delete_topic("a", deleting_owner="other")
    assert store.delete_observations_for_topic("a", deleting_owner="me") == 1
    assert store.delete_topic("a", deleting_owner="me") and store.get_topic("a") is None
    assert store.delete_topic("b"), "an unguarded delete still works for callers that own nothing"


def test_the_deletion_lease_is_acquired_released_and_taken_over_atomically(store):
    _seed(store)
    now = datetime.now(timezone.utc)
    live, past = now + timedelta(minutes=15), now - timedelta(seconds=1)
    lease = store.acquire_deletion_lease("a", "one", "reprocess", live)
    assert lease["owner"] == "one" and lease["prior_status"] == "emerging"
    assert store.acquire_deletion_lease("a", "two", "clean", live) is None, "a live lease is not taken"
    assert store.acquire_deletion_lease("a", "one", "reprocess", live)["owner"] == "one", "the holder may renew"
    assert store.acquire_deletion_lease("missing", "one", "reprocess", live) is None
    assert store.get_topic("a")["status"] == "deleting"
    assert [t["topic_id"] for t in store.list_topics(include_retired=False)] == ["b"], "leased topics are not live"
    assert not store.release_deletion_lease("a", "two")
    assert store.release_deletion_lease("a", "one") and store.get_topic("a")["status"] == "emerging"
    assert not store.get_topic("a").get("deleting")
    # an expired lease is taken over, keeping the original prior status
    store.update_topic("a", {"status": "active"})
    store.acquire_deletion_lease("a", "stalled", "reprocess", past)
    taken = store.acquire_deletion_lease("a", "recover", "recover", live)
    assert taken["owner"] == "recover" and taken["prior_status"] == "active"
    store.create_observation({"topic_id": "a", "item_url": "u1"})
    assert store.delete_observations_for_topic("a", deleting_owner="stalled") == 0, "the old owner's guarded delete is refused"
    assert not store.delete_topic("a", deleting_owner="stalled")
    assert store.count_observations_for_topic("a") == 1


def test_moving_an_observation_requires_a_live_destination(store):
    from topicsleuth.interfaces import DestinationNotLive
    _seed(store)
    oid = store.create_observation({"topic_id": "a", "item_url": "u1"})
    with pytest.raises(DestinationNotLive):
        store.update_observation(oid, {"topic_id": "r"})
    store.acquire_deletion_lease("b", "w", "reprocess", datetime.now(timezone.utc) + timedelta(minutes=15))
    with pytest.raises(DestinationNotLive):
        store.update_observation(oid, {"topic_id": "b"})
    with pytest.raises(DestinationNotLive):
        store.update_observation(oid, {"topic_id": "nowhere"})
    store.release_deletion_lease("b", "w")
    store.update_observation(oid, {"topic_id": "b", "strength": 0.4})
    moved = store.get_observations_for_item("u1")[0]
    assert moved["topic_id"] == "b" and moved["strength"] == 0.4


def test_folding_one_observation_into_another_is_one_guarded_write(store):
    """The store computes what the survivor keeps, from both observations
    as they are when the write commits: the note on its trace, the union of
    the plans, the source's history and topic on its own."""
    from topicsleuth.interfaces import DestinationNotLive, StoreError
    _seed(store)
    plan = lambda *tids: [{"topic_id": t, "connection_type": "supports", "strength": 0.8, "fit_questions": None} for t in tids]
    src = store.create_observation({"topic_id": "a", "item_url": "u1", "strength": 0.5, "decision_links": plan("a", "x"), "removed_from": ["z"]})
    dst = store.create_observation({"topic_id": "b", "item_url": "u1", "strength": 0.9, "reasoning_trace": "r"})
    store.acquire_deletion_lease("b", "w", "reprocess", datetime.now(timezone.utc) + timedelta(minutes=15))
    with pytest.raises(DestinationNotLive):
        store.fold_observation(src, dst, " folded")
    both = {o["topic_id"]: o for o in store.get_observations_for_item("u1")}
    assert sorted(both) == ["a", "b"] and both["b"]["reasoning_trace"] == "r" and "decision_links" not in both["b"], "a refused fold changes nothing"
    store.release_deletion_lease("b", "w")
    dead = store.create_observation({"topic_id": "b", "item_url": "dead"})
    store.delete_observation(dead)
    with pytest.raises(StoreError):
        store.fold_observation(src, dead, " folded")
    assert len(store.get_observations_for_item("u1")) == 2
    store.fold_observation(src, dst, " folded")
    left = store.get_observations_for_item("u1")
    assert len(left) == 1 and left[0]["topic_id"] == "b" and left[0]["strength"] == 0.9
    assert left[0]["reasoning_trace"] == "r folded" and left[0]["removed_from"] == ["z", "a"]
    assert [l["topic_id"] for l in left[0]["decision_links"]] == ["b", "a", "x"], "the survivor's own intent, then the folded plan"
    store.fold_observation(src, dst, " again")
    assert store.get_observations_for_item("u1")[0]["reasoning_trace"] == "r folded", "a source already gone is nothing to fold"


def test_delete_since_leases_every_topic_it_touches_including_old_ones(store):
    """Cleanup keeps no list of busy topics to go stale: each topic it will
    write to, old topics with new evidence or snapshots included, is leased
    just before the writes and given back after them, and one another
    operation holds is left alone."""
    now = datetime.now(timezone.utc)
    then = now - timedelta(days=3)
    store.create_topic({"topic_id": "old", "recorded_at": then, "status": "active"})
    store.create_observation({"topic_id": "old", "item_url": "u1", "recorded_at": then})
    store.create_observation({"topic_id": "old", "item_url": "u2"})
    store.append_snapshot("old", {"date": then, "evidence_count": 1, "recorded_at": then})
    store.append_snapshot("old", {"date": now, "evidence_count": 2})
    store.create_topic({"topic_id": "snap-only", "recorded_at": then})
    store.append_snapshot("snap-only", {"date": now, "evidence_count": 0})
    store.create_topic({"topic_id": "busy", "recorded_at": then, "status": "active"})
    store.create_observation({"topic_id": "busy", "item_url": "u3"})
    store.append_snapshot("busy", {"date": now, "evidence_count": 1})
    store.acquire_deletion_lease("busy", "w", "reprocess", now + timedelta(minutes=15))
    result = store.delete_since(now - timedelta(days=1))
    assert result == {"observations_deleted": 1, "topics_deleted": 0, "topics_kept": 0, "topics_busy": 1, "snapshots_cleared": 2}
    old = store.get_topic("old")
    assert old["status"] == "active" and not old.get("deleting"), "leased for its cleanup, then given back"
    assert [o["item_url"] for o in store.get_observations_for_topic("old")] == ["u1"]
    assert [s["evidence_count"] for s in old["snapshots"]] == [1]
    assert store.get_topic("snap-only")["snapshots"] == [] and store.get_topic("snap-only")["status"] == "emerging"
    busy = store.get_topic("busy")
    assert busy["deleting"]["owner"] == "w" and store.count_observations_for_topic("busy") == 1 and len(busy["snapshots"]) == 1


def test_publishing_an_embedding_is_a_compare_and_swap_on_its_revision(store):
    """Every embedding write bumps the revision; a publication names the
    revision it read and writes nothing if it moved."""
    store.create_topic({"topic_id": "t", "title": "T", "embedding": [1.0, 0.0, 0.0, 0.0]})
    assert store.get_topic("t").get("embedding_revision", 0) == 0
    assert store.publish_topic_embedding("t", {"embedding": [0.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u1"]}, expected_revision=0)
    t = store.get_topic("t")
    assert t["embedding_revision"] == 1 and t["embedding"] == [0.0, 1.0, 0.0, 0.0] and t["embedding_inputs"] == ["u1"]
    assert not store.publish_topic_embedding("t", {"embedding": [0.0, 0.0, 1.0, 0.0], "embedding_inputs": []}, expected_revision=0), "stale"
    t = store.get_topic("t")
    assert t["embedding_revision"] == 1 and t["embedding"] == [0.0, 1.0, 0.0, 0.0] and t["embedding_inputs"] == ["u1"], "a refused publication changes nothing"
    store.update_topic("t", {"embedding": [0.0, 0.0, 0.0, 1.0]})
    assert store.get_topic("t")["embedding_revision"] == 2, "any embedding write is a publication"
    store.update_topic("t", {"title": "T2"})
    assert store.get_topic("t")["embedding_revision"] == 2, "a write that does not touch the embedding is not"
    assert not store.publish_topic_embedding("t", {"embedding": [1.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u2"]}, expected_revision=1)
    assert store.publish_topic_embedding("t", {"embedding": [1.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u2"]}, expected_revision=2)
    assert store.get_topic("t")["embedding_revision"] == 3
    assert not store.publish_topic_embedding("missing", {"embedding": [1.0, 0.0, 0.0, 0.0]}, expected_revision=0)


def test_registering_owed_work_is_atomic_and_only_publication_clears_it(store):
    """Registration merges the inputs owed and bumps the revision, so a
    publication that read the topic earlier is refused and leaves the
    obligation; one that read it after clears it in the same write."""
    store.create_topic({"topic_id": "t", "title": "T", "embedding": [1.0, 0.0, 0.0, 0.0]})
    store.defer_topic_refresh("t", ["u1"])
    store.defer_topic_refresh("t", ["u2", "u1"])
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is True and t["embedding_refresh_missing"] == ["u1", "u2"] and t["embedding_revision"] == 2
    assert not store.publish_topic_embedding("t", {"embedding": [0.0, 1.0, 0.0, 0.0], "embedding_inputs": []}, expected_revision=0)
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is True and t["embedding_refresh_missing"] == ["u1", "u2"], "a refused publication clears nothing"
    assert store.publish_topic_embedding("t", {"embedding": [0.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u1", "u2"]}, expected_revision=2)
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is False and t["embedding_refresh_missing"] == [] and t["embedding_revision"] == 3
    store.defer_topic_refresh("t", [])
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is True and t["embedding_refresh_missing"] == [] and t["embedding_revision"] == 4


def test_a_publication_carries_what_the_refresh_could_not_load(store):
    store.create_topic({"topic_id": "t", "title": "T", "embedding": [1.0, 0.0, 0.0, 0.0]})
    store.defer_topic_refresh("t", ["old"])
    assert store.publish_topic_embedding("t", {"embedding": [0.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u1"]},
                                         expected_revision=1, missing=["u3", "u2", "u3"])
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is True and t["embedding_refresh_missing"] == ["u2", "u3"], "the refresh's own view replaces the registration it read under"
    assert store.publish_topic_embedding("t", {"embedding": [0.0, 1.0, 0.0, 0.0], "embedding_inputs": ["u1", "u2", "u3"]}, expected_revision=2)
    t = store.get_topic("t")
    assert t["embedding_refresh_pending"] is False and t["embedding_refresh_missing"] == []
