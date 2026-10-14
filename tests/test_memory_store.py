"""MemoryStore is the reference Store: these pin the semantics other
implementations must match."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from topicsleuth.backends.memory import MemoryStore
from topicsleuth.interfaces import Store


def _store():
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "A", "embedding": [1.0, 0.0]})
    st.create_topic({"topic_id": "b", "title": "B", "embedding": [0.0, 1.0]})
    st.create_topic({"topic_id": "r", "title": "R", "embedding": [1.0, 0.1], "status": "retired"})
    return st


def test_conforms_to_protocol():
    assert isinstance(MemoryStore(), Store)


def test_nearest_is_cosine_ranked_with_scores():
    st = _store()
    out = st.nearest_topics([1.0, 0.0], limit=2)
    assert [t["topic_id"] for t in out] == ["a", "r"]
    assert out[0]["vector_score"] == 1.0 and 0.9 < out[1]["vector_score"] < 1.0


def test_list_topics_can_hide_retired():
    st = _store()
    assert {t["topic_id"] for t in st.list_topics()} == {"a", "b", "r"}
    assert {t["topic_id"] for t in st.list_topics(include_retired=False)} == {"a", "b"}


def test_observations_round_trip():
    st = _store()
    oid = st.create_observation({"topic_id": "a", "item_url": "u1"})
    st.create_observation({"topic_id": "a", "item_url": "u2"})
    assert st.count_observations_for_topic("a") == 2
    assert [o["item_url"] for o in st.get_observations_for_item("u1")] == ["u1"]
    st.update_observation(oid, {"strength": 0.9})
    assert st.get_observations_for_item("u1")[0]["strength"] == 0.9
    assert st.delete_observations_for_topic("a") == 2


def test_returned_documents_are_copies():
    st = _store()
    t = st.get_topic("a")
    t["title"] = "mutated"
    assert st.get_topic("a")["title"] == "A"


def test_find_weak_topics_respects_status_and_age():
    st = MemoryStore()
    old = datetime.now(timezone.utc) - timedelta(days=90)
    st.create_topic({"topic_id": "old-single", "date_identified": old})
    st.create_topic({"topic_id": "young-single"})
    st.create_topic({"topic_id": "retired", "status": "retired", "date_identified": old})
    assert {t["topic_id"] for t in st.find_weak_topics(max_obs=1, older_than_days=None)} == {"old-single", "young-single"}
    assert [t["topic_id"] for t in st.find_weak_topics(max_obs=1, older_than_days=30)] == ["old-single"]


def test_item_embeddings_and_delete_since():
    st = _store()
    st.save_item_embedding("u1", [0.5, 0.5])
    assert st.get_item_embeddings(["u1", "missing"]) == {"u1": [0.5, 0.5]}
    assert st.get_items(["u1"])[0]["url"] == "u1"
    then = datetime.now(timezone.utc) - timedelta(days=2)
    st.create_observation({"topic_id": "a", "item_url": "u1", "date": then})
    st.create_observation({"topic_id": "a", "item_url": "u2"})
    st.append_snapshot("a", {"date": then, "evidence_count": 1})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 2})
    st.create_topic({"topic_id": "new"})
    result = st.delete_since(datetime.now(timezone.utc) - timedelta(days=1))
    # every topic in _store() was founded just now, so all four go; the
    # two-day-old observation and nothing else survives the sweep
    assert result == {"observations_deleted": 1, "topics_deleted": 4, "snapshots_cleared": 0}
    assert [o["item_url"] for o in st.list_observations()] == ["u1"]
