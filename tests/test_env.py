"""How the CLI's environment reaches the store: the names it reads and
the defaults it leaves alone, with the MongoDB client never built."""
from __future__ import annotations

import pytest

from topicsleuth import env
from topicsleuth.settings import load_settings

VARIABLES = (
    "TOPICSLEUTH_DATABASE",
    "TOPICSLEUTH_VECTOR_INDEX",
    "TOPICSLEUTH_TOPICS_COLLECTION",
    "TOPICSLEUTH_OBSERVATIONS_COLLECTION",
    "TOPICSLEUTH_ITEMS_COLLECTION",
    "TOPICSLEUTH_STORE",
)


@pytest.fixture
def from_uri(monkeypatch):
    """Capture the arguments store_from_env hands to MongoStore.from_uri."""
    from topicsleuth.backends.mongo import MongoStore

    calls = []
    monkeypatch.setattr(MongoStore, "from_uri", classmethod(lambda cls, *a, **kw: calls.append((a, kw)) or "store"))
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    for v in VARIABLES:
        monkeypatch.delenv(v, raising=False)
    return calls


def test_unset_collection_names_leave_the_store_defaults_alone(from_uri):
    assert env.store_from_env(load_settings()) == "store"
    (args, kwargs), = from_uri
    assert args == ("mongodb://example.invalid", "topicsleuth")
    assert kwargs["collections"] is None
    assert kwargs["vector_index"] == "topic_embedding_index"


def test_collection_names_from_the_environment_reach_the_store(from_uri, monkeypatch):
    monkeypatch.setenv("TOPICSLEUTH_DATABASE", "news_gather")
    monkeypatch.setenv("TOPICSLEUTH_TOPICS_COLLECTION", "briefing_topics")
    monkeypatch.setenv("TOPICSLEUTH_OBSERVATIONS_COLLECTION", "briefing_observations")
    monkeypatch.setenv("TOPICSLEUTH_VECTOR_INDEX", "briefing_topic_embedding_index")
    env.store_from_env(load_settings())
    (args, kwargs), = from_uri
    assert args == ("mongodb://example.invalid", "news_gather")
    # Only the names that are set are passed; the items collection keeps its default.
    assert kwargs["collections"] == {"topics": "briefing_topics", "observations": "briefing_observations"}
    assert kwargs["vector_index"] == "briefing_topic_embedding_index"
