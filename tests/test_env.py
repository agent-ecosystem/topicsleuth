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


def test_judge_effort_comes_from_the_environment(monkeypatch):
    pytest.importorskip("anthropic")
    from topicsleuth.env import judge_from_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("TOPICSLEUTH_JUDGE", raising=False)
    monkeypatch.delenv("TOPICSLEUTH_JUDGE_EFFORT", raising=False)
    assert judge_from_env().effort is None, "unset means the model's default"
    monkeypatch.setenv("TOPICSLEUTH_JUDGE_MODEL", "claude-x")
    monkeypatch.setenv("TOPICSLEUTH_JUDGE_EFFORT", "medium")
    j = judge_from_env()
    assert (j.model, j.effort) == ("claude-x", "medium")


def test_the_store_settings_name_the_judge_and_the_environment_overrides_for_one_run(monkeypatch, tmp_path):
    pytest.importorskip("anthropic")
    from topicsleuth.env import judge_from_env
    from topicsleuth.settings import load_settings
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    for v in ("TOPICSLEUTH_JUDGE", "TOPICSLEUTH_JUDGE_MODEL", "TOPICSLEUTH_JUDGE_EFFORT"):
        monkeypatch.delenv(v, raising=False)
    f = tmp_path / "store.yaml"
    f.write_text("domain: a field\njudge:\n  model: claude-sonnet-5-5\n  effort: low\n")
    s = load_settings(f)
    assert (s.judge.model, s.judge.effort) == ("claude-sonnet-5-5", "low") and s.judge.first_pass_max_tokens == 8192
    j = judge_from_env(s)
    assert (j.model, j.effort) == ("claude-sonnet-5-5", "low"), "the CLI judges with the model the feeder judges with"
    monkeypatch.setenv("TOPICSLEUTH_JUDGE_MODEL", "claude-opus-4-8")
    assert judge_from_env(s).model == "claude-opus-4-8" and judge_from_env(s).effort == "low", "the environment overrides one field at a time"
    monkeypatch.delenv("TOPICSLEUTH_JUDGE_MODEL")
    (tmp_path / "none.yaml").write_text("domain: x\n")
    assert judge_from_env(load_settings(tmp_path / "none.yaml")).model == "claude-opus-4-8", "nothing named: the library default"
