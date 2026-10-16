"""add-topic: definition parsing and the move plan (pure parts)."""
import pytest

from topicsleuth.add_topic import load_topic_definition, plan_moves


def test_definition_derives_id_from_title(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("title: Agent-Facing Error Interfaces\nsummary: s\ntags: [a]\n")
    d = load_topic_definition(p)
    assert d.topic_id == "agent-facing-error-interfaces"
    assert d.tags == ["a"] and d.link == []


def test_definition_keeps_explicit_id_and_links(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("topic_id: My.Id\ntitle: T\nsummary: s\nlink:\n  - http://x/1\n")
    d = load_topic_definition(p)
    assert d.topic_id == "my-id"
    assert d.link == ["http://x/1"]


def test_definition_rejects_seed_file_and_missing_fields(tmp_path):
    p = tmp_path / "seed.yaml"
    p.write_text("topics:\n  - title: T\n    summary: s\n")
    with pytest.raises(ValueError):
        load_topic_definition(p)
    p.write_text("title: T\n")
    with pytest.raises(ValueError):
        load_topic_definition(p)


class _Obs:
    def __init__(self, rows):
        self.rows = rows

    def find(self, q):
        return [dict(r) for r in self.rows if r["item_url"] == q["item_url"]]


class _Storage:
    def __init__(self, rows):
        self.observations = _Obs(rows)

    def get_observations_for_item(self, url):
        return self.observations.find({"item_url": url})


def test_plan_moves_skips_already_on_target_and_reports_missing():
    st = _Storage([
        {"_id": 1, "item_url": "u1", "topic_id": "old"},
        {"_id": 2, "item_url": "u2", "topic_id": "new"},
    ])
    moves, missing = plan_moves(st, ["u1", "u2", "u3"], "new")
    assert [m["_id"] for m in moves] == [1]
    assert missing == ["u2", "u3"]



from datetime import datetime, timezone  # noqa: E402

from topicsleuth.add_topic import move_observation, move_observations  # noqa: E402


class _MoveStorage:
    """Minimal Storage for move_observations / move_observation."""

    def __init__(self, obs, topics):
        self._obs = obs
        self._topics = topics
        self.observations = self
        self.snapshots: dict[str, int] = {}

    # observations collection surface
    def find(self, q):
        return [dict(o) for o in self._obs if all(o.get(k) == v for k, v in q.items())]

    def update_one(self, q, u):
        for o in self._obs:
            if o["_id"] == q["_id"]:
                o.update(u["$set"])

    # Store surface
    def get_observations_for_item(self, url):
        return self.find({"item_url": url})

    def update_observation(self, obs_id, updates):
        self.update_one({"_id": obs_id}, {"$set": updates})

    def count_observations_for_topic(self, tid):
        return sum(1 for o in self._obs if o["topic_id"] == tid)

    def get_observations_for_topic(self, tid, *, fields=None):
        rows = [o for o in self._obs if o["topic_id"] == tid]
        if fields is None:
            return rows
        return [{k: v for k, v in o.items() if k in {*fields, "_id"}} for o in rows]

    def get_topic(self, tid):
        return self._topics.get(tid)

    def update_topic(self, tid, updates):
        self._topics[tid].update(updates)

    def append_snapshot(self, tid, snap):
        self.snapshots[tid] = self.snapshots.get(tid, 0) + 1


def _fixture():
    obs = [
        {"_id": 1, "item_url": "u1", "item_title": "Only one", "topic_id": "lonely",
         "strength": 0.85, "connection_type": "supports", "reasoning_trace": "r", "date": datetime.now(timezone.utc)},
        {"_id": 2, "item_url": "u2", "item_title": "Two a", "topic_id": "busy", "strength": 0.8, "date": datetime.now(timezone.utc)},
        {"_id": 3, "item_url": "u3", "item_title": "Two b", "topic_id": "busy", "strength": 0.8, "date": datetime.now(timezone.utc)},
    ]
    topics = {t: {"topic_id": t, "status": "emerging", "snapshots": []} for t in ("lonely", "busy", "target")}
    return _MoveStorage(obs, topics)


def test_move_retires_emptied_source_with_provenance():
    st = _fixture()
    report = move_observation("u1", "target", st)
    assert st._obs[0]["topic_id"] == "target"
    assert st._obs[0]["moved_from"] == "lonely"
    assert "MOVED by move-observation" in st._obs[0]["reasoning_trace"]
    assert report["sources_retired"] == ["lonely"]
    assert st._topics["lonely"]["status"] == "retired"
    assert st._topics["lonely"]["merged_into"] == "target"
    assert st.snapshots == {"target": 1, "lonely": 1}


def test_move_keeps_source_with_remaining_evidence():
    st = _fixture()
    report = move_observation("u2", "target", st)
    assert report["sources_retired"] == []
    assert st._topics["busy"]["status"] == "emerging"
    assert st.count_observations_for_topic("busy") == 1


def test_move_dry_run_reports_without_writing():
    st = _fixture()
    report = move_observation("u1", "target", st, dry_run=True)
    assert report["sources_that_would_empty"] == ["lonely"]
    assert st._obs[0]["topic_id"] == "lonely" and st.snapshots == {}


def test_move_rejects_missing_or_retired_target_and_absent_observation():
    st = _fixture()
    with pytest.raises(ValueError):
        move_observation("u1", "nope", st)
    st._topics["target"]["status"] = "retired"
    with pytest.raises(ValueError):
        move_observation("u1", "target", st)
    st._topics["target"]["status"] = "emerging"
    with pytest.raises(ValueError):
        move_observation("u9", "target", st)


def test_add_topic_path_does_not_retire_sources():
    st = _fixture()
    retired = move_observations(st, st.find({"item_url": "u1"}), "target",
                                datetime.now(timezone.utc), reason="add-topic",
                                retire_empty_sources=False)
    assert retired == [] and st._topics["lonely"]["status"] == "emerging"
