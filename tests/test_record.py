"""Hermetic tests for the record-stage helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from topicsleuth.record import _strength_floor, _topical_tags
from topicsleuth.settings import Settings


@dataclass
class _Eval:
    """The ItemEval protocol: what a feeder hands the store per item."""
    url: str
    embedding: list[float]
    above_threshold: bool


class TestStrengthFloor:
    def test_below_threshold_no_floor(self):
        assert _strength_floor(0) is None
        assert _strength_floor(1) is None
        assert _strength_floor(19) is None

    def test_at_threshold_returns_floor(self):
        assert _strength_floor(20) == 0.7

    def test_above_threshold_returns_floor(self):
        assert _strength_floor(50) == 0.7
        assert _strength_floor(100) == 0.7


class TestTopicalTags:
    def _settings(self, ignored):
        s = Settings()
        s.candidates.ignored_tags = list(ignored)
        return s

    def test_strips_routing_tags(self):
        tags = ["feed-a", "benchmarking", "feed-b", "code-review"]
        assert _topical_tags(tags, self._settings(["feed-a", "feed-b", "feed-c"])) == [
            "benchmarking", "code-review",
        ]

    def test_nothing_ignored_is_identity(self):
        assert _topical_tags(["feed-a", "x"], self._settings([])) == ["feed-a", "x"]

    def test_empty_tags(self):
        assert _topical_tags([], self._settings(["feed-a"])) == []



def _FakeStorage():
    """The reference in-memory Store, with one topic founded in August."""
    from topicsleuth.backends.memory import MemoryStore
    st = MemoryStore()
    st.create_topic({"topic_id": "t1", "title": "T1", "date_identified": datetime(2026, 8, 1)})
    return st


def _one_link():
    from topicsleuth.synthesize import SynthesisDecision
    item = {"_id": 1, "url": "u", "title": "Item", "source_name": "arXiv", "source_type": "paper"}
    ev = _Eval(url="u", embedding=[0.0] * 4, above_threshold=True)
    d = SynthesisDecision(item_url="u", item_title="Item", decision="link_existing",
                          topic_id="t1", connection_type="supports", strength=0.8,
                          reasoning_trace="")
    return [item], [ev], [d]


class TestRecordBackfillFlag:
    def test_default_writes_no_backfill_field(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_one_link(), st, settings=Settings())
        assert len(_obs(st)) == 1
        assert "backfill" not in _obs(st)[0]

    def test_backfill_stamps_observations(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_one_link(), st, settings=Settings(), backfill=True)
        assert _obs(st)[0]["backfill"] is True


MARCH = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _obs(st):
    return st.list_observations()


def _dated_link(**item_extra):
    items, evs, ds = _one_link()
    items[0].update({"date_published": "2026-03-01T12:00:00+00:00", **item_extra})
    return items, evs, ds


class TestRecordItemDates:
    def test_default_dates_by_run_time(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_dated_link(), st, settings=Settings())
        assert _obs(st)[0]["date"] > MARCH
        assert st.topics["t1"]["date_identified"] == datetime(2026, 8, 1)

    def test_item_dates_flow_to_observation_snapshot_and_topic(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_dated_link(), st, settings=Settings(), date_from_item=True)
        assert _obs(st)[0]["date"] == MARCH
        assert st.topics["t1"]["snapshots"][-1]["date"] == MARCH
        # a February-to-March item predates the August founding: pulled back
        assert st.topics["t1"]["date_identified"] == MARCH

    def test_item_date_falls_back_to_gathered_then_now(self):
        from topicsleuth.record import _item_date
        assert _item_date({"date_gathered": datetime(2026, 4, 2)}) == datetime(2026, 4, 2, tzinfo=timezone.utc)
        assert _item_date({"date_published": "not a date", "date_gathered": None}) is None
        assert _item_date(None) is None

    def test_proposed_topic_dated_by_founding_item(self):
        from topicsleuth.record import record
        from topicsleuth.synthesize import SynthesisDecision
        st = _FakeStorage()
        item = {"_id": 2, "url": "n", "title": "New", "date_published": MARCH}
        d = SynthesisDecision(item_url="n", item_title="New", decision="propose_new",
                              topic_id=None, connection_type=None, strength=None,
                              reasoning_trace="", new_topic={"topic_id": "t2", "title": "T2",
                                                             "summary": "s", "embedding": []})
        record([item], [_Eval("n", [0.0] * 4, True)], [d], st, settings=Settings(), date_from_item=True)
        assert st.topics["t2"]["date_identified"] == MARCH
        assert _obs(st)[0]["topic_id"] == "t2" and _obs(st)[0]["date"] == MARCH


class TestRecordPassThrough:
    def test_lane_and_evidence_copied_from_item(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_dated_link(tags=["lane-a", "feed-a"], evidence="ships", lanes={"lane-a": 0.9}),
               st, settings=Settings())
        obs = _obs(st)[0]
        assert obs["tags"] == ["lane-a", "feed-a"]
        assert obs["evidence"] == "ships"
        assert obs["lanes"] == {"lane-a": 0.9}

    def test_absent_fields_are_empty_not_missing(self):
        from topicsleuth.record import record
        st = _FakeStorage()
        record(*_one_link(), st, settings=Settings())
        obs = _obs(st)[0]
        assert obs["tags"] == [] and obs["evidence"] is None and obs["lanes"] is None
