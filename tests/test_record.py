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


def _topic_with_evidence(st, topic_id, status, dates, title=None):
    """A topic in the given status with one observation per date."""
    st.create_topic({"topic_id": topic_id, "title": title or topic_id.upper(),
                     "status": status, "date_identified": min(dates)})
    for i, d in enumerate(dates):
        st.create_observation({
            "topic_id": topic_id, "item_url": f"{topic_id}-{i}", "item_title": f"{topic_id} {i}",
            "date": d, "strength": 0.8, "connection_type": "supports",
            "source_name": f"s{i}", "source_type": "paper",
        })


def _days_ago(n, now=None):
    from datetime import timedelta
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=n)


def _link_to(topic_id, url="x"):
    from topicsleuth.synthesize import SynthesisDecision
    item = {"_id": 9, "url": url, "title": "Item", "source_name": "arXiv", "source_type": "paper"}
    ev = _Eval(url=url, embedding=[0.0] * 4, above_threshold=True)
    d = SynthesisDecision(item_url=url, item_title="Item", decision="link_existing",
                          topic_id=topic_id, connection_type="supports", strength=0.8,
                          reasoning_trace="")
    return [item], [ev], [d]


class TestQuietTopicsFadeAndGoDormant:
    """Fading and dormant can only be reached by a topic that received
    nothing this run, so record checks the quiet topics too."""

    def test_active_topic_with_no_evidence_this_week_fades(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "t1", "active", [_days_ago(n) for n in (10, 12, 15, 20, 25)])
        result = record([], [], [], st, settings=Settings())
        assert st.topics["t1"]["status"] == "fading"
        assert [(t["from_status"], t["to_status"]) for t in result.transitions] == [("active", "fading")]
        assert result.topics_updated == ["t1"]
        snap = st.topics["t1"]["snapshots"][-1]
        assert snap["velocity"] == 0.0 and 9.9 < snap["days_since_evidence"] < 10.1

    def test_mature_topic_fades_too(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "t1", "mature", [_days_ago(8 + n) for n in range(20)])
        record([], [], [], st, settings=Settings())
        assert st.topics["t1"]["status"] == "fading"

    def test_active_topic_with_evidence_this_week_is_left_alone(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "t1", "active", [_days_ago(n) for n in (3, 12, 15, 20, 25)])
        result = record([], [], [], st, settings=Settings())
        assert st.topics["t1"]["status"] == "active"
        assert result.transitions == [] and st.topics["t1"]["snapshots"] == []

    def test_fading_topic_goes_dormant_after_thirty_days(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "soon", "fading", [_days_ago(20)])
        _topic_with_evidence(st, "gone", "fading", [_days_ago(31)])
        result = record([], [], [], st, settings=Settings())
        assert st.topics["soon"]["status"] == "fading"
        assert st.topics["gone"]["status"] == "dormant"
        assert [t["topic_id"] for t in result.transitions] == ["gone"]

    def test_fresh_evidence_revives_fading_and_dormant_topics(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "f", "fading", [_days_ago(20)])
        _topic_with_evidence(st, "d", "dormant", [_days_ago(60)])
        record(*_link_to("f", "x1"), st, settings=Settings())
        record(*_link_to("d", "x2"), st, settings=Settings())
        assert st.topics["f"]["status"] == "active"
        assert st.topics["d"]["status"] == "emerging"

    def test_emerging_and_retired_topics_never_fade(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "single", "emerging", [_days_ago(90)])
        _topic_with_evidence(st, "old", "retired", [_days_ago(90)])
        result = record([], [], [], st, settings=Settings())
        assert result.transitions == []
        assert st.topics["single"]["status"] == "emerging"
        assert st.topics["old"]["status"] == "retired"

    def test_touched_topic_is_not_checked_twice(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "t1", "active", [_days_ago(n) for n in (10, 12, 15, 20, 25)])
        result = record(*_link_to("t1"), st, settings=Settings())
        assert st.topics["t1"]["status"] == "active"
        assert result.topics_updated == ["t1"] and len(st.topics["t1"]["snapshots"]) == 1

    def test_history_load_judges_quiet_topics_as_of_the_batch(self):
        """Loading March items must not fade every topic against today's
        clock: the quiet check runs as of the latest date in the batch."""
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        from datetime import timedelta
        st = MemoryStore()
        march_1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        _topic_with_evidence(st, "quiet", "active", [march_1 - timedelta(days=n) for n in range(5)])
        st.create_topic({"topic_id": "other", "title": "Other"})

        items, evs, ds = _link_to("other", "a")
        items[0]["date_published"] = "2026-03-04T00:00:00+00:00"
        record(items, evs, ds, st, settings=Settings(), date_from_item=True)
        assert st.topics["quiet"]["status"] == "active"     # evidence 3 days before the batch

        items, evs, ds = _link_to("other", "b")
        items[0]["date_published"] = "2026-03-20T00:00:00+00:00"
        result = record(items, evs, ds, st, settings=Settings(), date_from_item=True)
        assert st.topics["quiet"]["status"] == "fading"
        assert st.topics["quiet"]["snapshots"][-1]["date"] == datetime(2026, 3, 20, tzinfo=timezone.utc)
        assert [t["topic_id"] for t in result.transitions] == ["quiet"]

    def test_empty_history_batch_does_not_check_quiet_topics(self):
        from topicsleuth.backends.memory import MemoryStore
        from topicsleuth.record import record
        st = MemoryStore()
        _topic_with_evidence(st, "t1", "active", [_days_ago(n) for n in (10, 12, 15, 20, 25)])
        result = record([], [], [], st, settings=Settings(), date_from_item=True)
        assert result.transitions == [] and st.topics["t1"]["status"] == "active"
