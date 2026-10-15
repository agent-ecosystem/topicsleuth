"""The write path under retry, failure and history: recording is idempotent,
every failure is reported per item and writes nothing for it, the judge's
answer is validated before it is written, backfills read as history, and a
run can be removed by its id."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from topicsleuth.backends.fake import HashEmbedder, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore
from topicsleuth.interfaces import ItemFailure, StoreError
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import Link, SynthesisDecision, intended_links, synthesize_items

DIM = 8
E_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
E_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAY, JUNE, JULY = (datetime(2026, m, 1, tzinfo=timezone.utc) for m in (5, 6, 7))


class _Eval:
    def __init__(self, url, embedding, above=True):
        self.url, self.embedding, self.above_threshold = url, embedding, above


def _settings():
    return load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM},
                                    "candidates": {"vector_limit": 5}})


def _store():
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A, "tags": ["alpha"]})
    for i in range(3):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "item_title": f"A {i}", "strength": 0.9})
    return st


def _item(url="u1", **extra):
    return {"url": url, "title": f"Item {url}", "summary": "About A.", "tags": ["alpha"], **extra}


def _link(url="u1", topic_id="a", strength=0.8, *more):
    """A link decision to topic_id, and to any further topics in ``more``."""
    links = [Link(topic_id=t, connection_type="supports", strength=strength) for t in (topic_id, *more)]
    return SynthesisDecision(item_url=url, item_title=f"Item {url}", decision="link_existing", topic_id=topic_id,
                             connection_type="supports", strength=strength, reasoning_trace="r", links=links)


def _propose(url="u2", topic_id="topic-c", title="Topic C"):
    return SynthesisDecision(item_url=url, item_title=f"Item {url}", decision="propose_new", topic_id=None,
                             connection_type=None, strength=0.85, reasoning_trace="r",
                             new_topic={"topic_id": topic_id, "title": title, "summary": "Tracks C.", "embedding": E_B})


FIT = {"q1_scope_expansion": False, "q2_representative_citation": True, "q3_distinguishing_evidence": True}
LINK = {"candidate_classifications": [{"topic_id": "a", "label": "same_question"}], "decision": "link_existing",
        "links": [{"topic_id": "a", "connection_type": "supports", "fit_questions": FIT}], "reasoning_trace": "fits"}
PROPOSE = {"candidate_classifications": [{"topic_id": "a", "label": "adjacent_question"}], "decision": "propose_new",
           "topic_id": None, "connection_type": None, "reasoning_trace": "new",
           "new_topic": {"title": "Topic C", "summary": "Tracks C.", "topic_id": "topic-c", "tags": ["gamma"]}}
CONFIRM = {"verdict": "confirm_new", "closest_candidate": "a", "reasoning": "right-sized"}


# ── 1. idempotent recording ──

def test_recording_the_same_decision_twice_writes_one_observation():
    st, s = _store(), _settings()
    items, evals, decisions = [_item("u1"), _item("u2")], [_Eval("u1", E_A), _Eval("u2", E_B)], [_link("u1"), _propose("u2")]
    first = record(items, evals, decisions, st, settings=s)
    assert len(first.observations_created) == 2 and len(first.topics_created) == 1 and first.failures == []
    second = record(items, evals, decisions, st, settings=s)
    assert second.observations_created == [] and second.topics_created == [] and second.failures == []
    assert [o["item_url"] for o in second.already_recorded] == ["u1", "u2"]
    assert st.count_observations_for_topic("a") == 4 and st.count_observations_for_topic("topic-c") == 1
    assert st.get_topic("a")["snapshots"][-1]["evidence_count"] == 4
    assert len(st.get_topic("a")["snapshots"]) == 1, "a retry does not add a snapshot either"


def test_two_feeders_admitting_the_same_document_record_it_once():
    st, s = _store(), _settings()
    record([_item("u1")], [_Eval("u1", E_A)], [_link("u1")], st, settings=s)
    other = record([_item("u1")], [_Eval("u1", E_A)], [_link("u1", strength=0.3)], st, settings=s)
    assert other.already_recorded and st.get_observations_for_item("u1")[0]["strength"] == 0.8, "first writer wins"


def test_a_colliding_proposal_reuses_the_topic_only_when_it_is_the_same_proposal():
    st, s = _store(), _settings()
    same = record([_item("u2")], [_Eval("u2", E_B)], [_propose("u2")], st, settings=s)
    assert same.topics_created and st.get_topic("topic-c")["title"] == "Topic C"
    again = record([_item("u3")], [_Eval("u3", E_B)], [_propose("u3", title="Topic C")], st, settings=s)
    assert again.topics_created == [] and again.failures == [] and st.count_observations_for_topic("topic-c") == 2
    other = record([_item("u4")], [_Eval("u4", E_B)], [_propose("u4", title="Something Else")], st, settings=s)
    assert other.failures == [ItemFailure("u4", "create_topic", "topic_id 'topic-c' already names a different topic")]
    assert st.count_observations_for_topic("topic-c") == 2 and st.get_topic("topic-c")["title"] == "Topic C"


# ── 2. failures are per item, reported, and write nothing ──

class _Failing(MemoryStore):
    def __init__(self, **fail_on):
        super().__init__()
        self.fail_on = fail_on

    def _maybe(self, op, key):
        if key in self.fail_on.get(op, ()):
            raise RuntimeError(f"injected {op} failure for {key}")

    def create_topic(self, topic_doc):
        self._maybe("create_topic", topic_doc["topic_id"])
        return super().create_topic(topic_doc)

    def create_observation(self, obs_doc):
        self._maybe("create_observation", obs_doc["item_url"])
        return super().create_observation(obs_doc)

    def save_item_embedding(self, item_url, embedding):
        self._maybe("save_item_embedding", item_url)
        super().save_item_embedding(item_url, embedding)


def test_a_failed_synthesis_is_a_failure_in_record_and_writes_nothing():
    st, s = _store(), _settings()
    failed = SynthesisDecision(item_url="u1", item_title="Item u1", decision="failed", topic_id=None,
                               connection_type=None, strength=0.0, reasoning_trace="", error="JSONDecodeError: bad")
    out = record([_item("u1")], [_Eval("u1", E_A)], [failed], st, settings=s)
    assert out.failures == [ItemFailure("u1", "synthesize", "JSONDecodeError: bad")] and out.failed_urls() == {"u1"}
    assert st.count_observations_for_topic("a") == 3 and st.get_item_embeddings(["u1"]) == {}


def test_a_proposal_that_cannot_be_created_leaves_no_orphan_observation():
    st = _Failing(create_topic={"topic-c"})
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A})
    items, evals = [_item("u2"), _item("u3")], [_Eval("u2", E_B), _Eval("u3", E_B)]
    # u3 linked to the batch-mate's proposal, which then fails to create
    out = record(items, evals, [_propose("u2"), _link("u3", topic_id="topic-c")], st, settings=_settings())
    assert {(f.item_url, f.stage) for f in out.failures} == {("u2", "create_topic"), ("u3", "create_topic")}
    assert st.list_observations() == [] and st.get_topic("topic-c") is None
    assert st.get_item_embeddings(["u2", "u3"]) == {}, "a failed item's embedding is not saved either"


def test_a_destination_that_vanished_or_retired_is_a_failure_not_an_orphan():
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "r", "title": "R", "summary": "s", "embedding": E_B, "status": "retired"})
    out = record([_item("u1"), _item("u2")], [_Eval("u1", E_A), _Eval("u2", E_B)],
                 [_link("u1", topic_id="gone"), _link("u2", topic_id="r")], st, settings=s)
    assert [(f.item_url, f.stage, f.error) for f in out.failures] == [
        ("u1", "create_observation", "topic 'gone' does not exist"),
        ("u2", "create_observation", "topic 'r' is retired"),
    ]
    assert st.list_observations()[-1]["item_url"] == "a2"


def test_store_failures_at_each_write_are_reported_per_item():
    st = _Failing(create_observation={"u1"}, save_item_embedding={"u2"})
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A})
    out = record([_item("u1"), _item("u2"), _item("u3")], [_Eval(u, E_A) for u in ("u1", "u2", "u3")],
                 [_link("u1"), _link("u2"), _link("u3")], st, settings=_settings())
    assert {(f.item_url, f.stage) for f in out.failures} == {("u1", "create_observation"), ("u2", "save_embedding")}
    assert out.failed_urls() == {"u1", "u2"} and [o["item_url"] for o in out.observations_created] == ["u2", "u3"]
    assert "injected" in out.failures[0].error


def test_synthesize_reports_judge_and_embedding_failures_as_failed_decisions():
    st, s = _store(), _settings()
    judge = ScriptedJudge(["not json", PROPOSE, CONFIRM, LINK])

    class _Emb(HashEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("embedding service down")
    items = [_item("u1"), _item("u2"), _item("u3")]
    evals = [_Eval("u1", E_A), _Eval("u2", E_B), _Eval("u3", E_A)]
    out = synthesize_items(items, evals, st, s, judge, _Emb(DIM))
    assert [d.decision for d in out] == ["failed", "failed", "link_existing"]
    assert "JSONDecodeError" in out[0].error and out[1].error.startswith("embed_proposal: RuntimeError")
    assert all(d.decision != "propose_new" for d in out), "a proposal without an embedding is never returned"
    rec = record(items, evals, out, st, settings=s)
    assert rec.failed_urls() == {"u1", "u2"} and [o["item_url"] for o in rec.observations_created] == ["u3"]


# ── 4. the judge's answer is validated before it is written ──

@pytest.mark.parametrize("answer, reason", [
    ({**LINK, "decision": "merge"}, "unknown decision 'merge'"),
    ({**LINK, "links": [{"topic_id": "nonexistent", "connection_type": "supports", "fit_questions": FIT}]}, "not among the shown candidates"),
    ({**LINK, "links": [{"topic_id": "a", "connection_type": "garbage", "fit_questions": FIT}]}, "unknown connection_type 'garbage'"),
    ({**LINK, "links": [{"topic_id": "a", "connection_type": "supports", "strength": 9}]}, "without the three fit questions"),
    ({**LINK, "links": []}, "link_existing with no links"),
    ({**LINK, "topic_id": "a", "connection_type": "supports", "fit_questions": FIT, "links": None}, "link_existing with no links"),
    ({**PROPOSE, "new_topic": {"topic_id": "x", "summary": "s"}}, "proposed topic has no title"),
    ({**PROPOSE, "new_topic": None}, "propose_new without a new_topic"),
])
def test_invalid_semantics_fail_the_item_instead_of_being_stored(answer, reason):
    st, s = _store(), _settings()
    out = synthesize_items([_item("u1")], [_Eval("u1", E_A)], st, s, ScriptedJudge([answer, CONFIRM]), HashEmbedder(DIM))
    assert out[0].decision == "failed" and reason in out[0].error, out[0].error
    rec = record([_item("u1")], [_Eval("u1", E_A)], out, st, settings=s)
    assert rec.failed_urls() == {"u1"} and st.count_observations_for_topic("a") == 3


def test_a_fabricated_classification_cannot_force_a_link_to_an_invented_topic():
    st, s = _store(), _settings()
    answer = {**PROPOSE, "candidate_classifications": [{"topic_id": "invented", "label": "same_question"},
                                                        {"topic_id": "a", "label": "adjacent_question"}]}
    out = synthesize_items([_item("u2")], [_Eval("u2", E_B)], st, s, ScriptedJudge([answer, CONFIRM]), HashEmbedder(DIM))
    d = out[0]
    assert d.decision == "propose_new" and "IGNORED: classifications of unshown topics ['invented']" in d.reasoning_trace
    assert [c["topic_id"] for c in d.diagnostics["classified"]] == ["a"]


def test_a_proposed_slug_that_is_taken_gets_a_distinct_one():
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "topic-c", "title": "Older C", "summary": "s", "embedding": E_A})
    out = synthesize_items([_item("u2")], [_Eval("u2", E_B)], st, s, ScriptedJudge([PROPOSE, CONFIRM]), HashEmbedder(DIM))
    assert out[0].new_topic["topic_id"] == "topic-c-2" and "created as 'topic-c-2'" in out[0].reasoning_trace
    rec = record([_item("u2")], [_Eval("u2", E_B)], out, st, settings=s)
    assert rec.failures == [] and st.get_topic("topic-c-2")["title"] == "Topic C"


# ── 5. backfill into a store that already holds newer evidence ──

def test_a_historical_snapshot_sees_only_evidence_up_to_its_date():
    st, s = _store(), _settings()
    st.create_observation({"topic_id": "a", "item_url": "later", "date": JUNE + timedelta(days=1)})
    out = record([_item("u1", date_published=JUNE)], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, date_from_item=True)
    snap = st.get_topic("a")["snapshots"][-1]
    assert out.failures == [] and snap["date"] == JUNE
    # the three seeded observations are dated today and "later" is June 2, all after the snapshot date
    assert snap["evidence_count"] == 1 and snap["days_since_evidence"] == 0.0
    assert st.count_observations_for_topic("a") == 5


def test_history_is_inserted_in_order_and_does_not_move_the_current_status():
    st, s = MemoryStore(), _settings()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_A, "status": "active",
                     "date_identified": MAY, "snapshots": [{"date": JULY, "evidence_count": 6, "velocity": 0.3}]})
    for i in range(6):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "date": JULY - timedelta(days=i)})
    out = record([_item("u1", date_published=JUNE)], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, date_from_item=True)
    topic = st.get_topic("a")
    assert out.transitions == [] and topic["status"] == "active", "a June batch does not re-judge a topic whose history runs to July"
    assert [snap["date"] for snap in topic["snapshots"]] == [JUNE, JULY], "the older snapshot is inserted before the newer one"
    june = topic["snapshots"][0]
    assert june["evidence_count"] == 1 and june["days_since_evidence"] == 0.0 and june["velocity"] == 1 / 7


def test_the_quiet_topic_pass_ignores_topics_that_did_not_exist_yet():
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "late", "title": "Late", "summary": "s", "embedding": E_B, "status": "active",
                     "date_identified": JULY})
    st.create_observation({"topic_id": "late", "item_url": "l0", "date": JULY})
    out = record([_item("u1", date_published=MAY)], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, date_from_item=True)
    assert out.transitions == [] and st.get_topic("late")["status"] == "active"


# ── 6. a run can be removed by its id ──

def test_a_run_id_is_stamped_on_everything_written_and_clean_removes_just_that_run():
    st, s = _store(), _settings()
    record([_item("u1"), _item("u2")], [_Eval("u1", E_A), _Eval("u2", E_B)], [_link("u1"), _propose("u2")], st, settings=s, run_id="run-1")
    record([_item("u3")], [_Eval("u3", E_A)], [_link("u3")], st, settings=s, run_id="run-2")
    assert st.get_topic("topic-c")["run_id"] == "run-1" and st.get_observations_for_item("u3")[0]["run_id"] == "run-2"
    assert all("recorded_at" in o for o in st.list_observations())
    assert [snap.get("run_id") for snap in st.get_topic("a")["snapshots"]] == ["run-1", "run-2"]
    result = st.delete_since(datetime.now(timezone.utc) - timedelta(hours=1), run_id="run-1")
    assert result == {"observations_deleted": 2, "topics_deleted": 1, "topics_kept": 0, "topics_busy": 0, "snapshots_cleared": 1}
    assert sorted(o["item_url"] for o in st.list_observations()) == ["a0", "a1", "a2", "u3"]
    assert st.get_topic("topic-c") is None and [snap["run_id"] for snap in st.get_topic("a")["snapshots"]] == ["run-2"]


def test_clean_keeps_a_topic_that_still_holds_older_evidence():
    from topicsleuth.add_topic import move_observations
    st, s = _store(), _settings()
    old = datetime.now(timezone.utc) - timedelta(days=10)
    for o in st.list_observations():
        st.update_observation(o["_id"], {"recorded_at": old})
    st.update_topic("a", {"recorded_at": old})
    record([_item("u2")], [_Eval("u2", E_B)], [_propose("u2")], st, settings=s, run_id="run-1")
    move_observations(st, st.get_observations_for_item("a0"), "topic-c", datetime.now(timezone.utc), reason="test")
    result = st.delete_since(datetime.now(timezone.utc) - timedelta(hours=1))
    assert result["topics_deleted"] == 0 and result["topics_kept"] == 1
    assert st.get_topic("topic-c") is not None and st.get_observations_for_item("a0")[0]["topic_id"] == "topic-c"


# ── settings and replay provenance ──

def test_settings_reject_unknown_keys_at_every_level():
    with pytest.raises(ValueError, match="unknown top-level settings: \\['store'\\]"):
        load_settings(overrides={"store": {"domain": "x"}})
    with pytest.raises(ValueError, match="unknown settings under 'candidates': \\['vector_limt'\\]"):
        load_settings(overrides={"candidates": {"vector_limt": 3}})


def test_replay_report_records_provenance_and_failures(tmp_path):
    import json
    from topicsleuth.tune import replay_synthesis
    st, s = _store(), _settings()
    st.items["u1"] = {"url": "u1", "title": "Item u1", "summary": "s", "embedding": E_A}
    st.items["u2"] = {"url": "u2", "title": "Item u2", "summary": "s", "embedding": E_A}
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"decisions": [{"item_url": "u1"}, {"item_url": "u2"}]}))
    report = replay_synthesis(st, ScriptedJudge([LINK, "garbage"]), HashEmbedder(DIM), settings=s,
                              replay_from=prior, output_path=tmp_path / "out.json", progress=False, label="t")
    assert report["sample_size"] == 2 and len(report["decisions"]) == 1
    assert report["failures"][0]["item_url"] == "u2" and "DecisionError" not in report["failures"][0]["error"]
    assert report["config"]["domain"] == "widget manufacturing" and report["config"]["rubric"]["topic_size"]
    assert "library_version" in report["config"] and report["config"]["decision_model"] is None


def test_a_history_batch_does_not_rejudge_a_quiet_topic_whose_history_runs_later():
    """The quiet-topic pass judges topics as of the batch date. A topic
    whose latest snapshot is already later than that date was judged by
    newer evidence; a May batch must not fade it on May's evidence."""
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "q", "title": "Quiet", "summary": "s", "embedding": E_B, "status": "active",
                     "date_identified": MAY - timedelta(days=60),
                     "snapshots": [{"date": JULY, "evidence_count": 5, "velocity": 0.5}]})
    for i in range(5):
        st.create_observation({"topic_id": "q", "item_url": f"q{i}", "date": JULY - timedelta(days=i)})
    out = record([_item("u1", date_published=MAY)], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, date_from_item=True)
    assert out.transitions == [] and st.get_topic("q")["status"] == "active"
    assert [snap["date"] for snap in st.get_topic("q")["snapshots"]] == [JULY], "no snapshot written either"


def test_a_quiet_topic_that_fades_during_a_run_carries_the_run_id():
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "q", "title": "Quiet", "summary": "s", "embedding": E_B, "status": "active",
                     "date_identified": datetime.now(timezone.utc) - timedelta(days=60)})
    for i in range(5):
        st.create_observation({"topic_id": "q", "item_url": f"q{i}", "date": datetime.now(timezone.utc) - timedelta(days=30 + i)})
    out = record([_item("u1")], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, run_id="run-9")
    assert [(t["topic_id"], t["to_status"]) for t in out.transitions] == [("q", "fading")]
    snap = st.get_topic("q")["snapshots"][-1]
    assert snap["run_id"] == "run-9" and "recorded_at" in snap
    assert st.delete_since(datetime.now(timezone.utc) - timedelta(hours=1), run_id="run-9")["snapshots_cleared"] == 2
    assert st.get_topic("q")["snapshots"] == [], "and the run's snapshot can be removed with the run"


# ── whole-pipeline retries, partial writes, maintenance interleaving ──

def _pipeline(st, items, judge, s=None, run_id=None, embedder=None):
    """synthesize then record, as a feeder does."""
    s = s or _settings()
    evals = [_Eval(i["url"], E_B if "C" in i.get("summary", "") else E_A) for i in items]
    decisions = synthesize_items(items, evals, st, s, judge, embedder or HashEmbedder(DIM))
    return decisions, record(items, evals, decisions, st, settings=s, run_id=run_id)


def test_rerunning_the_pipeline_on_a_recorded_item_does_not_rejudge_or_reopen_it():
    st = _store()
    item = _item("u2", summary="About C.")
    judge = ScriptedJudge([PROPOSE, CONFIRM, PROPOSE, CONFIRM])
    first_decisions, first = _pipeline(st, [item], judge)
    assert first_decisions[0].new_topic["topic_id"] == "topic-c" and len(first.topics_created) == 1
    second_decisions, second = _pipeline(st, [item], judge)
    assert len(judge.calls) == 2, "the judge was not called again"
    assert second_decisions[0].decision == "link_existing" and second_decisions[0].topic_id == "topic-c"
    assert "ALREADY RECORDED" in second_decisions[0].reasoning_trace
    assert second.topics_created == [] and second.observations_created == [] and second.failures == []
    assert len(st.get_observations_for_item("u2")) == 1
    assert [t["topic_id"] for t in st.list_topics()] == ["a", "topic-c"], "no topic-c-2"


def test_two_feeders_deciding_differently_about_one_document_leave_one_observation_and_no_empty_topic():
    """Feeder A links u2 to topic a; feeder B proposes a topic for it, and A
    records between B's pre-check and B's insert. B's proposal is a claim
    that nothing fits, now contested: B withdraws it, drops the empty topic,
    and fails the item so its next run resumes A's decision."""
    st, s = _store(), _settings()
    item, ev = _item("u2", summary="About C."), [_Eval("u2", E_B)]

    class _RacingStore(MemoryStore):
        def create_observation(self, obs_doc):
            if obs_doc["item_url"] == "u2" and not self.get_observations_for_item("u2"):
                super().create_observation({"topic_id": "a", "item_url": "u2", "item_title": "Item u2",
                                            "strength": 0.8, "connection_type": "supports"})
            return super().create_observation(obs_doc)
    racing = _RacingStore()
    racing.topics, racing.observations, racing._ids = st.topics, st.observations, st._ids
    out = record([item], ev, [_propose("u2")], racing, settings=s)
    assert out.topics_created == [] and out.observations_created == []
    assert [(f.stage, f.error) for f in out.failures] == [("create_observation", "another writer recorded this item on ['a'] during the write; proposal withdrawn")]
    assert racing.get_topic("topic-c") is None, "the empty topic opened for u2 was removed"
    assert [o["topic_id"] for o in racing.get_observations_for_item("u2")] == ["a"]
    again = record([item], ev, [_propose("u2")], racing, settings=s)
    assert again.failures == [] and again.already_recorded[0]["topic_id"] == "a", "the retry resumes A's decision"


class _FailOnce(MemoryStore):
    """Fails the first call of one operation, then behaves."""
    def __init__(self, op):
        super().__init__()
        self.op, self.tripped = op, False

    def _trip(self, op):
        if op == self.op and not self.tripped:
            self.tripped = True
            raise RuntimeError(f"injected {op} failure")

    def append_snapshot(self, topic_id, snapshot):
        self._trip("append_snapshot")
        super().append_snapshot(topic_id, snapshot)

    def update_topic(self, topic_id, updates):
        if "status" in updates:
            self._trip("status")
        super().update_topic(topic_id, updates)

    def save_item_embedding(self, item_url, embedding):
        self._trip("save_item_embedding")
        super().save_item_embedding(item_url, embedding)


@pytest.mark.parametrize("op, stage", [("append_snapshot", "snapshot"), ("status", "status"), ("save_item_embedding", "save_embedding")])
def test_a_retry_after_a_late_failure_finishes_the_work_without_duplicating_it(op, stage):
    st = _FailOnce(op)
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_A})
    for i in range(4):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "date": datetime.now(timezone.utc)})
    items = [_item("u1")]
    judge = ScriptedJudge([LINK])
    _, first = _pipeline(st, items, judge)
    assert first.failed_urls() == {"u1"} and first.failures[0].stage == stage
    assert len(st.get_observations_for_item("u1")) == 1, "the observation itself was written"
    _, second = _pipeline(st, items, judge)
    assert second.failures == [] and second.observations_created == [] and len(second.already_recorded) == 1
    assert len(judge.calls) == 1
    topic = st.get_topic("a")
    assert len(topic["snapshots"]) == 1 and topic["snapshots"][0]["evidence_count"] == 5
    assert topic["status"] == "active" and st.get_item_embeddings(["u1"]) == {"u1": E_A}
    _, third = _pipeline(st, items, judge)
    assert len(st.get_topic("a")["snapshots"]) == 1, "a further retry adds nothing"


def test_a_destination_deleted_or_retired_during_the_write_leaves_no_orphan():
    class _Maintenance(MemoryStore):
        def create_observation(self, obs_doc):
            oid = super().create_observation(obs_doc)
            if obs_doc["topic_id"] == "a":
                self.delete_topic("a")
            if obs_doc["topic_id"] == "s":
                self.update_topic("s", {"status": "retired", "split_into": ["s1", "s2"]})
            if obs_doc["topic_id"] == "m":
                self.update_topic("m", {"status": "retired", "merged_into": "t"})
            return oid
    st, s = _Maintenance(), _settings()
    for tid in ("a", "s", "m", "t"):
        st.create_topic({"topic_id": tid, "title": tid, "summary": "s", "embedding": E_A})
    out = record([_item("u1"), _item("u2"), _item("u3")], [_Eval(u, E_A) for u in ("u1", "u2", "u3")],
                 [_link("u1", "a"), _link("u2", "s"), _link("u3", "m")], st, settings=s)
    assert [(f.item_url, f.error) for f in out.failures] == [
        ("u1", "topic 'a' was deleted during the write"), ("u2", "topic 's' was retired during the write")]
    assert st.get_observations_for_item("u1") == [] and st.get_observations_for_item("u2") == []
    moved = st.get_observations_for_item("u3")[0]
    assert moved["topic_id"] == "t" and moved["moved_from"] == "m" and "merged into 't' during the write" in moved["reasoning_trace"]
    assert [o["topic_id"] for o in out.observations_created] == ["t"] and st.get_topic("t")["snapshots"][-1]["evidence_count"] == 1


def test_merge_and_split_carry_late_arrivals_with_them():
    from topicsleuth.merge import merge_topics
    from topicsleuth.split import split_topic
    st, s, emb = _store(), _settings(), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_A})
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "s", "embedding": E_B})
    st.create_observation({"topic_id": "b", "item_url": "b0"})

    class _LateJudge(ScriptedJudge):
        def complete(self, system, user, max_tokens=2048):
            st.create_observation({"topic_id": "b", "item_url": "late-b"})  # lands while the judge thinks
            return super().complete(system, user, max_tokens)
    merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
    out = merge_topics("b", "a", st, s, _LateJudge([merged]), emb)
    assert out["observations_reassigned"] == 2 and st.count_observations_for_topic("b") == 0
    assert st.get_observations_for_item("late-b")[0]["topic_id"] == "a"

    subs = {"sub_topics": [{"topic_id": "a-one", "title": "A one", "summary": "s1"}, {"topic_id": "a-two", "title": "A two", "summary": "s2"}]}
    plan_urls = [o["item_url"] for o in st.get_observations_for_topic("a")]
    reassign = [{"item_url": u, "new_topic_id": "a-one" if i % 2 else "a-two"} for i, u in enumerate(plan_urls)]

    class _LateSplitJudge(ScriptedJudge):
        def complete(self, system, user, max_tokens=2048):
            if "Observations to Reassign" in user:
                st.create_observation({"topic_id": "a", "item_url": "late-a"})
                st.save_item_embedding("late-a", E_A)
            return super().complete(system, user, max_tokens)
    out = split_topic("a", st, s, _LateSplitJudge([subs, reassign]), HashEmbedder(DIM, vectors={"A one\n\ns1": E_A, "A two\n\ns2": E_B}))
    late = st.get_observations_for_item("late-a")[0]
    assert late["topic_id"] == "a-one" and "arrived during the split; nearest sub-topic" in late["reasoning_trace"]
    assert st.count_observations_for_topic("a") == 0 and out["observations_reassigned"] == len(plan_urls) + 1


def test_a_backfill_recomputes_the_later_snapshots_it_lands_before():
    from topicsleuth.synthesize import _get_evidence_count
    st, s = MemoryStore(), _settings()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_A, "status": "active",
                     "date_identified": MAY, "snapshots": [{"date": JULY, "evidence_count": 6, "velocity": 6 / 7, "recorded_at": JULY, "run_id": "old"}]})
    for i in range(6):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "date": JULY - timedelta(days=i)})
    record([_item("u1", date_published=JUNE)], [_Eval("u1", E_A)], [_link("u1")], st, settings=s, date_from_item=True, run_id="new")
    snaps = st.get_topic("a")["snapshots"]
    assert [(x["date"], x["evidence_count"]) for x in snaps] == [(JUNE, 1), (JULY, 7)]
    assert snaps[1]["run_id"] == "old" and snaps[1]["recorded_at"] == JULY, "the later snapshot keeps its provenance"
    assert _get_evidence_count(st.get_topic("a")) == 7, "what the judge is shown includes the backfill"


def test_split_rejects_an_unknown_connection_type_before_writing():
    from topicsleuth.split import SplitPlanError, split_topic
    st, emb = _store(), HashEmbedder(DIM)
    subs = {"sub_topics": [{"topic_id": "a-one", "title": "A one", "summary": "s1"}, {"topic_id": "a-two", "title": "A two", "summary": "s2"}]}
    plan = [{"item_url": "a0", "new_topic_id": "a-one", "new_connection_type": "garbage"},
            {"item_url": "a1", "new_topic_id": "a-two"}, {"item_url": "a2", "new_topic_id": "a-two"}]
    with pytest.raises(SplitPlanError, match="'a0' has unknown connection type 'garbage'"):
        split_topic("a", st, _settings(), ScriptedJudge([subs, plan]), emb)
    assert st.get_topic("a-one") is None and st.count_observations_for_topic("a") == 3 and emb.calls == []


# ── a document is evidence for every question it bears on ──

def test_a_link_decision_with_two_topics_records_two_observations_each_scored_on_its_own():
    st = _store()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    for i in range(20):
        st.create_observation({"topic_id": "b", "item_url": f"b{i}", "strength": 0.9})
    yes, no = True, False
    two = {**LINK, "candidate_classifications": [{"topic_id": "a", "label": "same_question"}, {"topic_id": "b", "label": "same_question"}],
           "links": [{"topic_id": "a", "connection_type": "extends", "fit_questions": {"q1_scope_expansion": yes, "q2_representative_citation": yes, "q3_distinguishing_evidence": yes}},
                     {"topic_id": "b", "connection_type": "supports", "fit_questions": {"q1_scope_expansion": no, "q2_representative_citation": no, "q3_distinguishing_evidence": yes}}]}
    decisions, out = _pipeline(_store_with(st), [_item("u1", tags=["alpha", "beta"])], ScriptedJudge([two]))
    d = decisions[0]
    assert [(l.topic_id, l.strength, l.connection_type) for l in d.links] == [("a", 0.95, "extends"), ("b", 0.5, "supports")]
    assert d.topic_id == "a" and d.strength == 0.95, "the scalar fields mirror the first link"
    assert out.failures == [] and [(o["topic_id"], o["strength"], o["minor"]) for o in out.observations_created] == [("a", 0.95, False), ("b", 0.5, True)]
    assert out.observations_created[1]["fit_questions"]["q3_distinguishing_evidence"] is True
    assert sorted(out.topics_updated) == ["a", "b"] and st.get_topic("b")["snapshots"][-1]["evidence_count"] == 21
    assert "Fit questions for a:" in d.reasoning_trace and "Fit questions for b:" in d.reasoning_trace


def _store_with(st):
    """The store's topics need an embedding the hash embedder's candidate
    search will find; _store() already provides one."""
    return st


def test_the_gate_forces_links_to_every_same_question_candidate():
    st = _store()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    answer = {**PROPOSE, "candidate_classifications": [{"topic_id": "a", "label": "same_question"}, {"topic_id": "b", "label": "same_question"}]}
    decisions, out = _pipeline(st, [_item("u2", summary="About C.")], ScriptedJudge([answer]))
    d = decisions[0]
    assert d.decision == "link_existing" and sorted(l.topic_id for l in d.links) == ["a", "b"]
    assert d.links[0].topic_id == "a", "most evidence first"
    assert "Forced links to ['a', 'b']" in d.reasoning_trace and out.failures == []
    assert sorted(o["topic_id"] for o in out.observations_created) == ["a", "b"] and st.get_topic("topic-c") is None


def test_a_retry_of_a_multi_link_item_adds_nothing():
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    judge = ScriptedJudge([LINK])
    items = [_item("u1")]
    evals = [_Eval("u1", E_A)]
    first = record(items, evals, [_link("u1", "a", 0.8, "b")], st, settings=s)
    assert [o["topic_id"] for o in first.observations_created] == ["a", "b"]
    decisions = synthesize_items(items, evals, st, s, judge, HashEmbedder(DIM))
    assert judge.calls == [] and sorted(l.topic_id for l in decisions[0].links) == ["a", "b"]
    second = record(items, evals, decisions, st, settings=s)
    assert second.observations_created == [] and sorted(o["topic_id"] for o in second.already_recorded) == ["a", "b"]
    assert len(st.get_observations_for_item("u1")) == 2


def test_merging_two_topics_that_share_a_document_folds_it_into_one_observation():
    from topicsleuth.merge import merge_topics
    st, s = _store(), _settings()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    st.create_observation({"topic_id": "b", "item_url": "a0", "item_title": "A 0", "strength": 0.5, "connection_type": "extends"})
    st.create_observation({"topic_id": "b", "item_url": "only-b", "item_title": "Only B", "strength": 0.9})
    merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
    out = merge_topics("b", "a", st, s, ScriptedJudge([merged]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_A}))
    assert out["observations_reassigned"] == 2 and st.count_observations_for_topic("b") == 0
    kept = st.get_observations_for_item("a0")
    assert len(kept) == 1 and kept[0]["topic_id"] == "a" and kept[0]["strength"] == 0.9, "the target's observation survives"
    assert "FOLDED by merge" in kept[0]["reasoning_trace"] and "strength 0.5, extends" in kept[0]["reasoning_trace"]
    assert st.get_observations_for_item("only-b")[0]["topic_id"] == "a" and st.count_observations_for_topic("a") == 4


# ── the two floors are settings ──

def test_the_secondary_link_floor_drops_weak_later_links_but_never_the_first():
    st = _store()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    yes, no = True, False
    weak_first = {**LINK, "candidate_classifications": [{"topic_id": t, "label": "same_question"} for t in ("a", "b", "c")],
                  "links": [{"topic_id": "a", "connection_type": "supports", "fit_questions": {"q1_scope_expansion": no, "q2_representative_citation": no, "q3_distinguishing_evidence": no}},
                            {"topic_id": "b", "connection_type": "supports", "fit_questions": {"q1_scope_expansion": no, "q2_representative_citation": no, "q3_distinguishing_evidence": yes}},
                            {"topic_id": "c", "connection_type": "supports", "fit_questions": {"q1_scope_expansion": yes, "q2_representative_citation": yes, "q3_distinguishing_evidence": no}}]}
    s = load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM},
                                 "candidates": {"vector_limit": 5}, "synthesize": {"secondary_link_floor": 0.6}})
    decisions, out = _pipeline(st, [_item("u1")], ScriptedJudge([weak_first]), s=s)
    d = decisions[0]
    assert [(l.topic_id, l.strength) for l in d.links] == [("a", 0.3), ("c", 0.8)], "the first link stands at 0.3; b dropped at 0.5; c kept"
    assert d.diagnostics["dropped_links"] == [{"topic_id": "b", "strength": 0.5}]
    assert "SECONDARY LINKS DROPPED (below floor 0.6): b at 0.5" in d.reasoning_trace
    assert sorted(o["topic_id"] for o in out.observations_created) == ["a", "c"]
    off = load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM}, "candidates": {"vector_limit": 5}})
    decisions, _ = _pipeline(st, [_item("u9")], ScriptedJudge([weak_first]), s=off)
    assert len(decisions[0].links) == 3 and decisions[0].diagnostics["dropped_links"] == [], "the default floor keeps every link"


def test_the_minor_floor_comes_from_settings():
    st = _store()
    tight = load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM},
                                     "record": {"minor_floor_after": 3, "minor_floor_strength": 0.9}})
    out = record([_item("u1")], [_Eval("u1", E_A)], [_link("u1", "a", 0.8)], st, settings=tight)
    assert out.observations_created[0]["minor"] is True, "three observations already; 0.8 is under the 0.9 floor"
    out = record([_item("u2")], [_Eval("u2", E_A)], [_link("u2", "a", 0.8)], st, settings=_settings())
    assert out.observations_created[0]["minor"] is False, "the default floor applies from twenty"


def test_replay_summary_reports_secondary_links_and_drops(tmp_path):
    import json
    from topicsleuth.tune import replay_synthesis
    st = _store()
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    st.items["u1"] = {"url": "u1", "title": "Item u1", "summary": "s", "embedding": E_A, "tags": ["alpha"]}
    yes, no = True, False
    two = {**LINK, "candidate_classifications": [{"topic_id": "a", "label": "same_question"}, {"topic_id": "b", "label": "same_question"}],
           "links": [{"topic_id": "a", "connection_type": "supports", "fit_questions": FIT},
                     {"topic_id": "b", "connection_type": "supports", "fit_questions": {"q1_scope_expansion": no, "q2_representative_citation": no, "q3_distinguishing_evidence": yes}}]}
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"decisions": [{"item_url": "u1"}]}))
    s = load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM},
                                 "candidates": {"vector_limit": 5}, "synthesize": {"secondary_link_floor": 0.6}})
    report = replay_synthesis(st, ScriptedJudge([two]), HashEmbedder(DIM), settings=s, replay_from=prior,
                              output_path=tmp_path / "out.json", progress=False, label="t")
    assert report["summary"]["links"] == {"total": 1, "per_link_decision": 1.0, "multi_link_decisions": 0,
                                           "secondary": 0, "secondary_strength_buckets": {}, "dropped_by_floor": 1}
    assert report["config"]["synthesize"]["secondary_link_floor"] == 0.6


# ── the intended link set is durable, cleanup cannot orphan, derived state retries ──

class _FailLinkOnce(MemoryStore):
    """The first insert on ``topic`` fails; everything else behaves."""
    def __init__(self, topic):
        super().__init__()
        self.topic, self.tripped = topic, False

    def create_observation(self, obs_doc):
        if obs_doc["topic_id"] == self.topic and not self.tripped:
            self.tripped = True
            raise RuntimeError("injected insert failure")
        return super().create_observation(obs_doc)


def _two_topic_store(cls=MemoryStore, **kw):
    st = cls(**kw)
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A})
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B})
    return st


def test_a_direct_retry_finishes_the_intended_link_set_without_duplicates():
    st, s = _two_topic_store(_FailLinkOnce, topic="b"), _settings()
    items, evals, decisions = [_item("u1")], [_Eval("u1", E_A)], [_link("u1", "a", 0.8, "b")]
    first = record(items, evals, decisions, st, settings=s)
    assert [o["topic_id"] for o in first.observations_created] == ["a"] and first.failed_urls() == {"u1"}
    assert st.get_observations_for_item("u1")[0]["decision_links"] == [
        {"topic_id": "a", "connection_type": "supports", "strength": 0.8, "fit_questions": None},
        {"topic_id": "b", "connection_type": "supports", "strength": 0.8, "fit_questions": None}], "the whole intent travels with the observation"
    second = record(items, evals, decisions, st, settings=s)
    assert second.failures == [] and [o["topic_id"] for o in second.observations_created] == ["b"]
    assert [o["topic_id"] for o in second.already_recorded] == ["a"]
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u1")) == ["a", "b"]
    assert st.get_topic("b")["snapshots"][-1]["evidence_count"] == 1
    third = record(items, evals, decisions, st, settings=s)
    assert third.observations_created == [] and third.failures == [] and len(st.get_observations_for_item("u1")) == 2


def test_a_full_pipeline_retry_finishes_the_intended_link_set_without_rejudging():
    st = _two_topic_store(_FailLinkOnce, topic="b")
    yes, no = True, False
    two = {**LINK, "candidate_classifications": [{"topic_id": "a", "label": "same_question"}, {"topic_id": "b", "label": "same_question"}],
           "links": [{"topic_id": "a", "connection_type": "supports", "fit_questions": FIT},
                     {"topic_id": "b", "connection_type": "extends", "fit_questions": {"q1_scope_expansion": yes, "q2_representative_citation": yes, "q3_distinguishing_evidence": no}}]}
    judge = ScriptedJudge([two])
    _, first = _pipeline(st, [_item("u1")], judge)
    assert first.failed_urls() == {"u1"} and [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["a"]
    decisions, second = _pipeline(st, [_item("u1")], judge)
    assert len(judge.calls) == 1, "the judge was not called again"
    assert [(l.topic_id, l.strength, l.connection_type) for l in decisions[0].links] == [("a", 0.8, "supports"), ("b", 0.8, "extends")], "reconstructed from the stored intent, not from what was written"
    assert second.failures == [] and [o["topic_id"] for o in second.observations_created] == ["b"]
    written = {o["topic_id"]: o for o in st.get_observations_for_item("u1")}
    assert written["b"]["strength"] == 0.8 and written["b"]["connection_type"] == "extends" and written["b"]["fit_questions"]["q3_distinguishing_evidence"] is False
    _, third = _pipeline(st, [_item("u1")], judge)
    assert third.observations_created == [] and third.failures == [] and len(judge.calls) == 1


def test_an_incomplete_link_set_is_not_acknowledged_until_it_is_complete():
    st = _two_topic_store(_FailLinkOnce, topic="b")
    st.tripped = False
    class _AlwaysFailB(_FailLinkOnce):
        def create_observation(self, obs_doc):
            if obs_doc["topic_id"] == "b":
                raise RuntimeError("b is down")
            return MemoryStore.create_observation(self, obs_doc)
    down = _two_topic_store(_AlwaysFailB, topic="b")
    items, evals, decisions = [_item("u1")], [_Eval("u1", E_A)], [_link("u1", "a", 0.8, "b")]
    for _ in range(3):
        out = record(items, evals, decisions, down, settings=_settings())
        assert out.failed_urls() == {"u1"} and out.failures[-1].stage == "create_observation"
    assert [o["topic_id"] for o in down.get_observations_for_item("u1")] == ["a"], "a is written once, b never, and the item stays queued"


def test_cleanup_cannot_orphan_an_observation_that_lands_while_it_decides():
    """Cleanup marks the topic before counting. An insert that checks its
    destination after the mark is refused; one that landed before the mark
    is counted and the topic is kept. Either way nothing is orphaned."""
    class _Racing(MemoryStore):
        """During delete_since, a feeder records onto the topic right
        after cleanup's count comes back empty."""
        def __init__(self):
            super().__init__()
            self.in_cleanup = False
            self.feeder_result = None

        def count_observations_for_topic(self, topic_id):
            n = super().count_observations_for_topic(topic_id)
            if self.in_cleanup and topic_id == "arriving" and self.feeder_result is None:
                self.feeder_result = record([_item("late")], [_Eval("late", E_A)], [_link("late", "arriving")], self, settings=_settings())
            return n

        def delete_since(self, cutoff, run_id=None):
            self.in_cleanup = True
            try:
                return super().delete_since(cutoff, run_id)
            finally:
                self.in_cleanup = False
    st = _Racing()
    st.create_topic({"topic_id": "arriving", "title": "Arriving", "summary": "s", "embedding": E_A})
    result = st.delete_since(datetime.now(timezone.utc) - timedelta(hours=1))
    feeder = st.feeder_result
    orphans = [o for o in st.list_observations() if st.get_topic(o["topic_id"]) is None]
    assert orphans == [], "no observation points at a deleted topic"
    assert result["topics_deleted"] == 1 and feeder.failed_urls() == {"late"}
    assert "marked for deletion" in feeder.failures[0].error or "is deleting" in feeder.failures[0].error


def test_reprocess_cannot_orphan_an_observation_that_lands_while_it_runs():
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess
    class _Racing(MemoryStore):
        def __init__(self):
            super().__init__()
            self.fired = False

        def delete_observations_for_topic(self, topic_id, *, deleting_owner=None):
            if not self.fired:
                self.fired = True
                self.feeder_result = record([_item("late")], [_Eval("late", E_A)], [_link("late", topic_id)], self, settings=_settings())
            return super().delete_observations_for_topic(topic_id, deleting_owner=deleting_owner)
    st = _Racing()
    st.create_topic({"topic_id": "weak", "title": "Weak", "summary": "s", "embedding": E_A,
                     "date_identified": datetime.now(timezone.utc) - timedelta(days=90)})
    st.create_observation({"topic_id": "weak", "item_url": "w0"})
    out = execute_reprocess(st, plan_reprocess(st, max_obs=2))
    assert out["topics_deleted"] == 1 and st.list_observations() == []
    assert st.feeder_result.failed_urls() == {"late"}


def test_a_failed_later_snapshot_rebuild_is_retried_until_the_series_is_right():
    class _RebuildFailsOnce(MemoryStore):
        tripped = False
        def update_topic(self, topic_id, updates):
            if "snapshots" in updates and not self.tripped:
                self.tripped = True
                raise RuntimeError("injected rebuild failure")
            super().update_topic(topic_id, updates)
    st, s = _RebuildFailsOnce(), _settings()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_A, "status": "active",
                     "date_identified": MAY, "snapshots": [{"date": JULY, "evidence_count": 6, "velocity": 6 / 7, "recorded_at": JULY}]})
    for i in range(6):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "date": JULY - timedelta(days=i)})
    items, evals, decisions = [_item("u1", date_published=JUNE)], [_Eval("u1", E_A)], [_link("u1")]
    first = record(items, evals, decisions, st, settings=s, date_from_item=True)
    assert first.failures[0].stage == "snapshot_rebuild" and [x["evidence_count"] for x in st.get_topic("a")["snapshots"]] == [1, 6]
    second = record(items, evals, decisions, st, settings=s, date_from_item=True)
    assert second.failures == [] and second.observations_created == []
    assert [(x["date"], x["evidence_count"]) for x in st.get_topic("a")["snapshots"]] == [(JUNE, 1), (JULY, 7)]


def test_a_failed_embedding_refresh_is_retried_on_the_next_run_that_touches_the_topic():
    class _EmbedderFailsOnce(HashEmbedder):
        tripped = False
        def embed_documents(self, texts):
            if not self.tripped:
                self.tripped = True
                raise RuntimeError("embedding service down")
            return super().embed_documents(texts)
    st, s = MemoryStore(), _settings()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_B})
    for i in range(4):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}"})
        st.save_item_embedding(f"a{i}", E_A)
    emb = _EmbedderFailsOnce(DIM)
    items, evals, decisions = [_item("u1")], [_Eval("u1", E_A)], [_link("u1")]
    first = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert first.failures[0].stage == "embedding_refresh" and st.get_topic("a")["embedding_refresh_pending"] is True
    assert st.get_topic("a")["embedding"] == E_B, "the fifth observation crossed the threshold; the refresh did not land"
    second = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert second.failures == [] and second.observations_created == [] and len(st.get_observations_for_item("u1")) == 1
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is False and topic["embedding"] != E_B, "recomputed from the store, no observation added"


# ── curation revises what a retry may finish; deletion marks survive a crash; refreshes wait for their inputs ──

MERGED = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}


def test_a_retry_after_a_merge_follows_the_merge_and_never_writes_to_the_retired_topic():
    from topicsleuth.merge import merge_topics
    st = _two_topic_store()
    judge = ScriptedJudge([LINK])
    _, first = _pipeline(st, [_item("u1")], judge)
    assert first.failures == [] and st.get_observations_for_item("u1")[0]["topic_id"] == "a"
    merge_topics("a", "b", st, _settings(), ScriptedJudge([MERGED]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_B}))
    decisions, retry = _pipeline(st, [_item("u1")], judge)
    assert len(judge.calls) == 1 and [l.topic_id for l in decisions[0].links] == ["b"]
    assert retry.failures == [] and retry.observations_created == []
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["b"]


def test_a_retry_after_a_move_keeps_the_move_and_does_not_recreate_the_old_link():
    from topicsleuth.add_topic import move_observation
    st = _two_topic_store()
    st.create_observation({"topic_id": "a", "item_url": "keeps-a-live"})
    judge = ScriptedJudge([LINK])
    _pipeline(st, [_item("u1")], judge)
    move_observation("u1", "b", st)
    assert st.get_topic("a")["status"] != "retired"
    decisions, retry = _pipeline(st, [_item("u1")], judge)
    assert len(judge.calls) == 1 and [l.topic_id for l in decisions[0].links] == ["b"]
    assert retry.failures == [] and retry.observations_created == []
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["b"], "a was taken off on purpose"


def test_curating_one_link_of_a_multi_topic_item_does_not_revive_it_from_a_sibling_copy_of_the_plan():
    from topicsleuth.add_topic import move_observation
    st = _two_topic_store()
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    for i in range(2):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}"})
    items, evals = [_item("u1")], [_Eval("u1", E_A)]
    record(items, evals, [_link("u1", "a", 0.8, "b")], st, settings=_settings())
    plans = [o["decision_links"] for o in st.get_observations_for_item("u1")]
    assert all([l["topic_id"] for l in p] == ["a", "b"] for p in plans), "both observations carry the full plan"
    move_observation("u1", "c", st)  # moves every observation of u1 that is not already on c
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u1")) == ["c"]
    retry = record(items, evals, [_link("u1", "a", 0.8, "b")], st, settings=_settings())
    assert retry.failures == [] and retry.observations_created == []
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u1")) == ["c"], "neither a nor b is revived from the stale copies"
    decisions = synthesize_items(items, evals, st, _settings(), ScriptedJudge([LINK]), HashEmbedder(DIM))
    assert [l.topic_id for l in decisions[0].links] == ["c"]


def test_a_retry_after_a_split_does_not_restore_the_retired_parent():
    from topicsleuth.split import split_topic
    st = _store()
    judge = ScriptedJudge([LINK])
    _pipeline(st, [_item("u1")], judge)
    urls = [o["item_url"] for o in st.get_observations_for_topic("a")]
    subs = {"sub_topics": [{"topic_id": "a-one", "title": "A one", "summary": "s1"}, {"topic_id": "a-two", "title": "A two", "summary": "s2"}]}
    plan = [{"item_url": u, "new_topic_id": "a-one" if u == "u1" else "a-two"} for u in urls]
    split_topic("a", st, _settings(), ScriptedJudge([subs, plan]), HashEmbedder(DIM, vectors={"A one\n\ns1": E_A, "A two\n\ns2": E_B}))
    decisions, retry = _pipeline(st, [_item("u1")], judge)
    assert len(judge.calls) == 1 and [l.topic_id for l in decisions[0].links] == ["a-one"]
    assert retry.failures == [] and retry.observations_created == []
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["a-one"]


def test_an_unfinished_link_to_a_topic_merged_meanwhile_is_finished_on_the_merge_target():
    from topicsleuth.merge import merge_topics
    st = _two_topic_store(_FailLinkOnce, topic="b")
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    items, evals, decisions = [_item("u1")], [_Eval("u1", E_A)], [_link("u1", "a", 0.8, "b")]
    first = record(items, evals, decisions, st, settings=_settings())
    assert first.failed_urls() == {"u1"} and [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["a"]
    merge_topics("b", "c", st, _settings(), ScriptedJudge([MERGED]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_B}))
    retry = record(items, evals, decisions, st, settings=_settings())
    assert retry.failures == [] and [o["topic_id"] for o in retry.observations_created] == ["c"], "the approved link, where its topic now lives"
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u1")) == ["a", "c"]


class _CountFailsOnce(MemoryStore):
    tripped = False

    def count_observations_for_topic(self, topic_id):
        if topic_id == "kept" and not self.tripped:
            self.tripped = True
            raise RuntimeError("injected count failure")
        return super().count_observations_for_topic(topic_id)


def test_an_interrupted_cleanup_restores_the_original_status_on_retry():
    st = _CountFailsOnce()
    st.create_topic({"topic_id": "kept", "title": "Kept", "summary": "s", "embedding": E_A, "status": "active", "run_id": "cleanup-run"})
    st.create_observation({"topic_id": "kept", "item_url": "x", "run_id": "other-run"})
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    with pytest.raises(RuntimeError):
        st.delete_since(cutoff, run_id="cleanup-run")
    marked = st.get_topic("kept")
    assert marked["status"] == "deleting" and marked["deleting"]["prior_status"] == "active" and marked["deleting"]["by"] == "clean"
    immediate = st.delete_since(cutoff, run_id="cleanup-run")
    assert immediate["topics_busy"] == 1 and st.get_topic("kept")["status"] == "deleting", "the lease is still live; a second cleanup leaves it alone"
    st.topics["kept"]["deleting"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)  # the lease lapses
    result = st.delete_since(cutoff, run_id="cleanup-run")
    assert result["topics_kept"] == 1 and result["topics_deleted"] == 0 and result["topics_busy"] == 0
    topic = st.get_topic("kept")
    assert topic["status"] == "active" and "deleting" not in topic, "the mark is never taken for the status to restore"
    assert [t["topic_id"] for t in st.list_topics(include_retired=False)] == ["kept"], "visible again"
    out = record([_item("u1")], [_Eval("u1", E_A)], [_link("u1", "kept")], st, settings=_settings())
    assert out.failures == [] and len(out.observations_created) == 1, "and writable"


def test_a_fresh_caller_recovers_deletions_a_crashed_run_left_marked():
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    st = MemoryStore()
    st.create_topic({"topic_id": "holds", "title": "H", "summary": "s", "embedding": E_A, "status": "mature"})
    st.create_observation({"topic_id": "holds", "item_url": "h0"})
    st.create_topic({"topic_id": "empty", "title": "E", "summary": "s", "embedding": E_B})
    assert mark_deleting(st, "holds", by="reprocess", owner="run-1")
    assert mark_deleting(st, "empty", by="clean", owner="run-2")
    assert not mark_deleting(st, "holds", by="clean", owner="run-3"), "a live lease is not taken over"
    assert st.get_topic("holds")["deleting"]["by"] == "reprocess" and st.get_topic("holds")["deleting"]["owner"] == "run-1"
    assert st.list_topics(include_retired=False) == [], "marked topics are hidden from the live listing"
    assert recover_interrupted_deletions(st) == {"restored": [], "deleted": [], "busy": ["holds", "empty"]}, "while the leases are live, nothing is touched"
    for tid in ("holds", "empty"):  # the leases lapse
        st.topics[tid]["deleting"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
    out = recover_interrupted_deletions(st)  # knows nothing about the runs that set the marks
    assert out == {"restored": ["holds"], "deleted": ["empty"], "busy": []}
    assert st.get_topic("holds")["status"] == "mature" and not st.get_topic("holds").get("deleting")
    assert st.get_topic("empty") is None
    assert recover_interrupted_deletions(st) == {"restored": [], "deleted": [], "busy": []}
    assert not st.delete_topic("holds", deleting_owner="run-1"), "the stalled owner, waking, cannot finish what was undone"


def test_the_cleanup_race_still_leaves_no_orphan_with_the_durable_mark():
    test_cleanup_cannot_orphan_an_observation_that_lands_while_it_decides()


class _SaveFailsOnce(MemoryStore):
    tripped = False

    def save_item_embedding(self, item_url, embedding):
        if item_url == "u5" and not self.tripped:
            self.tripped = True
            raise RuntimeError("injected save failure")
        super().save_item_embedding(item_url, embedding)


def _four_plus_seed(cls):
    st = cls()
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "s", "embedding": E_A})
    for i in range(4):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}"})
        st.save_item_embedding(f"a{i}", E_A)
    return st


def test_a_refresh_waits_for_the_item_embedding_this_run_failed_to_save():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(_SaveFailsOnce), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    items, evals, decisions = [_item("u5")], [_Eval("u5", E_B)], [_link("u5")]
    first = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert [f.stage for f in first.failures] == ["save_embedding"]
    assert st.get_topic("a")["embedding_refresh_pending"] is True and st.get_topic("a")["embedding"] == E_A, "no refresh from four of five inputs"
    retry = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert retry.failures == [] and retry.observations_created == [] and len(st.get_observations_for_item("u5")) == 1
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is False and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    after_retry = list(topic["embedding"])
    assert after_retry != E_A and after_retry[1] > 0, "the fifth input pulled the embedding toward E_B"
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == after_retry, "equal to a recomputation from all five persisted inputs"


def test_a_resumed_pending_refresh_uses_the_input_the_same_call_recovers():
    st, s = _four_plus_seed(MemoryStore), _settings()
    st.create_observation({"topic_id": "a", "item_url": "u5", "recorded_at": datetime.now(timezone.utc)})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 5, "velocity": 5 / 7, "days_since_evidence": 0.0})
    st.update_topic("a", {"embedding_refresh_pending": True})
    assert st.get_item_embeddings(["u5"]) == {}, "u5's embedding never landed"
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    out = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5")], st, settings=s, embedder=emb)
    assert out.failures == [] and out.observations_created == [] and len(out.already_recorded) == 1
    topic = st.get_topic("a")
    assert st.get_item_embeddings(["u5"]) == {"u5": E_B} and topic["embedding_refresh_pending"] is False
    assert topic["embedding"][1] > 0, "the refresh ran after the embedding was saved and saw it"


# ── cumulative curation, owned deletion, blocked is not removed, refresh inputs across batches ──

def _three_live_topics():
    st = MemoryStore()
    for tid, vec in (("a", E_A), ("b", E_B), ("c", E_B)):
        st.create_topic({"topic_id": tid, "title": f"Topic {tid.upper()}", "summary": f"Tracks {tid.upper()}.", "embedding": vec})
        st.create_observation({"topic_id": tid, "item_url": f"{tid}-keeps-it-live"})
    return st


def test_a_second_move_does_not_revive_the_first_source():
    from topicsleuth.add_topic import move_observation
    st = _three_live_topics()
    judge = ScriptedJudge([LINK])
    _pipeline(st, [_item("u1")], judge)
    move_observation("u1", "b", st)
    move_observation("u1", "c", st)
    obs = st.get_observations_for_item("u1")[0]
    assert obs["moved_from"] == "b" and obs["removed_from"] == ["a", "b"], "the history is cumulative"
    decisions, retry = _pipeline(st, [_item("u1")], judge)
    assert len(judge.calls) == 1 and [l.topic_id for l in decisions[0].links] == ["c"]
    assert retry.failures == [] and retry.observations_created == []
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["c"]


def test_successive_curation_ending_in_a_fold_keeps_every_removal():
    from topicsleuth.add_topic import move_observation
    st = _three_live_topics()
    items, evals = [_item("u1")], [_Eval("u1", E_A)]
    record(items, evals, [_link("u1", "a", 0.8, "c")], st, settings=_settings())  # u1 on a and c, plan [a, c]
    move_observation("u1", "c", st)   # moves every observation of u1 not on c: a folds into the existing c one
    survivors = st.get_observations_for_item("u1")
    assert [o["topic_id"] for o in survivors] == ["c"] and survivors[0]["removed_from"] == ["a"]
    move_observation("u1", "b", st)   # c -> b: the fold's history must come along
    moved = st.get_observations_for_item("u1")
    assert [o["topic_id"] for o in moved] == ["b"] and moved[0]["removed_from"] == ["a", "c"]
    retry = record(items, evals, [_link("u1", "a", 0.8, "c")], st, settings=_settings())
    assert retry.failures == [] and retry.observations_created == []
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["b"], "neither a nor c is revived"


def test_reprocess_records_the_removal_on_the_items_other_observations():
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess
    st = _two_topic_store()
    st.update_topic("b", {"date_identified": datetime.now(timezone.utc) - timedelta(days=90)})
    items, evals = [_item("u1")], [_Eval("u1", E_A)]
    record(items, evals, [_link("u1", "a", 0.8, "b")], st, settings=_settings())
    out = execute_reprocess(st, plan_reprocess(st, max_obs=1, older_than_days=30))
    assert out["topics_deleted"] == 1 and out["item_urls"] == ["u1"]
    st.create_topic({"topic_id": "b", "title": "B again", "summary": "s", "embedding": E_B})  # the id comes back
    retry = record(items, evals, [_link("u1", "a", 0.8, "b")], st, settings=_settings())
    assert retry.observations_created == [] and st.get_observations_for_item("u1")[0]["removed_from"] == ["b"]


def test_recovery_cannot_take_over_a_live_deletion_and_strand_new_evidence():
    """The interleaving: a second caller runs recovery while reprocess
    holds the lease, then a feeder writes, then reprocess deletes the topic.
    With the lease, recovery leaves the topic alone, the feeder's write is
    refused, and nothing is orphaned."""
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess, recover_interrupted_deletions

    class _Racing(MemoryStore):
        fired = False

        def delete_observations_for_topic(self, topic_id, *, deleting_owner=None):
            if not self.fired:
                self.fired = True
                self.recovery = recover_interrupted_deletions(self)
                n = super().delete_observations_for_topic(topic_id, deleting_owner=deleting_owner)
                self.feeder = record([_item("late")], [_Eval("late", E_A)], [_link("late", topic_id)], self, settings=_settings())
                return n
            return super().delete_observations_for_topic(topic_id, deleting_owner=deleting_owner)
    st = _Racing()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A,
                     "date_identified": datetime.now(timezone.utc) - timedelta(days=90)})
    st.create_observation({"topic_id": "a", "item_url": "original"})
    out = execute_reprocess(st, plan_reprocess(st, max_obs=2))
    assert st.recovery == {"restored": [], "deleted": [], "busy": ["a"]}, "a live lease is not an abandoned one"
    assert st.feeder.failed_urls() == {"late"} and "deleting" in st.feeder.failures[0].error
    assert out["topics_deleted"] == 1 and st.list_observations() == []


def test_a_stalled_owner_cannot_continue_after_its_lease_was_taken_over():
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "status": "active"})
    st.create_observation({"topic_id": "a", "item_url": "x"})
    assert mark_deleting(st, "a", by="reprocess", owner="stalled", lease_seconds=1, now=datetime.now(timezone.utc) - timedelta(seconds=5))
    assert recover_interrupted_deletions(st)["restored"] == ["a"]
    assert st.delete_observations_for_topic("a", deleting_owner="stalled") == 0
    assert not st.delete_topic("a", deleting_owner="stalled")
    assert st.get_topic("a")["status"] == "active" and len(st.get_observations_for_topic("a")) == 1


def test_automatic_cli_recovery_leaves_a_live_lease_alone(capsys, monkeypatch, tmp_path):
    from topicsleuth import cli
    from topicsleuth.backends import fake
    from topicsleuth.reprocess import mark_deleting
    import sys
    monkeypatch.chdir(tmp_path)
    for k, v in (("TOPICSLEUTH_STORE", "memory"), ("TOPICSLEUTH_JUDGE", "scripted"), ("TOPICSLEUTH_EMBEDDER", "hash")):
        monkeypatch.setenv(k, v)
    fake.reset_shared()
    st = fake.shared("store")
    st.create_topic({"topic_id": "held", "title": "H", "summary": "s", "embedding": [1.0] + [0.0] * 1023, "status": "active"})
    st.create_observation({"topic_id": "held", "item_url": "h0"})
    assert mark_deleting(st, "held", by="reprocess", owner="other-process")
    sys.argv = ["topicsleuth", "clean", "--hours", "1"]
    try:
        cli.main()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "holds ['held']; left alone" in out and st.get_topic("held")["status"] == "deleting"
    fake.reset_shared()


def test_a_link_blocked_by_a_deletion_mark_keeps_the_item_failed_until_the_mark_clears():
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    st = _two_topic_store(_FailLinkOnce, topic="b")
    MemoryStore.create_observation(st, {"topic_id": "b", "item_url": "b-keeps-it-live"})
    items, evals, decisions = [_item("u1")], [_Eval("u1", E_A)], [_link("u1", "a", 0.8, "b")]
    first = record(items, evals, decisions, st, settings=_settings())
    assert first.failed_urls() == {"u1"}
    assert mark_deleting(st, "b", by="clean", owner="cleanup", lease_seconds=1, now=datetime.now(timezone.utc) - timedelta(seconds=5))
    blocked = record(items, evals, decisions, st, settings=_settings())
    assert blocked.failed_urls() == {"u1"} and blocked.failures[0].error == "topic 'b' is deleting", "blocked, not complete"
    assert [o["topic_id"] for o in st.get_observations_for_item("u1")] == ["a"]
    decisions_now = synthesize_items(items, evals, st, _settings(), ScriptedJudge([LINK]), HashEmbedder(DIM))
    assert [l.topic_id for l in decisions_now[0].links] == ["a", "b"], "the blocked link is still intended"
    assert recover_interrupted_deletions(st)["restored"] == ["b"]
    done = record(items, evals, decisions, st, settings=_settings())
    assert done.failures == [] and [o["topic_id"] for o in done.observations_created] == ["b"]
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u1")) == ["a", "b"]


def test_a_batch_in_between_cannot_clear_a_refresh_still_waiting_on_another_batchs_input():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(_SaveFailsOnce), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    u5 = ([_item("u5")], [_Eval("u5", E_B)], [_link("u5")])
    first = record(*u5, st, settings=s, embedder=emb)
    assert [f.stage for f in first.failures] == ["save_embedding"] and st.get_topic("a")["embedding_refresh_missing"] == ["u5"]
    between = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6")], st, settings=s, embedder=emb)
    topic = st.get_topic("a")
    assert between.failures == [] and topic["embedding_refresh_pending"] is True and topic["embedding"] == E_A, "u6's batch waited for u5's input"
    retry = record(*u5, st, settings=s, embedder=emb)
    topic = st.get_topic("a")
    assert retry.failures == [] and topic["embedding_refresh_pending"] is False and topic["embedding_refresh_missing"] == []
    final = list(topic["embedding"])
    assert final[1] > 0
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == final, "equal to a recomputation from all six persisted inputs"
    assert len(st.list_observations()) == 6


# ── the lease is atomic and every protected mutation honours it ──

def test_two_callers_racing_for_one_lease_get_one_winner():
    """Acquisition is one store operation: there is no read-then-write for
    a second caller to slip between. The second caller, racing, loses."""
    from topicsleuth.reprocess import mark_deleting
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "status": "active"})
    results = {}
    original = st.acquire_deletion_lease

    def racing_acquire(topic_id, owner, by, expires_at):
        # Worker two arrives while worker one is inside its own acquisition;
        # the store orders them, and worker one's answer reflects the order.
        if owner == "worker-one" and "worker-two" not in results:
            results["worker-two"] = original(topic_id, "worker-two", by, expires_at)
        return original(topic_id, owner, by, expires_at)
    st.acquire_deletion_lease = racing_acquire
    results["worker-one"] = mark_deleting(st, "a", by="reprocess", owner="worker-one")
    assert results["worker-two"] is not None and results["worker-one"] is False
    assert st.get_topic("a")["deleting"]["owner"] == "worker-two", "the live lease belongs to exactly one caller"
    assert st.get_topic("a")["deleting"]["prior_status"] == "active"


def test_a_stale_owner_cannot_delete_evidence_after_a_takeover_restored_the_topic():
    """The guarded delete and the lease are one store operation: once the
    takeover has committed, the old owner's delete deletes nothing, however
    long ago it last checked."""
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "status": "active"})
    st.create_observation({"topic_id": "a", "item_url": "old"})
    assert mark_deleting(st, "a", by="reprocess", owner="stalled", lease_seconds=1, now=datetime.now(timezone.utc) - timedelta(seconds=5))
    assert recover_interrupted_deletions(st)["restored"] == ["a"]
    feeder = record([_item("new")], [_Eval("new", E_A)], [_link("new", "a")], st, settings=_settings())
    assert feeder.failures == []
    assert st.delete_observations_for_topic("a", deleting_owner="stalled") == 0
    assert not st.delete_topic("a", deleting_owner="stalled")
    assert not st.release_deletion_lease("a", "stalled")
    assert sorted(o["item_url"] for o in st.get_observations_for_topic("a")) == ["new", "old"]
    assert st.get_topic("a")["status"] == "active"


def test_cleanup_leaves_a_busy_topics_observations_and_snapshots_untouched():
    from topicsleuth.reprocess import mark_deleting
    st = MemoryStore()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "run_id": "r"})
    st.create_observation({"topic_id": "a", "item_url": "x", "run_id": "r"})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 1, "run_id": "r"})
    st.create_topic({"topic_id": "free", "title": "F", "summary": "s", "embedding": E_B, "run_id": "r"})
    st.create_observation({"topic_id": "free", "item_url": "y", "run_id": "r"})
    assert mark_deleting(st, "a", by="reprocess", owner="active-worker")
    result = st.delete_since(datetime.now(timezone.utc) - timedelta(hours=1), run_id="r")
    assert result == {"observations_deleted": 1, "topics_deleted": 1, "topics_kept": 0, "topics_busy": 1, "snapshots_cleared": 0}
    assert [o["item_url"] for o in st.get_observations_for_topic("a")] == ["x"] and len(st.get_topic("a")["snapshots"]) == 1
    assert st.get_topic("a")["deleting"]["owner"] == "active-worker" and st.get_topic("free") is None


def test_a_move_onto_a_topic_leased_for_deletion_is_refused():
    from topicsleuth.add_topic import move_observation
    from topicsleuth.interfaces import DestinationNotLive
    from topicsleuth.reprocess import mark_deleting
    st = _two_topic_store()
    st.create_observation({"topic_id": "b", "item_url": "u1"})
    assert mark_deleting(st, "a", by="reprocess", owner="w")
    with pytest.raises(ValueError, match="target topic is deleting"):
        move_observation("u1", "a", st)
    oid = st.get_observations_for_item("u1")[0]["_id"]
    with pytest.raises(DestinationNotLive):
        st.update_observation(oid, {"topic_id": "a"}), "and the write itself refuses, whatever the caller checked"
    assert st.get_observations_for_item("u1")[0]["topic_id"] == "b"


def test_a_move_interleaved_with_a_reprocess_cannot_strand_evidence():
    from topicsleuth.add_topic import move_observation
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess

    class _Racing(MemoryStore):
        fired = False

        def delete_observations_for_topic(self, topic_id, *, deleting_owner=None):
            n = super().delete_observations_for_topic(topic_id, deleting_owner=deleting_owner)
            if not self.fired:
                self.fired = True
                try:
                    move_observation("moved", topic_id, self)
                    self.move = "succeeded"
                except ValueError as e:
                    self.move = str(e)
            return n
    st = _Racing()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A,
                     "date_identified": datetime.now(timezone.utc) - timedelta(days=90)})
    st.create_observation({"topic_id": "a", "item_url": "old-a"})
    st.create_topic({"topic_id": "b", "title": "B", "summary": "s", "embedding": E_B})
    st.create_observation({"topic_id": "b", "item_url": "moved"})
    st.create_observation({"topic_id": "b", "item_url": "keeps-b"})
    st.create_observation({"topic_id": "b", "item_url": "keeps-b-too"})
    out = execute_reprocess(st, plan_reprocess(st, max_obs=2, older_than_days=30))
    assert out["topics_deleted"] == 1 and "target topic is deleting" in st.move
    assert st.get_observations_for_item("moved")[0]["topic_id"] == "b"
    assert [o for o in st.list_observations() if st.get_topic(o["topic_id"]) is None] == []


def test_an_embedding_save_skipped_because_another_link_failed_still_counts_as_owed():
    from topicsleuth.embeddings import refresh_topic_embedding
    st = _four_plus_seed(lambda: _FailLinkOnce("b"))
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "s", "embedding": E_B})
    s = _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    items, evals, decisions = [_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a", 0.8, "b")]
    first = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert [f.stage for f in first.failures] == ["create_observation"], "b failed; a landed and crossed the threshold"
    topic = st.get_topic("a")
    assert st.get_item_embeddings(["u5"]) == {"u5": E_B}, "the observation on a is in the store, so its embedding is saved whatever b did"
    assert topic["embedding_refresh_pending"] is False and topic["embedding"][1] > 0, "and the refresh saw it"
    retry = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert retry.failures == [] and [o["topic_id"] for o in retry.observations_created] == ["b"]
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u5")) == ["a", "b"]
    final = list(st.get_topic("a")["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == final, "equal to a recomputation from all five persisted inputs"


# ── no cached busy list, folds are one write, atomic across threads ──

def test_cleanup_honours_a_lease_that_lands_after_it_began():
    """Cleanup has no busy list to go stale: it leases each topic just before
    writing to it, and a lease another operation took first, however late,
    leaves that topic's evidence and snapshots alone."""
    class _Late(MemoryStore):
        fired = False

        def acquire_deletion_lease(self, topic_id, owner, by, expires_at):
            if by == "clean" and topic_id == "a" and not self.fired:
                self.fired = True
                assert super().acquire_deletion_lease(topic_id, "arrived-after-scan", "reprocess", expires_at)
            return super().acquire_deletion_lease(topic_id, owner, by, expires_at)
    st = _Late()
    now = datetime.now(timezone.utc)
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "run_id": "r"})
    st.create_observation({"topic_id": "a", "item_url": "x", "run_id": "r"})
    st.append_snapshot("a", {"date": now, "evidence_count": 1, "run_id": "r"})
    result = st.delete_since(now - timedelta(hours=1), run_id="r")
    assert result == {"observations_deleted": 0, "topics_deleted": 0, "topics_kept": 0, "topics_busy": 1, "snapshots_cleared": 0}
    assert st.get_topic("a")["deleting"]["owner"] == "arrived-after-scan"
    assert [o["item_url"] for o in st.get_observations_for_topic("a")] == ["x"] and len(st.get_topic("a")["snapshots"]) == 1


def test_cleanup_writes_nothing_to_a_topic_after_giving_it_back():
    """Every write cleanup makes to a topic happens under its lease; once the
    lease is released the topic is another operation's to take, and nothing
    that operation writes is touched."""
    now = datetime.now(timezone.utc)

    class _Taken(MemoryStore):
        def release_deletion_lease(self, topic_id, owner):
            released = super().release_deletion_lease(topic_id, owner)
            assert super().acquire_deletion_lease(topic_id, "new-owner", "reprocess", now + timedelta(minutes=15))
            self.append_snapshot(topic_id, {"date": now, "evidence_count": 1, "run_id": "r", "by": "new-owner"})
            return released
    st = _Taken()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "run_id": "r"})
    st.create_observation({"topic_id": "a", "item_url": "retained", "run_id": "other"})
    st.append_snapshot("a", {"date": now, "evidence_count": 1, "run_id": "r"})
    result = st.delete_since(now - timedelta(hours=1), run_id="r")
    assert result == {"observations_deleted": 0, "topics_deleted": 0, "topics_kept": 1, "topics_busy": 0, "snapshots_cleared": 1}
    topic = st.get_topic("a")
    assert topic["deleting"]["owner"] == "new-owner" and [s.get("by") for s in topic["snapshots"]] == ["new-owner"]
    assert [o["item_url"] for o in st.get_observations_for_topic("a")] == ["retained"]


class _LeasedBeforeFold(MemoryStore):
    """The destination's lease lands after the caller's check, right before
    the fold would write."""
    lease_target = "b"

    def fold_observation(self, obs_id, into_obs_id, note):
        from topicsleuth.reprocess import mark_deleting
        assert mark_deleting(self, self.lease_target, by="reprocess", owner="active-worker")
        return super().fold_observation(obs_id, into_obs_id, note)


def test_a_fold_onto_a_topic_leased_after_the_callers_check_is_refused_whole():
    from topicsleuth.add_topic import move_observation
    from topicsleuth.interfaces import DestinationNotLive
    st = _two_topic_store(_LeasedBeforeFold)
    for topic_id, url in (("a", "u"), ("b", "u"), ("a", "keeps-a")):
        st.create_observation({"topic_id": topic_id, "item_url": url, "reasoning_trace": "r"})
    with pytest.raises(DestinationNotLive):
        move_observation("u", "b", st)
    both = {o["topic_id"]: o for o in st.get_observations_for_item("u")}
    assert sorted(both) == ["a", "b"] and both["b"]["reasoning_trace"] == "r", "neither half of the fold happened"
    assert st.delete_observations_for_topic("b", deleting_owner="active-worker") == 1
    assert st.delete_topic("b", deleting_owner="active-worker")
    assert [o["topic_id"] for o in st.get_observations_for_item("u")] == ["a"], "u keeps its evidence on a"


def test_a_merge_whose_fold_meets_a_lease_leaves_both_observations():
    from topicsleuth.interfaces import DestinationNotLive
    from topicsleuth.merge import merge_topics
    st, s = _two_topic_store(_LeasedBeforeFold), _settings()
    st.lease_target = "a"
    for topic_id, url in (("a", "u"), ("b", "u"), ("b", "only-b")):
        st.create_observation({"topic_id": topic_id, "item_url": url})
    merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
    with pytest.raises(DestinationNotLive):
        merge_topics("b", "a", st, s, ScriptedJudge([merged]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_A}))
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u")) == ["a", "b"]
    assert st.get_observations_for_item("only-b")[0]["topic_id"] == "b" and st.get_topic("b")["status"] != "retired"


def test_a_takeover_waits_for_a_guarded_delete_already_in_progress():
    """The reference store's operations are atomic across threads, not just
    within one call: a recovery that arrives while a stale owner's guarded
    delete is between its check and its write waits for it to commit."""
    import threading
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    inside, resume = threading.Event(), threading.Event()

    class _Paused(MemoryStore):
        def _held_by(self, topic_id, owner):
            held = super()._held_by(topic_id, owner)
            if owner == "stalled" and not inside.is_set():
                inside.set()
                resume.wait(5)
            return held
    st = _Paused()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "status": "active"})
    st.create_observation({"topic_id": "a", "item_url": "old"})
    assert mark_deleting(st, "a", by="reprocess", owner="stalled", lease_seconds=1, now=datetime.now(timezone.utc) - timedelta(seconds=5))
    outcome = {}
    worker = threading.Thread(target=lambda: outcome.update(deleted=st.delete_observations_for_topic("a", deleting_owner="stalled")))
    worker.start()
    assert inside.wait(5)
    recovery = threading.Thread(target=lambda: outcome.update(recovery=recover_interrupted_deletions(st)))
    recovery.start()
    recovery.join(0.3)
    assert recovery.is_alive(), "the takeover waits for the delete in progress"
    resume.set()
    worker.join(5)
    recovery.join(5)
    assert outcome == {"deleted": 1, "recovery": {"restored": [], "deleted": ["a"], "busy": []}}, \
        "the delete committed first, so recovery found an empty topic and finished the abandoned deletion"
    assert st.list_observations() == [] and st.get_topic("a") is None


def test_a_stale_owner_blocked_behind_a_takeover_deletes_nothing():
    """The other order: the takeover is in progress when the stale owner's
    delete arrives. The delete waits, then finds the lease gone, and the
    evidence recorded after the takeover survives with the old."""
    import threading
    from topicsleuth.reprocess import mark_deleting, recover_interrupted_deletions
    inside, resume = threading.Event(), threading.Event()

    class _Paused(MemoryStore):
        def acquire_deletion_lease(self, topic_id, owner, by, expires_at):
            # Pause inside the acquisition: under the store's lock, as the
            # real one runs, with the lease about to change hands.
            with self._lock:
                if by == "recover":
                    inside.set()
                    resume.wait(5)
                return super().acquire_deletion_lease(topic_id, owner, by, expires_at)
    st = _Paused()
    st.create_topic({"topic_id": "a", "title": "A", "summary": "s", "embedding": E_A, "status": "active"})
    st.create_observation({"topic_id": "a", "item_url": "old"})
    assert mark_deleting(st, "a", by="reprocess", owner="stalled", lease_seconds=1, now=datetime.now(timezone.utc) - timedelta(seconds=5))
    outcome = {}
    recovery = threading.Thread(target=lambda: outcome.update(recovery=recover_interrupted_deletions(st)))
    recovery.start()
    assert inside.wait(5)
    worker = threading.Thread(target=lambda: outcome.update(deleted=st.delete_observations_for_topic("a", deleting_owner="stalled")))
    worker.start()
    worker.join(0.3)
    assert worker.is_alive(), "the stale owner's delete waits for the takeover in progress"
    resume.set()
    recovery.join(5)
    worker.join(5)
    assert outcome == {"recovery": {"restored": ["a"], "deleted": [], "busy": []}, "deleted": 0}
    feeder = record([_item("new")], [_Eval("new", E_A)], [_link("new", "a")], st, settings=_settings())
    assert feeder.failures == [] and st.get_topic("a")["status"] == "active"
    assert sorted(o["item_url"] for o in st.get_observations_for_topic("a")) == ["new", "old"]


# ── a write that lands as its topic is merged, with the successor leased ──

class _MergedUnderneath(MemoryStore):
    """Between recording's check of ``a`` and its insert for item ``u``: the
    real merge of ``a`` into ``b`` completes (it cannot see an observation
    not yet inserted), then ``b`` is leased for deletion, as cleanup would."""
    fired = False

    def create_observation(self, obs_doc):
        if obs_doc.get("item_url") == "u" and not self.fired:
            self.fired = True
            from topicsleuth.merge import merge_topics
            from topicsleuth.reprocess import mark_deleting
            merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
            merge_topics("a", "b", self, _settings(), ScriptedJudge([merged]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_B}))
            assert mark_deleting(self, "b", by="clean", owner="cleanup")
        return super().create_observation(obs_doc)


def test_a_write_landing_on_a_topic_merged_into_a_leased_successor_fails_the_item_and_keeps_the_decision():
    st, s = _two_topic_store(_MergedUnderneath), _settings()
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    st.create_observation({"topic_id": "a", "item_url": "old-a"})
    items, evals = [_item("u"), _item("v")], [_Eval("u", E_A), _Eval("v", E_B)]
    decisions = [_link("u", "a"), _link("v", "c")]
    first = record(items, evals, decisions, st, settings=s)
    assert [(f.item_url, f.stage) for f in first.failures] == [("u", "create_observation")], "a failure, not an escaping exception"
    assert "'a' was merged into 'b' during the write" in first.failures[0].error and "waits on 'a'" in first.failures[0].error
    assert [o["item_url"] for o in first.observations_created] == ["v"], "the other item's work is unaffected"
    assert [o["topic_id"] for o in st.get_observations_for_item("u")] == ["a"] and st.get_topic("a")["status"] == "retired"
    # retried while b is still leased: still failed, nothing moved, nothing duplicated
    snapshots_on_a = len(st.get_topic("a")["snapshots"])
    again = record(items, evals, decisions, st, settings=s)
    assert [(f.item_url, f.stage) for f in again.failures] == [("u", "create_observation")] and "left for the next run" in again.failures[0].error
    assert [o["topic_id"] for o in st.get_observations_for_item("u")] == ["a"]
    assert [o["item_url"] for o in again.already_recorded] == ["v"], "the stranded observation is not resumed as if recorded"
    assert len(st.get_topic("a")["snapshots"]) == snapshots_on_a, "nothing was written to the retired source"
    # the lease is resolved; a full pipeline retry, from a fresh caller, converges on b
    assert st.release_deletion_lease("b", "cleanup")
    judge = ScriptedJudge([])
    retried, third = _pipeline(st, [_item("u")], judge)
    assert judge.calls == [] and [l.topic_id for l in retried[0].links] == ["b"], "the stored decision stands for its live successor"
    assert third.failures == [] and [o["item_url"] for o in third.already_recorded] == ["u"]
    obs = st.get_observations_for_item("u")
    assert len(obs) == 1 and obs[0]["topic_id"] == "b" and obs[0]["moved_from"] == "a" and obs[0]["removed_from"] == ["a"]
    assert "had been merged into 'b' when this landed" in obs[0]["reasoning_trace"]
    assert st.count_observations_for_topic("a") == 0 and st.get_topic("b")["snapshots"][-1]["evidence_count"] == 2


def test_a_fresh_caller_rehomes_an_observation_left_on_a_retired_merge_source():
    """As a crash between the insert and its post-check leaves things: the
    observation sits on the retired source. The next run folds it into the
    item's observation on the successor; one with no successor is withdrawn
    and the item judged again."""
    st, s = _two_topic_store(), _settings()
    st.create_observation({"topic_id": "a", "item_url": "u", "strength": 0.5, "connection_type": "extends", "reasoning_trace": "r",
                           "decision_links": [{"topic_id": "a", "connection_type": "extends", "strength": 0.5, "fit_questions": None}]})
    st.create_observation({"topic_id": "b", "item_url": "u", "strength": 0.9, "reasoning_trace": "r", "removed_from": ["x"]})
    st.update_topic("a", {"status": "retired", "merged_into": "b"})
    out = record([_item("u")], [_Eval("u", E_A)], [_link("u", "a")], st, settings=s)
    assert out.failures == []
    obs = st.get_observations_for_item("u")
    assert len(obs) == 1 and obs[0]["topic_id"] == "b" and obs[0]["strength"] == 0.9 and obs[0]["removed_from"] == ["x", "a"]
    assert "FOLDED by record" in obs[0]["reasoning_trace"] and "strength 0.5, extends" in obs[0]["reasoning_trace"]
    st.create_observation({"topic_id": "a", "item_url": "w"})
    st.update_topic("a", {"merged_into": "gone"})
    out = record([_item("w")], [_Eval("w", E_A)], [_link("w", "a")], st, settings=s)
    assert [(f.item_url, f.stage) for f in out.failures] == [("w", "create_observation")] and "no live successor" in out.failures[0].error
    assert st.get_observations_for_item("w") == []
    to_b = {**LINK, "candidate_classifications": [{"topic_id": "b", "label": "same_question"}],
            "links": [{"topic_id": "b", "connection_type": "supports", "fit_questions": FIT}]}
    judge = ScriptedJudge([to_b])
    _, fresh = _pipeline(st, [_item("w")], judge)
    assert len(judge.calls) == 1 and fresh.failures == [] and st.get_observations_for_item("w")[0]["topic_id"] == "b"


# ── a fold keeps the plan; a relocation is a mutation of its destination ──

class _TwoWritersThenMerge(MemoryStore):
    """Before the first writer's insert of ``u`` on ``a``: a second writer
    records ``u`` on ``b`` with plan [b]; the real merge of ``a`` into ``b``
    completes; ``b`` is leased. The first writer's insert of ``u`` on ``c``
    then fails ``c_failures`` times."""
    fired = False

    def __init__(self, c_failures=1):
        super().__init__()
        self.c_failures = c_failures

    def create_observation(self, obs_doc):
        if obs_doc.get("item_url") == "u" and obs_doc.get("topic_id") == "a" and not self.fired:
            self.fired = True
            from topicsleuth.merge import merge_topics
            from topicsleuth.reprocess import mark_deleting
            second = record([_item("u")], [_Eval("u", E_B)], [_link("u", "b")], self, settings=_settings())
            assert second.failures == []
            merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
            merge_topics("a", "b", self, _settings(), ScriptedJudge([merged]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_B}))
            assert mark_deleting(self, "b", by="clean", owner="cleanup")
        if obs_doc.get("item_url") == "u" and obs_doc.get("topic_id") == "c" and self.c_failures:
            self.c_failures -= 1
            raise RuntimeError("injected insert failure on c")
        return super().create_observation(obs_doc)


def _two_writers(c_failures=1):
    st = _TwoWritersThenMerge(c_failures)
    for tid, vec in (("a", E_A), ("b", E_B), ("c", E_B)):
        st.create_topic({"topic_id": tid, "title": f"Topic {tid.upper()}", "summary": f"Tracks {tid.upper()}.", "embedding": vec})
        st.create_observation({"topic_id": tid, "item_url": f"{tid}-keeps-it-live"})
    return st


def test_a_repair_fold_keeps_the_link_only_the_folded_observation_carried():
    st, s = _two_writers(c_failures=1), _settings()
    first = record([_item("u")], [_Eval("u", E_A)], [_link("u", "a", 0.8, "c")], st, settings=s)
    assert [f.stage for f in first.failures] == ["create_observation", "create_observation"], "a's relocation blocked; c's insert failed"
    plans = {o["topic_id"]: [l["topic_id"] for l in o["decision_links"]] for o in st.get_observations_for_item("u")}
    assert plans == {"b": ["b"], "a": ["a", "c"]}, "the only durable copy of the c link is on the stranded observation"
    assert st.release_deletion_lease("b", "cleanup")
    judge = ScriptedJudge([])
    retried, out = _pipeline(st, [_item("u")], judge)
    assert judge.calls == [] and [l.topic_id for l in retried[0].links] == ["b", "c"]
    assert out.failures == [] and [o["topic_id"] for o in out.observations_created] == ["c"]
    final = {o["topic_id"]: o for o in st.get_observations_for_item("u")}
    assert sorted(final) == ["b", "c"] and st.count_observations_for_topic("a") == 0
    assert [l["topic_id"] for l in final["b"]["decision_links"]] == ["b", "a", "c"] and final["b"]["removed_from"] == ["a"]
    assert "FOLDED by record" in final["b"]["reasoning_trace"]
    assert len(st.list_observations()) == len({(o["item_url"], o["topic_id"]) for o in st.list_observations()}), "no duplicate pair"


def test_the_item_stays_failed_until_the_link_the_fold_preserved_is_written():
    """The fold happens on the first retry, c fails again on it: the
    survivor's plan is the only thing that knows c is intended, and a
    fresh caller finishes it."""
    st, s = _two_writers(c_failures=2), _settings()
    record([_item("u")], [_Eval("u", E_A)], [_link("u", "a", 0.8, "c")], st, settings=s)
    assert st.release_deletion_lease("b", "cleanup")
    _, folded = _pipeline(st, [_item("u")], ScriptedJudge([]))
    assert [f.stage for f in folded.failures] == ["create_observation"] and folded.failed_urls() == {"u"}, "not acknowledged with b alone"
    only = st.get_observations_for_item("u")
    assert [o["topic_id"] for o in only] == ["b"] and [l["topic_id"] for l in only[0]["decision_links"]] == ["b", "a", "c"]
    fresh = record([_item("u")], [_Eval("u", E_A)], [_link("u", "b")], st, settings=s)
    assert fresh.failures == [] and [o["topic_id"] for o in fresh.observations_created] == ["c"]
    assert sorted(o["topic_id"] for o in st.get_observations_for_item("u")) == ["b", "c"]


def test_a_curation_fold_keeps_the_plan_too():
    """The same invariant on the shared curation fold: merging a into b
    folds u's a observation into its b one, and the c link only the a
    observation planned is still intended afterward."""
    from topicsleuth.merge import merge_topics
    st, s = _three_live_topics(), _settings()
    st.create_observation({"topic_id": "a", "item_url": "u", "decision_links": [
        {"topic_id": "a", "connection_type": "supports", "strength": 0.8, "fit_questions": None},
        {"topic_id": "c", "connection_type": "supports", "strength": 0.8, "fit_questions": None}]})
    st.create_observation({"topic_id": "b", "item_url": "u", "strength": 0.9})
    merged = {"title": "Merged", "summary": "AB.", "tags": [], "suggested_directions": [], "rationale": "r"}
    merge_topics("a", "b", st, s, ScriptedJudge([merged]), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_B}))
    survivor = st.get_observations_for_item("u")
    assert len(survivor) == 1 and [l["topic_id"] for l in survivor[0]["decision_links"]] == ["b", "a", "c"]
    assert [l.topic_id for l in intended_links(survivor, st)] == ["b", "c"]
    out = record([_item("u")], [_Eval("u", E_A)], [_link("u", "b")], st, settings=s)
    assert out.failures == [] and [o["topic_id"] for o in out.observations_created] == ["c"]


def test_merged_plans_do_not_revive_a_link_removed_on_purpose():
    from topicsleuth.history import merged_plan
    survivor = {"topic_id": "b", "strength": 0.9, "removed_from": ["x"]}
    folded = {"topic_id": "a", "decision_links": [{"topic_id": "a", "connection_type": "supports", "strength": 0.8, "fit_questions": None},
                                                   {"topic_id": "x", "connection_type": "extends", "strength": 0.5, "fit_questions": None}]}
    plan = merged_plan(survivor, folded)
    assert [l["topic_id"] for l in plan] == ["b", "a", "x"] and plan[0]["strength"] == 0.9 and plan[2]["connection_type"] == "extends"
    st = _three_live_topics()
    st.create_topic({"topic_id": "x", "title": "X", "summary": "s", "embedding": E_A})
    obs = {"topic_id": "b", "item_url": "u", "decision_links": plan, "removed_from": ["x", "a"]}
    assert [l.topic_id for l in intended_links([obs], st)] == ["b"], "x and a were removed on purpose; the union does not bring them back"


def test_relocating_stranded_evidence_refreshes_the_destination_it_pushes_over_a_threshold():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(MemoryStore), _settings()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    st.create_topic({"topic_id": "src", "title": "Source", "summary": "s", "embedding": E_A, "status": "retired", "merged_into": "a"})
    st.create_observation({"topic_id": "src", "item_url": "u5", "date": yesterday, "recorded_at": yesterday})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 4, "velocity": 4 / 7, "days_since_evidence": 0.0})
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    out = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert out.failures == [] and out.observations_created == [] and [o["item_url"] for o in out.already_recorded] == ["u5"]
    assert [o["topic_id"] for o in st.get_observations_for_item("u5")] == ["a"] and st.count_observations_for_topic("src") == 0
    topic = st.get_topic("a")
    assert topic["snapshots"][-1]["evidence_count"] == 5 and st.count_observations_for_topic("a") == 5
    assert st.get_item_embeddings(["u5"]) == {"u5": E_B} and topic["embedding"][1] > 0, "the fifth input pulled the embedding toward E_B"
    assert topic["embedding_refresh_pending"] is False
    after = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == after, "equal to a recomputation from all five saved inputs"


# ── folds into one survivor compose; a relocation's work is durable ──

class _SecondFoldInBetween(MemoryStore):
    """A second caller folds u's c observation into b between the first
    caller's reads and its fold of u's a observation into b. The insert of
    u on x then fails ``x_failures`` times."""
    fired = False
    x_failures = 0

    def fold_observation(self, obs_id, into_obs_id, note):
        if not self.fired:
            self.fired = True
            from topicsleuth.add_topic import move_observations
            on_c = [o for o in self.get_observations_for_item("u") if o["topic_id"] == "c"]
            move_observations(self, on_c, "b", datetime.now(timezone.utc), reason="second move")
        return super().fold_observation(obs_id, into_obs_id, note)

    def create_observation(self, obs_doc):
        if obs_doc.get("item_url") == "u" and obs_doc.get("topic_id") == "x" and self.x_failures:
            self.x_failures -= 1
            raise RuntimeError("injected insert failure on x")
        return super().create_observation(obs_doc)


def test_two_folds_into_one_survivor_each_keep_what_the_other_added():
    from topicsleuth.add_topic import move_observations
    st, s = _SecondFoldInBetween(), _settings()
    for tid, vec in (("a", E_A), ("b", E_B), ("c", E_B), ("x", E_A), ("y", E_B)):
        st.create_topic({"topic_id": tid, "title": f"Topic {tid.upper()}", "summary": f"Tracks {tid.upper()}.", "embedding": vec})
        st.create_observation({"topic_id": tid, "item_url": f"{tid}-keeps-it-live"})
    plan = lambda *tids: [{"topic_id": t, "connection_type": "supports", "strength": 0.8, "fit_questions": None} for t in tids]
    st.create_observation({"topic_id": "a", "item_url": "u", "decision_links": plan("a", "x")})
    st.create_observation({"topic_id": "b", "item_url": "u", "decision_links": plan("b")})
    st.create_observation({"topic_id": "c", "item_url": "u", "decision_links": plan("c", "y")})
    on_a = [o for o in st.get_observations_for_item("u") if o["topic_id"] == "a"]
    move_observations(st, on_a, "b", datetime.now(timezone.utc), reason="first move")
    survivor = st.get_observations_for_item("u")
    assert len(survivor) == 1 and survivor[0]["topic_id"] == "b"
    assert [l["topic_id"] for l in survivor[0]["decision_links"]] == ["b", "c", "y", "a", "x"]
    assert survivor[0]["removed_from"] == ["c", "a"]
    assert survivor[0]["reasoning_trace"].count("FOLDED by") == 2, "both folds' provenance"
    assert sorted(l.topic_id for l in intended_links(survivor, st)) == ["b", "x", "y"]
    assert st.count_observations_for_topic("a") == 1 and st.count_observations_for_topic("c") == 1, "only u's observations moved"
    st.x_failures = 1
    first = record([_item("u")], [_Eval("u", E_A)], [_link("u", "b")], st, settings=s)
    assert [f.stage for f in first.failures] == ["create_observation"] and [o["topic_id"] for o in first.observations_created] == ["y"]
    second = record([_item("u")], [_Eval("u", E_A)], [_link("u", "b")], st, settings=s)
    assert second.failures == [] and [o["topic_id"] for o in second.observations_created] == ["x"]
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert sorted(t for u_, t in pairs if u_ == "u") == ["b", "x", "y"] and len(pairs) == len(set(pairs))


class _SnapshotFailsOnce(MemoryStore):
    armed = False

    def append_snapshot(self, topic_id, snapshot):
        if self.armed and topic_id == "a":
            self.armed = False
            raise RuntimeError("snapshot unavailable once")
        super().append_snapshot(topic_id, snapshot)


def test_a_relocations_derived_work_survives_a_snapshot_failure_after_it():
    """The move stamps the observation; until a snapshot of the destination
    postdates the stamp, a fresh caller still owes the snapshot and the
    threshold refresh, whatever the observation's own older timestamps say."""
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(_SnapshotFailsOnce), _settings()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    st.create_topic({"topic_id": "src", "title": "Source", "summary": "s", "embedding": E_A, "status": "retired", "merged_into": "a"})
    st.create_observation({"topic_id": "src", "item_url": "u5", "date": yesterday, "recorded_at": yesterday})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 4, "velocity": 4 / 7, "days_since_evidence": 0.0})
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    st.armed = True
    first = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert [f.stage for f in first.failures] == ["snapshot"]
    moved = st.get_observations_for_item("u5")
    assert len(moved) == 1 and moved[0]["topic_id"] == "a" and moved[0]["relocated_at"] > moved[0]["recorded_at"]
    assert st.get_topic("a")["embedding"][1] > 0, "the refresh the move owed ran, snapshot or no snapshot"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and retry.observations_created == [] and [o["item_url"] for o in retry.already_recorded] == ["u5"]
    topic = st.get_topic("a")
    assert topic["snapshots"][-1]["evidence_count"] == 5 and st.count_observations_for_topic("a") == 5
    assert len(st.get_observations_for_item("u5")) == 1 and st.count_observations_for_topic("src") == 0
    assert topic["embedding"][1] > 0 and topic["embedding_refresh_pending"] is False
    after = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == after, "equal to a recomputation from all five saved inputs"
    third = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert third.failures == [] and third.topics_updated == [], "the work is done once; the stamp is satisfied by the snapshot"


# ── another batch's snapshot does not hide a relocation's refresh ──

def test_a_relocations_refresh_is_owed_until_it_runs_whatever_snapshots_other_batches_write():
    """The refresh obligation is registered on the topic with the move and
    cleared only by a refresh that ran; a later batch that snapshots the
    topic without crossing a threshold performs it instead of hiding it."""
    st, s = _four_plus_seed(_SnapshotFailsOnce), _settings()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    st.create_topic({"topic_id": "src", "title": "Source", "summary": "s", "embedding": E_A, "status": "retired", "merged_into": "a"})
    st.create_observation({"topic_id": "src", "item_url": "u5", "date": yesterday, "recorded_at": yesterday})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 4, "velocity": 4 / 7, "days_since_evidence": 0.0})
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    st.armed = True
    first = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert [f.stage for f in first.failures] == ["snapshot"]
    topic = st.get_topic("a")
    moved_at = st.get_observations_for_item("u5")[0]["relocated_at"]
    assert topic["embedding"][1] > 0 and topic["embedding_refreshed_at"] >= moved_at, "the move's refresh ran in the same call, snapshot or no snapshot"
    at_five = list(topic["embedding"])
    # another batch records a sixth item on a: five to six crosses nothing and nothing is owed
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == [] and st.get_topic("a")["snapshots"][-1]["evidence_count"] == 6
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and retry.observations_created == [] and [o["item_url"] for o in retry.already_recorded] == ["u5"]
    final = st.get_topic("a")
    assert final["snapshots"][-1]["evidence_count"] == 6 and st.count_observations_for_topic("a") == 6 and st.count_observations_for_topic("src") == 0
    assert final["embedding_refresh_pending"] is False and list(final["embedding"]) == at_five, "nothing owed: the refresh at five stands"
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert len(pairs) == len(set(pairs)) and len(st.get_observations_for_item("u5")) == 1


def test_an_intervening_batch_cannot_clear_a_relocations_refresh_before_its_input_is_saved():
    """If the relocated item's embedding never landed, a later batch keeps
    the obligation and the inputs owed; the retry that saves it refreshes."""
    from topicsleuth.embeddings import refresh_topic_embedding

    class _SaveFailsForU5Once(_SnapshotFailsOnce):
        tripped = False

        def save_item_embedding(self, item_url, embedding):
            if item_url == "u5" and not self.tripped:
                self.tripped = True
                raise RuntimeError("injected save failure")
            super().save_item_embedding(item_url, embedding)
    st, s = _four_plus_seed(_SaveFailsForU5Once), _settings()
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    st.create_topic({"topic_id": "src", "title": "Source", "summary": "s", "embedding": E_A, "status": "retired", "merged_into": "a"})
    st.create_observation({"topic_id": "src", "item_url": "u5", "date": yesterday, "recorded_at": yesterday})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 4, "velocity": 4 / 7, "days_since_evidence": 0.0})
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    st.armed = True
    first = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert sorted(f.stage for f in first.failures) == ["save_embedding", "snapshot"]
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == []
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is True and topic["embedding_refresh_missing"] == ["u5"] and topic["embedding"] == E_A, "kept: u5's input is still unsaved"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    final = st.get_topic("a")
    assert final["embedding_refresh_pending"] is False and final["embedding"][1] > 0
    settled = list(final["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six saved inputs"


# ── the move is the whole record of what it owes ──

def _stranded_into_four(cls):
    st = _four_plus_seed(cls)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    st.create_topic({"topic_id": "src", "title": "Source", "summary": "s", "embedding": E_A, "status": "retired", "merged_into": "a"})
    st.create_observation({"topic_id": "src", "item_url": "u5", "date": yesterday, "recorded_at": yesterday})
    st.append_snapshot("a", {"date": datetime.now(timezone.utc), "evidence_count": 4, "velocity": 4 / 7, "days_since_evidence": 0.0})
    return st


def test_a_fresh_caller_finishes_a_relocation_the_process_died_right_after():
    """What a crash right after the move leaves: the observation on its
    destination with its stamp, and nothing else. A later batch's snapshot
    does not answer the stamp; the retry does the refresh from all inputs."""
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _stranded_into_four(MemoryStore), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    stranded = st.get_observations_for_item("u5")[0]
    st.update_observation(stranded["_id"], {"topic_id": "a", "moved_from": "src", "removed_from": ["src"], "relocated_at": datetime.now(timezone.utc)})
    assert st.get_item_embeddings(["u5"]) == {} and "embedding_refreshed_at" not in st.get_topic("a")
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == [] and st.get_topic("a")["embedding"] == E_A, "five to six crosses nothing; the snapshot is newer than the stamp"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and retry.observations_created == [] and [o["item_url"] for o in retry.already_recorded] == ["u5"]
    topic = st.get_topic("a")
    assert topic["embedding"][1] > 0 and topic["embedding_refreshed_at"] > stranded.get("recorded_at")
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six saved inputs"
    assert topic["snapshots"][-1]["evidence_count"] == 6 and st.count_observations_for_topic("a") == 6 and st.count_observations_for_topic("src") == 0
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert len(pairs) == len(set(pairs)) and len(st.get_observations_for_item("u5")) == 1
    again = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert again.failures == [] and list(st.get_topic("a")["embedding"]) == settled, "answered: nothing more is owed"


def test_a_store_failure_right_after_a_relocation_fails_that_item_only_and_is_recoverable():
    """The store fails on the first read after the move commits: the item
    fails at the recording boundary, the unrelated item in the batch is
    recorded, and a fresh caller finds the moved observation and finishes
    its refresh."""
    from topicsleuth.embeddings import refresh_topic_embedding
    from topicsleuth.interfaces import StoreError

    class _FailsAfterTheMove(MemoryStore):
        tripped = False

        def get_observations_for_item(self, item_url):
            found = super().get_observations_for_item(item_url)
            if item_url == "u5" and not self.tripped and any(o["topic_id"] == "a" for o in found):
                self.tripped = True
                raise StoreError("store unavailable once, right after the move")
            return found
    st, s = _stranded_into_four(_FailsAfterTheMove), _settings()
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    items, evals, decisions = [_item("u5"), _item("v")], [_Eval("u5", E_B), _Eval("v", E_B)], [_link("u5", "src"), _link("v", "c")]
    first = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert [(f.item_url, f.stage) for f in first.failures] == [("u5", "relocate")] and "unavailable once" in first.failures[0].error
    assert [o["item_url"] for o in first.observations_created] == ["v"], "the unrelated item was recorded"
    moved = st.get_observations_for_item("u5")
    assert len(moved) == 1 and moved[0]["topic_id"] == "a" and moved[0].get("relocated_at"), "the move committed, with its stamp"
    assert st.get_topic("a")["embedding"] == E_A and st.get_item_embeddings(["u5"]) == {}
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == []
    retry = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert retry.failures == [] and sorted(o["item_url"] for o in retry.already_recorded) == ["u5", "v"]
    topic = st.get_topic("a")
    assert topic["embedding"][1] > 0 and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six saved inputs"
    assert st.count_observations_for_topic("a") == 6 and st.count_observations_for_topic("src") == 0


# ── a refresh answers only for the inputs it included ──

class _FailsAfterTheMoveOfU5(MemoryStore):
    """One StoreError on the first read of u5 after its move to a commits."""
    tripped = False

    def get_observations_for_item(self, item_url):
        found = super().get_observations_for_item(item_url)
        if item_url == "u5" and not self.tripped and any(o["topic_id"] == "a" for o in found):
            self.tripped = True
            raise StoreError("store unavailable once, right after the move")
        return found


def test_a_threshold_refresh_that_left_the_relocated_input_out_does_not_answer_for_it():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _stranded_into_four(_FailsAfterTheMoveOfU5), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    first = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert [(f.item_url, f.stage) for f in first.failures] == [("u5", "relocate")] and st.get_item_embeddings(["u5"]) == {}
    more = [f"u{i}" for i in range(6, 16)]
    batch = record([_item(u) for u in more], [_Eval(u, E_A) for u in more], [_link(u, "a") for u in more], st, settings=s, embedder=emb)
    assert batch.failures == [] and st.count_observations_for_topic("a") == 15
    topic = st.get_topic("a")
    assert len(topic["embedding_inputs"]) == 14 and "u5" not in topic["embedding_inputs"], "the threshold refresh ran from what was saved"
    assert topic["embedding_refreshed_at"] > st.get_observations_for_item("u5")[0]["relocated_at"], "and finished after the move"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    topic = st.get_topic("a")
    assert "u5" in topic["embedding_inputs"] and len(topic["embedding_inputs"]) == 15 and topic["embedding"][1] > 0
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all fifteen saved inputs"
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert len(pairs) == len(set(pairs)) and st.count_observations_for_topic("src") == 0


class _PausingEmbedder(HashEmbedder):
    """Pauses the seed embedding of Topic A: a refresh that has read its
    observations and inputs, and has not yet written."""
    def __init__(self, *a, **kw):
        import threading
        super().__init__(*a, **kw)
        self.inside, self.resume = threading.Event(), threading.Event()

    def embed_documents(self, texts):
        if any(t.startswith("Topic A") for t in texts) and not self.inside.is_set():
            self.inside.set()
            self.resume.wait(5)
        return super().embed_documents(texts)


def _overlapping_refresh(st, s, emb):
    """Start a refresh of a on a worker thread and return once it has read
    the four observations and is paused before writing."""
    import threading
    from topicsleuth.embeddings import refresh_topic_embedding
    outcome = {}
    worker = threading.Thread(target=lambda: outcome.update(refreshed=refresh_topic_embedding("a", st, s, emb)))
    worker.start()
    assert emb.inside.wait(5)
    return worker, outcome


def test_a_refresh_that_read_the_topic_before_the_move_and_finished_after_it_answers_nothing():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _stranded_into_four(_FailsAfterTheMoveOfU5), _settings()
    st.save_item_embedding("u5", E_B)
    emb = _PausingEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    worker, outcome = _overlapping_refresh(st, s, emb)
    first = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert [(f.item_url, f.stage) for f in first.failures] == [("u5", "relocate")]
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == []
    assert st.get_topic("a")["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"], \
        "the sixth item's batch found a topic never refreshed at six and refreshed it"
    emb.resume.set()
    worker.join(5)
    topic = st.get_topic("a")
    assert outcome == {"refreshed": True} and topic["embedding_revision"] == 2
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"], \
        "the old refresh, computed from before the move, lost its publication and recomputed"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == []
    topic = st.get_topic("a")
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"] and topic["embedding"][1] > 0
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six saved inputs"


def test_an_old_refresh_landing_after_the_repair_does_not_overwrite_it():
    """The other order: the repair's refresh is written first; the old
    refresh, about to write a result computed from fewer inputs, recomputes
    from the current state instead."""
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _stranded_into_four(_FailsAfterTheMoveOfU5), _settings()
    st.save_item_embedding("u5", E_B)
    emb = _PausingEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    worker, outcome = _overlapping_refresh(st, s, emb)
    record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert retry.failures == [] and st.get_topic("a")["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"]
    repaired = list(st.get_topic("a")["embedding"])
    emb.resume.set()
    worker.join(5)
    assert outcome == {"refreshed": True}
    topic = st.get_topic("a")
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"] and list(topic["embedding"]) == repaired, "the old refresh recomputed; it did not narrow the result"
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == repaired


# ── publication is a compare-and-swap ──

def test_an_old_refresh_cannot_overwrite_a_repair_acknowledged_while_it_computed():
    """The old refresh has computed and would publish next; the repair's
    refresh lands first. Publication is conditional on the revision the
    refresh read, so the old one recomputes instead of overwriting, and a
    later non-crossing item finds nothing stale waiting."""
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _stranded_into_four(MemoryStore), _settings()
    emb = _PausingEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    worker, outcome = _overlapping_refresh(st, s, emb)
    repaired = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert repaired.failures == [] and st.get_topic("a")["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5"]
    acknowledged = list(st.get_topic("a")["embedding"])
    assert acknowledged[1] > 0 and st.get_topic("a")["embedding_revision"] == 1
    emb.resume.set()
    worker.join(5)
    assert outcome == {"refreshed": True}
    topic = st.get_topic("a")
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5"] and list(topic["embedding"]) == acknowledged, "recomputed, not overwritten"
    assert topic["embedding_revision"] == 2 and len(emb.calls) == 3, "two seed embeddings for the old refresh: one wasted, one that published"
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == [] and st.get_topic("a")["embedding_refresh_pending"] is False
    topic = st.get_topic("a")
    assert "u5" in topic["embedding_inputs"] and list(topic["embedding"]) == acknowledged, "nothing stale is left waiting for a threshold"
    refresh_topic_embedding("a", st, s, emb)
    assert "u6" in st.get_topic("a")["embedding_inputs"], "a refresh now includes the sixth item, as a recomputation would"


def test_a_refresh_that_keeps_losing_the_race_reports_a_conflict_rather_than_publish():
    from topicsleuth.embeddings import REFRESH_ATTEMPTS, RefreshConflict, refresh_topic_embedding

    class _AlwaysBeaten(MemoryStore):
        """Something publishes to a between every read and write."""
        beaten = 0

        def publish_topic_embedding(self, topic_id, fields, *, expected_revision, missing=()):
            if topic_id == "a":
                self.beaten += 1
                super().update_topic(topic_id, {"embedding": E_A})
            return super().publish_topic_embedding(topic_id, fields, expected_revision=expected_revision, missing=missing)
    st, s = _stranded_into_four(_AlwaysBeaten), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    with pytest.raises(RefreshConflict):
        refresh_topic_embedding("a", st, s, emb)
    topic = st.get_topic("a")
    assert st.beaten == REFRESH_ATTEMPTS and "embedding_inputs" not in topic and topic["embedding"] == E_A, "nothing of this refresh was published"
    out = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "src")], st, settings=s, embedder=emb)
    assert [(f.item_url, f.stage) for f in out.failures] == [("u5", "embedding_refresh")] and "published ahead" in out.failures[0].error
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is True and "u5" not in (topic.get("embedding_inputs") or []), "still owed, not acknowledged"
    assert [o["topic_id"] for o in st.get_observations_for_item("u5")] == ["a"]


# ── a refresh clears only the obligation it satisfied ──

def test_an_older_refresh_cannot_clear_an_obligation_registered_after_it_published():
    """Ordinary observations, no relocation: the older batch's refresh has
    published; before it returns, a newer batch crosses fifteen with one
    input unsaved and registers that. Publication and clearing are one
    write, so the older batch has nothing left to write and the newer
    obligation survives; the retry that saves the input refreshes."""
    import threading
    from topicsleuth.embeddings import refresh_topic_embedding
    inside, resume = threading.Event(), threading.Event()

    class _PausesAfterPublishing(MemoryStore):
        paused = tripped = False

        def publish_topic_embedding(self, topic_id, fields, *, expected_revision, missing=()):
            ok = super().publish_topic_embedding(topic_id, fields, expected_revision=expected_revision, missing=missing)
            if ok and not self.paused:
                self.paused = True
                inside.set()
                resume.wait(5)
            return ok

        def save_item_embedding(self, item_url, embedding):
            if item_url == "u15" and not self.tripped:
                self.tripped = True
                raise RuntimeError("injected save failure")
            super().save_item_embedding(item_url, embedding)
    st, s = _four_plus_seed(_PausesAfterPublishing), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    outcome = {}
    older = threading.Thread(target=lambda: outcome.update(
        older=record([_item("u5")], [_Eval("u5", E_A)], [_link("u5", "a")], st, settings=s, embedder=emb)))
    older.start()
    assert inside.wait(5)
    assert st.get_topic("a")["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5"], "the older refresh has published"
    more = [f"u{i}" for i in range(6, 16)]
    newer = record([_item(u) for u in more], [_Eval(u, E_B if u == "u15" else E_A) for u in more],
                   [_link(u, "a") for u in more], st, settings=s, embedder=emb)
    assert [(f.item_url, f.stage) for f in newer.failures] == [("u15", "save_embedding")]
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is True and topic["embedding_refresh_missing"] == ["u15"], "registered by the newer batch"
    resume.set()
    older.join(5)
    assert outcome["older"].failures == []
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is True and topic["embedding_refresh_missing"] == ["u15"], "the older batch cleared nothing it did not satisfy"
    retry = record([_item("u15")], [_Eval("u15", E_B)], [_link("u15", "a")], st, settings=s, embedder=emb)
    assert retry.failures == [] and st.get_item_embeddings(["u15"]) == {"u15": E_B}
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is False and topic["embedding_refresh_missing"] == []
    assert len(topic["embedding_inputs"]) == 15 and "u15" in topic["embedding_inputs"] and topic["embedding"][1] > 0
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all fifteen inputs"


def test_a_refresh_publishes_what_it_has_and_keeps_owing_what_it_could_not_load():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(MemoryStore), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    st.create_observation({"topic_id": "a", "item_url": "u5"})
    st.defer_topic_refresh("a", ["u5", "gone"])
    assert refresh_topic_embedding("a", st, s, emb)
    topic = st.get_topic("a")
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3"] and topic["embedding_refreshed_count"] == 5
    assert topic["embedding_refresh_pending"] is True and topic["embedding_refresh_missing"] == ["u5"], \
        "published from four, still owing the fifth; an owed item no longer on the topic is dropped"
    st.save_item_embedding("u5", E_B)
    assert refresh_topic_embedding("a", st, s, emb)
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is False and topic["embedding_refresh_missing"] == [] and "u5" in topic["embedding_inputs"]


# ── a threshold refresh is owed by the store's own count, not by what a call remembered ──

class _Interrupted(BaseException):
    """What a process kill looks like from inside: not an Exception."""


def test_an_interruption_after_the_snapshot_and_before_the_refresh_is_recovered_by_the_next_run(monkeypatch):
    import topicsleuth.record as record_module
    from topicsleuth.embeddings import refresh_topic_embedding
    st, s = _four_plus_seed(MemoryStore), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})

    def die(*a, **k):
        raise _Interrupted()
    monkeypatch.setattr(record_module, "_refresh_or_defer", die)
    with pytest.raises(_Interrupted):
        record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st, settings=s, embedder=emb)
    monkeypatch.undo()
    topic = st.get_topic("a")
    assert topic["snapshots"][-1]["evidence_count"] == 5 and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    assert "embedding_refresh_pending" not in topic and "embedding_refreshed_count" not in topic, "nothing durable says a refresh is owed, except the count itself"
    # a later batch crossing nothing writes a newer snapshot; it also finds the crossing and does the work
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == [] and st.get_topic("a")["embedding_refreshed_count"] == 6 and st.get_topic("a")["embedding"][1] > 0
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st, settings=s, embedder=emb)
    assert retry.failures == [] and retry.observations_created == [] and [o["item_url"] for o in retry.already_recorded] == ["u5"]
    topic = st.get_topic("a")
    assert len(topic["embedding_inputs"]) == 6 and "u5" in topic["embedding_inputs"]
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six inputs"
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert len(pairs) == len(set(pairs)) == 6
    # and with no intervening batch at all, the retry itself does the work
    st2 = _four_plus_seed(MemoryStore)
    monkeypatch.setattr(record_module, "_refresh_or_defer", die)
    with pytest.raises(_Interrupted):
        record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st2, settings=s, embedder=emb)
    monkeypatch.undo()
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st2, settings=s, embedder=emb)
    assert retry.failures == [] and st2.get_topic("a")["embedding_refreshed_count"] == 5 and st2.get_topic("a")["embedding"][1] > 0


def test_a_failed_registration_fails_the_item_and_the_next_run_still_finds_the_refresh_owed():
    from topicsleuth.embeddings import refresh_topic_embedding
    from topicsleuth.interfaces import StoreError

    class _RegistrationFailsOnce(MemoryStore):
        tripped = False

        def defer_topic_refresh(self, topic_id, missing):
            if not self.tripped:
                self.tripped = True
                raise StoreError("injected obligation-registration failure")
            super().defer_topic_refresh(topic_id, missing)
    st, s = _four_plus_seed(_RegistrationFailsOnce), _settings()
    st.create_topic({"topic_id": "c", "title": "Topic C", "summary": "Tracks C.", "embedding": E_B})
    items, evals = [_item("u5"), _item("v")], [_Eval("u5", E_B), _Eval("v", E_B)]
    first = record(items, evals, [_link("u5", "a"), _link("v", "c")], st, settings=s)  # no embedder: the supported deferral path
    assert [(f.item_url, f.stage) for f in first.failures] == [("u5", "embedding_refresh")] and "registration failure" in first.failures[0].error
    assert [o["item_url"] for o in first.observations_created] == ["u5", "v"], "the unrelated item is unaffected"
    topic = st.get_topic("a")
    assert topic["snapshots"][-1]["evidence_count"] == 5 and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    assert topic["embedding_refresh_pending"] is True, "the fallback registration after the failure did land this time"
    st.update_topic("a", {"embedding_refresh_pending": False})  # and even if it had not:
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    retry = record(items, evals, [_link("u5", "a"), _link("v", "c")], st, settings=s, embedder=emb)
    assert retry.failures == [] and sorted(o["item_url"] for o in retry.already_recorded) == ["u5", "v"]
    topic = st.get_topic("a")
    assert topic["embedding_refreshed_count"] == 5 and topic["embedding"][1] > 0 and topic["embedding_refresh_pending"] is False
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all five inputs"


def test_the_threshold_policy_is_kept_and_a_shrunken_topic_refreshes_again_from_zero():
    from topicsleuth.embeddings import refresh_owed
    assert refresh_owed({}, 4) is False and refresh_owed({}, 5) is True
    assert refresh_owed({"embedding_refreshed_count": 5}, 6) is False, "five to six crosses nothing"
    assert refresh_owed({"embedding_refreshed_count": 5}, 15) is True
    assert refresh_owed({"embedding_refreshed_count": 15}, 15) is False
    assert refresh_owed({"embedding_refreshed_count": 15}, 3) is False and refresh_owed({"embedding_refreshed_count": 15}, 5) is True, \
        "evidence removed: the next threshold from zero owes a refresh"


# ── a partial refresh never claims the whole count ──

def test_a_refresh_that_ran_between_another_writers_insert_and_save_leaves_that_input_owed():
    """Two writers, no failures: the fifth item's writer has inserted its
    observation and not yet saved its embedding when the sixth item's
    writer refreshes. That refresh publishes from what it could load and
    records the fifth input as owed; the fifth writer, saving it, finds
    the refresh still owed and finishes it."""
    import threading
    from topicsleuth.embeddings import refresh_topic_embedding
    inside, resume = threading.Event(), threading.Event()

    class _PausesBeforeSavingU5(MemoryStore):
        def save_item_embedding(self, item_url, embedding):
            if item_url == "u5" and not inside.is_set():
                inside.set()
                resume.wait(5)
            super().save_item_embedding(item_url, embedding)
    st, s = _four_plus_seed(_PausesBeforeSavingU5), _settings()
    emb = HashEmbedder(DIM, vectors={"Topic A\n\ns": E_A})
    outcome = {}
    fifth = threading.Thread(target=lambda: outcome.update(
        fifth=record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st, settings=s, embedder=emb)))
    fifth.start()
    assert inside.wait(5)
    sixth = record([_item("u6")], [_Eval("u6", E_A)], [_link("u6", "a")], st, settings=s, embedder=emb)
    assert sixth.failures == []
    topic = st.get_topic("a")
    assert topic["embedding_refreshed_count"] == 6 and topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u6"]
    assert topic["embedding_refresh_pending"] is True and topic["embedding_refresh_missing"] == ["u5"], "published from five, owing the fifth"
    resume.set()
    fifth.join(5)
    assert outcome["fifth"].failures == [] and st.get_item_embeddings(["u5"]) == {"u5": E_B}
    topic = st.get_topic("a")
    assert topic["embedding_refresh_pending"] is False and topic["embedding_refresh_missing"] == []
    assert topic["embedding_inputs"] == ["a0", "a1", "a2", "a3", "u5", "u6"] and topic["embedding"][1] > 0
    settled = list(topic["embedding"])
    refresh_topic_embedding("a", st, s, emb)
    assert list(st.get_topic("a")["embedding"]) == settled, "equal to a recomputation from all six inputs, with no manual repair"
    retry = record([_item("u5")], [_Eval("u5", E_B)], [_link("u5", "a")], st, settings=s, embedder=emb)
    assert retry.failures == [] and list(st.get_topic("a")["embedding"]) == settled and "u5" in st.get_topic("a")["embedding_inputs"]
    pairs = [(o["item_url"], o["topic_id"]) for o in st.list_observations()]
    assert len(pairs) == len(set(pairs)) == 6
