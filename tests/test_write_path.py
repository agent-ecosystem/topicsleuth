"""The write path under retry, failure and history: recording is idempotent,
every failure is reported per item and writes nothing for it, the judge's
answer is validated before it is written, backfills read as history, and a
run can be removed by its id."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from topicsleuth.backends.fake import HashEmbedder, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore
from topicsleuth.interfaces import ItemFailure
from topicsleuth.record import record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import Link, SynthesisDecision, synthesize_items

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
    assert result == {"observations_deleted": 2, "topics_deleted": 1, "topics_kept": 0, "snapshots_cleared": 1}
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


# ── round 2: whole-pipeline retries, partial writes, maintenance interleaving ──

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
