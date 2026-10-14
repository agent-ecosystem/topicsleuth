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
from topicsleuth.synthesize import SynthesisDecision, synthesize_items

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


def _link(url="u1", topic_id="a", strength=0.8):
    return SynthesisDecision(item_url=url, item_title=f"Item {url}", decision="link_existing", topic_id=topic_id,
                             connection_type="supports", strength=strength, reasoning_trace="r")


def _propose(url="u2", topic_id="topic-c", title="Topic C"):
    return SynthesisDecision(item_url=url, item_title=f"Item {url}", decision="propose_new", topic_id=None,
                             connection_type=None, strength=0.85, reasoning_trace="r",
                             new_topic={"topic_id": topic_id, "title": title, "summary": "Tracks C.", "embedding": E_B})


FIT = {"q1_scope_expansion": False, "q2_representative_citation": True, "q3_distinguishing_evidence": True}
LINK = {"candidate_classifications": [{"topic_id": "a", "label": "same_question"}], "decision": "link_existing",
        "topic_id": "a", "connection_type": "supports", "fit_questions": FIT, "reasoning_trace": "fits"}
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
    ({**LINK, "topic_id": "nonexistent"}, "not among the shown candidates"),
    ({**LINK, "connection_type": "garbage"}, "unknown connection_type 'garbage'"),
    ({**LINK, "fit_questions": None, "strength": 9}, "without the three fit questions"),
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
