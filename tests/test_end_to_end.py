"""The decision and record path end to end, offline: MemoryStore, a hashing
embedder and a scripted judge. Each test is one story the prompts describe:
link, propose and confirm, propose and get overridden, skip, the gate, the
in-batch proposal, the decision-model reviewer, cold start, leave-one-out,
and the curation flows on top of a populated store."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from topicsleuth.backends.fake import HashEmbedder, ScriptedDecisionModel, ScriptedJudge
from topicsleuth.backends.memory import MemoryStore
from topicsleuth.interfaces import Completion
from topicsleuth.record import QUIET_TOPIC_FIELDS, record
from topicsleuth.settings import load_settings
from topicsleuth.synthesize import (
    cold_start_system_prompt,
    CONFIRM_LINK_STRENGTH,
    GATE_OVERRIDE_DEFAULT_STRENGTH,
    SynthesisDecision,
    TOPIC_HEAD_FIELDS,
    synthesize_item,
    synthesize_items,
)

DIM = 8
E_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
E_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _Eval:
    def __init__(self, url, embedding, above=True):
        self.url, self.embedding, self.above_threshold = url, embedding, above


def _settings(**over):
    return load_settings(overrides={"domain": "widget manufacturing", "embedding": {"dimensions": DIM},
                                    "candidates": {"vector_limit": 5, "ignored_tags": ["feed"]}, **over})


def _store(st=None):
    st = MemoryStore() if st is None else st
    st.create_topic({"topic_id": "a", "title": "Topic A", "summary": "Tracks A.", "embedding": E_A, "tags": ["alpha"],
                     "snapshots": [{"evidence_count": 3}]})
    st.create_topic({"topic_id": "b", "title": "Topic B", "summary": "Tracks B.", "embedding": E_B, "tags": ["beta"]})
    for i in range(3):
        st.create_observation({"topic_id": "a", "item_url": f"a{i}", "item_title": f"A evidence {i}", "strength": 0.9})
    return st


def _item(url="u1", **extra):
    return {"url": url, "title": f"Item {url}", "summary": "About A.", "source_type": "rss", "evidence": "ships",
            "tags": ["feed", "alpha"], "date_published": datetime(2026, 5, 1, tzinfo=timezone.utc), **extra}


LINK = {
    "candidate_classifications": [{"topic_id": "a", "candidate_question": "A?", "item_question": "A?", "label": "same_question"}],
    "decision": "link_existing",
    "links": [{"topic_id": "a", "connection_type": "supports",
               "fit_questions": {"q1_scope_expansion": False, "q2_representative_citation": True, "q3_distinguishing_evidence": True}}],
    "reasoning_trace": "fits A",
}
PROPOSE = {
    "candidate_classifications": [{"topic_id": "a", "label": "adjacent_question"}],
    "decision": "propose_new", "topic_id": None, "connection_type": None,
    "reasoning_trace": "new question",
    "new_topic": {"title": "Topic C", "summary": "Tracks C.", "topic_id": "Topic C!!", "tags": ["feed", "gamma"],
                  "suggested_directions": ["d1"], "scope_test_question": "C?", "differentiation_from_closest": "a tracks A"},
}
CONFIRM = {"verdict": "confirm_new", "closest_candidate": "a", "reasoning": "right-sized"}


def test_link_decision_records_observation_and_snapshot():
    st, judge, emb, s = _store(), ScriptedJudge([LINK]), HashEmbedder(DIM), _settings()
    decisions = synthesize_items([_item()], [_Eval("u1", E_A)], st, s, judge, emb)
    d = decisions[0]
    assert d.decision == "link_existing" and d.topic_id == "a" and d.strength == 0.80 and d.connection_type == "supports"
    assert d.diagnostics["chosen_channel"] == "vector" and d.diagnostics["chosen_vector_rank"] == 1
    assert d.diagnostics["input_tokens"] and d.diagnostics["confirmation"] is None
    system, user = judge.calls[0]
    assert "follows widget manufacturing" in system and "Topic A" in user and "Source type: rss" in user and "Evidence kind: ships" in user
    assert "A evidence 0" in user, "representative observations are shown"

    result = record([_item()], [_Eval("u1", E_A)], decisions, st, settings=s, embedder=emb)
    assert len(result.observations_created) == 1 and result.topics_created == []
    obs = st.get_observations_for_item("u1")[0]
    assert obs["topic_id"] == "a" and obs["strength"] == 0.80 and obs["tags"] == ["feed", "alpha"] and obs["evidence"] == "ships"
    assert st.get_topic("a")["snapshots"][-1]["evidence_count"] == 4
    assert st.get_item_embeddings(["u1"]) == {"u1": E_A}


class _Metered(MemoryStore):
    """Records each read the hot paths make and the fields it declared."""

    def __init__(self):
        super().__init__()
        self.reads: list[tuple[str, tuple | None]] = []

    def list_topics(self, include_retired=True, *, fields=None):
        self.reads.append(("list_topics", fields))
        return super().list_topics(include_retired, fields=fields)

    def nearest_topics(self, embedding, limit, *, fields=None):
        self.reads.append(("nearest_topics", fields))
        return super().nearest_topics(embedding, limit, fields=fields)

    def get_observations_for_topic(self, topic_id, *, fields=None):
        self.reads.append(("get_observations_for_topic", fields))
        return super().get_observations_for_topic(topic_id, fields=fields)


def test_hot_paths_declare_their_fields_and_list_topics_once_per_batch():
    # The reads that pull whole documents are what exhausts a metered
    # store's transfer allowance (issue #5): every read synthesize and
    # record make of the topic list, the vector candidates and a topic's
    # observations says what it will look at, and the topic list, the
    # largest of them, is read once for the batch, not once per item.
    st, emb, s = _store(_Metered()), HashEmbedder(DIM), _settings()
    items, evals = [_item("u1"), _item("u2")], [_Eval("u1", E_A), _Eval("u2", E_A)]
    decisions = synthesize_items(items, evals, st, s, ScriptedJudge([LINK, LINK]), emb)
    assert [d.topic_id for d in decisions] == ["a", "a"]
    assert [f for m, f in st.reads if m == "list_topics"] == [TOPIC_HEAD_FIELDS], "once, for the batch"
    assert all(f is not None for _, f in st.reads), st.reads

    st.reads.clear()
    result = record(items, evals, decisions, st, settings=s, embedder=emb)
    assert len(result.observations_created) == 2 and st.get_topic("a")["embedding_refreshed_count"] == 5, "the threshold refresh ran"
    assert ("list_topics", QUIET_TOPIC_FIELDS) in st.reads, "the quiet-topic pass"
    assert all(f is not None for _, f in st.reads), st.reads
    assert "A evidence 0" in st.get_observations_for_item("a0")[0]["item_title"], "whole documents are still there to read"


def test_below_threshold_items_are_not_synthesized():
    judge = ScriptedJudge([])
    assert synthesize_items([_item()], [_Eval("u1", E_A, above=False)], _store(), _settings(), judge, HashEmbedder(DIM)) == []
    assert judge.calls == []


def test_propose_confirmed_creates_topic_with_embedding_and_strips_routing_tags():
    st, judge, emb, s = _store(), ScriptedJudge([PROPOSE, CONFIRM]), HashEmbedder(DIM), _settings()
    decisions = synthesize_items([_item()], [_Eval("u1", E_B)], st, s, judge, emb)
    d = decisions[0]
    assert d.decision == "propose_new" and d.new_topic["topic_id"] == "topic-c" and d.diagnostics["confirmation"] == "confirmed_new"
    assert d.diagnostics["confirm_input_tokens"] and len(judge.calls) == 2
    assert "Proposed new topic" in judge.calls[1][1] and "Topic C" in judge.calls[1][1]
    assert emb.calls == [["Topic C\n\nTracks C."]], "the proposed topic was embedded once"

    result = record([_item()], [_Eval("u1", E_B)], decisions, st, settings=s, embedder=emb)
    t = st.get_topic("topic-c")
    assert t and t["tags"] == ["gamma"] and t["embedding"] == d.new_topic["embedding"] and t["status"] == "emerging"
    assert result.topics_created[0]["topic_id"] == "topic-c"
    assert st.get_observations_for_topic("topic-c")[0]["strength"] == 0.85


def test_reviewer_overrides_proposal_to_link():
    st, s = _store(), _settings()
    judge = ScriptedJudge([PROPOSE, {"verdict": "link_existing", "topic_id": "a", "connection_type": "extends",
                                     "closest_candidate": "a", "broadened_summary": "Tracks A, including C.", "reasoning": "a facet"}])
    d = synthesize_items([_item()], [_Eval("u1", E_B)], st, s, judge, HashEmbedder(DIM))[0]
    assert d.decision == "link_existing" and d.topic_id == "a" and d.connection_type == "extends"
    assert d.strength == CONFIRM_LINK_STRENGTH and d.new_topic is None
    assert d.diagnostics["confirmation"] == "overridden_to_link" and "Broadened summary" in d.reasoning_trace


def test_reviewer_link_to_unshown_topic_is_ignored():
    judge = ScriptedJudge([PROPOSE, {"verdict": "link_existing", "topic_id": "nowhere", "reasoning": "?"}])
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), _settings(), judge, HashEmbedder(DIM))[0]
    assert d.decision == "propose_new" and d.diagnostics["confirmation"] == "confirmed_new"


def test_reviewer_skip_records_nothing():
    st, s = _store(), _settings()
    judge = ScriptedJudge([PROPOSE, {"verdict": "skip", "closest_candidate": "a", "reasoning": "not right-sized"}])
    decisions = synthesize_items([_item()], [_Eval("u1", E_B)], st, s, judge, HashEmbedder(DIM))
    assert decisions[0].decision == "skip" and decisions[0].diagnostics["confirmation"] == "overridden_to_skip"
    result = record([_item()], [_Eval("u1", E_B)], decisions, st, settings=s)
    assert result.observations_created == [] and len(st.list_topics()) == 2


def test_reviewer_off_accepts_proposal_without_second_call():
    judge = ScriptedJudge([PROPOSE])
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), _settings(synthesize={"confirm_proposals": False}), judge, HashEmbedder(DIM))[0]
    assert d.decision == "propose_new" and len(judge.calls) == 1 and d.diagnostics["confirmation"] is None


def test_gate_forces_link_when_model_proposes_despite_same_question():
    gated = dict(PROPOSE, candidate_classifications=[{"topic_id": "a", "label": "same_question"}, {"topic_id": "b", "label": "subset_question"}])
    judge = ScriptedJudge([gated])
    d = synthesize_items([_item()], [_Eval("u1", E_A)], _store(), _settings(), judge, HashEmbedder(DIM))[0]
    assert d.decision == "link_existing" and d.topic_id == "a" and d.strength == GATE_OVERRIDE_DEFAULT_STRENGTH
    assert d.diagnostics["gate_override"] is True and len(judge.calls) == 1, "no reviewer once the gate linked"


def test_second_item_can_link_to_a_topic_proposed_earlier_in_the_batch():
    st, emb, s = _store(), HashEmbedder(DIM), _settings()
    link_c = dict(LINK, candidate_classifications=[{"topic_id": "topic-c", "label": "same_question"}],
                  links=[dict(LINK["links"][0], topic_id="topic-c")])
    judge = ScriptedJudge([PROPOSE, CONFIRM, link_c])
    items, evals = [_item("u1"), _item("u2")], [_Eval("u1", E_B), _Eval("u2", E_B)]
    decisions = synthesize_items(items, evals, st, s, judge, emb)
    assert [d.decision for d in decisions] == ["propose_new", "link_existing"]
    assert decisions[1].topic_id == "topic-c" and decisions[1].diagnostics["chosen_channel"] == "batch"
    assert "topic-c" in judge.calls[2][1], "the batch proposal was shown as a candidate"
    record(items, evals, decisions, st, settings=s, embedder=emb)
    assert st.count_observations_for_topic("topic-c") == 2
    assert st.get_topic("topic-c")["snapshots"][-1]["evidence_count"] == 2


def test_decision_model_reviewer_demotes_on_confident_match():
    s = _settings(synthesize={"confirm_backend": "decision", "confirm_link_threshold": 0.7})
    dm = ScriptedDecisionModel({"Topic A": 0.9, "Topic B": 0.2})
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), s, ScriptedJudge([PROPOSE]), HashEmbedder(DIM), decision_model=dm)[0]
    assert d.decision == "link_existing" and d.topic_id == "a" and d.connection_type == "extends" and d.strength == CONFIRM_LINK_STRENGTH
    assert d.diagnostics["confirmation"] == "overridden_to_link" and len(d.diagnostics["decision_scores"]) == 2
    assert {c["topic"]["title"] for c in dm.calls} == {"Topic A", "Topic B"}
    assert all(c["item"]["url"] == "u1" and isinstance(c["representative_titles"], list) for c in dm.calls)


def test_decision_model_reviewer_keeps_proposal_below_threshold_or_when_absent():
    s = _settings(synthesize={"confirm_backend": "decision"})
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), s, ScriptedJudge([PROPOSE]), HashEmbedder(DIM),
                         decision_model=ScriptedDecisionModel({"Topic A": 0.3}))[0]
    assert d.decision == "propose_new" and d.diagnostics["confirmation"] == "confirmed_new"
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), s, ScriptedJudge([PROPOSE]), HashEmbedder(DIM), decision_model=None)[0]
    assert d.decision == "propose_new" and "no candidate scores" in d.reasoning_trace


def test_cold_start_uses_the_cold_start_prompt():
    judge = ScriptedJudge([PROPOSE, CONFIRM])
    st = MemoryStore()
    d = synthesize_items([_item()], [_Eval("u1", E_B)], st, _settings(), judge, HashEmbedder(DIM))[0]
    assert d.decision == "propose_new" and len(judge.calls) == 1, "nothing to review against on an empty store"
    assert judge.calls[0][0] == cold_start_system_prompt(_settings())


def test_leave_one_out_hides_the_items_own_singleton_topic():
    st = MemoryStore()
    st.create_topic({"topic_id": "mine", "title": "Mine", "summary": "founded by u1", "embedding": E_A})
    st.create_observation({"topic_id": "mine", "item_url": "u1", "item_title": "Item u1"})
    st.create_topic({"topic_id": "other", "title": "Other", "summary": "o", "embedding": E_B})
    st.create_observation({"topic_id": "other", "item_url": "u9", "item_title": "Item u9"})
    judge = ScriptedJudge([{"decision": "skip", "candidate_classifications": [], "reasoning_trace": "x"}])
    synthesize_item(_item("u1"), _Eval("u1", E_A), st, _settings(), judge, [], leave_one_out=True)
    user = judge.calls[0][1]
    assert "mine" not in user and "other" in user
    assert user.count("Item u1") == 1 and "Item u9" in user, "the item's own title appears only as the item, never as evidence"


def test_unparseable_judge_output_is_reported_not_recorded(caplog):
    judge = ScriptedJudge(["this is not json at all"])
    decisions = synthesize_items([_item()], [_Eval("u1", E_A)], _store(), _settings(), judge, HashEmbedder(DIM))
    assert len(decisions) == 1 and decisions[0].decision == "failed" and "JSONDecodeError" in decisions[0].error
    assert "Failed to synthesize item" in caplog.text


def test_first_pass_cut_off_at_max_tokens_is_reported_as_truncation_not_a_parse_error():
    # Issue #3: with vector_limit candidates each classified and questioned,
    # the first-pass JSON outgrew the cap and surfaced as a parse error.
    cut = Completion(text='{\n  "candidate_classifications": [\n    {\n      "topic_id": "a",\n      "candidate_que',
                     input_tokens=900, output_tokens=2048, truncated=True)
    st = _store()
    decisions = synthesize_items([_item()], [_Eval("u1", E_A)], st, _settings(), ScriptedJudge([cut]), HashEmbedder(DIM))
    d = decisions[0]
    assert d.decision == "failed" and d.error == "JudgeTruncated: judge reply for the first-pass decision was truncated at 2048 output tokens; raise settings judge.first_pass_max_tokens"
    assert "JSONDecodeError" not in d.error and st.get_observations_for_item("u1") == []


def test_judge_calls_leave_room_for_every_shown_candidate():
    caps = []

    class _CapJudge(ScriptedJudge):
        def complete(self, system, user, max_tokens=2048):
            caps.append(max_tokens)
            return super().complete(system, user, max_tokens)

    synthesize_items([_item()], [_Eval("u1", E_B)], _store(), _settings(), _CapJudge([PROPOSE, CONFIRM]), HashEmbedder(DIM))
    first_pass, confirm = caps
    assert first_pass >= 8192, "30 candidates each with a candidate_question ran past 2048"
    assert confirm >= 2048

    caps.clear()
    raised = _settings(judge={"first_pass_max_tokens": 32000, "confirm_max_tokens": 5000})
    synthesize_items([_item()], [_Eval("u1", E_B)], _store(), raised, _CapJudge([PROPOSE, CONFIRM]), HashEmbedder(DIM))
    assert caps == [32000, 5000], "a feeder raises a cap in settings, no judge subclass needed"
    with pytest.raises(ValueError, match="unknown settings under 'judge'"):
        _settings(judge={"first_pass": 32000})


def test_truncated_reviewer_keeps_the_proposal_like_a_garbled_one(caplog):
    cut = Completion(text='{"verdict": "link_existing", "topic_id": "a", "reas', truncated=True)
    d = synthesize_items([_item()], [_Eval("u1", E_B)], _store(), _settings(), ScriptedJudge([PROPOSE, cut]), HashEmbedder(DIM))[0]
    assert d.decision == "propose_new" and d.diagnostics["confirmation"] == "confirmed_new"
    assert "Confirmation pass was truncated" in caplog.text


def test_record_status_transition_and_embedding_refresh():
    st, emb, s = _store(), HashEmbedder(DIM), _settings()
    for i in range(3):
        st.save_item_embedding(f"a{i}", E_A)
    st.create_observation({"topic_id": "a", "item_url": "a3", "item_title": "A 3", "strength": 0.9})
    st.save_item_embedding("a3", E_A)
    d = SynthesisDecision(item_url="u1", item_title="Item", decision="link_existing", topic_id="a",
                          connection_type="supports", strength=0.8, reasoning_trace="")
    result = record([_item()], [_Eval("u1", E_A)], [d], st, settings=s, embedder=emb)
    assert result.transitions == [{"topic_id": "a", "title": "Topic A", "from_status": "emerging", "to_status": "active"}]
    assert st.get_topic("a")["status"] == "active"
    assert emb.calls == [["Topic A\n\nTracks A."]], "crossing 5 observations refreshed the embedding"
    assert abs(sum(x * x for x in st.get_topic("a")["embedding"]) - 1.0) < 1e-5


def test_record_marks_weak_links_minor_on_well_evidenced_topics():
    st, s = MemoryStore(), _settings()
    st.create_topic({"topic_id": "big", "title": "Big", "embedding": E_A})
    for i in range(20):
        st.create_observation({"topic_id": "big", "item_url": f"b{i}", "strength": 0.9})
    d = SynthesisDecision(item_url="u1", item_title="Item", decision="link_existing", topic_id="big",
                          connection_type="supports", strength=0.5, reasoning_trace="")
    record([_item()], [_Eval("u1", E_A)], [d], st, settings=s)
    assert st.get_observations_for_item("u1")[0]["minor"] is True


# ── curation on a populated store ──

def test_merge_topics_rewrites_target_and_retires_source():
    from topicsleuth.merge import merge_topics, suggest_merges
    st, emb = _store(), HashEmbedder(DIM, vectors={"Merged\n\nAB.": E_A})
    st.create_observation({"topic_id": "b", "item_url": "b0", "item_title": "B evidence", "strength": 0.7})
    judge = ScriptedJudge([{"title": "Merged", "summary": "AB.", "tags": ["alpha", "beta"], "suggested_directions": ["d"], "rationale": "same question"}] * 2)
    dry = merge_topics("b", "a", st, _settings(), judge, emb, dry_run=True)
    assert dry["dry_run"] and dry["merged_title"] == "Merged" and st.get_topic("b")["status"] == "emerging"
    out = merge_topics("b", "a", st, _settings(), judge, emb)
    assert out["observations_reassigned"] == 1 and st.count_observations_for_topic("a") == 4
    moved = st.get_observations_for_item("b0")[0]
    assert moved["topic_id"] == "a" and moved["moved_from"] == "b" and "MOVED by merge" in moved["reasoning_trace"]
    a, b = st.get_topic("a"), st.get_topic("b")
    assert a["title"] == "Merged" and a["embedding"] == E_A and a["snapshots"][-1]["evidence_count"] == 4
    assert b["status"] == "retired" and b["merged_into"] == "a"
    with pytest.raises(ValueError):
        merge_topics("b", "a", st, _settings(), judge, emb)
    assert suggest_merges(st, threshold=0.5) == [], "a retired topic is never suggested"


def test_suggest_merges_pairs_near_duplicates():
    from topicsleuth.merge import suggest_merges
    st = MemoryStore()
    st.create_topic({"topic_id": "x", "embedding": E_A, "tags": ["t"]})
    st.create_topic({"topic_id": "y", "embedding": [0.99, 0.1, 0, 0, 0, 0, 0, 0], "tags": ["t"]})
    st.create_topic({"topic_id": "z", "embedding": E_B, "tags": ["t"]})
    st.create_observation({"topic_id": "y", "item_url": "u"})
    out = suggest_merges(st, threshold=0.9, min_tag_overlap=1)
    assert len(out) == 1 and out[0]["source_topic_id"] == "x" and out[0]["target_topic_id"] == "y"


def test_split_topic_creates_subtopics_and_retires_parent():
    from topicsleuth.split import split_topic
    st, emb = _store(), HashEmbedder(DIM)
    subs = {"sub_topics": [{"topic_id": "A One", "title": "A one", "summary": "s1", "tags": ["alpha"], "suggested_directions": []},
                           {"topic_id": "a-two", "title": "A two", "summary": "s2", "tags": [], "suggested_directions": []}]}
    yes, no = True, False
    before = {o["item_url"]: o for o in st.get_observations_for_topic("a")}
    reassign = [{"item_url": "a0", "new_topic_id": "a-one", "new_connection_type": "supports", "reasoning": "r",
                 "fit_questions": {"q1_scope_expansion": yes, "q2_representative_citation": yes, "q3_distinguishing_evidence": yes}},
                {"item_url": "a1", "new_topic_id": "a-two", "new_connection_type": "extends", "reasoning": "r",
                 "fit_questions": {"q1_scope_expansion": yes, "q2_representative_citation": no, "q3_distinguishing_evidence": no}},
                {"item_url": "a2", "new_topic_id": "a-two", "new_connection_type": "supports", "reasoning": "r",
                 "fit_questions": {"q1_scope_expansion": no, "q2_representative_citation": yes, "q3_distinguishing_evidence": no}}]
    judge = ScriptedJudge([subs, reassign])
    out = split_topic("a", st, _settings(), judge, emb)
    assert out["sub_topics_created"] == ["a-one", "a-two"] and out["observations_reassigned"] == 3
    assert st.get_topic("a")["status"] == "retired" and st.get_topic("a")["split_into"] == ["a-one", "a-two"]
    assert st.count_observations_for_topic("a") == 0 and st.count_observations_for_topic("a-one") == 1 and st.count_observations_for_topic("a-two") == 2
    moved = st.get_observations_for_topic("a-one")[0]
    assert moved["_id"] == before["a0"]["_id"] and moved["date"] == before["a0"]["date"], "the observation is moved, not replaced"
    assert moved.get("tags") == before["a0"].get("tags") and moved["moved_from"] == "a" and "MOVED by split-topic" in moved["reasoning_trace"]
    assert moved["strength"] == 0.95 and moved["fit_questions"]["q2_representative_citation"] is True
    by_url = {o["item_url"]: o for o in st.get_observations_for_topic("a-two")}
    assert by_url["a1"]["strength"] == 0.5 and by_url["a2"]["strength"] == 0.65, "derived from the fit questions, not emitted"
    assert st.get_topic("a-one")["snapshots"][-1]["evidence_count"] == 1
    assert "widget manufacturing" in judge.calls[0][0] and "widget manufacturing" in judge.calls[1][0]
    assert len(emb.calls) == 2


def test_split_rejects_an_incomplete_or_malformed_plan_before_writing():
    from topicsleuth.split import SplitPlanError, split_topic, validate_split_plan
    st, emb = _store(), HashEmbedder(DIM)
    subs = {"sub_topics": [{"topic_id": "a-one", "title": "A one", "summary": "s1"},
                           {"topic_id": "a-two", "title": "A two", "summary": "s2"}]}
    partial = [{"item_url": "a0", "new_topic_id": "a-one"}, {"item_url": "a1", "new_topic_id": "nowhere"},
               {"item_url": "a1", "new_topic_id": "a-two"}, {"item_url": "ghost", "new_topic_id": "a-one"}]
    plan = split_topic("a", st, _settings(), ScriptedJudge([subs, partial]), emb, dry_run=True)
    problems = "\n".join(plan["problems"])
    assert "'a1' is assigned to unknown sub-topic 'nowhere'" in problems and "'a1' is assigned 2 times" in problems
    assert "not hold: 'ghost'" in problems and "1 observation(s) not assigned: ['a2']" in problems
    with pytest.raises(SplitPlanError):
        split_topic("a", st, _settings(), ScriptedJudge([subs, partial]), emb)
    assert st.count_observations_for_topic("a") == 3 and st.get_topic("a")["status"] == "emerging", "nothing written"
    assert st.get_topic("a-one") is None and emb.calls == []
    taken = {"sub_topics": [{"topic_id": "b", "title": "B", "summary": "s"}, {"topic_id": "a", "title": "", "summary": "s"}]}
    problems = validate_split_plan("a", st.get_observations_for_topic("a"), taken["sub_topics"], [], st)
    assert any("'b' is already taken" in p for p in problems) and any("reuses the parent" in p for p in problems)
    assert any("has no title" in p for p in problems)


def test_add_topic_and_move_observation_on_a_real_store(tmp_path):
    from topicsleuth.add_topic import add_topic, load_topic_definition, move_observation
    st, emb = _store(), HashEmbedder(DIM)
    p = tmp_path / "t.yaml"
    p.write_text("title: Topic D\nsummary: Tracks D.\ntags: [delta]\nlink: [a0, missing]\n")
    report = add_topic(load_topic_definition(p), st, emb)
    assert report["topic_id"] == "topic-d" and report["observations_moved"] == 1 and report["urls_without_observation"] == ["missing"]
    assert st.get_topic("topic-d")["added_manually"] is True and st.count_observations_for_topic("a") == 2
    moved = st.get_observations_for_item("a0")[0]
    assert moved["topic_id"] == "topic-d" and moved["moved_from"] == "a" and "MOVED by add-topic" in moved["reasoning_trace"]
    out = move_observation("a1", "b", st)
    assert out["sources_retired"] == [] and st.count_observations_for_topic("b") == 1


def test_reprocess_tail_removes_weak_topics_and_reports_items():
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess
    st = _store()
    st.create_observation({"topic_id": "b", "item_url": "b0"})
    plan = plan_reprocess(st, max_obs=1)
    assert [t["topic_id"] for t in plan.weak_topics] == ["b"] and plan.item_urls == ["b0"]
    out = execute_reprocess(st, plan)
    assert out == {"topics_deleted": 1, "observations_deleted": 1, "topics_skipped": [], "item_urls": ["b0"]}
    assert st.get_topic("b") is None and st.get_topic("a") is not None


def test_refresh_topic_embedding_blends_centroid_and_seed():
    from topicsleuth.embeddings import refresh_topic_embedding
    st, emb, s = _store(), HashEmbedder(DIM, vectors={"Topic A\n\nTracks A.": E_B}), _settings()
    assert refresh_topic_embedding("a", st, s, emb) is False, "no item embeddings yet"
    for i in range(3):
        st.save_item_embedding(f"a{i}", E_A)
    assert refresh_topic_embedding("a", st, s, emb) is True
    e = st.get_topic("a")["embedding"]
    assert e[0] > e[1] > 0, "centroid (A) dominates, seed (B) present"
    assert refresh_topic_embedding("missing", st, s, emb) is False


def test_audit_report_counts_the_store():
    from topicsleuth.audit import build_audit_report, write_audit_report
    st = _store()
    text = build_audit_report(st)
    assert "Total: 2" in text and "Topic A" in text and "0.8-0.9" in text
    import os
    cwd = os.getcwd()
    try:
        import tempfile
        d = tempfile.mkdtemp(); os.chdir(d)
        path = write_audit_report(st, label="t")
        assert path.name == "audit-t.md" and path.read_text().startswith("# Store state")
    finally:
        os.chdir(cwd)


def test_replay_harness_reproduces_a_prior_report(tmp_path):
    from topicsleuth.tune import replay_synthesis
    st, emb, s = _store(), HashEmbedder(DIM), _settings()
    st.items["u1"] = {"url": "u1", "title": "Item u1", "summary": "About A.", "embedding": E_A, "tags": ["alpha"]}
    st.items["u2"] = {"url": "u2", "title": "Item u2", "summary": "About B.", "embedding": E_B}
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"decisions": [{"item_url": "u1"}, {"item_url": "u2"}, {"item_url": "gone"}]}))
    judge = ScriptedJudge([LINK, PROPOSE, CONFIRM])
    out = tmp_path / "out.json"
    report = replay_synthesis(st, judge, emb, settings=s, replay_from=prior, output_path=out, progress=False, label="t")
    assert report["sample_size"] == 2 and report["summary"]["decisions"] == {"link_existing": 1, "propose_new": 1, "skip": 0}
    assert report["config"]["model"] == "scripted" and report["topic_universe"] == 2 and report["replay_from"] == str(prior)
    assert report["summary"]["diagnostics"]["chosen_channel"] == {"vector": 1}
    assert json.loads(out.read_text())["label"] == "t"
    assert len(st.list_topics()) == 2, "a replay writes nothing to the store"
    with pytest.raises(ValueError, match="cannot sample"):
        replay_synthesis(st, judge, emb, settings=s, sample_query={}, progress=False)
