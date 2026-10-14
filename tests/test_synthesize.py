"""Hermetic unit tests for topicsleuth.synthesize.

These cover the pure functions that make up the calibration / decision
pipeline: strength derivation, JSON recovery, topic_id sanitization, the
link-gate override, and end-to-end response parsing.
"""
from __future__ import annotations

import json

import pytest

from topicsleuth.rubric import derive_strength
from topicsleuth.synthesize import (
    _enforce_link_gate,
    _extract_json_object,
    _normalize_json_text,
    _parse_decision,
    _robust_json_parse,
    _sanitize_topic_id,
    GATE_OVERRIDE_DEFAULT_STRENGTH,
)


class TestDeriveStrength:
    """The 3-question rubric should map to the documented strength table."""

    def test_all_three_yes_returns_top_band(self):
        assert derive_strength({
            "q1_scope_expansion": True,
            "q2_representative_citation": True,
            "q3_distinguishing_evidence": True,
        }) == 0.95

    def test_q2_plus_q1_returns_strong_band(self):
        assert derive_strength({
            "q1_scope_expansion": True,
            "q2_representative_citation": True,
            "q3_distinguishing_evidence": False,
        }) == 0.80

    def test_q2_plus_q3_returns_strong_band(self):
        assert derive_strength({
            "q1_scope_expansion": False,
            "q2_representative_citation": True,
            "q3_distinguishing_evidence": True,
        }) == 0.80

    def test_q2_only_returns_mid_band(self):
        assert derive_strength({
            "q1_scope_expansion": False,
            "q2_representative_citation": True,
            "q3_distinguishing_evidence": False,
        }) == 0.65

    def test_q1_alone_returns_low_band(self):
        # Q2=no but at least one other yes.
        assert derive_strength({
            "q1_scope_expansion": True,
            "q2_representative_citation": False,
            "q3_distinguishing_evidence": False,
        }) == 0.50

    def test_q3_alone_returns_low_band(self):
        assert derive_strength({
            "q1_scope_expansion": False,
            "q2_representative_citation": False,
            "q3_distinguishing_evidence": True,
        }) == 0.50

    def test_all_three_no_returns_floor(self):
        assert derive_strength({
            "q1_scope_expansion": False,
            "q2_representative_citation": False,
            "q3_distinguishing_evidence": False,
        }) == 0.30

    def test_none_input_returns_none(self):
        assert derive_strength(None) is None

    def test_empty_dict_returns_none(self):
        assert derive_strength({}) is None

    def test_missing_keys_returns_none(self):
        assert derive_strength({"q1_scope_expansion": True}) is None

    def test_non_dict_returns_none(self):
        assert derive_strength("yes") is None
        assert derive_strength(["q1", "q2", "q3"]) is None

    def test_truthy_values_treated_as_yes(self):
        # The function calls bool() on each value, so non-boolean truthy
        # values should still produce the all-three-yes outcome.
        assert derive_strength({
            "q1_scope_expansion": 1,
            "q2_representative_citation": "yes",
            "q3_distinguishing_evidence": [1],
        }) == 0.95


class TestSanitizeTopicId:
    def test_lowercases(self):
        assert _sanitize_topic_id("Foo Bar") == "foo-bar"

    def test_replaces_special_chars_with_hyphens(self):
        assert _sanitize_topic_id("foo.bar/baz") == "foo-bar-baz"

    def test_collapses_multiple_hyphens(self):
        assert _sanitize_topic_id("foo--bar---baz") == "foo-bar-baz"

    def test_strips_leading_trailing_hyphens(self):
        assert _sanitize_topic_id("-foo-bar-") == "foo-bar"

    def test_preserves_alphanumeric_and_hyphens(self):
        assert _sanitize_topic_id("foo-bar-123") == "foo-bar-123"

    def test_strips_whitespace(self):
        assert _sanitize_topic_id("  foo-bar  ") == "foo-bar"


class TestNormalizeJsonText:
    def test_smart_quotes_replaced(self):
        assert '"' in _normalize_json_text("“key”")

    def test_python_literals_replaced(self):
        assert _normalize_json_text(': True') == ': true'
        assert _normalize_json_text(': False') == ': false'
        assert _normalize_json_text(': None') == ': null'

    def test_trailing_comma_in_object_removed(self):
        assert _normalize_json_text('{"a": 1,}') == '{"a": 1}'

    def test_trailing_comma_in_array_removed(self):
        assert _normalize_json_text('[1, 2, 3,]') == '[1, 2, 3]'

    def test_no_op_on_clean_input(self):
        assert _normalize_json_text('{"a": 1}') == '{"a": 1}'


class TestExtractJsonObject:
    def test_extracts_top_level_object(self):
        assert _extract_json_object('prefix {"a": 1} suffix') == '{"a": 1}'

    def test_handles_nested_objects(self):
        assert _extract_json_object('{"a": {"b": 2}}') == '{"a": {"b": 2}}'

    def test_ignores_braces_in_strings(self):
        # The brace-counter must skip over } inside string values.
        result = _extract_json_object('{"key": "value with } brace"}')
        assert result == '{"key": "value with } brace"}'

    def test_returns_none_when_no_object(self):
        assert _extract_json_object("not an object") is None


class TestRobustJsonParse:
    def test_plain_json(self):
        assert _robust_json_parse('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self):
        assert _robust_json_parse('```json\n{"a": 1}\n```') == {"a": 1}

    def test_handles_trailing_comma(self):
        assert _robust_json_parse('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_handles_python_literals(self):
        assert _robust_json_parse(
            '{"a": True, "b": False, "c": None}'
        ) == {"a": True, "b": False, "c": None}

    def test_handles_smart_quotes(self):
        assert _robust_json_parse("{“a”: 1}") == {"a": 1}

    def test_extracts_from_prose_wrapping(self):
        assert _robust_json_parse(
            "Here is the response:\n{\"a\": 1}\nDone."
        ) == {"a": 1}

    def test_handles_newline_in_string(self):
        # strict=False allows literal control chars inside string values.
        assert _robust_json_parse('{"a": "line1\nline2"}') == {
            "a": "line1\nline2"
        }

    def test_handles_nested_objects(self):
        assert _robust_json_parse(
            '{"outer": {"inner": "value"}}'
        ) == {"outer": {"inner": "value"}}

    def test_raises_on_garbage(self):
        with pytest.raises(json.JSONDecodeError):
            _robust_json_parse("not json at all")


class TestEnforceLinkGate:
    """Code-level override: propose_new -> link_existing when subset/same
    matches exist in the candidate classifications."""

    def test_overrides_when_subset_match_exists(self):
        parsed = {
            "decision": "propose_new",
            "topic_id": None,
            "reasoning_trace": "novel",
            "candidate_classifications": [
                {"topic_id": "memory-arch", "label": "subset_question"},
            ],
            "new_topic": {"topic_id": "comprehensive-memory"},
        }
        candidates = [
            {"topic_id": "memory-arch", "snapshots": [{"evidence_count": 5}]},
        ]
        result = _enforce_link_gate(parsed, candidates)
        assert result["decision"] == "link_existing"
        assert [l["topic_id"] for l in result["links"]] == ["memory-arch"]
        assert result["new_topic"] is None
        assert result["gate_override"] is True
        assert "GATE OVERRIDE" in result["reasoning_trace"]

    def test_overrides_when_same_question_match_exists(self):
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                {"topic_id": "exact", "label": "same_question"},
            ],
            "new_topic": {"topic_id": "x"},
        }
        candidates = [
            {"topic_id": "exact", "snapshots": [{"evidence_count": 3}]},
        ]
        result = _enforce_link_gate(parsed, candidates)
        assert result["decision"] == "link_existing"
        assert [l["topic_id"] for l in result["links"]] == ["exact"]

    def test_prefers_same_question_over_subset(self):
        # When both labels are present, same_question wins regardless of
        # evidence count.
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                {"topic_id": "subset-with-many", "label": "subset_question"},
                {"topic_id": "same-with-few", "label": "same_question"},
            ],
            "new_topic": {"topic_id": "x"},
        }
        candidates = [
            {"topic_id": "subset-with-many", "snapshots": [{"evidence_count": 50}]},
            {"topic_id": "same-with-few", "snapshots": [{"evidence_count": 2}]},
        ]
        result = _enforce_link_gate(parsed, candidates)
        assert [l["topic_id"] for l in result["links"]] == ["same-with-few"], "subset matches are not linked when a same_question exists"

    def test_links_every_match_within_the_label_most_evidence_first(self):
        # With multiple subset matches, the item is evidence for each.
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                {"topic_id": "small", "label": "subset_question"},
                {"topic_id": "large", "label": "subset_question"},
            ],
            "new_topic": {"topic_id": "x"},
        }
        candidates = [
            {"topic_id": "small", "snapshots": [{"evidence_count": 3}]},
            {"topic_id": "large", "snapshots": [{"evidence_count": 30}]},
        ]
        result = _enforce_link_gate(parsed, candidates)
        assert [l["topic_id"] for l in result["links"]] == ["large", "small"]
        assert all(l["strength"] == GATE_OVERRIDE_DEFAULT_STRENGTH for l in result["links"])

    def test_no_override_when_only_adjacent_or_surface(self):
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                {"topic_id": "a", "label": "adjacent_question"},
                {"topic_id": "b", "label": "surface_overlap"},
            ],
            "new_topic": {"topic_id": "genuinely-new"},
        }
        result = _enforce_link_gate(parsed, [])
        assert result["decision"] == "propose_new"
        assert not result.get("gate_override")

    def test_no_override_when_no_classifications(self):
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [],
            "new_topic": {"topic_id": "fresh"},
        }
        result = _enforce_link_gate(parsed, [])
        assert result["decision"] == "propose_new"

    def test_no_override_for_link_existing(self):
        parsed = {
            "decision": "link_existing",
            "links": [{"topic_id": "foo", "connection_type": "supports", "fit_questions": None, "strength": 0.8}],
            "reasoning_trace": "good link",
            "candidate_classifications": [
                {"topic_id": "foo", "label": "same_question"}
            ],
            "new_topic": None,
        }
        result = _enforce_link_gate(parsed, [])
        assert result["decision"] == "link_existing"
        assert result["links"][0]["topic_id"] == "foo"
        assert not result.get("gate_override")

    def test_no_override_for_skip(self):
        parsed = {
            "decision": "skip",
            "reasoning_trace": "",
            "candidate_classifications": [],
        }
        result = _enforce_link_gate(parsed, [])
        assert result["decision"] == "skip"

    def test_default_strength_set_on_override(self):
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                {"topic_id": "foo", "label": "subset_question"},
            ],
            "new_topic": {"topic_id": "x"},
        }
        candidates = [{"topic_id": "foo", "snapshots": [{"evidence_count": 5}]}]
        result = _enforce_link_gate(parsed, candidates)
        assert result["strength"] == GATE_OVERRIDE_DEFAULT_STRENGTH
        assert result["links"][0]["connection_type"] == "supports"

    def test_skips_invalid_classification_entries(self):
        # Defensive: classifications list might contain non-dict entries.
        parsed = {
            "decision": "propose_new",
            "reasoning_trace": "",
            "candidate_classifications": [
                "not a dict",
                {"topic_id": "valid", "label": "subset_question"},
            ],
            "new_topic": {"topic_id": "x"},
        }
        candidates = [{"topic_id": "valid", "snapshots": [{"evidence_count": 4}]}]
        result = _enforce_link_gate(parsed, candidates)
        assert result["decision"] == "link_existing"
        assert result["links"][0]["topic_id"] == "valid"


class TestParseDecisionEndToEnd:
    """End-to-end parse: JSON -> sanitize -> derive strength -> fold trace."""

    def test_link_existing_with_fit_questions_derives_strength(self):
        raw = json.dumps({
            "candidate_classifications": [
                {"topic_id": "a", "label": "same_question",
                 "candidate_question": "Q1", "item_question": "Q2"}
            ],
            "decision": "link_existing",
            "links": [
                {"topic_id": "a", "connection_type": "supports",
                 "fit_questions": {"q1_scope_expansion": False, "q2_representative_citation": True, "q3_distinguishing_evidence": True}},
                {"topic_id": "B Two", "connection_type": "extends",
                 "fit_questions": {"q1_scope_expansion": True, "q2_representative_citation": True, "q3_distinguishing_evidence": True}},
            ],
            "reasoning_trace": "base trace",
        })
        out = _parse_decision(raw)
        assert out["decision"] == "link_existing"
        assert [(l["topic_id"], l["strength"]) for l in out["links"]] == [("a", 0.80), ("b-two", 0.95)], "scored per link, ids sanitized"
        assert out["strength"] == 0.80, "the first link's, for readers that want one number"
        assert "Candidate classifications:" in out["reasoning_trace"]
        assert "Fit questions for a:" in out["reasoning_trace"] and "Fit questions for b-two:" in out["reasoning_trace"]
        assert "derived_strength: 0.8" in out["reasoning_trace"]

    def test_propose_new_uses_default_strength(self):
        raw = json.dumps({
            "candidate_classifications": [],
            "decision": "propose_new",
            "topic_id": None,
            "reasoning_trace": "novel direction",
            "new_topic": {
                "title": "Foo",
                "summary": "A topic",
                "topic_id": "foo",
                "tags": [],
                "suggested_directions": [],
                "scope_test_question": "What is X?",
                "differentiation_from_closest": "none",
            },
        })
        out = _parse_decision(raw)
        assert out["strength"] == 0.85
        assert "Scope test question: What is X?" in out["reasoning_trace"]
        assert "Differentiation from closest: none" in out["reasoning_trace"]

    def test_skip_strength_is_zero(self):
        raw = json.dumps({"decision": "skip", "reasoning_trace": "no fit"})
        out = _parse_decision(raw)
        assert out["strength"] == 0.0

    def test_sanitizes_topic_ids(self):
        raw = json.dumps({
            "decision": "propose_new",
            "new_topic": {
                "topic_id": "Foo.Bar/Baz",
                "title": "T",
                "summary": "S",
            },
            "reasoning_trace": "",
        })
        out = _parse_decision(raw)
        assert out["new_topic"]["topic_id"] == "foo-bar-baz"

    def test_link_without_fit_questions_has_no_strength(self):
        # The parser leaves it unscored; validation then fails the item.
        raw = json.dumps({
            "decision": "link_existing",
            "links": [{"topic_id": "Foo Bar", "connection_type": "supports"}],
            "reasoning_trace": "",
        })
        out = _parse_decision(raw)
        assert out["links"] == [{"topic_id": "foo-bar", "connection_type": "supports", "fit_questions": None, "strength": None}]
        assert out["strength"] == 0.0

    def test_link_without_a_links_list_is_an_empty_link_decision(self):
        out = _parse_decision(json.dumps({"decision": "link_existing", "topic_id": "a", "reasoning_trace": ""}))
        assert out["links"] == [] and "topic_id" not in out.get("links", [{}])[0:0]


from topicsleuth.synthesize import (  # noqa: E402
    CONFIRM_LINK_STRENGTH,
    _apply_confirmation,
    _confirmation_candidates,
)


def _proposal():
    return {
        "decision": "propose_new",
        "topic_id": None,
        "connection_type": None,
        "strength": 0.85,
        "reasoning_trace": "first pass",
        "new_topic": {"topic_id": "new-one", "title": "New", "summary": "S"},
        "candidate_classifications": [
            {"topic_id": "a", "label": "adjacent_question"},
            {"topic_id": "z", "label": "surface_overlap"},
        ],
    }


class TestApplyConfirmation:
    """The second pass may demote a proposal to a link or a skip, never the
    reverse, and may only link to a topic it was actually shown."""

    def test_confirm_new_keeps_proposal(self):
        out = _apply_confirmation(_proposal(), {"verdict": "confirm_new", "reasoning": "ok"}, {"a"})
        assert out["decision"] == "propose_new"
        assert out["new_topic"]["topic_id"] == "new-one"
        assert out["confirmation"] == "confirmed_new"
        assert "CONFIRMATION PASS" in out["reasoning_trace"]

    def test_link_to_shown_candidate(self):
        confirm = {
            "verdict": "link_existing", "topic_id": "a", "connection_type": "extends",
            "broadened_summary": "A, including B.", "reasoning": "facet",
        }
        out = _apply_confirmation(_proposal(), confirm, {"a"})
        assert out["decision"] == "link_existing"
        assert out["links"] == [{"topic_id": "a", "connection_type": "extends", "fit_questions": None, "strength": CONFIRM_LINK_STRENGTH}]
        assert out["strength"] == CONFIRM_LINK_STRENGTH
        assert out["new_topic"] is None
        assert out["confirmation"] == "overridden_to_link"
        assert "Broadened summary: A, including B." in out["reasoning_trace"]

    def test_reviewer_fit_questions_are_ignored(self):
        # The reviewer is not asked for fit questions; if it emits them anyway
        # they must not change the strength, which is fixed by construction.
        confirm = {
            "verdict": "link_existing", "topic_id": "a",
            "fit_questions": {
                "q1_scope_expansion": True,
                "q2_representative_citation": True,
                "q3_distinguishing_evidence": True,
            },
        }
        out = _apply_confirmation(_proposal(), confirm, {"a"})
        assert out["strength"] == CONFIRM_LINK_STRENGTH
        assert out["links"][0]["fit_questions"] is None
        assert out["links"][0]["connection_type"] == "extends"

    def test_link_strength_sits_between_minor_floor_and_core_evidence(self):
        from topicsleuth.settings import RecordConfig
        assert CONFIRM_LINK_STRENGTH >= RecordConfig().minor_floor_strength
        assert CONFIRM_LINK_STRENGTH < 0.80

    def test_link_to_unshown_topic_is_ignored(self):
        out = _apply_confirmation(
            _proposal(), {"verdict": "link_existing", "topic_id": "ghost"}, {"a"}
        )
        assert out["decision"] == "propose_new"
        assert out["confirmation"] == "confirmed_new"
        assert "not among the shown candidates" in out["reasoning_trace"]

    def test_topic_id_is_sanitized_before_matching(self):
        out = _apply_confirmation(
            _proposal(), {"verdict": "link_existing", "topic_id": "A"}, {"a"}
        )
        assert out["decision"] == "link_existing" and out["links"][0]["topic_id"] == "a"

    def test_skip_clears_topic_fields(self):
        out = _apply_confirmation(_proposal(), {"verdict": "skip", "closest_candidate": "a"}, {"a"})
        assert out["decision"] == "skip"
        assert out["links"] == [] and out["new_topic"] is None
        assert out["strength"] == 0.0
        assert out["confirmation"] == "overridden_to_skip"

    def test_unknown_verdict_treated_as_confirm(self):
        out = _apply_confirmation(_proposal(), {"verdict": "maybe"}, {"a"})
        assert out["decision"] == "propose_new"
        assert out["confirmation"] == "confirmed_new"


class TestConfirmationCandidates:
    def test_top_n_vector_plus_classified(self):
        vec = [{"topic_id": t} for t in ("v1", "v2", "v3", "v4")]
        tag = [{"topic_id": "t1"}, {"topic_id": "a"}]
        parsed = {"candidate_classifications": [
            {"topic_id": "v2", "label": "adjacent_question"},
            {"topic_id": "a", "label": "surface_overlap"},
            {"topic_id": "missing", "label": "surface_overlap"},
        ]}
        shown = _confirmation_candidates(parsed, vec, vec + tag, n=2)
        assert [c["topic_id"] for c in shown] == ["v1", "v2", "a"]

    def test_no_classifications(self):
        vec = [{"topic_id": "v1"}]
        assert [c["topic_id"] for c in _confirmation_candidates({}, vec, vec, n=5)] == ["v1"]


from topicsleuth.synthesize import _apply_decision_confirmation  # noqa: E402


class TestApplyDecisionConfirmation:
    def test_best_candidate_above_threshold_links(self):
        scores = [
            {"topic_id": "a", "link_probability": 0.42, "label": "adjacent_question", "confidence": 0.5},
            {"topic_id": "b", "link_probability": 0.81, "label": "subset_question", "confidence": 0.7},
        ]
        out = _apply_decision_confirmation(_proposal(), scores, 0.7)
        assert out["decision"] == "link_existing" and out["links"][0]["topic_id"] == "b"
        assert out["links"][0]["connection_type"] == "extends"
        assert out["strength"] == CONFIRM_LINK_STRENGTH
        assert out["new_topic"] is None
        assert out["confirmation"] == "overridden_to_link"
        assert "CONFIRMATION PASS (decision model)" in out["reasoning_trace"]

    def test_below_threshold_keeps_proposal(self):
        scores = [{"topic_id": "a", "link_probability": 0.69, "label": "subset_question", "confidence": 0.6}]
        out = _apply_decision_confirmation(_proposal(), scores, 0.7)
        assert out["decision"] == "propose_new"
        assert out["confirmation"] == "confirmed_new"

    def test_no_scores_keeps_proposal(self):
        out = _apply_decision_confirmation(_proposal(), [], 0.7)
        assert out["decision"] == "propose_new"
        assert out["confirmation"] == "confirmed_new"

    def test_malformed_scores_are_ignored(self):
        scores = [{"topic_id": "a", "link_probability": None}, {"topic_id": None, "link_probability": 0.9}]
        out = _apply_decision_confirmation(_proposal(), scores, 0.7)
        assert out["decision"] == "propose_new"


from topicsleuth.synthesize import _select_batch_candidates, BATCH_CANDIDATE_LIMIT  # noqa: E402


class TestSelectBatchCandidates:
    def _topics(self):
        return [
            {"topic_id": "far", "embedding": [0.0, 1.0]},
            {"topic_id": "near", "embedding": [1.0, 0.1]},
            {"topic_id": "mid", "embedding": [1.0, 1.0]},
        ]

    def test_ranks_by_cosine_and_caps(self):
        out = _select_batch_candidates([1.0, 0.0], self._topics(), k=2)
        assert [t["topic_id"] for t in out] == ["near", "mid"]
        assert out[0]["vector_score"] > out[1]["vector_score"]

    def test_does_not_mutate_input(self):
        topics = self._topics()
        _select_batch_candidates([1.0, 0.0], topics, k=1)
        assert "vector_score" not in topics[1]

    def test_missing_embeddings_fall_back_to_unranked(self):
        topics = [{"topic_id": "a"}, {"topic_id": "b", "embedding": [1.0, 0.0]}]
        assert _select_batch_candidates([1.0, 0.0], topics, k=1) == topics
        assert _select_batch_candidates(None, self._topics(), k=1) == self._topics()

    def test_empty(self):
        assert _select_batch_candidates([1.0, 0.0], [], k=BATCH_CANDIDATE_LIMIT) == []
