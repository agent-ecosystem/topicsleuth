"""The rubric is the single source: every prompt is assembled from it, every
rendering carries the same labels, questions and types, and the prose that
describes a store's field comes from settings."""
from __future__ import annotations

import pytest

from topicsleuth import rubric
from topicsleuth.rubric import (
    CONNECTION_TYPES,
    DEFAULT_COUNTER_EXAMPLES,
    FIT_QUESTIONS,
    PROMPTS,
    RELATIONSHIP_CRITERIA,
    RELATIONSHIP_LABELS,
    STRENGTH_BANDS,
    RubricConfig,
    derive_strength,
    relationship_rubric_text,
    right_sized_topic,
)
from topicsleuth.settings import Settings, load_settings

DECISION_PROMPTS = ("synthesize", "cold_start")
SCOPE_PROMPTS = ("synthesize", "cold_start", "confirm", "merge", "split_propose")


def _settings(**rubric_over):
    s = Settings()
    s.domain = "widget manufacturing"
    for key, value in rubric_over.items():
        setattr(s.rubric, key, value)
    return s


@pytest.mark.parametrize("name", PROMPTS)
def test_every_prompt_carries_the_domain(name):
    out = PROMPTS[name](_settings())
    assert "widget manufacturing" in out and "{domain}" not in out


@pytest.mark.parametrize("name", DECISION_PROMPTS)
def test_decision_prompts_carry_every_label_question_and_type(name):
    text = PROMPTS[name](_settings())
    for label in RELATIONSHIP_LABELS:
        assert f"`{label}`: {RELATIONSHIP_CRITERIA[label]['what']}" in text
    for key, question in FIT_QUESTIONS.items():
        assert f"`{key}`: {question}" in text and f'"{key}": <bool>' in text
    for ctype, meaning in CONNECTION_TYPES.items():
        assert f"`{ctype}`: {meaning}" in text
    assert '"label": ' + " | ".join(f'"{l}"' for l in RELATIONSHIP_LABELS) in text
    assert '"connection_type": ' + " | ".join(f'"{c}"' for c in CONNECTION_TYPES) + " | null" in text


def test_the_two_decision_prompts_share_their_steps_verbatim():
    s = _settings()
    main, cold = PROMPTS["synthesize"](s), PROMPTS["cold_start"](s)
    for marker in ("## Step 2: Decide", "## Step 3: Strength assessment", "## Step 4: Connection type", "## Response format"):
        section = lambda t: t[t.index(marker):].split("\n## ")[0]
        assert section(main) == section(cold), marker


@pytest.mark.parametrize("name", SCOPE_PROMPTS)
def test_every_prompt_that_judges_scope_carries_the_same_topic_definition(name):
    s = _settings()
    assert right_sized_topic(s.rubric) in PROMPTS[name](s)


def test_split_reassignment_asks_the_fit_questions_not_for_a_number():
    text = PROMPTS["split_reassign"](_settings())
    for key in FIT_QUESTIONS:
        assert f'"{key}": <bool>' in text
    assert "new_strength" not in text
    for ctype in CONNECTION_TYPES:
        assert f'"{ctype}"' in text


def test_reviewer_links_only_as_an_extension():
    text = PROMPTS["confirm"](_settings())
    assert '"connection_type": "extends" | "supports" | null' in text
    assert "contradicts" not in text


def test_relationship_prose_is_rendered_from_the_criteria():
    r = RubricConfig()
    text = relationship_rubric_text(r)
    assert text.count("- `") == len(RELATIONSHIP_LABELS)
    assert "(Example: " + r.subset_example + ")" in text
    assert RELATIONSHIP_CRITERIA["adjacent_question"]["tie_break"] in text
    for label in RELATIONSHIP_LABELS:
        assert "Not for: " + RELATIONSHIP_CRITERIA[label]["not_for"] in text
    assert "(Example:" not in relationship_rubric_text(RubricConfig(subset_example=""))


def test_decision_model_backend_renders_the_same_criteria():
    from topicsleuth.backends.jev import PAIR_QUESTIONS
    assert PAIR_QUESTIONS["relationship"]["criteria"] is RELATIONSHIP_CRITERIA
    assert set(PAIR_QUESTIONS["relationship"]["criteria"]) == set(RELATIONSHIP_LABELS)
    assert PAIR_QUESTIONS["link"]["instructions"] == rubric.LINK_CRITERION


def test_strength_bands_are_ordered_and_derive_strength_uses_them():
    values = list(STRENGTH_BANDS.values())
    assert values == sorted(values, reverse=True)
    q = lambda a, b, c: {"q1_scope_expansion": a, "q2_representative_citation": b, "q3_distinguishing_evidence": c}
    assert derive_strength(q(True, True, True)) == STRENGTH_BANDS["all_three_yes"]
    assert derive_strength(q(True, True, False)) == STRENGTH_BANDS["citable_plus_one"]
    assert derive_strength(q(False, True, False)) == STRENGTH_BANDS["citable_only"]
    assert derive_strength(q(True, False, True)) == STRENGTH_BANDS["not_citable_some"]
    assert derive_strength(q(False, False, False)) == STRENGTH_BANDS["all_three_no"]
    assert derive_strength(None) is None and derive_strength({"q1_scope_expansion": True}) is None


# ── the store's own field ──

def test_the_rubric_block_reaches_every_prompt_that_uses_it():
    s = _settings(
        source_kinds="council minutes, local news, planning filings",
        evidence_kinds=["reports", "proposes", "enacts"],
        reading_by_kind="a filing bears on a question by proposing a change, a minute by recording a vote",
        topic_size="20-80 observations over a year",
        question_examples=[
            {"question": "Are cities loosening parking minimums near transit?", "subject": "parking"},
            {"question": "Is inclusionary zoning producing units?", "subject": "zoning"},
        ],
        subset_example="The candidate asks whether parking minimums are loosening; the item is one city's ordinance.",
        counter_examples=["A council agenda that lists every item of the month: skip, not a meta-topic."],
    )
    main = PROMPTS["synthesize"](s)
    assert "Items arrive from every kind of source: council minutes, local news, planning filings." in main
    assert "evidence kind (`reports`, `proposes`, `enacts`)" in main
    assert "Read the item for what it is: a filing bears on a question by proposing a change, a minute by recording a vote. " in main
    assert "accumulate 20-80 observations over a year without" in main
    assert ('("Are cities loosening parking minimums near transit?" is a question a topic can track. '
            '"parking" is a subject. "Is inclusionary zoning producing units?" is a question; '
            '"zoning" is a subject.)') in main
    assert "(Example: The candidate asks whether parking minimums are loosening; the item is one city's ordinance.)" in main
    assert "- A council agenda that lists every item of the month: skip, not a meta-topic." in main
    for leftover in ("harness vendors", "static analysis", "release notes", "agentic"):
        assert leftover not in main, leftover
    confirm = PROMPTS["confirm"](s)
    assert "Items are council minutes, local news, planning filings;" in confirm
    assert "parking minimums near transit" in confirm
    for name in ("merge", "split_propose"):
        assert "parking minimums near transit" in PROMPTS[name](s)


def test_an_empty_example_list_renders_without_a_parenthetical():
    text = right_sized_topic(RubricConfig(question_examples=[]))
    assert "from different angles.\n- Could plausibly" in text and "(" not in text.split("\n")[1]


def test_the_defaults_are_the_committed_rubric():
    r = RubricConfig()
    assert r.counter_examples == list(DEFAULT_COUNTER_EXAMPLES) and r.counter_examples is not DEFAULT_COUNTER_EXAMPLES
    assert load_settings().rubric == r, "the packaged settings.yaml must not drift from RubricConfig"


def test_the_rubric_block_is_validated_when_loaded():
    assert load_settings(overrides={"rubric": {"topic_size": "a dozen a month"}}).rubric.topic_size == "a dozen a month"
    with pytest.raises(ValueError, match="unknown rubric settings"):
        load_settings(overrides={"rubric": {"source_kind": "x"}})
    with pytest.raises(ValueError, match="must be a list"):
        load_settings(overrides={"rubric": {"counter_examples": "just one"}})
    with pytest.raises(ValueError, match="'question' and 'subject'"):
        load_settings(overrides={"rubric": {"question_examples": [{"question": "q?"}]}})
