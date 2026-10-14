"""The relationship rubric: how an item relates to a candidate topic.

The judge prompts in ``synthesize`` carry this rubric in prose, for a
language model that writes text. This module carries the same rubric as
data, for a decision model that scores fixed options, and for anything
else that needs the label set. One rubric, two renderings, so they cannot
drift apart.
"""
from __future__ import annotations

RELATIONSHIP_LABELS = (
    "same_question",
    "subset_question",
    "adjacent_question",
    "surface_overlap",
)

RELATIONSHIP_INSTRUCTIONS = (
    "Name the question the item bears on and the question the candidate "
    "topic investigates, then pick the structural relationship between them. "
    "The label is descriptive, not a quality judgment. Weigh the topic's "
    "representative observations alongside its one-sentence summary."
)

RELATIONSHIP_CRITERIA = {
    "same_question": {
        "what": "The topic's question is the question this item bears on. The item's primary contribution is an answer or partial answer to it.",
        "not_for": "An item that only touches the topic's subject, or that asks a narrower sub-question (that is subset_question).",
    },
    "subset_question": {
        "what": "The item investigates a specific instance of the topic's question; the topic's question subsumes it. A different method, artifact type, or subdomain inside the same question is still a subset.",
        "example": "Topic asks how agents fail in ways static analysis misses; item proposes a detector for one failure class.",
        "not_for": "An item whose question the topic's summary, read fairly, does not cover (that is adjacent_question).",
    },
    "adjacent_question": {
        "what": "A different question in the same domain. The topic's summary does not cover what the item is about, even though they share keywords.",
        "tie_break": "If broadening the topic summary by one clause would cover the item, prefer subset_question.",
        "not_for": "Pairs that share only tags, model names or subfield with no shared direction (that is surface_overlap).",
    },
    "surface_overlap": {
        "what": "Shared tags, model names or subfield, but the item's question lies elsewhere. A literature review of the topic would not cite this item.",
        "not_for": "Pairs in the same area with a genuinely neighbouring question (that is adjacent_question).",
    },
}

LINK_CRITERION = (
    "This item belongs as evidence under this topic: the topic's question is "
    "the same as, or subsumes, the item's question."
)


def pair_view(item: dict, topic: dict, representative_titles: list[str]) -> dict:
    """What a decision model is shown for one item-topic pair: the item's
    title, summary and an excerpt of its text; the candidate's title,
    summary, evidence count and representative observations. A backend
    renders this into whatever request shape its API takes."""
    return {
        "item": {
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "abstract_excerpt": (item.get("content_raw") or "")[:1800],
        },
        "candidate_topic": {
            "title": topic.get("title", ""),
            "summary": topic.get("summary", ""),
            "observations_so_far": topic.get("evidence_count"),
            "representative_observations": list(representative_titles),
        },
    }
