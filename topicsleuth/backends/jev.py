"""TypeSafe Jev as the decision model, and the item-topic pair questions the
proposal reviewer asks it.

Jev scores a fixed option list in one forward pass and returns a probability
distribution plus a confidence, with no text. There is no system prompt; the
rubric is carried in the option descriptions. No SDK: plain HTTPS.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

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

PAIR_QUESTIONS = {
    "relationship": {
        "type": "choice",
        "instructions": (
            "Name the question the item bears on and the question the "
            "candidate topic investigates, then pick the structural relationship between them. "
            "The label is descriptive, not a quality judgment. Weigh the topic's representative "
            "observations alongside its one-sentence summary."
        ),
        "criteria": RELATIONSHIP_CRITERIA,
    },
    "link": {
        "type": "noul",
        "instructions": (
            "This item belongs as evidence under this topic: the topic's question is the "
            "same as, or subsumes, the item's question."
        ),
    },
}


class JevError(RuntimeError):
    pass


class JevDecisionModel:
    """Implements the DecisionModel protocol over the TypeSafe API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, retries: int = 3, timeout: float = 30.0):
        if not api_key:
            raise ValueError("JevDecisionModel needs an API key")
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self.timeout = timeout

    def decide(self, state: dict, questions: dict) -> dict:
        """One call. Returns the parsed response ({model, answers, usage})."""
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise JevError(f"HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise JevError(str(e)) from None
        raise JevError("unreachable")


def pair_state(item: dict, topic: dict, representative_titles: list[str]) -> dict:
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
            "representative_observations": representative_titles,
        },
    }


def score_pair(decision_model, item: dict, topic: dict, representative_titles: list[str]) -> dict:
    """Score one item-topic pair with any DecisionModel. Returns {label,
    probabilities, confidence, link_probability, input_tokens, model}."""
    resp = decision_model.decide(pair_state(item, topic, representative_titles), PAIR_QUESTIONS)
    rel = resp["answers"]["relationship"]
    link = resp["answers"].get("link") or {}
    return {
        "label": rel.get("choice"),
        "probabilities": rel.get("probabilities"),
        "confidence": rel.get("confidence"),
        "link_probability": link.get("noul"),
        "input_tokens": (resp.get("usage") or {}).get("input_tokens"),
        "model": resp.get("model", getattr(decision_model, "model", None)),
    }
