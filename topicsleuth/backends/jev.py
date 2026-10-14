"""TypeSafe Jev as the decision model.

Jev scores a fixed option list in one forward pass and returns a probability
distribution plus a confidence, with no text. There is no system prompt; the
rubric from ``topicsleuth.rubric`` is carried in the option descriptions,
which Jev reads field by field. No SDK: plain HTTPS.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from topicsleuth.interfaces import PairScore
from topicsleuth.rubric import (
    LINK_CRITERION,
    RELATIONSHIP_CRITERIA,
    RELATIONSHIP_INSTRUCTIONS,
    pair_view,
)

logger = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# The library's rubric in Jev's question format: a choice over the
# relationship labels and a yes/no ("noul") on whether the item belongs.
PAIR_QUESTIONS = {
    "relationship": {
        "type": "choice",
        "instructions": RELATIONSHIP_INSTRUCTIONS,
        "criteria": RELATIONSHIP_CRITERIA,
    },
    "link": {
        "type": "noul",
        "instructions": LINK_CRITERION,
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

    def score_pair(self, item: dict, topic: dict, representative_titles: list[str]) -> PairScore:
        resp = self.decide(pair_view(item, topic, representative_titles), PAIR_QUESTIONS)
        answers = resp.get("answers") or {}
        rel = answers.get("relationship") or {}
        link = answers.get("link") or {}
        return PairScore(
            label=rel.get("choice"),
            link_probability=link.get("noul"),
            confidence=rel.get("confidence"),
            probabilities=rel.get("probabilities"),
            input_tokens=(resp.get("usage") or {}).get("input_tokens"),
            model=resp.get("model", self.model),
        )

    def decide(self, state: dict, questions: dict) -> dict:
        """One raw call: Jev's own request and reply shapes."""
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
