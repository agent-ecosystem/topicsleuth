"""Deterministic backends for tests, yours and ours.

``HashEmbedder`` maps text to a stable unit vector, so similar strings do
not embed similarly but the same string always does and nothing needs a
network. ``ScriptedJudge`` returns canned completions in order, or whatever a
callable decides from the prompt; ``ScriptedDecisionModel`` answers the pair
questions from a table of link probabilities. Together with
``topicsleuth.backends.memory.MemoryStore`` they run the whole decision and
record path offline.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable

from topicsleuth.interfaces import Completion, PairScore


class HashEmbedder:
    """Stable pseudo-embeddings. Pass ``vectors`` to pin specific texts to
    specific vectors when a test needs two items to be neighbours."""

    def __init__(self, dimensions: int = 8, vectors: dict[str, list[float]] | None = None):
        self.model = "hash"
        self.dimensions = dimensions
        self.vectors = dict(vectors or {})
        self.calls: list[list[str]] = []

    def _one(self, text: str) -> list[float]:
        if text in self.vectors:
            return list(self.vectors[text])
        raw: list[float] = []
        block = 0
        while len(raw) < self.dimensions:  # a digest is 32 bytes; chain as many as needed
            digest = hashlib.sha256(f"{block}:{text}".encode()).digest()
            raw.extend((b / 255.0) * 2 - 1 for b in digest)
            block += 1
        raw = raw[: self.dimensions]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._one(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class ScriptedJudge:
    """Returns completions from a script: a list consumed in order, or a
    callable ``(system, user) -> str | dict | Completion``. Dicts are
    JSON-encoded; a ``Completion`` is returned as-is, which is how a test
    scripts a truncated reply. Every call is recorded in ``calls`` as
    ``(system, user)``."""

    def __init__(self, script: Iterable[str | dict] | Callable[[str, str], str | dict] = (), model: str = "scripted"):
        self.model = model
        self._fn = script if callable(script) else None
        self._queue = list(script) if not callable(script) else []
        self.calls: list[tuple[str, str]] = []

    def queue(self, *items: str | dict) -> "ScriptedJudge":
        """Append completions to the script. Returns self for chaining."""
        self._queue.extend(items)
        return self

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> Completion:
        self.calls.append((system, user))
        if self._fn is not None:
            out = self._fn(system, user)
        elif self._queue:
            out = self._queue.pop(0)
        else:
            raise AssertionError(f"ScriptedJudge ran out of script at call {len(self.calls)}")
        if isinstance(out, Completion):
            return out
        text = json.dumps(out) if isinstance(out, (dict, list)) else str(out)
        return Completion(text=text, input_tokens=len(system) + len(user), output_tokens=len(text))


class ScriptedDecisionModel:
    """Scores item-topic pairs from ``link_probabilities``, a map of
    candidate topic title to p(belongs). Unknown titles score 0.1. Each
    call is recorded as ``{item, topic, representative_titles}``."""

    def __init__(self, link_probabilities: dict[str, float] | None = None, model: str = "scripted-decision"):
        self.model = model
        self.link_probabilities = dict(link_probabilities or {})
        self.calls: list[dict] = []

    def score_pair(self, item: dict, topic: dict, representative_titles: list[str]) -> PairScore:
        self.calls.append({"item": item, "topic": topic, "representative_titles": list(representative_titles)})
        p = self.link_probabilities.get(topic.get("title", ""), 0.1)
        label = "same_question" if p >= 0.7 else "adjacent_question"
        return PairScore(
            label=label, link_probability=p, confidence=0.9,
            probabilities={label: p, "surface_overlap": 1 - p},
            input_tokens=100, model=self.model,
        )


# Process-wide instances the CLI picks up when TOPICSLEUTH_STORE=memory,
# TOPICSLEUTH_JUDGE=scripted or TOPICSLEUTH_EMBEDDER=hash, so a test can seed
# a store and queue completions, then run a command in-process.
_shared: dict[str, object] = {}


def shared(kind: str):
    """``shared("store")``, ``shared("judge")``, ``shared("embedder")``,
    ``shared("decision_model")``: the process-wide fake of that kind,
    created on first use."""
    if kind not in _shared:
        if kind == "store":
            from topicsleuth.backends.memory import MemoryStore
            _shared[kind] = MemoryStore()
        elif kind == "judge":
            _shared[kind] = ScriptedJudge()
        elif kind == "embedder":
            _shared[kind] = HashEmbedder()
        elif kind == "decision_model":
            _shared[kind] = ScriptedDecisionModel()
        else:
            raise KeyError(kind)
    return _shared[kind]


def reset_shared() -> None:
    _shared.clear()
