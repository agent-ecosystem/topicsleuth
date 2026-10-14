"""Decision settings: everything two feeders writing one topic store must
agree on.

Which services to use (the store, the embedding model, the judge) are
objects the caller constructs; see topicsleuth.interfaces. What lives here
is the embedding dimensionality the store's index was built for, the
candidate assembly rules and the link-or-propose decision controls, because
a store written under two different versions of these is two stores.

Defaults ship with the package (config/settings.yaml). A feeder may layer a
YAML file or a dict of overrides on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from topicsleuth.rubric import RubricConfig

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "config" / "settings.yaml"


@dataclass
class EmbeddingConfig:
    """dimensions: what the store's vector index was built for; an Embedder
    must produce vectors of this length."""
    dimensions: int = 1024


@dataclass
class CandidateConfig:
    """How the synthesize stage assembles candidate topics for an item.

    tag_overlap: include every non-retired topic sharing a tag with the item
        (in addition to the vector-search results).
    ignored_tags: tags excluded from overlap matching. A routing tag that
        every item carries (the tag a feeder selects items by, say) would
        otherwise turn the tag channel into "every topic that ever copied
        that tag"; list such tags here.
    """
    tag_overlap: bool = True
    ignored_tags: list[str] = field(default_factory=list)
    # How many nearest topics the store returns per item, and how wide the
    # approximate search casts its net before ranking (passed to stores that
    # take it, such as Atlas numCandidates).
    vector_limit: int = 30
    vector_num_candidates: int = 200


@dataclass
class SynthesizeConfig:
    """Decision-stage controls.

    confirm_proposals: when the first pass chooses propose_new, run a second,
        independent call that sees only the closest candidates and the
        proposed topic, and must choose between confirming the new topic,
        linking the item as an extension of a candidate, or skipping. A
        wrong link is a slightly noisy observation; a wrong proposal is a
        permanent singleton topic, so proposals have to clear two draws.
    confirm_candidates: how many top vector candidates the confirmation
        call sees (plus any the first pass classified).
    """
    confirm_proposals: bool = True
    confirm_candidates: int = 5
    # "judge": a second call to the Judge reviews the proposal (default).
    # "decision": score each shown candidate with the DecisionModel's
    #        "belongs under this topic" question; demote the proposal to an
    #        extends-link when the best candidate clears confirm_link_threshold.
    #        Falls back to confirming the proposal (never to the judge path)
    #        if no decision model was supplied or a call fails.
    confirm_backend: str = "judge"
    confirm_link_threshold: float = 0.7


@dataclass
class Settings:
    # What the store follows, as a noun phrase the prompts are rendered
    # with: "the coding agent ecosystem", "climate policy", "my industry".
    domain: str = "the field this store follows"
    # The prose in the prompts that describes that field: what kinds of
    # source arrive, what a right-sized question looks like there, the
    # counter-examples. See RubricConfig in topicsleuth.rubric.
    rubric: RubricConfig = field(default_factory=RubricConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    synthesize: SynthesizeConfig = field(default_factory=SynthesizeConfig)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _rubric_from_raw(raw: dict[str, Any]) -> RubricConfig:
    known = {f.name for f in fields(RubricConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown rubric settings: {sorted(unknown)}; known: {sorted(known)}")
    for key in ("evidence_kinds", "question_examples", "counter_examples"):
        if key in raw and not isinstance(raw[key], list):
            raise ValueError(f"rubric.{key} must be a list")
    for ex in raw.get("question_examples", []):
        if not isinstance(ex, dict) or set(ex) != {"question", "subject"}:
            raise ValueError(
                "each rubric.question_examples entry needs exactly the keys "
                f"'question' and 'subject', got {ex!r}"
            )
    return RubricConfig(**raw)


def _from_raw(raw: dict[str, Any]) -> Settings:
    return Settings(
        domain=raw.get("domain", Settings.domain),
        rubric=_rubric_from_raw(raw.get("rubric") or {}),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        candidates=CandidateConfig(**raw.get("candidates", {})),
        synthesize=SynthesizeConfig(**raw.get("synthesize", {})),
    )


def load_settings(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Packaged defaults, then the YAML at ``path`` (if given), then
    ``overrides``, each layered over the last, nested keys merged."""
    with open(DEFAULT_SETTINGS_PATH) as f:
        raw = yaml.safe_load(f) or {}
    if path is not None:
        with open(path) as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _from_raw(raw)


@dataclass
class SeedTopic:
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    suggested_directions: list[str] = field(default_factory=list)


def load_seed_topics(path: Path | str) -> list[SeedTopic]:
    """Read a ``topics:`` list from YAML, the shape ``topicsleuth seed`` takes."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    topics = []
    for t in raw.get("topics", []):
        topics.append(SeedTopic(
            title=t["title"],
            summary=t["summary"],
            tags=t.get("tags", []),
            suggested_directions=t.get("suggested_directions", []),
        ))
    return topics
