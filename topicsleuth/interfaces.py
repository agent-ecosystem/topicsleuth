"""The seams a feeder fills in.

topicsleuth owns the matching, recording and measuring logic. Everything
that talks to a service is behind one of these protocols, and the caller
supplies the implementation: which vector store, which embedding model,
which language model judges, and optionally which decision model reviews
proposals. Default implementations for MongoDB Atlas, Voyage, Anthropic and
TypeSafe Jev live under ``topicsleuth.backends`` behind optional extras;
``topicsleuth.backends.memory`` is an in-memory store for tests.

These are ``typing.Protocol``s: anything with the right methods qualifies,
no subclassing needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


class StoreError(RuntimeError):
    """A store could not be reached or refused the operation."""


class DuplicateTopic(StoreError):
    """create_topic was given a topic_id the store already holds."""


class DuplicateObservation(StoreError):
    """create_observation was given an (item_url, topic_id) the store already
    holds. An item is evidence for a topic once; recording treats this as
    "already recorded", which is what makes a retried batch and two feeders
    admitting one document safe."""


@dataclass
class ItemFailure:
    """One item the pipeline could not finish. ``stage`` names where it
    stopped (synthesize, embed_proposal, create_topic, create_observation,
    save_embedding); the item should be retried, not acknowledged."""

    item_url: str
    stage: str
    error: str


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. ``dimensions`` must match what the store's
    vector index was built for. Models with no document/query asymmetry
    implement both methods the same way."""

    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class Judge(Protocol):
    """A language model asked for text: the link-or-propose decision, its
    reviewer, and the title and summary rewrites that merge and split make.
    ``model`` is informational, for replay reports."""

    model: str

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> Completion: ...


@dataclass
class PairScore:
    """A decision model's reading of one item against one candidate topic.
    ``label`` is one of ``topicsleuth.rubric.RELATIONSHIP_LABELS`` and
    ``link_probability`` is p(the item belongs as evidence under the topic),
    which the proposal reviewer compares with ``confirm_link_threshold``.
    The rest is kept for reports."""

    label: str | None
    link_probability: float | None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    input_tokens: int | None = None
    model: str | None = None


@runtime_checkable
class DecisionModel(Protocol):
    """A decision-only model: scores one item against one candidate topic
    and returns probabilities, no text. It applies the rubric in
    ``topicsleuth.rubric`` in whatever form its API takes. Optional; used
    by the proposal reviewer when settings select the ``decision`` backend.
    ``model`` is informational, for replay reports."""

    model: str

    def score_pair(
        self, item: dict, topic: dict, representative_titles: list[str]
    ) -> PairScore: ...


@runtime_checkable
class Store(Protocol):
    """The topic store. Topics and observations are the library's; the
    only item-side operations are the embedding the store keeps per item
    (written at record time, read back for the centroid refresh) and
    fetching items by URL for replays.

    Documents are plain dicts. ``topic_id`` identifies a topic and is
    unique: ``create_topic`` raises ``DuplicateTopic`` on a collision. An
    observation is identified by ``(item_url, topic_id)``: an item may be
    evidence for several topics, once each. ``create_observation`` raises
    ``DuplicateObservation`` on a collision, atomically, so two writers
    cannot both insert the same evidence, and ``update_observation`` raises
    it if an update would create that pair twice. A store returns its own
    identifier for a created observation.

    Every datetime a store hands back is timezone-aware, in UTC. A store
    given a naive datetime treats it as UTC. The library does its date
    arithmetic on that contract and never normalizes what a store returns.

    A topic's ``snapshots`` are kept in order of their ``date``, whatever
    order they were appended in, so a history load that writes an older
    snapshot after a newer one still reads as a sequence.

    Every topic, observation and snapshot carries ``recorded_at``, when it
    was written, which a store stamps itself if the caller did not. It is
    distinct from ``date`` and ``date_identified``, which are when the
    evidence happened, and it is what ``delete_since`` works from.
    """

    # ── topics ──
    def get_topic(self, topic_id: str) -> dict | None: ...

    def list_topics(self, include_retired: bool = True) -> list[dict]: ...

    def create_topic(self, topic_doc: dict) -> str: ...

    def update_topic(self, topic_id: str, updates: dict) -> None: ...

    def delete_topic(self, topic_id: str) -> bool: ...

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None: ...

    def nearest_topics(self, embedding: list[float], limit: int) -> list[dict]:
        """Topics nearest to the embedding, each with a ``vector_score``
        (cosine similarity, higher is closer). May include retired topics;
        callers filter."""
        ...

    def seed_topics_if_empty(self, topic_docs: list[dict]) -> int: ...

    def find_weak_topics(self, max_obs: int, older_than_days: int | None) -> list[dict]: ...

    # ── observations ──
    def create_observation(self, obs_doc: dict) -> Any: ...

    def get_observations_for_topic(self, topic_id: str) -> list[dict]: ...

    def count_observations_for_topic(self, topic_id: str) -> int: ...

    def get_observations_for_item(self, item_url: str) -> list[dict]: ...

    def list_observations(self) -> list[dict]: ...

    def update_observation(self, obs_id: Any, updates: dict) -> None: ...

    def delete_observation(self, obs_id: Any) -> bool: ...

    def delete_observations_for_topic(self, topic_id: str) -> int: ...

    # ── item embeddings ──
    def save_item_embedding(self, item_url: str, embedding: list[float]) -> None: ...

    def get_item_embeddings(self, item_urls: list[str]) -> dict[str, list[float]]: ...

    def get_items(self, item_urls: list[str]) -> list[dict]: ...

    # ── maintenance ──
    def delete_since(self, cutoff: datetime, run_id: str | None = None) -> dict:
        """Remove what runs wrote at or after ``cutoff``: observations and
        topics by their ``recorded_at`` stamp, and snapshots by theirs. With
        ``run_id``, only documents that run stamped. A topic in the window
        that still holds observations from outside it (evidence moved onto
        it, or recorded earlier) is kept, not deleted, and counted under
        ``topics_kept``. Returns counts: observations_deleted,
        topics_deleted, topics_kept, snapshots_cleared.

        This removes documents; it does not restore a status, embedding or
        description a run changed."""
        ...
