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

    Documents are plain dicts. ``topic_id`` identifies a topic;
    observations carry ``topic_id`` and ``item_url``. A store returns its
    own identifier for a created observation.

    Every datetime a store hands back is timezone-aware, in UTC. A store
    given a naive datetime treats it as UTC. The library does its date
    arithmetic on that contract and never normalizes what a store returns.
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

    def delete_observations_for_topic(self, topic_id: str) -> int: ...

    def reassign_observations(self, source_topic_id: str, target_topic_id: str) -> int: ...

    # ── item embeddings ──
    def save_item_embedding(self, item_url: str, embedding: list[float]) -> None: ...

    def get_item_embeddings(self, item_urls: list[str]) -> dict[str, list[float]]: ...

    def get_items(self, item_urls: list[str]) -> list[dict]: ...

    # ── maintenance ──
    def delete_since(self, cutoff: datetime) -> dict:
        """Remove observations and topics created at or after ``cutoff`` and
        clear snapshots taken since then on older topics. For undoing a bad
        run. Returns counts."""
        ...
