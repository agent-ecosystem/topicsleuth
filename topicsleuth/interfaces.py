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
from typing import Any, Protocol, Sequence, runtime_checkable


# How long a maintenance operation's deletion mark on a topic is honoured
# before another caller may take it over as abandoned.
DELETING_LEASE_SECONDS = 900


class StoreError(RuntimeError):
    """A store could not be reached or refused the operation."""


class DestinationNotLive(StoreError):
    """update_observation or fold_observation was asked to put an
    observation on a topic that is retired, marked deleting, or gone. The
    check and the write are one operation, so a destination leased for
    deletion in between is refused and nothing is written."""


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
    relocate, save_embedding, snapshot, snapshot_rebuild, status,
    embedding_refresh); the item should be retried, not acknowledged."""

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


class JudgeTruncated(RuntimeError):
    """The judge stopped at its output-token cap, so the reply is cut off
    and nothing in it can be trusted as complete. Raised instead of a parse
    error so the cause is visible; it is deterministic for a given input,
    so a retry with the same cap fails the same way."""


class JudgeRefused(RuntimeError):
    """The judge declined to answer: the provider's safety classifier
    stopped the call before any text was produced. Raised instead of an
    index or parse error so the cause is visible. Like truncation it is
    deterministic for a given prompt, so a feeder should park the item
    rather than spend its retry budget on it; a transport error, which a
    retry may fix, never takes this type."""


@dataclass
class Completion:
    """What a judge returned. ``truncated`` is True when the reply was cut
    off at the ``max_tokens`` cap, False when the model finished on its
    own, and None when the backend did not say (fakes may leave it unset).
    ``refused`` is True when the provider declined to answer and ``text``
    is empty, False when it answered, None when the backend did not say.
    Each backend maps its own vocabulary for why generation stopped to
    these bits; the core never sees a provider's string."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool | None = None
    refused: bool | None = None

    def require_complete(self, what: str, setting: str = "max_tokens") -> "Completion":
        """Return self, or raise ``JudgeRefused`` if the judge declined to
        answer, or ``JudgeTruncated`` if the reply was cut off at the cap.
        ``what`` names the call and ``setting`` the cap to raise, for the
        message."""
        if self.refused:
            raise JudgeRefused(f"judge refused to answer the {what}; the prompt trips the provider's input classifier")
        if self.truncated:
            at = f"{self.output_tokens} output tokens" if self.output_tokens is not None else "the output-token cap"
            raise JudgeTruncated(f"judge reply for the {what} was truncated at {at}; raise {setting}")
        return self


@runtime_checkable
class Judge(Protocol):
    """A language model asked for text: the link-or-propose decision, its
    reviewer, and the title and summary rewrites that merge and split make.
    ``model`` is informational, for replay reports. ``complete`` should set
    ``Completion.truncated`` from whatever the provider reports about why
    generation stopped, so callers can tell a reply cut off at the cap from
    a malformed one. The caps callers pass are generous: an output-token cap costs nothing
    until the reply reaches it."""

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

    Several operations below are guarded: they test a condition on the
    topic (that a lease is held, that a destination is live) and write
    only if it holds. The test and the write are one operation, atomic
    with respect to every other caller of the store, whichever threads or
    processes they run in. ``topicsleuth.backends.memory`` is the
    reference: one lock around every operation.

    Three reads take ``fields``: ``list_topics``, ``nearest_topics`` and
    ``get_observations_for_topic``. Given field names, a store returns each
    document with exactly those fields and its identity (``topic_id`` for a
    topic, ``_id`` for an observation; ``nearest_topics`` keeps its
    ``vector_score`` too). A field the document lacks is absent, not None.
    Without ``fields``, whole documents. It is how a caller on a hot path
    says what it reads, so a store that pays per byte transfers no
    embeddings or reasoning traces nobody will look at; a store with no
    native projection filters what it fetched. ``topicsleuth.backends.memory``
    is the reference here too: it returns exactly what was asked for, so a
    caller that reads a field it did not declare fails in the test suite.
    """

    # ── topics ──
    def get_topic(self, topic_id: str) -> dict | None: ...

    def list_topics(self, include_retired: bool = True, *, fields: Sequence[str] | None = None) -> list[dict]: ...

    def create_topic(self, topic_doc: dict) -> str: ...

    def update_topic(self, topic_id: str, updates: dict) -> None:
        """Apply ``updates``. One that sets ``embedding`` is a publication
        of the embedding and bumps ``embedding_revision``."""
        ...

    def publish_topic_embedding(self, topic_id: str, fields: dict, *, expected_revision: int,
                                missing: list[str] = ()) -> bool:
        """Write ``fields`` (an embedding and what describes it) and set
        the topic's refresh obligation to ``missing``, the inputs the
        refresh wanted and could not load (``embedding_refresh_missing``
        becomes exactly that list, ``embedding_refresh_pending`` whether it
        is non-empty), only if the topic's ``embedding_revision`` is still
        ``expected_revision`` (0 for a topic never published to), and bump
        it, as one operation. Returns False, writing nothing, if an
        embedding was published or an obligation registered since the
        caller read that revision: the refresh's view of what is owed is
        only valid for the state it read."""
        ...

    def defer_topic_refresh(self, topic_id: str, missing: list[str]) -> None:
        """Register that the topic owes an embedding refresh: set
        ``embedding_refresh_pending``, add ``missing`` (item urls whose
        embeddings the refresh needs and the store does not have) to
        ``embedding_refresh_missing`` without dropping what is there, and
        bump ``embedding_revision``, as one operation, so a publication in
        flight cannot clear this obligation."""
        ...

    def delete_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> bool:
        """Delete the topic. With ``deleting_owner``, only if the topic is
        marked ``deleting`` by that owner, atomically, so an owner that lost
        its lease cannot finish a deletion someone else undid."""
        ...

    # ── the deletion lease ──
    # A maintenance operation that will delete from a topic takes a lease
    # on it first: the topic is marked ``deleting`` with the owner, an
    # expiry and the status to restore. Recording refuses a leased
    # destination, other maintenance leaves it alone, and every destructive
    # write the owner makes names its lease and is refused once the lease
    # is gone. The three operations below are atomic with respect to one
    # another and to the lease state they test; nothing else writes the
    # lease.

    def acquire_deletion_lease(self, topic_id: str, owner: str, by: str, expires_at: datetime) -> dict | None:
        """Take or renew the lease, atomically: succeeds if the topic has no
        lease, an expired one, or one held by ``owner``. Returns the lease
        (with ``prior_status``, kept from the first acquisition) or None if
        another owner's lease is live or the topic does not exist."""
        ...

    def release_deletion_lease(self, topic_id: str, owner: str) -> bool:
        """Give the topic back with its prior status, atomically, only if
        ``owner`` still holds the lease."""
        ...

    def append_snapshot(self, topic_id: str, snapshot: dict) -> None: ...

    def nearest_topics(self, embedding: list[float], limit: int, *, fields: Sequence[str] | None = None) -> list[dict]:
        """Topics nearest to the embedding, each with a ``vector_score``
        (cosine similarity, higher is closer). May include retired topics;
        callers filter."""
        ...

    def seed_topics_if_empty(self, topic_docs: list[dict]) -> int: ...

    def find_weak_topics(self, max_obs: int, older_than_days: int | None) -> list[dict]: ...

    # ── observations ──
    def create_observation(self, obs_doc: dict) -> Any: ...

    def get_observations_for_topic(self, topic_id: str, *, fields: Sequence[str] | None = None) -> list[dict]: ...

    def count_observations_for_topic(self, topic_id: str) -> int: ...

    def get_observations_for_item(self, item_url: str) -> list[dict]: ...

    def list_observations(self) -> list[dict]: ...

    def update_observation(self, obs_id: Any, updates: dict) -> None:
        """Apply ``updates``. When they move the observation to another
        topic, that topic must be live (not retired, deleting or gone) at
        the moment of the write, atomically, else ``DestinationNotLive``;
        and the new (item_url, topic_id) must be free, else
        ``DuplicateObservation``."""
        ...

    def fold_observation(self, obs_id: Any, into_obs_id: Any, note: str) -> None:
        """Merge observation ``obs_id`` into ``into_obs_id`` (one item,
        evidence on both sides of a move or merge): the survivor is
        written with ``topicsleuth.history.fold_fields`` of the two as
        they are at the moment of the write (``note`` appended to its
        trace, the union of their plans, the source's history and topic
        added to its own) and the other is deleted, as one operation. The
        survivor's topic must be live then, else ``DestinationNotLive``
        and nothing changes; a missing survivor is a ``StoreError``; a
        source already gone is nothing to fold."""
        ...

    def delete_observation(self, obs_id: Any) -> bool: ...

    def delete_observations_for_topic(self, topic_id: str, *, deleting_owner: str | None = None) -> int:
        """Delete the topic's observations. With ``deleting_owner``, only
        while the topic is marked ``deleting`` by that owner, atomically
        with the lease: a lease that changes hands before the delete
        commits means nothing is deleted."""
        ...

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
        ``topics_kept``.

        Every topic the window touches, including an older topic that only
        has an observation or snapshot in it, is leased for the whole of
        the writes made to it and given back, or deleted, under that same
        lease; nothing is written to it after. A topic another operation
        holds is left entirely alone, document, observations and
        snapshots, and counted under ``topics_busy``. Returns counts:
        observations_deleted, topics_deleted, topics_kept, topics_busy,
        snapshots_cleared.

        This removes documents; it does not restore a status, embedding or
        description a run changed."""
        ...
