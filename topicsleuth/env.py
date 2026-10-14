"""Build the default backends from environment variables, for the CLI and
for feeders that are happy with the defaults.

    MONGODB_URI             connection string (required for the store)
    TOPICSLEUTH_DATABASE    database name (default: topicsleuth)
    TOPICSLEUTH_JUDGE_MODEL      default claude-opus-4-8  (ANTHROPIC_API_KEY read by the SDK)
    TOPICSLEUTH_EMBEDDING_MODEL  default voyage-4         (VOYAGE_API_KEY read by the SDK)
    TOPICSLEUTH_DECISION_MODEL   default jev-latest       (TYPESAFE_API_KEY; optional)

For tests: TOPICSLEUTH_STORE=memory, TOPICSLEUTH_JUDGE=scripted,
TOPICSLEUTH_EMBEDDER=hash and TOPICSLEUTH_DECISION=scripted select the
process-wide fakes from topicsleuth.backends.fake instead.

Library code never reads these; only this module and the CLI do.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from topicsleuth.interfaces import DecisionModel, Embedder, Judge, Store
from topicsleuth.settings import Settings


@dataclass
class Backends:
    store: Store
    embedder: Embedder
    judge: Judge
    decision_model: DecisionModel | None = None


def store_from_env(settings: Settings) -> Store:
    if os.environ.get("TOPICSLEUTH_STORE") == "memory":
        from topicsleuth.backends.fake import shared
        return shared("store")
    from topicsleuth.backends.mongo import MongoStore
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    return MongoStore.from_uri(
        uri,
        os.environ.get("TOPICSLEUTH_DATABASE", "topicsleuth"),
        vector_index=os.environ.get("TOPICSLEUTH_VECTOR_INDEX", "topic_embedding_index"),
        num_candidates=settings.candidates.vector_num_candidates,
    )


def embedder_from_env(settings: Settings) -> Embedder:
    if os.environ.get("TOPICSLEUTH_EMBEDDER") == "hash":
        from topicsleuth.backends.fake import shared
        embedder = shared("embedder")
        embedder.dimensions = settings.embedding.dimensions
        return embedder
    from topicsleuth.backends.voyage import VoyageEmbedder
    return VoyageEmbedder(
        model=os.environ.get("TOPICSLEUTH_EMBEDDING_MODEL", "voyage-4"),
        dimensions=settings.embedding.dimensions,
    )


def judge_from_env() -> Judge:
    if os.environ.get("TOPICSLEUTH_JUDGE") == "scripted":
        from topicsleuth.backends.fake import shared
        return shared("judge")
    from topicsleuth.backends.anthropic import AnthropicJudge
    return AnthropicJudge(model=os.environ.get("TOPICSLEUTH_JUDGE_MODEL", "claude-opus-4-8"))


def decision_model_from_env() -> DecisionModel | None:
    if os.environ.get("TOPICSLEUTH_DECISION") == "scripted":
        from topicsleuth.backends.fake import shared
        return shared("decision_model")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        return None
    from topicsleuth.backends.jev import JevDecisionModel
    return JevDecisionModel(key, model=os.environ.get("TOPICSLEUTH_DECISION_MODEL", "jev-latest"))


def backends_from_env(settings: Settings, *, need_judge: bool = True, need_embedder: bool = True) -> Backends:
    return Backends(
        store=store_from_env(settings),
        embedder=embedder_from_env(settings) if need_embedder else None,  # type: ignore[arg-type]
        judge=judge_from_env() if need_judge else None,  # type: ignore[arg-type]
        decision_model=decision_model_from_env(),
    )
