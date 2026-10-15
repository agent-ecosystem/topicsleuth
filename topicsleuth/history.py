"""What an observation carries of its own past and its decision's intent,
and how two observations combine when one is folded into the other.

``removed_from`` is the cumulative list of topics an item was deliberately
taken off; ``decision_links`` is the whole decision the observation was
part of. A fold (one item already evidence on both sides of a move or
merge) keeps one observation and must lose neither: the store computes
``fold_fields`` from the two observations as they are at the moment of
the write, inside the same operation that deletes the folded one, so two
folds into one survivor each see what the other added.
"""
from __future__ import annotations

from topicsleuth.rubric import DEFAULT_CONNECTION_TYPE


def removed_history(observation: dict, *topics: str) -> list[str]:
    """The observation's cumulative ``removed_from`` history with ``topics``
    added: every topic this item was deliberately taken off, in order, so a
    second move does not forget the first."""
    history = list(observation.get("removed_from") or [])
    for t in topics:
        if t and t not in history:
            history.append(t)
    return history


def merged_plan(survivor: dict, *folded: dict) -> list[dict]:
    """The ``decision_links`` the survivor of a fold must carry: its own
    plan, then every link of the folded observations' plans it does not
    already have, each with its own metadata. An observation without a
    plan is its own intent. A link the folded observation held the only
    copy of, approved and not yet written, is not lost with it; whether it
    is still to be written is for ``intended_links`` to decide against
    the merged ``removed_from`` history, so nothing removed on purpose is
    revived by the union."""
    def plan(o: dict) -> list[dict]:
        if o.get("decision_links"):
            return [dict(l) for l in o["decision_links"]]
        if not o.get("topic_id"):
            return []
        return [{"topic_id": o["topic_id"], "connection_type": o.get("connection_type") or DEFAULT_CONNECTION_TYPE,
                 "strength": o.get("strength") if o.get("strength") is not None else 0.0,
                 "fit_questions": o.get("fit_questions")}]
    out: list[dict] = []
    seen: set[str] = set()
    for o in (survivor, *folded):
        for l in plan(o):
            if l.get("topic_id") and l["topic_id"] not in seen:
                seen.add(l["topic_id"])
                out.append(l)
    return out


def fold_fields(survivor: dict, source: dict, note: str) -> dict:
    """The fields the survivor of a fold is written with: its trace with
    ``note`` appended, the union of the two plans, and its history with
    the source's history and the source's own topic added."""
    return {
        "reasoning_trace": (survivor.get("reasoning_trace") or "") + note,
        "decision_links": merged_plan(survivor, source),
        "removed_from": removed_history(survivor, *(source.get("removed_from") or []), source.get("topic_id")),
    }
