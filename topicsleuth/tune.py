"""Tuning harness for prompt and config experimentation.

Replays the decision against already-recorded items without writing to
the store. Use this to compare decision distributions, strength calibration, and
new-topic proposal rates across prompt or config changes.

Reads stored item embeddings so the feeder's evaluate stage doesn't have to
be re-run (no embedding spend). Synthesize is replayed live since that's
typically what we're tuning. Items come either from a prior report
(``replay_from``) or from a seeded sample the store draws for a query.

Output is a JSON file under tuning_runs/ for diffing across runs.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topicsleuth.interfaces import DecisionModel, Embedder, Judge, Store
from topicsleuth.settings import Settings, load_settings
from topicsleuth.synthesize import SynthesisDecision, synthesize_item

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("tuning_runs")  # relative to the working directory


@dataclass
class _StoredEval:
    """The store's view of an already-evaluated item (see ItemEval)."""
    url: str
    embedding: list[float]
    above_threshold: bool = True


def _eval_from_item(item: dict) -> _StoredEval | None:
    if not item.get("embedding"):
        return None
    return _StoredEval(url=item["url"], embedding=item["embedding"])


def _library_version() -> str:
    from topicsleuth import __version__
    return __version__


def _replay_items(store: Store, prior_run: Path) -> list[dict]:
    """Load the exact items a prior run decided on, in the same order.

    Lets two runs be diffed item-by-item. Note the topic universe still
    differs between runs, so decisions can legitimately change even with
    identical items and prompts.
    """
    prior = json.loads(Path(prior_run).read_text())
    urls = [d["item_url"] for d in prior.get("decisions", [])]
    found = {i["url"]: i for i in store.get_items(urls)}
    missing = [u for u in urls if u not in found]
    if missing:
        logger.warning("replay: %d of %d prior items not found", len(missing), len(urls))
    return [found[u] for u in urls if u in found]


def _sample_items(store: Store, n: int, seed: int, query: dict) -> list[dict]:
    """A seeded sample of items matching ``query``. A store that can sample
    server-side exposes ``sample_items``; with any other store, replay from
    a prior report."""
    sampler = getattr(store, "sample_items", None)
    if sampler is None:
        raise ValueError("this store cannot sample items; use replay_from")
    return sampler(query, n, seed)


def _decision_to_dict(d: SynthesisDecision, item: dict) -> dict[str, Any]:
    out = {
        "item_url": d.item_url,
        "item_title": d.item_title,
        "item_source": item.get("source_name", ""),
        "decision": d.decision,
        "topic_id": d.topic_id,
        "connection_type": d.connection_type,
        "strength": d.strength,
        "links": [{"topic_id": l.topic_id, "connection_type": l.connection_type, "strength": l.strength} for l in d.links],
        "reasoning_trace": d.reasoning_trace,
    }
    if d.new_topic:
        # Drop the embedding to keep JSON readable
        nt = {k: v for k, v in d.new_topic.items() if k != "embedding"}
        out["new_topic"] = nt
    if d.diagnostics:
        out["diagnostics"] = d.diagnostics
    return out


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def _summarize_diagnostics(decisions: list[dict]) -> dict[str, Any]:
    """Aggregate per-call retrieval facts so runs can be compared on prompt
    size and on whether linked topics came from vector or tag channels."""
    diags = [d["diagnostics"] for d in decisions if d.get("diagnostics")]
    if not diags:
        return {}
    by_decision: dict[str, dict] = {}
    for d in decisions:
        g = d.get("diagnostics")
        if not g:
            continue
        b = by_decision.setdefault(d["decision"], {"n": 0, "top1": [], "labels": Counter()})
        b["n"] += 1
        b["top1"].append(g.get("vector_top1"))
        b["labels"].update(c.get("label") for c in g.get("classified", []))
    return {
        "mean_total_candidates": _mean([g["total_candidates"] for g in diags]),
        "mean_tag_only_candidates": _mean([g["tag_only_candidates"] for g in diags]),
        "mean_input_tokens": _mean([g.get("input_tokens") for g in diags]),
        "mean_output_tokens": _mean([g.get("output_tokens") for g in diags]),
        "mean_classified_per_call": _mean([len(g.get("classified", [])) for g in diags]),
        "chosen_channel": dict(Counter(
            g["chosen_channel"] for g in diags if g.get("chosen_channel")
        )),
        "chosen_vector_rank": dict(Counter(
            g["chosen_vector_rank"] for g in diags if g.get("chosen_vector_rank")
        )),
        "gate_overrides": sum(1 for g in diags if g.get("gate_override")),
        "confirmation": dict(Counter(
            g["confirmation"] for g in diags if g.get("confirmation")
        )),
        "mean_confirm_input_tokens": _mean([g.get("confirm_input_tokens") for g in diags]),
        "by_decision": {
            k: {
                "n": v["n"],
                "mean_vector_top1": _mean(v["top1"]),
                "labels": dict(v["labels"]),
            }
            for k, v in by_decision.items()
        },
    }


def _summarize(decisions: list[dict]) -> dict[str, Any]:
    counts = Counter(d["decision"] for d in decisions)
    total = len(decisions) or 1

    # Every link counts: a decision may put the item on more than one topic.
    all_links = [l for d in decisions if d["decision"] == "link_existing" for l in d["links"]]
    link_decisions = sum(1 for d in decisions if d["decision"] == "link_existing")
    strengths = [l["strength"] or 0 for l in all_links]
    strength_buckets = {
        "<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0,
        "0.7-0.8": 0, "0.8-0.9": 0, "0.9+": 0,
    }
    for s in strengths:
        if s < 0.3: strength_buckets["<0.3"] += 1
        elif s < 0.5: strength_buckets["0.3-0.5"] += 1
        elif s < 0.7: strength_buckets["0.5-0.7"] += 1
        elif s < 0.8: strength_buckets["0.7-0.8"] += 1
        elif s < 0.9: strength_buckets["0.8-0.9"] += 1
        else: strength_buckets["0.9+"] += 1

    connection_types = Counter(l["connection_type"] for l in all_links if l["connection_type"])
    secondary = [l for d in decisions if d["decision"] == "link_existing" for l in d["links"][1:]]
    secondary_buckets = Counter()
    for l in secondary:
        s = l["strength"] or 0
        secondary_buckets["<0.5" if s < 0.5 else "0.5-0.7" if s < 0.7 else "0.7-0.9" if s < 0.9 else "0.9+"] += 1
    links = {
        "total": len(all_links),
        "per_link_decision": round(len(all_links) / link_decisions, 3) if link_decisions else None,
        "multi_link_decisions": sum(1 for d in decisions if d["decision"] == "link_existing" and len(d["links"]) > 1),
        "secondary": len(secondary),
        "secondary_strength_buckets": dict(secondary_buckets),
        "dropped_by_floor": sum(len((d.get("diagnostics") or {}).get("dropped_links") or []) for d in decisions),
    }

    new_topics = [
        {
            "topic_id": d["new_topic"].get("topic_id"),
            "title": d["new_topic"].get("title"),
            "summary": d["new_topic"].get("summary"),
            "scope_test_question": d["new_topic"].get("scope_test_question"),
            "differentiation_from_closest": d["new_topic"].get("differentiation_from_closest"),
        }
        for d in decisions
        if d["decision"] == "propose_new" and d.get("new_topic")
    ]

    link_targets = Counter(l["topic_id"] for l in all_links if l["topic_id"])

    return {
        "total": total,
        "decisions": {
            "link_existing": counts.get("link_existing", 0),
            "propose_new": counts.get("propose_new", 0),
            "skip": counts.get("skip", 0),
        },
        "decisions_pct": {
            k: round(100 * counts.get(k, 0) / total, 1)
            for k in ("link_existing", "propose_new", "skip")
        },
        "strength_buckets": strength_buckets,
        "connection_types": dict(connection_types),
        "links": links,
        "new_topics": new_topics,
        "link_targets": dict(link_targets.most_common()),
        "diagnostics": _summarize_diagnostics(decisions),
    }


def replay_synthesis(
    store: Store,
    judge: Judge,
    embedder: Embedder,
    *,
    decision_model: DecisionModel | None = None,
    n: int = 30,
    seed: int = 42,
    sample_query: dict | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
    progress: bool = True,
    replay_from: Path | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Replay synthesize against a sample of already-processed items.

    With replay_from, reuse the exact items of that prior run instead of
    drawing a fresh sample. Returns the run report dict and writes it to
    output_path as JSON.

    Every item is replayed leave-one-out: the topic it founded, if that
    topic holds nothing else, and its own title are hidden from the judge.
    Without that a replay shows an item its historical answer and flatters
    itself, so it is not an option.
    """
    settings = settings or load_settings()
    sample_query = sample_query or {}

    if replay_from is not None:
        sample = _replay_items(store, replay_from)
    else:
        sample = _sample_items(store, n=n, seed=seed, query=sample_query)
    if progress:
        if replay_from is not None:
            print(f"sample: {len(sample)} items replayed from {replay_from}")
        else:
            print(f"sample: {len(sample)} items (seed={seed}, query={sample_query})")
        print(
            f"model={getattr(judge, 'model', '?')} "
            f"vector_limit={settings.candidates.vector_limit} "
            f"tag_overlap={settings.candidates.tag_overlap} "
            f"ignored_tags={settings.candidates.ignored_tags} "
            f"confirm_proposals={settings.synthesize.confirm_proposals} "
            f"confirm_backend={settings.synthesize.confirm_backend}"
        )
        print()

    decisions_dicts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    proposed_in_batch: list[dict] = []

    for i, item in enumerate(sample, 1):
        eval_result = _eval_from_item(item)
        if eval_result is None:
            if progress:
                print(f"  [{i}/{len(sample)}] SKIP (missing embedding)")
            continue

        try:
            decision = synthesize_item(
                item, eval_result, store, settings, judge,
                proposed_in_batch, leave_one_out=True,
                decision_model=decision_model,
            )
        except Exception as e:
            logger.exception("synthesize_item failed for %s", item.get("url"))
            failures.append({"item_url": item.get("url"), "item_title": item.get("title", ""),
                             "error": f"{type(e).__name__}: {e}"})
            if progress:
                print(f"  [{i}/{len(sample)}] FAIL: {e}")
            continue

        decisions_dicts.append(_decision_to_dict(decision, item))

        if decision.decision == "propose_new" and decision.new_topic:
            topic_text = (
                f"{decision.new_topic['title']}\n\n"
                f"{decision.new_topic['summary']}"
            )
            decision.new_topic["embedding"] = embedder.embed_documents([topic_text])[0]
            proposed_in_batch.append(decision.new_topic)

        if progress:
            marker = "+" if decision.decision == "propose_new" else "="
            target = decision.topic_id or (
                decision.new_topic.get("topic_id") if decision.new_topic else "(skip)"
            )
            strength = decision.strength if decision.strength is not None else 0.0
            print(
                f"  [{i}/{len(sample)}] {marker} {decision.decision:13} "
                f"strength={strength:.2f} -> {target or '(skip)':45} | "
                f"{item.get('title','')[:55]}"
            )

    summary = _summarize(decisions_dicts)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "stage": "synthesize",
        "sample_size": len(sample),
        "seed": seed,
        "sample_query": sample_query,
        "replay_from": str(replay_from) if replay_from else None,
        "label": label,
        "topic_universe": len(store.list_topics(include_retired=False)),
        "config": {
            "library_version": _library_version(),
            "model": getattr(judge, "model", None),
            "embedding_model": getattr(embedder, "model", None),
            "decision_model": getattr(decision_model, "model", None) if decision_model else None,
            "domain": settings.domain,
            "rubric": asdict(settings.rubric),
            "candidates": {
                "tag_overlap": settings.candidates.tag_overlap,
                "ignored_tags": list(settings.candidates.ignored_tags),
                "vector_limit": settings.candidates.vector_limit,
                "vector_num_candidates": settings.candidates.vector_num_candidates,
            },
            "synthesize": {
                "confirm_proposals": settings.synthesize.confirm_proposals,
                "confirm_candidates": settings.synthesize.confirm_candidates,
                "confirm_backend": settings.synthesize.confirm_backend,
                "confirm_link_threshold": settings.synthesize.confirm_link_threshold,
                "secondary_link_floor": settings.synthesize.secondary_link_floor,
            },
            "judge": asdict(settings.judge),
            "record": {
                "minor_floor_after": settings.record.minor_floor_after,
                "minor_floor_strength": settings.record.minor_floor_strength,
            },
        },
        "summary": summary,
        "decisions": decisions_dicts,
        "failures": failures,
    }

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        stem = label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tail = "replay" if replay_from else f"n{n}-seed{seed}"
        output_path = DEFAULT_OUTPUT_DIR / f"synthesize-{stem}-{tail}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(report, indent=2, default=str))

    if progress:
        _print_summary(summary, output_path)

    return report


def _print_summary(summary: dict[str, Any], output_path: Path) -> None:
    print()
    print("=" * 70)
    print("DECISION DISTRIBUTION")
    print("=" * 70)
    for k in ("link_existing", "propose_new", "skip"):
        n = summary["decisions"][k]
        pct = summary["decisions_pct"][k]
        print(f"  {k:<15} {n:>3}  ({pct}%)")

    print()
    print("STRENGTH BUCKETS (link_existing only)")
    for b, v in summary["strength_buckets"].items():
        print(f"  {b:<10} {v}")

    print()
    print("CONNECTION TYPES")
    for k, v in summary["connection_types"].items():
        print(f"  {k:<14} {v}")
    links = summary["links"]
    print()
    print(f"LINKS  total={links['total']}  per link decision={links['per_link_decision']}  "
          f"decisions with more than one={links['multi_link_decisions']}")
    print(f"  secondary links {links['secondary']}  strengths {links['secondary_strength_buckets']}  "
          f"dropped by floor {links['dropped_by_floor']}")

    if summary["new_topics"]:
        print()
        print(f"NEW TOPICS PROPOSED ({len(summary['new_topics'])})")
        for nt in summary["new_topics"]:
            print(f"  - {nt['topic_id']}: {nt['title']}")

    diag = summary.get("diagnostics") or {}
    if diag:
        print()
        print("RETRIEVAL / PROMPT DIAGNOSTICS")
        print(f"  mean candidates per call   {diag.get('mean_total_candidates')}"
              f"  (tag-only: {diag.get('mean_tag_only_candidates')})")
        print(f"  mean input tokens per call {diag.get('mean_input_tokens')}")
        print(f"  mean classified per call   {diag.get('mean_classified_per_call')}")
        print(f"  linked topic channel       {diag.get('chosen_channel')}")
        print(f"  linked topic vector rank   {diag.get('chosen_vector_rank')}")
        print(f"  gate overrides             {diag.get('gate_overrides')}")
        print(f"  confirmation outcomes      {diag.get('confirmation')}"
              f"  (mean confirm tokens {diag.get('mean_confirm_input_tokens')})")
        for k, v in (diag.get("by_decision") or {}).items():
            print(f"  {k:<14} n={v['n']:<3} mean vector top1={v['mean_vector_top1']}"
                  f"  labels={v['labels']}")

    print()
    print(f"report: {output_path}")
