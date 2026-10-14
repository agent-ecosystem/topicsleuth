"""Snapshot the current store state for before/after comparison.

Produces a markdown report covering topic counts, observation distribution,
strength buckets and connection types. Useful for
diffing across major operations (reprocess-tail, merge-topics, prompt
changes, etc.).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from topicsleuth.interfaces import Store

DEFAULT_OUTPUT_DIR = Path("tuning_runs")  # relative to the working directory


def _strength_bucket(s: float) -> str:
    if s < 0.3: return "<0.3"
    if s < 0.5: return "0.3-0.5"
    if s < 0.7: return "0.5-0.7"
    if s < 0.8: return "0.7-0.8"
    if s < 0.9: return "0.8-0.9"
    return "0.9+"


def _evidence_bucket(n: int) -> str:
    if n <= 1: return "0-1"
    if n <= 4: return "2-4"
    if n <= 9: return "5-9"
    if n <= 19: return "10-19"
    if n <= 29: return "20-29"
    return "30+"


def build_audit_report(store: Store) -> str:
    """Compute the audit and return it as a markdown string."""
    topics = store.list_topics()
    observations = store.list_observations()

    lines: list[str] = []
    lines.append(f"# Store state — {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Topics")
    status_counts = Counter(t.get("status", "?") for t in topics)
    lines.append(f"Total: {len(topics)}")
    for status, n in status_counts.most_common():
        lines.append(f"  {status:<12} {n}")
    lines.append("")

    obs_by_topic: Counter = Counter()
    for o in observations:
        obs_by_topic[o["topic_id"]] += 1

    lines.append("## Topics by observation count (non-retired)")
    bucket_counts: Counter = Counter()
    for topic in topics:
        if topic.get("status") == "retired":
            continue
        n = obs_by_topic.get(topic["topic_id"], 0)
        bucket_counts[_evidence_bucket(n)] += 1
    for bucket in ("0-1", "2-4", "5-9", "10-19", "20-29", "30+"):
        lines.append(f"  {bucket:<6} {bucket_counts.get(bucket, 0)}")
    lines.append("")

    lines.append("## Top 15 topics by observation count")
    topic_lookup = {t["topic_id"]: t for t in topics}
    for tid, n in obs_by_topic.most_common(15):
        t = topic_lookup.get(tid)
        title = t["title"] if t else tid
        status = t.get("status", "?") if t else "?"
        lines.append(f"  {n:>4}  [{status:<8}] {title[:60]}")
    lines.append("")

    lines.append("## Observations")
    minor = sum(1 for o in observations if o.get("minor"))
    lines.append(f"Total: {len(observations)}")
    lines.append(f"  primary: {len(observations) - minor}")
    lines.append(f"  minor:   {minor}")
    lines.append("")

    sb: Counter = Counter()
    for o in observations:
        sb[_strength_bucket(o.get("strength") or 0)] += 1
    lines.append("## Strength distribution")
    for b in ("<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9+"):
        lines.append(f"  {b:<10} {sb.get(b, 0)}")
    lines.append("")

    lines.append("## Connection types")
    ct = Counter(o.get("connection_type", "?") for o in observations)
    for k, n in ct.most_common():
        lines.append(f"  {k:<14} {n}")
    lines.append("")

    return "\n".join(lines)


def write_audit_report(
    store: Store,
    output_path: Path | None = None,
    label: str | None = None,
) -> Path:
    """Run the audit and write it to disk. Returns the output path.

    If output_path is None, writes to tuning_runs/audit-<label or
    timestamp>.md.
    """
    report = build_audit_report(store)
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)
        if label:
            output_path = DEFAULT_OUTPUT_DIR / f"audit-{label}.md"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            output_path = DEFAULT_OUTPUT_DIR / f"audit-{ts}.md"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return output_path
