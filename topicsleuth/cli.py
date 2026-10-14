"""Curate and tune a topicsleuth store from the command line.

The CLI is the one place that reads the environment: it builds the default
backends (MongoDB Atlas, Voyage, Anthropic, optionally Jev) from the
variables documented in topicsleuth.env and hands them to the library.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from topicsleuth.env import backends_from_env
from topicsleuth.settings import load_settings


def _settings():
    """Packaged defaults, with the operator's settings file layered on top
    when TOPICSLEUTH_SETTINGS names one."""
    return load_settings(os.environ.get("TOPICSLEUTH_SETTINGS") or None)


def _backends(need_judge: bool = False, need_embedder: bool = False):
    settings = _settings()
    try:
        return settings, backends_from_env(settings, need_judge=need_judge, need_embedder=need_embedder)
    except (RuntimeError, ImportError) as e:
        raise SystemExit(f"error: {e}")


def cmd_seed(args):
    from topicsleuth.settings import load_seed_topics

    seed_topics = load_seed_topics(args.definitions)
    if not seed_topics:
        print(f"No seed topics defined in {args.definitions}")
        return
    settings, b = _backends(need_embedder=not args.dry_run)

    topic_docs = []
    for st in seed_topics:
        topic_docs.append({
            "topic_id": re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9-]+", "-", st.title.lower())).strip("-"),
            "title": st.title,
            "summary": st.summary,
            "tags": st.tags,
            "suggested_directions": st.suggested_directions,
        })

    if args.dry_run:
        for doc in topic_docs:
            print(f"  {doc['topic_id']:<45} {doc['title']}")
        existing = len(b.store.list_topics())
        if existing:
            print(f"\n[DRY RUN] store already holds {existing} topics; seed writes only to an empty store.")
        else:
            print(f"\n[DRY RUN] would embed and write {len(topic_docs)} topics.")
        return

    for doc, st in zip(topic_docs, seed_topics):
        doc["embedding"] = b.embedder.embed_documents([f"{st.title}\n\n{st.summary}"])[0]

    inserted = b.store.seed_topics_if_empty(topic_docs)
    if inserted:
        print(f"Seeded {inserted} topics.")
    else:
        print("Topics collection already has data, skipping seed.")


def _preview_delete_since(store, cutoff: datetime, run_id: str | None = None) -> dict:
    """What Store.delete_since would remove, counted through the read side
    of the protocol so no adapter needs a second method for it."""
    def in_window(doc: dict) -> bool:
        stamp = doc.get("recorded_at")
        return bool(stamp and stamp >= cutoff and (run_id is None or doc.get("run_id") == run_id))

    doomed_obs = [o for o in store.list_observations() if in_window(o)]
    doomed_keys = {(o.get("item_url"), o.get("topic_id")) for o in doomed_obs}
    surviving_by_topic: dict[str, int] = {}
    for o in store.list_observations():
        if (o.get("item_url"), o.get("topic_id")) not in doomed_keys:
            surviving_by_topic[o.get("topic_id")] = surviving_by_topic.get(o.get("topic_id"), 0) + 1
    deleted = kept = snapshots = 0
    for t in store.list_topics():
        if in_window(t):
            if surviving_by_topic.get(t["topic_id"]):
                kept += 1
            else:
                deleted += 1
                continue
        snapshots += sum(1 for s in t.get("snapshots", []) if in_window(s))
    return {"observations_deleted": len(doomed_obs), "topics_deleted": deleted,
            "topics_kept": kept, "snapshots_cleared": snapshots}


def cmd_clean(args):
    settings, b = _backends()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    scope = f"the last {args.hours}h" + (f" by run {args.run_id}" if args.run_id else "")
    if args.dry_run:
        result = _preview_delete_since(b.store, cutoff, args.run_id)
        print(f"[DRY RUN] would remove what was written in {scope}:")
        for k, v in result.items():
            print(f"  {k:<24} {v}")
        return
    result = b.store.delete_since(cutoff, run_id=args.run_id)
    print(f"Removed what was written in {scope}:")
    for k, v in result.items():
        print(f"  {k:<24} {v}")
    print("Statuses, embeddings and descriptions the run changed are not restored.")
    print("Items are the feeder's: reset their consumed markers there if they should re-run.")


def cmd_split_topic(args):
    from topicsleuth.split import split_topic

    settings, b = _backends(need_judge=True, need_embedder=True)
    result = split_topic(
        topic_id=args.topic_id,
        store=b.store,
        settings=settings,
        judge=b.judge,
        embedder=b.embedder,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"\n[DRY RUN] Proposed split for '{result['original_topic']}'"
              f" ({result['observation_count']} observations):\n")
        for st in result["sub_topics"]:
            print(f"  Sub-topic: {st['title']}")
            print(f"    topic_id: {st['topic_id']}")
            print(f"    summary: {st['summary'][:120]}")
            print()
        assigned_counts: dict[str, int] = {}
        for r in result["reassignments"]:
            assigned_counts[r["new_topic_id"]] = assigned_counts.get(r["new_topic_id"], 0) + 1
        print("  Reassignment distribution:")
        for tid, count in sorted(assigned_counts.items(), key=lambda x: -x[1]):
            print(f"    {tid}: {count} observations")
        if result["problems"]:
            print(f"\n  PLAN REJECTED, {len(result['problems'])} problem(s); nothing would be written:")
            for problem in result["problems"]:
                print(f"    - {problem}")
    else:
        print(f"\nSplit complete for '{result['original_topic']}':")
        print(f"  Sub-topics created:      {', '.join(result['sub_topics_created'])}")
        print(f"  Observations moved:      {result['observations_reassigned']}")
        print(f"  Original topic retired:  {result['original_topic_id']}")


def cmd_merge_topics(args):
    from topicsleuth.merge import merge_topics

    settings, b = _backends(need_judge=True, need_embedder=True)
    result = merge_topics(
        source_topic_id=args.source,
        target_topic_id=args.target,
        store=b.store,
        settings=settings,
        judge=b.judge,
        embedder=b.embedder,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY RUN] Proposed merge")
        print(f"  source: {result['source_topic_id']} ({result['source_observation_count']} obs)")
        print(f"          title: {result['source_title']}")
        print(f"          summary: {(result['source_summary'] or '')[:120]}")
        print(f"  target: {result['target_topic_id']} "
              f"({result['target_observation_count_before']} obs -> "
              f"{result['target_observation_count_after']} obs)")
        print(f"          title before: {result['target_title_before']}")
        print(f"          summary before: {(result['target_summary_before'] or '')[:120]}")
        print()
        print(f"  merged title:   {result['merged_title']}")
        print(f"  merged summary: {result['merged_summary']}")
        if result.get("merged_tags"):
            print(f"  merged tags:    {', '.join(result['merged_tags'])}")
        print()
        print(f"  rationale: {result.get('rationale','')}")
        if result.get("merged_suggested_directions"):
            print("\n  suggested directions:")
            for d in result["merged_suggested_directions"]:
                print(f"    - {d}")
    else:
        print("\nMerge complete:")
        print(f"  source retired:        {result['source_topic_id']}")
        print(f"  target updated:        {result['target_topic_id']}")
        print(f"  merged title:          {result['merged_title']}")
        print(f"  observations reassigned: {result['observations_reassigned']}")
        print(f"  target now has:        {result['target_observation_count_after']} observations")


def cmd_suggest_merges(args):
    from topicsleuth.merge import suggest_merges

    settings, b = _backends()
    suggestions = suggest_merges(
        b.store,
        threshold=args.threshold,
        min_tag_overlap=args.min_tag_overlap,
    )

    if not suggestions:
        print(f"No pairs above cosine {args.threshold} "
              f"with tag overlap >= {args.min_tag_overlap}.")
        return

    print(f"Found {len(suggestions)} candidate merge pairs "
          f"(cosine >= {args.threshold}, tag_overlap >= {args.min_tag_overlap}):\n")

    for s in suggestions:
        print(f"  cosine={s['cosine']:.3f}  tag_overlap={s['tag_overlap']}")
        print(f"    source ({s['source_obs']:>3} obs): {s['source_topic_id']}")
        print(f"      {s['source_title']}")
        print(f"    target ({s['target_obs']:>3} obs): {s['target_topic_id']}")
        print(f"      {s['target_title']}")
        print(f"    suggested: topicsleuth merge-topics "
              f"{s['source_topic_id']} {s['target_topic_id']} --dry-run")
        print()


def cmd_reprocess_tail(args):
    from topicsleuth.reprocess import execute_reprocess, plan_reprocess

    settings, b = _backends()
    plan = plan_reprocess(
        b.store, max_obs=args.max_obs, older_than_days=args.older_than_days
    )

    age = f", identified > {args.older_than_days}d ago" if args.older_than_days else ""
    print(f"Weak topics (<= {args.max_obs} obs, status in emerging/active{age}): {plan.topic_count}")
    print(f"  observations on those topics: {plan.observation_count}")
    print(f"  unique items that lose their home: {plan.item_count}")

    if plan.topic_count == 0:
        print("\nNothing to do.")
        return

    sample = plan.weak_topics[:25]
    print()
    print("Sample of affected topics:")
    for t in sample:
        n = b.store.count_observations_for_topic(t["topic_id"])
        print(f"  - {t['topic_id']:<55} status={t.get('status','?'):<10} obs={n}")
    if plan.topic_count > len(sample):
        print(f"  ... and {plan.topic_count - len(sample)} more")

    if args.dry_run:
        print("\n[DRY RUN] no changes written.")
        print("Drop --dry-run to execute. The affected item URLs are written to a "
              "file for the feeder to re-queue.")
        return

    print()
    result = execute_reprocess(b.store, plan)
    out = Path(args.urls_out or f"tuning_runs/reprocess-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.urls")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(result["item_urls"]) + "\n")
    print(f"Topics deleted:       {result['topics_deleted']}")
    print(f"Observations deleted: {result['observations_deleted']}")
    print(f"Items to re-queue:    {len(result['item_urls'])} (written to {out})")
    print()
    print("Cleanup complete. Hand the URL file to the feeder so its next run re-homes the items.")


def cmd_add_topic(args):
    from topicsleuth.add_topic import add_topic, load_topic_definition

    definition = load_topic_definition(Path(args.definition))
    if args.link:
        definition.link.extend(args.link)
    settings, b = _backends(need_embedder=not args.dry_run)

    report = add_topic(definition, b.store, b.embedder, dry_run=args.dry_run)

    print(f"Topic: {report['topic_id']}  ({report['title']})")
    moves = report["observations_to_move"]
    print(f"Observations to move: {len(moves)}")
    for m in moves:
        print(f"  - {m['item_title'][:60]:<60} from {m['from_topic']} "
              f"({m['connection_type']}, {m['strength']})")
    for url in report["urls_without_observation"]:
        print(f"  ! no observation for {url} (not moved)")
    if args.dry_run:
        print("\n[DRY RUN] no changes written.")
        return
    print(f"\nCreated topic and moved {report['observations_moved']} observations. "
          f"Source topics re-snapshotted: {', '.join(report['source_topics_affected']) or 'none'}")


def cmd_move_observation(args):
    from topicsleuth.add_topic import move_observation

    settings, b = _backends()
    report = move_observation(args.item_url, args.target_topic_id, b.store, dry_run=args.dry_run)
    for m in report["moves"]:
        print(f"Move: {m['item_title'][:60]}  from {m['from_topic']}  -> {report['target_topic_id']}  "
              f"({m['connection_type']}, {m['strength']})")
    if report["sources_that_would_empty"]:
        print(f"Source topic(s) left empty, will be retired with merged_into: "
              f"{', '.join(report['sources_that_would_empty'])}")
    if args.dry_run:
        print("\n[DRY RUN] no changes written.")
        return
    print(f"\nMoved. Retired: {', '.join(report['sources_retired']) or 'none'}")


def cmd_audit(args):
    from topicsleuth.audit import build_audit_report, write_audit_report

    settings, b = _backends()
    report = build_audit_report(b.store)
    print(report)

    if not args.no_save:
        path = write_audit_report(
            b.store,
            output_path=Path(args.output) if args.output else None,
            label=args.label,
        )
        print()
        print(f"Written: {path}")


REFRESH_MIN_OBS = 5


def cmd_backfill_embeddings(args):
    """Repair topic embeddings.

    Two operations in one pass:
    - Topics missing a valid embedding get one computed from title+summary.
    - Topics that already have an embedding AND have >= REFRESH_MIN_OBS
      non-minor observations get a centroid-blend refresh. This catches up
      topics that grew before the threshold-triggered refresh in record.py
      was wired in.

    Topics that have a valid embedding and fewer than REFRESH_MIN_OBS
    observations are left alone: the centroid would be too noisy at low
    counts, and the title+summary embedding is fine for new topics.
    """
    from topicsleuth.embeddings import refresh_topic_embedding

    settings, b = _backends(need_embedder=not args.dry_run)
    topics = b.store.list_topics(include_retired=False)
    needs_seed: list[dict] = []
    needs_refresh: list[dict] = []

    for t in topics:
        has_valid_embedding = (
            t.get("embedding")
            and len(t["embedding"]) == settings.embedding.dimensions
        )
        non_minor_obs_count = sum(
            1 for o in b.store.get_observations_for_topic(t["topic_id"]) if not o.get("minor")
        )
        if not has_valid_embedding:
            needs_seed.append(t)
        elif non_minor_obs_count >= REFRESH_MIN_OBS:
            t["_obs_count"] = non_minor_obs_count
            needs_refresh.append(t)

    print(f"Total topics (non-retired): {len(topics)}")
    print(f"Need seed embedding (missing): {len(needs_seed)}")
    print(f"Need centroid refresh (>= {REFRESH_MIN_OBS} non-minor obs): {len(needs_refresh)}")

    if not needs_seed and not needs_refresh:
        return

    if args.dry_run:
        for t in needs_seed:
            print(f"  would seed: {t['topic_id']}")
        for t in needs_refresh:
            print(f"  would refresh: {t['topic_id']} ({t['_obs_count']} non-minor obs)")
        return

    failures = 0
    for i, t in enumerate(needs_seed, 1):
        topic_text = f"{t.get('title','')}\n\n{t.get('summary','')}"
        try:
            b.store.update_topic(t["topic_id"], {"embedding": b.embedder.embed_documents([topic_text])[0]})
            print(f"  [{i}/{len(needs_seed)}] seeded {t['topic_id']}")
        except Exception as e:
            failures += 1
            print(f"  [{i}/{len(needs_seed)}] FAILED {t['topic_id']}: {e}")

    refresh_failures = 0
    refreshed = 0
    for i, t in enumerate(needs_refresh, 1):
        try:
            ok = refresh_topic_embedding(t["topic_id"], b.store, settings, b.embedder)
            if ok:
                refreshed += 1
                print(f"  [{i}/{len(needs_refresh)}] refreshed {t['topic_id']} ({t['_obs_count']} obs)")
            else:
                print(f"  [{i}/{len(needs_refresh)}] no-op {t['topic_id']} (no usable item embeddings)")
        except Exception as e:
            refresh_failures += 1
            print(f"  [{i}/{len(needs_refresh)}] FAILED {t['topic_id']}: {e}")

    total_attempted = len(needs_seed) + len(needs_refresh)
    print(
        f"\nDone. seeded={len(needs_seed) - failures}/{len(needs_seed)} "
        f"refreshed={refreshed}/{len(needs_refresh)} "
        f"failures={failures + refresh_failures}/{total_attempted}"
    )


def cmd_show_prompts(args):
    from topicsleuth.rubric import PROMPTS

    settings = _settings()
    names = [args.name] if args.name else list(PROMPTS)
    for i, name in enumerate(names):
        if i:
            print()
        if len(names) > 1:
            print(f"{'=' * 70}\n{name}\n{'=' * 70}")
        print(PROMPTS[name](settings))


def cmd_tune(args):
    from topicsleuth.tune import replay_synthesis

    if args.stage != "synthesize":
        raise SystemExit(f"unknown tune stage: {args.stage}")

    settings, b = _backends(need_judge=True, need_embedder=True)
    if args.limit:
        settings.candidates.vector_limit = args.limit
    if args.no_tag_overlap:
        settings.candidates.tag_overlap = False
    if args.no_confirm:
        settings.synthesize.confirm_proposals = False
    if args.confirm_backend:
        settings.synthesize.confirm_backend = args.confirm_backend
    if args.confirm_threshold is not None:
        settings.synthesize.confirm_link_threshold = args.confirm_threshold
    if args.secondary_link_floor is not None:
        settings.synthesize.secondary_link_floor = args.secondary_link_floor
    if args.ignore_tags:
        settings.candidates.ignored_tags = [
            t.strip() for t in args.ignore_tags.split(",") if t.strip()
        ]
    if args.model:
        b.judge.model = args.model
    sample_query = json.loads(args.sample_query) if args.sample_query else None
    if not args.replay and sample_query is None:
        raise SystemExit("tune synthesize needs --replay <report> or --sample-query '<json>'")
    replay_synthesis(
        b.store, b.judge, b.embedder,
        decision_model=b.decision_model,
        n=args.n,
        seed=args.seed,
        sample_query=sample_query,
        output_path=Path(args.output) if args.output else None,
        settings=settings,
        replay_from=Path(args.replay) if args.replay else None,
        label=args.label,
    )


def main():
    # Search from the working directory: the default searches from this
    # file's directory, which is the installed package, not the operator's
    # project.
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(
        prog="topicsleuth",
        description="Curate and tune a topicsleuth store: seed, split, merge, move, audit, replay",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser(
        "seed", help="Load seed topics into an empty store from a YAML file"
    )
    seed_parser.add_argument(
        "definitions", help="YAML file with a topics: list (title, summary, tags, suggested_directions)"
    )
    seed_parser.add_argument("--dry-run", action="store_true", help="List the topics that would be written, without embedding or writing")
    seed_parser.set_defaults(func=cmd_seed)

    clean_parser = subparsers.add_parser(
        "clean", help="Remove the observations, topics and snapshots written in the last N hours (not a full undo)"
    )
    clean_parser.add_argument(
        "--hours", type=float, default=28,
        help="Remove store output created within this many hours (default: 28)"
    )
    clean_parser.add_argument("--run-id", type=str, default=None, help="Only what the run with this id wrote (the run_id a feeder passed to record)")
    clean_parser.add_argument("--dry-run", action="store_true", help="Count what would be removed without removing it")
    clean_parser.set_defaults(func=cmd_clean)

    split_parser = subparsers.add_parser(
        "split-topic", help="Split an oversized topic into more specific sub-topics"
    )
    split_parser.add_argument("topic_id", help="The topic_id of the topic to split")
    split_parser.add_argument(
        "--dry-run", action="store_true", help="Show proposed split without executing it"
    )
    split_parser.set_defaults(func=cmd_split_topic)

    merge_parser = subparsers.add_parser(
        "merge-topics",
        help="Merge a near-duplicate topic into another (source retires, target absorbs)"
    )
    merge_parser.add_argument("source", help="topic_id of the topic to retire (its observations move to target)")
    merge_parser.add_argument("target", help="topic_id of the topic that survives and absorbs source's observations")
    merge_parser.add_argument(
        "--dry-run", action="store_true", help="Preview the merged title/summary without writing"
    )
    merge_parser.set_defaults(func=cmd_merge_topics)

    suggest_parser = subparsers.add_parser(
        "suggest-merges", help="Find candidate topic pairs for merging via cosine similarity"
    )
    suggest_parser.add_argument("--threshold", type=float, default=0.85, help="Cosine similarity threshold (default: 0.85)")
    suggest_parser.add_argument("--min-tag-overlap", type=int, default=0, help="Minimum number of shared tags (default: 0)")
    suggest_parser.set_defaults(func=cmd_suggest_merges)

    reprocess_parser = subparsers.add_parser(
        "reprocess-tail",
        help="Clear weak topics (those with few observations) and list their items "
             "for the feeder to re-queue"
    )
    reprocess_parser.add_argument("--max-obs", type=int, default=2, help="Topics with this many or fewer observations are eligible (default: 2)")
    reprocess_parser.add_argument("--older-than-days", type=int, default=None, help="Only topics identified at least this many days ago (default: no age filter)")
    reprocess_parser.add_argument("--urls-out", type=str, default=None, help="Where to write the item URLs to re-queue (default: tuning_runs/reprocess-<ts>.urls)")
    reprocess_parser.add_argument("--dry-run", action="store_true", help="Preview the cleanup without writing")
    reprocess_parser.set_defaults(func=cmd_reprocess_tail)

    add_topic_parser = subparsers.add_parser(
        "add-topic",
        help="Add one topic to a populated store from a YAML definition, optionally moving existing observations onto it"
    )
    add_topic_parser.add_argument("definition", help="YAML file: title, summary, optional topic_id, tags, suggested_directions, link (item URLs to move)")
    add_topic_parser.add_argument("--link", action="append", default=None, metavar="URL", help="Item URL whose observation should move onto the new topic (repeatable)")
    add_topic_parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing")
    add_topic_parser.set_defaults(func=cmd_add_topic)

    move_parser = subparsers.add_parser(
        "move-observation",
        help="Move one item's observation onto an existing topic; a source topic left empty is retired with merged_into set"
    )
    move_parser.add_argument("item_url", help="URL of the item whose observation moves")
    move_parser.add_argument("target_topic_id", help="topic_id to move it onto")
    move_parser.add_argument("--dry-run", action="store_true", help="Show the plan without writing")
    move_parser.set_defaults(func=cmd_move_observation)

    audit_parser = subparsers.add_parser(
        "audit", help="Snapshot the store (topics, observations, strengths) to tuning_runs/ for before/after comparison"
    )
    audit_parser.add_argument("--label", type=str, default=None, help="Label suffix for the output file. Default: timestamp.")
    audit_parser.add_argument("--output", type=str, default=None, help="Explicit output path. Overrides --label if both are given.")
    audit_parser.add_argument("--no-save", action="store_true", help="Print to stdout only; do not write a file.")
    audit_parser.set_defaults(func=cmd_audit)

    backfill_parser = subparsers.add_parser(
        "backfill-embeddings", help="Seed missing topic embeddings and centroid-refresh strong topics"
    )
    backfill_parser.add_argument("--dry-run", action="store_true", help="List topics that would be embedded without calling the embedder")
    backfill_parser.set_defaults(func=cmd_backfill_embeddings)

    tune_parser = subparsers.add_parser(
        "tune", help="Replay a stage against already-recorded items without writing"
    )
    tune_parser.add_argument("stage", choices=["synthesize"], help="Stage to replay")
    tune_parser.add_argument("--replay", type=str, default=None, help="Reuse the exact items from a prior tuning_runs/*.json")
    tune_parser.add_argument("--sample-query", type=str, default=None,
                             help="JSON query selecting items to sample from (store-specific; "
                                  "MongoDB filter syntax for MongoStore). Required unless --replay.")
    tune_parser.add_argument("--n", type=int, default=30, help="Sample size (default: 30)")
    tune_parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling (default: 42)")
    tune_parser.add_argument("--output", type=str, default=None, help="Path to write JSON report (default: tuning_runs/synthesize-<ts>.json)")
    tune_parser.add_argument("--label", type=str, default=None, help="Short label used in the output filename instead of a timestamp")
    tune_parser.add_argument("--model", type=str, default=None, help="Override the judge model for this run")
    tune_parser.add_argument("--limit", type=int, default=None, help="Override candidates.vector_limit for this run")
    tune_parser.add_argument("--no-tag-overlap", action="store_true", help="Disable the tag-overlap candidate channel (vector search only)")
    tune_parser.add_argument("--confirm-backend", choices=["judge", "decision"], default=None, help="Which reviewer checks propose_new decisions (default: settings.yaml)")
    tune_parser.add_argument("--confirm-threshold", type=float, default=None, help="decision backend: p(belongs) needed to demote a proposal")
    tune_parser.add_argument("--secondary-link-floor", type=float, default=None, help="Drop a second or later link scoring below this strength")
    tune_parser.add_argument("--no-confirm", action="store_true", help="Disable the second-pass confirmation of propose_new decisions")
    tune_parser.add_argument("--ignore-tags", type=str, default=None, help="Comma-separated tags excluded from overlap matching")
    tune_parser.set_defaults(func=cmd_tune)

    from topicsleuth.rubric import PROMPTS
    prompts_parser = subparsers.add_parser(
        "show-prompts",
        help="Print the judge prompts as rendered with the current settings (domain and rubric)",
    )
    prompts_parser.add_argument("name", nargs="?", choices=list(PROMPTS), default=None,
                                help="One prompt; default: all of them")
    prompts_parser.set_defaults(func=cmd_show_prompts)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.verbose:
        logging.getLogger("topicsleuth").setLevel(logging.DEBUG)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
