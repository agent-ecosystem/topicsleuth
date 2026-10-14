#!/usr/bin/env python3
"""Label item-candidate pairs with TypeSafe's Jev decision model.

Jev does not generate text: it scores a fixed option list in one forward pass
and returns a probability distribution plus a confidence. Each pair is sent
as the `state`, with two questions evaluated in parallel:

  relationship  choice  same_question | subset_question | adjacent_question | surface_overlap
  link          noul    "This item belongs as evidence under this topic."

The rubric is the library's own (``topicsleuth.rubric``), rendered into
Jev's option descriptions by ``topicsleuth.backends.jev``; it lives in the
option descriptions, not in a system prompt, since there is none.
Output has the same shape as the other rater files, with Jev's probabilities
and confidence kept per pair. The API key comes from TYPESAFE_API_KEY.

Examples:
  python scripts/label_pairs_jev.py --dry-run             # print the payload for the first pair
  python scripts/label_pairs_jev.py --limit 3
  python scripts/label_pairs_jev.py --out data/labels/jev_labels.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from topicsleuth.backends.jev import API_URL as API, PAIR_QUESTIONS as QUESTIONS  # noqa: E402
from topicsleuth.rubric import RELATIONSHIP_LABELS  # noqa: E402

PAIRS = ROOT / "data" / "labels" / "pairs.json"
LABELS = list(RELATIONSHIP_LABELS)


def state_for(p: dict) -> dict:
    it, th = p["item"], (p.get("topic") or p["theme"])  # pairs.json predates the rename
    return {
        "item": {
            "title": it["title"],
            "summary": it.get("summary", ""),
            "abstract_excerpt": it.get("excerpt", ""),
        },
        "candidate_topic": {
            "title": th["title"],
            "summary": th.get("summary", ""),
            "observations_so_far": th.get("evidence_count"),
            "representative_observations": th.get("representative_observations") or [],
        },
    }


def call(key: str, model: str, state: dict, retries: int = 4) -> dict:
    body = json.dumps({"model": model, "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        API, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(str(e)) from None
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--out", default=None, help="Default: data/labels/jev_labels.json")
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="Print the first pair's request payload and exit (no key needed)")
    args = ap.parse_args()

    if not Path(args.pairs).exists():
        sys.exit(f"pair set not found at {args.pairs}; pass --pairs <pairs.json> (see data/examples/)")
    pairs = json.load(open(args.pairs))["pairs"]
    if args.limit:
        pairs = pairs[: args.limit]

    if args.dry_run:
        print(json.dumps({"model": args.model, "state": state_for(pairs[0]), "questions": QUESTIONS}, indent=1)[:6000])
        return

    key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("TYPESAFE_API_KEY is not set")
    out_path = Path(args.out) if args.out else ROOT / "data" / "labels" / "jev_labels.json"
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    served_model = None

    def work(p):
        try:
            return p["id"], call(key, args.model, state_for(p)), None
        except Exception as e:
            return p["id"], None, str(e)

    print(f"model={args.model} pairs={len(pairs)} workers={args.workers}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (pid, resp, err) in enumerate(ex.map(work, pairs), 1):
            if err:
                errors[pid] = err
                print(f"  [{i}/{len(pairs)}] {pid} ERROR {err[:140]}", flush=True)
                continue
            served_model = resp.get("model", served_model)
            rel = resp["answers"]["relationship"]
            link = resp["answers"].get("link", {})
            results[pid] = {
                "label": rel["choice"],
                "confidence": rel.get("confidence"),
                "probabilities": rel.get("probabilities"),
                "link_probability": link.get("noul"),
                "rater": served_model or args.model,
                "date": date.today().isoformat(),
                "input_tokens": (resp.get("usage") or {}).get("input_tokens"),
            }
            probs = rel.get("probabilities") or {}
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
            print(f"  [{i}/{len(pairs)}] {pid} {rel['choice']:<18} conf={rel.get('confidence'):.2f} "
                  f"link={link.get('noul', float('nan')):.2f} top2={[(k[:7], round(v, 2)) for k, v in top]}", flush=True)

    out_path.write_text(json.dumps({
        "rater": served_model or args.model,
        "note": "Decision-model labels (TypeSafe Jev). Rubric carried in option descriptions; probabilities and confidence kept per pair.",
        "questions": QUESTIONS,
        "labels": results,
        "errors": errors,
    }, indent=1))
    print(f"\nlabels: {dict(Counter(r['label'] for r in results.values()))}")
    print(f"errors: {len(errors)}  elapsed: {time.time() - t0:.1f}s  "
          f"input tokens: {sum(r.get('input_tokens') or 0 for r in results.values())}  written: {out_path}")


if __name__ == "__main__":
    main()
