#!/usr/bin/env python3
"""Label item-candidate pairs with an OpenAI model, as an independent rater.

Reads data/labels/pairs.json, asks the model the same question a human
labeler is asked (same rubric text), and writes a label file in the same
shape as the other rater files. The API key comes from OPENAI_API_KEY.

Examples:
  python scripts/label_pairs_openai.py --list-models
  python scripts/label_pairs_openai.py --model gpt-5.5 --out data/labels/openai_labels.json
  python scripts/label_pairs_openai.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "labels" / "pairs.json"
API = "https://api.openai.com/v1"

LABELS = ["same_question", "subset_question", "adjacent_question", "surface_overlap"]

RUBRIC = """\
You are labeling pairs for a research-topic taxonomy of the coding agent \
ecosystem. For each pair, name the research question the item investigates \
and the research question the candidate topic investigates, then pick the \
structural relationship between them. The label is descriptive, not a \
quality judgment.

- same_question: The topic's research question is the question this item \
investigates. The item's primary contribution is an answer or partial answer \
to it.
- subset_question: The item investigates a specific instance of the topic's \
question; the topic's question subsumes it. Example: topic asks "how do \
agents fail in ways static analysis misses?", item proposes a detector for \
one failure class. A different method, artifact type, or subdomain inside the \
same question is still a subset.
- adjacent_question: A different research question in the same domain. The \
topic's summary, read fairly, does not cover what the item is about, even \
though they share keywords. If broadening the summary by one clause would \
cover the item, lean toward subset rather than adjacent.
- surface_overlap: Shared tags, model names, or subfield, but the item's \
question lies elsewhere.

Representative observations show what the topic actually contains; weigh \
them alongside the one-sentence summary. Respond with JSON only.
"""


def api_key(args) -> str:
    return os.environ.get("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY is not set")


def request(path: str, key: str, body: dict | None = None, retries: int = 4) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API}{path}", data=data, method="POST" if data else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(str(e)) from None
    raise RuntimeError("unreachable")


def format_pair(p: dict) -> str:
    it, th = p["item"], p["topic"]
    reps = "\n".join(f"  * {r}" for r in th.get("representative_observations") or []) or "  (none besides this item)"
    return (
        f"## Item\nTitle: {it['title']}\nSummary: {it.get('summary','')}\n"
        f"Abstract excerpt:\n{it.get('excerpt','')}\n\n"
        f"## Candidate topic\nTitle: {th['title']}\nSummary: {th.get('summary','')}\n"
        f"Observations so far: {th.get('evidence_count')}\nRepresentative observations:\n{reps}\n\n"
        f"How does the item's research question relate to this topic's?"
    )


SCHEMA = {
    "name": "pair_label",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "item_question": {"type": "string"},
            "candidate_question": {"type": "string"},
            "label": {"type": "string", "enum": LABELS},
            "reason": {"type": "string"},
        },
        "required": ["item_question", "candidate_question", "label", "reason"],
    },
}


def label_one(p: dict, key: str, model: str, extra: dict) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": format_pair(p)},
        ],
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        **extra,
    }
    resp = request("/chat/completions", key, body)
    content = resp["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    usage = resp.get("usage") or {}
    return {
        "label": parsed["label"],
        "reason": parsed.get("reason", ""),
        "item_question": parsed.get("item_question", ""),
        "candidate_question": parsed.get("candidate_question", ""),
        "rater": model,
        "date": date.today().isoformat(),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.5", help="OpenAI model id (default: gpt-5.5; see --list-models)")
    ap.add_argument("--out", default=None, help="Output path (default: data/labels/openai_labels.json)")
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--limit", type=int, default=None, help="Label only the first N pairs (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=None, help="Omit for the model default; some models reject it")
    ap.add_argument("--reasoning-effort", default=None, help="For reasoning models: low|medium|high")
    ap.add_argument("--list-models", action="store_true", help="Print available model ids and exit")
    args = ap.parse_args()

    key = api_key(args)

    if args.list_models:
        models = sorted(m["id"] for m in request("/models", key)["data"])
        print("\n".join(models))
        return

    if not Path(args.pairs).exists():
        sys.exit(f"pair set not found at {args.pairs}; pass --pairs <pairs.json> (see data/examples/)")
    pairs = json.load(open(args.pairs))["pairs"]
    if args.limit:
        pairs = pairs[: args.limit]
    extra: dict = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort

    out_path = Path(args.out) if args.out else ROOT / "data" / "labels" / "openai_labels.json"
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def work(p):
        try:
            return p["id"], label_one(p, key, args.model, extra), None
        except Exception as e:  # keep going; report at the end
            return p["id"], None, str(e)

    print(f"model={args.model} pairs={len(pairs)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (pid, res, err) in enumerate(ex.map(work, pairs), 1):
            if err:
                errors[pid] = err
                print(f"  [{i}/{len(pairs)}] {pid} ERROR {err[:120]}", flush=True)
            else:
                results[pid] = res
                print(f"  [{i}/{len(pairs)}] {pid} {res['label']}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "rater": args.model,
        "note": "Independent rater labels, blind to model_key.json. Same rubric text as the labeling page.",
        "settings": {"temperature": args.temperature, "reasoning_effort": args.reasoning_effort},
        "labels": results,
        "errors": errors,
    }, indent=1))
    from collections import Counter
    print("\nlabels:", dict(Counter(r["label"] for r in results.values())))
    tok = sum(r.get("input_tokens") or 0 for r in results.values())
    print(f"errors: {len(errors)}  input tokens: {tok}  written: {out_path}")


if __name__ == "__main__":
    main()
