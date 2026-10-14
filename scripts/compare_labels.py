#!/usr/bin/env python3
"""Compare label files for the pair set against each other and the model key.

Usage:
  python scripts/compare_labels.py data/labels/claude_labels.json data/labels/openai_labels.json [more.json]

Each file has {"rater": ..., "labels": {pair_id: {"label": ...}}}. The model
key (data/labels/model_key.json) supplies the stratum and, where the first
pass emitted one, the first pass's own label.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["same_question", "subset_question", "adjacent_question", "surface_overlap"]
LINK = {"same_question", "subset_question"}


def load(path):
    d = json.load(open(path))
    return d.get("rater", Path(path).stem), {k: v["label"] for k, v in d["labels"].items() if v.get("label") in LABELS}


def confusion(a: dict, b: dict, name_a: str, name_b: str) -> None:
    common = [k for k in a if k in b]
    if not common:
        print(f"  no common pairs between {name_a} and {name_b}")
        return
    exact = sum(a[k] == b[k] for k in common) / len(common)
    dec = sum((a[k] in LINK) == (b[k] in LINK) for k in common) / len(common)
    # Cohen's kappa on the 4-way label
    pa = exact
    ma, mb = Counter(a[k] for k in common), Counter(b[k] for k in common)
    pe = sum(ma[l] * mb[l] for l in LABELS) / (len(common) ** 2)
    kappa = (pa - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"  {name_a} vs {name_b}: n={len(common)} exact={exact:.2f} link/not-link={dec:.2f} kappa={kappa:.2f}")
    print(f"    rows={name_a[:12]}, cols={name_b[:12]}: " + " ".join(f"{l[:7]:>8}" for l in LABELS))
    for la in LABELS:
        row = [sum(1 for k in common if a[k] == la and b[k] == lb) for lb in LABELS]
        print(f"    {la[:16]:<16} " + " ".join(f"{n:8d}" for n in row))


def main() -> None:
    args = sys.argv[1:]
    key_path = ROOT / "data" / "labels" / "model_key.json"
    if "--key" in args:
        i = args.index("--key")
        key_path = Path(args[i + 1])
        del args[i:i + 2]
    files = args
    if not files:
        sys.exit(__doc__)
    if not key_path.exists():
        sys.exit(f"model key not found at {key_path}; pass --key <model_key.json>")
    key = json.load(open(key_path))
    raters = [load(f) for f in files]
    model = {pid: v["model_label"] for pid, v in key.items() if v.get("model_label") in LABELS}

    print("LABEL DISTRIBUTION")
    for name, labs in raters:
        print(f"  {name:<28} n={len(labs):3d} {dict(Counter(labs.values()))}")
    print(f"  {'first pass':<28} n={len(model):3d} {dict(Counter(model.values()))}")

    print("\nPAIRWISE AGREEMENT")
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            confusion(raters[i][1], raters[j][1], raters[i][0], raters[j][0])
    for name, labs in raters:
        confusion(model, labs, "first pass", name)

    print("\nLINK-WORTHY SHARE BY STRATUM (same or subset)")
    strata = sorted({v["stratum"] for v in key.values()})
    header = f"  {'stratum':<28} " + " ".join(f"{name[:14]:>14}" for name, _ in raters) + f" {'first pass':>10}"
    print(header)
    for s in strata:
        ids = [pid for pid, v in key.items() if v["stratum"] == s]
        cells = []
        for _, labs in raters:
            have = [pid for pid in ids if pid in labs]
            cells.append(f"{sum(labs[p] in LINK for p in have):3d}/{len(have):<3d}" if have else "   -   ")
        mh = [pid for pid in ids if pid in model]
        mp = f"{sum(model[p] in LINK for p in mh):3d}/{len(mh):<3d}" if mh else "   -   "
        print(f"  {s:<28} " + " ".join(f"{c:>14}" for c in cells) + f" {mp:>10}")

    if len(raters) >= 2:
        print("\nUNANIMOUS vs CONTESTED (across rater files)")
        common = set.intersection(*(set(l) for _, l in raters))
        unanimous = [k for k in common if len({l[k] for _, l in raters}) == 1]
        dec_unan = [k for k in common if len({l[k] in LINK for _, l in raters}) == 1]
        print(f"  common pairs={len(common)} unanimous label={len(unanimous)} unanimous link/not-link={len(dec_unan)}")
        contested = sorted(k for k in common if k not in dec_unan)
        if contested:
            print("  contested on link/not-link:", ", ".join(contested))


if __name__ == "__main__":
    main()
