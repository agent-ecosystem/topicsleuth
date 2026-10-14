"""Compare confirmation backends on one seed: baseline (no confirm), opus, jev."""
import json, sys

def tid(d): return d.get("topic_id", d.get("theme_id"))  # reports before the rename say theme_id
def new_tid(d): return tid(d.get("new_topic") or d.get("new_theme") or {})
from collections import Counter
base, opus, jev = [json.load(open(f)) for f in sys.argv[1:4]]
def by(r): return {d["item_url"]: d for d in r["decisions"]}
B, O, J = by(base), by(opus), by(jev)
def dist(r): s=r["summary"]["decisions"]; return f"{s['link_existing']:2d} link / {s['propose_new']:2d} propose / {s['skip']:2d} skip"
print(f"baseline (no confirm): {dist(base)}")
print(f"opus confirm:          {dist(opus)}   outcomes {opus['summary']['diagnostics'].get('confirmation')}")
print(f"jev confirm:           {dist(jev)}   outcomes {jev['summary']['diagnostics'].get('confirmation')}")
print("\nFIRST-PASS PROPOSALS IN THE JEV RUN (reviewer decisions, with opus run's decision on the same item)")
for u, d in J.items():
    g = d.get("diagnostics") or {}
    if not g.get("confirmation"): continue
    scores = g.get("jev_scores") or []
    best = max(scores, key=lambda s: s["link_probability"]) if scores else None
    o = O.get(u); og = (o or {}).get("diagnostics") or {}
    otxt = f"{o['decision']}:{(tid(o) or new_tid(o) or '-')[:28]} [{og.get('confirmation') or 'first-pass'}]" if o else "-"
    jtxt = f"{d['decision']}:{(tid(d) or new_tid(d) or '-')[:28]}"
    btxt = f"best={tid(best)[:24]} p={best['link_probability']:.2f} {best.get('label','?')[:7]}" if best else "no scores"
    print(f"  {d['item_title'][:42]:<42} jev-> {jtxt:<42} {btxt:<52} | opus-> {otxt}")
print("\nITEM-BY-ITEM DECISION AGREEMENT")
common = [u for u in J if u in O and u in B]
print("  opus vs jev:", dict(Counter((O[u]["decision"], J[u]["decision"]) for u in common)))
same_target = sum(1 for u in common if O[u]["decision"]==J[u]["decision"]=="link_existing" and tid(O[u])==tid(J[u]))
links_both = sum(1 for u in common if O[u]["decision"]==J[u]["decision"]=="link_existing")
print(f"  both link: {links_both}, same topic: {same_target}")
