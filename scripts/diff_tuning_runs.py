"""Diff two tuning runs: summary side-by-side, and item-by-item where items overlap."""
import json, sys
from collections import Counter
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
def tid(d): return d.get("topic_id", d.get("theme_id"))  # reports before the rename say theme_id
def new_tid(d): return tid(d.get("new_topic") or d.get("new_theme") or {})
def hdr(r): return f"{r.get('label') or r['run_at'][:10]} model={r['config']['model']} topics={r.get('topic_universe', r.get('theme_universe', '?'))} tag_overlap={r['config'].get('candidates',{}).get('tag_overlap','?')}"
print("A:", hdr(a)); print("B:", hdr(b))
for k in ("link_existing","propose_new","skip"):
    print(f"  {k:<14} A={a['summary']['decisions'][k]:>3} ({a['summary']['decisions_pct'][k]}%)   B={b['summary']['decisions'][k]:>3} ({b['summary']['decisions_pct'][k]}%)")
da = {d["item_url"]: d for d in a["decisions"]}; db = {d["item_url"]: d for d in b["decisions"]}
common = [u for u in da if u in db]
if common:
    print(f"\nitem-by-item ({len(common)} common items): A -> B")
    trans = Counter((da[u]["decision"], db[u]["decision"]) for u in common)
    for (x,y),n in trans.most_common(): print(f"  {x:>13} -> {y:<13} {n}")
    print()
    for u in common:
        x, y = da[u], db[u]
        if x["decision"] != y["decision"] or (x["decision"]=="link_existing" and tid(x)!=tid(y)):
            tx = tid(x) or new_tid(x) or "-"
            ty = tid(y) or new_tid(y) or "-"
            print(f"  {x['item_title'][:55]:<55} | {x['decision']:>13}:{tx[:30]:<30} -> {y['decision']:>13}:{ty[:30]}")
