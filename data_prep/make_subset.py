"""Materialise the shared evaluation subsets so every method -- DSRE-LLM and all
baselines -- is trained and scored on exactly the same instances."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre.data import load_nyt10, load_gids, stratified_subsample

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["nyt10", "gids"], required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--limit", type=int, required=True)
ap.add_argument("--data-root", default=os.path.expanduser("~/dsre/data"))
ap.add_argument("--cache-root", default=os.path.expanduser("~/dsre/cache"))
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

if a.dataset == "nyt10":
    insts = load_nyt10(os.path.join(a.data_root, "nyt10", "nyt10_%s.txt" % a.split))
else:
    insts = load_gids(os.path.join(a.data_root, "gids", "%s.tsv" % a.split))
print("[load] %d" % len(insts))
sub = stratified_subsample(insts, a.limit, seed=a.seed)
out = os.path.join(a.cache_root, "%s_%s_subset.json" % (a.dataset, a.split))
os.makedirs(a.cache_root, exist_ok=True)
json.dump([{"text": x["text"], "relation": x["relation"], "h": x["h"], "t": x["t"]} for x in sub],
          open(out, "w"), ensure_ascii=False)
from collections import Counter
c = Counter(x["relation"] for x in sub)
print("[subset] %d instances -> %s" % (len(sub), out))
print("[subset] NA=%.1f%%  distinct relations=%d" % (100 * c.get("NA", 0) / len(sub), len(c)))
