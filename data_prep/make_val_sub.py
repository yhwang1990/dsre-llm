"""NYT10 validation subset used to calibrate the zero-shot / ICL baselines (val_sub15k.jsonl, about 15k sentences).
Whole entity pairs are drawn from val.jsonl in a fixed random order (seed 42) until at least 15,000 sentences are kept.
Usage: python make_val_sub.py <data_prepared>/nyt10/val.jsonl <data_prepared>/nyt10/val_sub15k.jsonl"""
import json, random, sys

X = [json.loads(l) for l in open(sys.argv[1])]
key = lambda x: (x["h"]["name"], x["t"]["name"])
pairs = sorted({key(x) for x in X}); random.Random(42).shuffle(pairs)
cnt = {}
for x in X:
    cnt[key(x)] = cnt.get(key(x), 0) + 1
keep, n = set(), 0
for p in pairs:
    if n >= 15000:
        break
    keep.add(p); n += cnt[p]
with open(sys.argv[2], "w") as f:
    for x in X:
        if key(x) in keep:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("val subset", n)
