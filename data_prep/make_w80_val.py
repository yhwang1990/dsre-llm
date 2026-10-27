"""Validation set for selecting c_max on Wiki80: from the sentences of the official wiki80_train.txt that are not in train_main,
70 sentences per relation are drawn (stratified, seed 7) = 5,600 sentences, disjoint from the test set (official val).
Label noise is injected with the same protocol as the training set (same inject_noise, same asymmetric transition matrix, seed 7), so c_max is selected on noisy labels only.
Writes val_cmax.jsonl (clean labels + uid) and val_cmax_labels.json (noisy label of every sentence for each noise setting)."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D

R = os.environ.get("DSRE_ROOT", "workspace"); W = R + "/data_prepared/wiki80"
wtr = D.load_wiki80(R + "/data/wiki80/wiki80_train.txt")
used = {x["uid"] for x in D.load_jsonl(W + "/train_main.jsonl")}
test = {x["uid"] for x in D.load_jsonl(W + "/test.jsonl")}
pool = [x for x in wtr if x["uid"] not in used and x["uid"] not in test]
rng = random.Random(7); by = {}
for x in pool:
    by.setdefault(x["relation"], []).append(x)
val = []
for r in sorted(by):
    xs = by[r][:]; rng.shuffle(xs); val += xs[:70]
D.save_jsonl(val, W + "/val_cmax.jsonl")
T = json.load(open(R + "/data/wiki80_asym_T.json"))["transition"]
rels = sorted(json.load(open(W + "/rel2id.json")))
labels = {"sym_0": [x["relation"] for x in val]}
for mode, short in (("symmetric", "sym"), ("asymmetric", "asym")):
    for rate in (0.1, 0.2, 0.3, 0.4):
        noisy = D.inject_noise(val, rate, mode, seed=7, transition=T, relations=rels)
        labels["%s_%d" % (short, int(rate * 100))] = [x["relation"] for x in noisy]
json.dump(labels, open(W + "/val_cmax_labels.json", "w"))
print("pool", len(pool), "val", len(val), {k: round(sum(a != b for a, b in zip(v, labels["sym_0"])) / len(val), 3) for k, v in labels.items()})
