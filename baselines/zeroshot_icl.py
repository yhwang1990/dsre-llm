"""Few-shot in-context learning (ICL) baseline: demonstrations are inserted into the prompt of the probabilistic zero-shot version (zeroshot_lp.py) after the relation list and before the test sentence,
one per non-NA relation and --n-na for NA. Demonstrations come from the training set with a fixed rule (no test data, no anchors or ontology of our method):
the shortest sentence of each relation, and NA demonstrations drawn with a fixed seed.
The demonstration block is identical for all test sentences and placed in the prefix (vLLM prefix caching); scoring is identical to the zero-shot version.
Usage as zeroshot_lp.py, plus --train-file / --n-na."""
import argparse, json, os, random, sys, threading, time
import concurrent.futures as cf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E
from train import load_cache
from zeroshot_lp import HEAD, BODY, CHAT, complete


def score(url, model, head, x, n_opt, prune=0.002):
    """As zeroshot_lp.score, but two-digit branches whose first-digit probability is below `prune` are not expanded (fewer calls per sentence);
    the pruned mass (< prune) is kept on the one-digit number, which has a negligible effect on the ranking."""
    base = CHAT.format(u=head + BODY.format(sent=x["text"][:1200], h=x["h"]["name"], t=x["t"]["name"]))
    p1 = complete(url, model, base); P = np.zeros(n_opt)
    for d1 in range(min(10, n_opt)):
        P[d1] = p1[str(d1)]
    for d1 in range(1, (n_opt - 1) // 10 + 1):
        if p1[str(d1)] < prune:
            continue
        q = complete(url, model, base + str(d1))
        P[d1] = p1[str(d1)] * max(1.0 - sum(q.values()), 0.0)
        for d2 in range(10):
            n = 10 * d1 + d2
            if n < n_opt:
                P[n] = p1[str(d1)] * q[str(d2)]
    return P


def demos(train_file, rel2id, n_na, seed=42):
    X = D.load_jsonl(train_file); by = {}
    for x in X:
        if len(x["text"]) >= 40:
            by.setdefault(x["relation"], []).append(x)
    out = []
    for r in sorted(rel2id, key=rel2id.get):
        if r != "NA" and by.get(r):
            out.append(min(by[r], key=lambda x: (len(x["text"]), x["uid"])))
    out += random.Random(seed).sample(by.get("NA", []), min(n_na, len(by.get("NA", []))))
    random.Random(seed).shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True); ap.add_argument("--rel2id", required=True); ap.add_argument("--train-file", required=True)
    ap.add_argument("--n-na", type=int, default=3)
    ap.add_argument("--base-url", default="http://localhost:8003"); ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="Qwen3-8B-ICL-Prob")
    ap.add_argument("--tiers", default=None); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=200); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id["NA"]
    rel_rows = [r for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    opt_of = {"NA": 0}; opt_of.update({r: i + 1 for i, r in enumerate(rel_rows)})
    opt2cls = [na_id] + [rel2id[r] for r in rel_rows]
    ex = demos(a.train_file, rel2id, a.n_na)
    block = "Examples:\n" + "\n".join(BODY.format(sent=x["text"][:400], h=x["h"]["name"], t=x["t"]["name"]) + "\nAnswer: %d\n" % opt_of[x["relation"]] for x in ex) + "\nNow the question:\n"
    head = HEAD.format(opts="\n".join("%d. %s" % (i, r) for i, r in enumerate(["NA"] + rel_rows))) + block
    insts = D.load_jsonl(a.test_file, limit=a.limit)
    cache = load_cache(a.cache); lock = threading.Lock(); fh = open(a.cache, "a", encoding="utf-8")
    todo = [x for x in insts if x["uid"] not in cache]
    print("[icl] %d instances, %d to query, K=%d, %d demos, head %d chars" % (len(insts), len(todo), K, len(ex), len(head)), flush=True)

    def work(x):
        P = score(a.base_url, a.model, head, x, len(opt2cls))
        with lock:
            cache[x["uid"]] = [round(float(v), 7) for v in P]
            fh.write(json.dumps({"k": x["uid"], "v": cache[x["uid"]]}) + "\n"); fh.flush()

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 2000 == 0:
                r = i / (time.time() - t0)
                print("  %d/%d  %.1f it/s  eta %.1f min" % (i, len(todo), r, (len(todo) - i) / r / 60), flush=True)
    fh.close()
    probs = np.full((len(insts), K), 1e-8)
    for i, x in enumerate(insts):
        for o, v in enumerate(cache[x["uid"]]):
            probs[i, opt2cls[o]] += v
    mass = probs.sum(1); probs /= mass[:, None]
    gold = np.array([rel2id.get(x["relation"], na_id) for x in insts]); bags = [D.bag_key(x) for x in insts]
    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]; tier_of_class = {rel2id[r]: v for r, v in t.items() if r in rel2id}
    res = E.evaluate(probs, gold, bags, K, na_id=na_id, tier_of_class=tier_of_class)
    if a.bootstrap:
        res["bootstrap"] = E.bootstrap_ci(probs, gold, bags, K, na_id=na_id, n_boot=a.bootstrap)
    res.update({"answered_NA_frac": float((probs.argmax(1) == na_id).mean()), "mean_option_mass": float(np.mean(mass)), "n_demos": len(ex)})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.name, "test_file": a.test_file, "demos": [x["uid"] for x in ex], "result": res}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
