"""Retrieval-based few-shot ICL (kNN demonstrations, in the spirit of GPT-RE): for each test sentence the k most similar training sentences are used as demonstrations.
Similarity: TF-IDF cosine (1-2 grams, sublinear tf) after replacing the head/tail mentions by placeholders; retrieval only from the training set, no test/validation labels.
Prompt: numbered relation list (shared by all sentences, prefix-cache friendly) -> k demonstrations (sentence, entities, answer number) -> test sentence; scoring as in the probabilistic zero-shot version.
Usage as zeroshot_icl.py, plus --k."""
import argparse, json, os, re, sys, threading, time
import concurrent.futures as cf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E
from train import load_cache
from zeroshot_lp import HEAD, BODY
from zeroshot_icl import score


def masked(x):
    s = x["text"]
    for name, tag in sorted([(x["h"]["name"], " headent "), (x["t"]["name"], " tailent ")], key=lambda p: -len(p[0])):
        if name:
            s = re.sub(re.escape(name), tag, s, flags=re.I)
    return s


def neighbors(train, insts, k, bs=4096):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)
    Xtr = vec.fit_transform([masked(x) for x in train]); Xq = vec.transform([masked(x) for x in insts])
    out = np.zeros((len(insts), k), dtype=np.int64)
    for s in range(0, len(insts), bs):
        S = (Xq[s:s + bs] @ Xtr.T).toarray()
        top = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.take_along_axis(S, top, 1).argsort(1)            # ascending: the most similar demonstration is placed last, next to the test sentence
        out[s:s + bs] = np.take_along_axis(top, order, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True); ap.add_argument("--rel2id", required=True); ap.add_argument("--train-file", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--base-url", default="http://localhost:8003"); ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="Qwen3-8B-kNN-ICL-Prob")
    ap.add_argument("--tiers", default=None); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=200); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id["NA"]
    rel_rows = [r for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    opt_of = {"NA": 0}; opt_of.update({r: i + 1 for i, r in enumerate(rel_rows)})
    opt2cls = [na_id] + [rel2id[r] for r in rel_rows]
    head0 = HEAD.format(opts="\n".join("%d. %s" % (i, r) for i, r in enumerate(["NA"] + rel_rows)))
    train = D.load_jsonl(a.train_file); insts = D.load_jsonl(a.test_file, limit=a.limit)
    cache = load_cache(a.cache)
    key = lambda x: (x["text"], x["h"]["name"], x["t"]["name"])
    first = {}                                      # identical (sentence, head, tail) triples are queried once (greedy decoding gives the same answer) and copied
    for i, x in enumerate(insts):
        first.setdefault(key(x), i)
    todo_i = [i for i, x in enumerate(insts) if x["uid"] not in cache and first[key(x)] == i]
    t0 = time.time(); nb = neighbors(train, [insts[i] for i in todo_i], a.k) if todo_i else None
    print("[knn] %d instances, %d to query, K=%d, k=%d, retrieval %.0fs" % (len(insts), len(todo_i), K, a.k, time.time() - t0), flush=True)
    lock = threading.Lock(); fh = open(a.cache, "a", encoding="utf-8")

    def work(j):
        x = insts[todo_i[j]]
        demo = "Examples:\n" + "\n".join(BODY.format(sent=train[n]["text"][:400], h=train[n]["h"]["name"], t=train[n]["t"]["name"])
                                         + "\nAnswer: %d\n" % opt_of.get(train[n]["relation"], 0) for n in nb[j]) + "\nNow the question:\n"
        P = score(a.base_url, a.model, head0 + demo, x, len(opt2cls))
        with lock:
            cache[x["uid"]] = [round(float(v), 7) for v in P]
            fh.write(json.dumps({"k": x["uid"], "v": cache[x["uid"]]}) + "\n"); fh.flush()

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(work, range(len(todo_i))), 1):
            if i % 2000 == 0:
                r = i / (time.time() - t0)
                print("  %d/%d  %.1f it/s  eta %.1f min" % (i, len(todo_i), r, (len(todo_i) - i) / r / 60), flush=True)
    for i, x in enumerate(insts):
        if x["uid"] not in cache:
            src = insts[first[key(x)]]["uid"]; cache[x["uid"]] = cache[src]
            fh.write(json.dumps({"k": x["uid"], "v": cache[src]}) + "\n")
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
    res.update({"answered_NA_frac": float((probs.argmax(1) == na_id).mean()), "mean_option_mass": float(np.mean(mass)), "k": a.k})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.name, "test_file": a.test_file, "result": res}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
