"""Probabilistic zero-shot baseline:
the prompt only states that 0 = none of the listed relations, and
instead of the self-reported confidence it reads the generation probabilities of the candidate numbers (top-20 logprobs of vLLM completions),
giving a continuous distribution over all K relations that is evaluated with the same bag-level protocol as the other methods.
With more than 10 options, two-digit numbers are scored by chaining: P(n=d1d2) = P(d1) * P(d2|d1), P(n=d1) = P(d1) * P(end|d1).
Usage: python zeroshot_lp.py --test-file test.jsonl --rel2id rel2id.json --base-url http://localhost:8003 --model Qwen3-8B --out <dir>
"""
import argparse, json, math, os, sys, threading, time, urllib.request
import concurrent.futures as cf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E
from train import load_cache

HEAD = ("Below is a numbered list of relation types. Option 0 means that none of the listed relations "
        "holds between the two entities (NA).\n{opts}\n\n")
BODY = ('Sentence: "{sent}"\nHead entity: "{h}"\nTail entity: "{t}"\n'
        "Which relation does the sentence express between the head entity and the tail entity? "
        "Answer with the option number only.")
CHAT = "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DIG = [str(d) for d in range(10)]


def complete(url, model, prompt, timeout=600):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": 20}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]
            p = dict.fromkeys(DIG, 0.0)
            for tok, lp in top.items():
                if tok.strip() in p:
                    p[tok.strip()] += math.exp(lp)
            return p
        except Exception as e:
            last = e; time.sleep(2 * (attempt + 1))
    raise last


def score(url, model, head, x, n_opt):
    base = CHAT.format(u=head + BODY.format(sent=x["text"][:1200], h=x["h"]["name"], t=x["t"]["name"]))
    p1 = complete(url, model, base); P = np.zeros(n_opt)
    for d1 in range(10):
        if d1 < n_opt:
            P[d1] = p1[str(d1)]
    for d1 in range(1, (n_opt - 1) // 10 + 1):   # two-digit numbers need P(d2|d1) and P(end|d1)
        q = complete(url, model, base + str(d1))
        P[d1] = p1[str(d1)] * max(1.0 - sum(q.values()), 0.0)
        for d2 in range(10):
            n = 10 * d1 + d2
            if n < n_opt:
                P[n] = p1[str(d1)] * q[str(d2)]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True); ap.add_argument("--rel2id", required=True)
    ap.add_argument("--base-url", default="http://localhost:8003"); ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--cache", required=True, help="per-instance probability cache (resumable)")
    ap.add_argument("--out", required=True); ap.add_argument("--name", default="Qwen3-8B-ZeroShot-Prob")
    ap.add_argument("--tiers", default=None); ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=200); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id["NA"]
    rel_rows = [r for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    opt2cls = [na_id] + [rel2id[r] for r in rel_rows]           # option 0 = NA, the others follow rel2id order
    head = HEAD.format(opts="\n".join("%d. %s" % (i, r) for i, r in enumerate(["NA"] + rel_rows)))
    insts = D.load_jsonl(a.test_file, limit=a.limit)
    cache = load_cache(a.cache); lock = threading.Lock(); fh = open(a.cache, "a", encoding="utf-8")
    todo = [x for x in insts if x["uid"] not in cache]
    print("[zs-prob] %d instances, %d to query, K=%d" % (len(insts), len(todo), K), flush=True)

    def work(x):
        P = score(a.base_url, a.model, head, x, len(opt2cls))
        with lock:
            cache[x["uid"]] = [round(float(v), 7) for v in P]
            fh.write(json.dumps({"k": x["uid"], "v": cache[x["uid"]]}) + "\n"); fh.flush()

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, todo), 1):
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
    res.update({"answered_NA_frac": float((probs.argmax(1) == na_id).mean()),
                "mean_option_mass": float(np.mean(mass)), "n_queried": len(todo)})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.name, "test_file": a.test_file, "result": res}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
