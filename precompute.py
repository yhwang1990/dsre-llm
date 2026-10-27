"""LLM pre-computation (all results cached on disk by instance uid / entity name / relation name; resumable):
  types    : entity -> coarse type (CoT step 1)
  probe    : ontology leakage probe (mean |R_constr|, share collapsed to a single relation)
  augment  : non-NA training instance -> 2 paraphrases + 1 hard negative
  anchors  : relations with fewer than N_min (or zero) training sentences -> description sentences (centroid anchors)
  votes    : ontology-guided six-step CoT relation prediction (--uids restricts it to the on-demand instances)
  zeroshot : zero-shot baseline (all relations as candidates, no CoT)
Call counts and token usage of each stage are written to cache-root/usage_<stage>.json.
"""
import argparse, json, os, sys, threading, time
import concurrent.futures as cf
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre.ontology import Ontology
from dsre import llm as L


class Cache:
    def __init__(self, path):
        self.path = path; self.d = {}; self.lock = threading.Lock()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line); self.d[r["k"]] = r["v"]
                    except Exception:
                        pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.fh.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n"); self.fh.flush()


def run_pool(fn, items, workers, desc):
    done, t0, n = 0, time.time(), len(items)
    with cf.ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(fn, items):
            done += 1
            if done % 500 == 0 or done == n:
                el = time.time() - t0; r = done / max(el, 1e-9)
                print("  %s %d/%d  %.1f it/s  eta %.1f min" % (desc, done, n, r, (n - done) / max(r, 1e-9) / 60), flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="jsonl instance file (produced by prepare_data.py)")
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--tag", default=None, help="cache file prefix (default: basename of --file)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--stages", default="types,probe", help="types,probe,augment,anchors,votes,zeroshot")
    ap.add_argument("--uids", default=None, help="votes/zeroshot: process only the uids in this file (one per line or a json list)")
    ap.add_argument("--n-min", type=int, default=30, help="anchors: generate anchors for relations with fewer training sentences")
    ap.add_argument("--n-anchor", type=int, default=8)
    ap.add_argument("--ontology-variant", default=None, help="alternative ontology json for votes (coarse / noisy)")
    ap.add_argument("--no-onto", action="store_true", help="votes: candidates = all relations (CoT kept)")
    a = ap.parse_args()

    tag = a.tag or os.path.splitext(os.path.basename(a.file))[0]
    insts = D.load_jsonl(a.file)
    rel2id = json.load(open(a.rel2id)); all_rels = sorted(rel2id, key=lambda r: rel2id[r])
    has_na = "NA" in rel2id
    onto = Ontology(a.ontology_variant or a.ontology, relations=[r for r in all_rels if r != "NA"], has_na=has_na)
    client = L.LLMClient(a.base_url, a.model)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    os.makedirs(a.cache_root, exist_ok=True)
    print("[load] %s: %d instances, K=%d, stages=%s" % (a.file, len(insts), len(rel2id), stages), flush=True)
    usage = {}; _last = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def record(stage, secs):
        tot = client.usage(); u = {k: tot[k] - _last[k] for k in _last}; _last.update(tot)
        u["seconds"] = secs; usage[stage] = u
        json.dump(usage, open(os.path.join(a.cache_root, "usage_%s.json" % tag), "w"), indent=1)

    # ---------------------------------------------------------------- types
    tcache = Cache(os.path.join(a.cache_root, "entity_types.jsonl"))
    ents = {}
    for x in insts:
        for side in ("h", "t"):
            ents.setdefault(x[side]["name"], x["text"])
    if "types" in stages:
        todo = [(n, c) for n, c in ents.items() if tcache.get(n) is None]
        print("[types] unique=%d to_query=%d" % (len(ents), len(todo)), flush=True)

        def do_type(nc):
            name, ctx = nc
            try:
                tcache.put(name, L.type_entity(client, name, ctx))
            except Exception:
                tcache.put(name, "OTHER")
        record("types", run_pool(do_type, todo, a.workers, "types"))
    types = {n: (tcache.get(n) or "OTHER") for n in ents}

    # ---------------------------------------------------------------- probe
    if "probe" in stages:
        pairs = [(types[x["h"]["name"]], types[x["t"]["name"]]) for x in insts]
        probe = onto.stats(pairs)
        probe["type_distribution"] = dict(Counter(types.values()))
        json.dump(probe, open(os.path.join(a.cache_root, "%s_onto_probe.json" % tag), "w"), indent=1)
        print("[probe] mean|R_constr|=%.2f  single-relation frac=%.4f  (n=%d)"
              % (probe["mean_size"], probe["frac_single_relation"], probe["n"]), flush=True)

    # ---------------------------------------------------------------- augment (non-NA only)
    if "augment" in stages:
        acache = Cache(os.path.join(a.cache_root, "%s_aug.jsonl" % tag))
        todo = [x for x in insts if x["relation"] != "NA" and acache.get(x["uid"]) is None]
        print("[augment] nonNA=%d to_query=%d" % (sum(x["relation"] != "NA" for x in insts), len(todo)), flush=True)

        def do_aug(x):
            try:
                pos, neg = L.augment(client, x["text"], x["h"]["name"], x["t"]["name"], x["relation"])
            except Exception:
                pos, neg = [], None
            acache.put(x["uid"], {"pos": pos, "neg": neg})
        record("augment", run_pool(do_aug, todo, a.workers, "augment"))

    # ---------------------------------------------------------------- anchors (long-tail relation descriptions)
    if "anchors" in stages:
        ccache = Cache(os.path.join(a.cache_root, "%s_anchors.jsonl" % tag))
        cnt = Counter(x["relation"] for x in insts)
        rare = [r for r in all_rels if r != "NA" and cnt.get(r, 0) < a.n_min]
        todo = [r for r in rare if ccache.get(r) is None]
        print("[anchors] rare(<%d)=%d to_query=%d" % (a.n_min, len(rare), len(todo)), flush=True)

        def do_anchor(r):
            try:
                ccache.put(r, L.anchor_sentences(client, r, a.n_anchor))
            except Exception:
                ccache.put(r, [])
        record("anchors", run_pool(do_anchor, todo, min(a.workers, 16), "anchors"))

    # ---------------------------------------------------------------- votes / zeroshot
    want = None
    if a.uids:
        txt = open(a.uids).read().strip()
        want = set(json.loads(txt)) if txt.startswith("[") else set(txt.split())
    if "votes" in stages:
        name = "%s_votes%s.jsonl" % (tag, ("_" + os.path.splitext(os.path.basename(a.ontology_variant))[0]) if a.ontology_variant
                                      else ("_noonto" if a.no_onto else ""))
        vcache = Cache(os.path.join(a.cache_root, name))
        todo = [x for x in insts if vcache.get(x["uid"]) is None and (want is None or x["uid"] in want)]
        print("[votes] to_query=%d / %d  (restricted=%s)" % (len(todo), len(insts), want is not None), flush=True)

        def do_vote(x):
            ht, tt = types[x["h"]["name"]], types[x["t"]["name"]]
            cands = all_rels if a.no_onto else onto.constrained_space(ht, tt)
            try:
                rel, conf, why = L.predict_relation(client, x["text"], x["h"]["name"], x["t"]["name"], ht, tt, cands)
            except Exception:
                rel, conf, why = ("NA" if has_na else cands[0]), 0.0, "error"
            vcache.put(x["uid"], {"rel": rel, "conf": conf, "n_cand": len(cands), "why": why})
        record("votes", run_pool(do_vote, todo, a.workers, "votes"))

    if "zeroshot" in stages:
        zcache = Cache(os.path.join(a.cache_root, "%s_zeroshot.jsonl" % tag))
        todo = [x for x in insts if zcache.get(x["uid"]) is None and (want is None or x["uid"] in want)]
        print("[zeroshot] to_query=%d / %d" % (len(todo), len(insts)), flush=True)

        def do_zs(x):
            try:
                rel, conf = L.predict_relation_zeroshot(client, x["text"], x["h"]["name"], x["t"]["name"], all_rels)
            except Exception:
                rel, conf = ("NA" if has_na else all_rels[0]), 0.0
            zcache.put(x["uid"], {"rel": rel, "conf": conf})
        record("zeroshot", run_pool(do_zs, todo, a.workers, "zeroshot"))

    print("[done] usage=%s" % json.dumps(client.usage()), flush=True)


if __name__ == "__main__":
    main()
