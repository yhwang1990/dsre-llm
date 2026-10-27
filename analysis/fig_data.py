"""Data of Figure 5 (PR curves) and Figure 7 (long-tail tiers and per-relation F1), from the same runs as Table 2.
DSRE-LLM is re-fused at the selected c_max; Qwen3-8B-ICL is rebuilt from the cached LLM scores with the calibration selected
on the validation set. PR curves use seed 42; tier Macro-F1 and per-relation F1 are averaged over the available seeds.
Usage: python fig_data.py --ds nyt10|gids    Output: results/figdata/figdata_<ds>.json"""
import argparse, collections, json, os
import numpy as np
from common import D, E, R, RUNS, P, C, RES, FN, CMAX, FTLR, BEST, ONTO, Ontology, load_cache

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True, choices=["nyt10", "gids"])
a = ap.parse_args(); ds = a.ds
cfg = {"nyt10": dict(tf=f"{P}/nyt10/test_full.jsonl", tr=f"{P}/nyt10/train_main.jsonl", ev="eval_test_full", votes=f"{C}/nyt10/test_full_votes.jsonl",
                     tiers=f"{P}/nyt10/tiers_main.json", knn=(f"{P}/nyt10/val_sub15k.jsonl", f"{C}/nyt10/val_sub15k_knn.jsonl", f"{C}/nyt10/test_full_knn.jsonl", f"{RUNS}/bl_knncal_8b")),
       "gids": dict(tf=f"{P}/gids/test.jsonl", tr=f"{P}/gids/train.jsonl", ev="eval_test", votes=f"{C}/gids/test_votes.jsonl", tiers=None,
                    knn=(f"{P}/gids/dev.jsonl", f"{C}/gids/dev_knn.jsonl", f"{C}/gids/test_knn.jsonl", f"{RUNS}/gids_bl_knncal_8b"))}[ds]
rel2id = json.load(open(f"{P}/{ds}/rel2id.json")); K = len(rel2id); na = rel2id["NA"]; id2rel = {v: k for k, v in rel2id.items()}
X = D.load_jsonl(cfg["tf"]); gold = np.array([rel2id.get(x["relation"], na) for x in X]); bags = [D.bag_key(x) for x in X]
onto = Ontology(ONTO, relations=[r for r in sorted(rel2id, key=rel2id.get) if r != "NA"], has_na=True)
types = load_cache(f"{C}/{ds}/entity_types.jsonl")


def ours(s):
    ev = f"{RUNS}/{ds}_{FN}_8b_s{s}_v3/{cfg['ev']}"
    if not os.path.exists(f"{ev}/probs.npy"):
        return None
    Pc = np.load(f"{ev}/probs.npy"); votes = load_cache(cfg["votes"]); c = CMAX[ds]; fused = Pc.copy()
    for i in np.nonzero(E.needs_llm(Pc, c))[0]:
        v = votes.get(X[i]["uid"])
        if v:
            ok = bool(onto.consistent(types.get(X[i]["h"]["name"], "OTHER"), types.get(X[i]["t"]["name"], "OTHER"), v["rel"]))
            fused[i] = E.fuse(Pc[i], v["rel"], v["conf"], rel2id, ok, c)
    return fused


def npy(path):
    return np.load(path) if os.path.exists(path) else None


def bl(m):
    return lambda s: npy(f"{RUNS}/tune/{ds}_{m}_{BEST[(ds, m)]}_s{s}/probs.npy")


def icl(s):
    vf, vc, tc, run = cfg["knn"]
    if s != 42 or not os.path.exists(f"{run}/result.json"):
        return None
    sel = json.load(open(f"{run}/result.json"))["selected"]
    rows = [na] + [rel2id[r] for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    Cv, Ct = load_cache(vc), load_cache(tc)
    prior = np.mean([Cv[x["uid"]] for x in D.load_jsonl(vf) if x["uid"] in Cv], 0) + 1e-8
    O = np.array([Ct[x["uid"]] for x in X]) + 1e-8; Q = O / prior ** sel["beta"]; Q[:, 0] *= sel["w"]
    Pm = np.zeros((len(X), K)); Pm[:, rows] = Q
    return Pm / Pm.sum(1, keepdims=True)


M = [("PCNN+ATT", bl("pcnn")), ("BERT-Focal", bl("focal")), ("RoBERTa-MIL", bl("mil")), ("BERT-NLI", bl("nli")),
     ("CGRE", bl("cgre")), ("LLM-CG", bl("llmcg")), ("Qwen3-8B-ICL", icl),
     ("Qwen3-8B-FT", lambda s: npy(f"{RUNS}/{ds}_ft_lr{FTLR}_8b_s{s}_v3/{cfg['ev']}/probs.npy")), ("DSRE-LLM (Ours)", ours)]
tier_of = None
if cfg["tiers"]:
    t = json.load(open(cfg["tiers"]))["tier_of_relation"]; tier_of = {rel2id[r]: v for r, v in t.items() if r in rel2id}
freq = collections.Counter(x["relation"] for x in D.load_jsonl(cfg["tr"]))
out = {"ds": ds, "cmax": CMAX[ds], "train_freq": {r: freq.get(r, 0) for r in rel2id if r != "NA"}, "methods": {}}
for name, fn in M:
    rec = {"seeds": []}; tiers, pcf = collections.defaultdict(list), collections.defaultdict(list)
    for s in (42, 13, 123):
        Pm = fn(s)
        if Pm is None:
            continue
        assert Pm.shape == (len(X), K), (name, s, Pm.shape)
        rec["seeds"].append(s)
        keys, bg, bp = E.bag_aggregate(Pm, bags)
        bgold = [set(int(gold[i]) for i in bg[k]) for k in keys]
        sc, lab = E._facts(bp, bgold, K, na)
        if s == 42:
            auc, pr, rc = E.pr_curve_auc(sc, lab)
            idx = np.flatnonzero(np.diff(np.concatenate([[0.0], rc])) > 0)   # one point per recalled positive
            rec["pr"] = {"auc": auc, "precision": pr[idx].round(5).tolist(), "recall": rc[idx].round(5).tolist()}
        yp = Pm.argmax(1)
        for k, (f1, sup) in E.per_class_f1(gold, yp, K).items():
            if k != na:
                pcf[id2rel[k]].append(f1)
        if tier_of:
            for t_, (v, n) in E.tier_macro_f1(gold, yp, K, tier_of, na).items():
                tiers[t_].append(v)
    if not rec["seeds"]:
        continue
    rec["per_class_f1"] = {r: float(np.mean(v)) for r, v in pcf.items()}
    if tier_of:
        rec["tier_macro_f1"] = {t_: float(np.mean(v)) for t_, v in tiers.items()}
    out["methods"][name] = rec
    print(name, rec["seeds"], "AUC(s42) %.4f" % rec.get("pr", {}).get("auc", float("nan")), flush=True)
os.makedirs(f"{RES}/figdata", exist_ok=True)
json.dump(out, open(f"{RES}/figdata/figdata_{ds}.json", "w"))
