"""Validation-based selection of the trust ceiling c_max (uses existing classifier probabilities and LLM votes only, no retraining).
A smaller c_max sends fewer instances to the LLM (fusion_margin is monotone), so values <= the original c_max never miss votes; instances without a vote keep the classifier output and are counted.
c_max = 0 is equivalent to no fusion (classifier_only). Writes the fused metrics of every c_max and the combined score AUC + Mi-F1 + Ma-F1."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre import evaluate as E
from dsre.ontology import Ontology
from train import load_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True); ap.add_argument("--rel2id", required=True)
    ap.add_argument("--ontology", required=True); ap.add_argument("--cache-root", required=True)
    ap.add_argument("--eval-dir", required=True); ap.add_argument("--votes", required=True)
    ap.add_argument("--cmaxes", default="0,0.05,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--out", default="cmax_sweep.json")
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); K = len(rel2id); na_id = rel2id["NA"]
    rel_rows = [r for r in sorted(rel2id, key=rel2id.get) if r != "NA"]
    insts = D.load_jsonl(a.test_file); uids = [x["uid"] for x in insts]
    assert uids == json.load(open(os.path.join(a.eval_dir, "uids.json")))
    gold = np.array([rel2id.get(x["relation"], na_id) for x in insts]); bags = [D.bag_key(x) for x in insts]
    P = np.load(os.path.join(a.eval_dir, "probs.npy"))
    votes = load_cache(a.votes); onto = Ontology(a.ontology, relations=rel_rows, has_na=True)
    types = load_cache(os.path.join(a.cache_root, "entity_types.jsonl"))
    out = {}
    for c in [float(x) for x in a.cmaxes.split(",")]:
        fused = P.copy(); miss = 0; nd = E.needs_llm(P, c) if c > 0 else np.zeros(len(P), bool)
        for i in np.nonzero(nd)[0]:
            v = votes.get(uids[i])
            if not v:
                miss += 1; continue
            x = insts[i]
            ok = bool(onto.consistent(types.get(x["h"]["name"], "OTHER"), types.get(x["t"]["name"], "OTHER"), v["rel"]))
            fused[i] = E.fuse(P[i], v["rel"], v["conf"], rel2id, ok, c)
        r = E.evaluate(fused, gold, bags, K, na_id=na_id)
        r.update(votes_missing=miss, call_fraction=float(nd.mean()))
        out["%.2f" % c] = r
        print("cmax %.2f  AUC %.2f  P@100 %.1f  MiF1 %.2f  MaF1 %.2f  score %.2f  calls %.4f miss %d"
              % (c, 100 * r["auc_pr"], 100 * r["p@100"], 100 * r["micro_f1_nonNA"], 100 * r["macro_f1"],
                 100 * (r["auc_pr"] + r["micro_f1_nonNA"] + r["macro_f1"]), nd.mean(), miss), flush=True)
    json.dump(out, open(os.path.join(a.eval_dir, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
