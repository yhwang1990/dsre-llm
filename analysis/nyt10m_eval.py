"""Re-evaluation of trained runs on the human-annotated NYT10m test set
(OpenNRE, Gao et al., 2021; 11,086 sentences, 25 classes).
All NYT10m test sentences come from the NYT10 test set and are matched to test_full.jsonl by (text, head, tail), so the existing
probs.npy / probs_fused.npy of each run are reused without re-encoding or new LLM calls.
Label-space mapping (identical for all models): relations with the same name map directly; deceased_person/* and
major_shareholders map to their renamed NYT10m labels; the first-level administrative */capital relations (excluding
/location/country/capital) merge into /location/region/capital; /business/company/locations -> /business/location; all other
NYT10 relations (mostly inverse or finer-grained ones) merge into NA. Probabilities of merged classes are summed.
Eval directories are aligned through uids.json, so either full-test-set results or results encoded on the NYT10m subset can be used.
Also reports a paired bootstrap (reference run - other run) with two-sided p."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E

RENAME = {"/people/deceased_person/place_of_death": "/people/deceasedperson/place_of_death",
          "/people/deceased_person/place_of_burial": "/people/deceasedperson/place_of_burial",
          "/business/company/major_shareholders": "/business/company/majorshareholders",
          "/business/company/locations": "/business/location"}


def build_map(ours, m):
    M = np.zeros((len(ours), len(m)), dtype=np.float32)
    for r, i in ours.items():
        if r in m:
            j = m[r]
        elif r in RENAME:
            j = m[RENAME[r]]
        elif r.endswith("/capital") and r != "/location/country/capital":
            j = m["/location/region/capital"]
        else:
            j = m["NA"]
        M[i, j] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-full", required=True)
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--nyt10m-test", required=True)
    ap.add_argument("--nyt10m-rel2id", required=True)
    ap.add_argument("--ref", required=True, help="name=<eval dir>")
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--tiers", default=None, help="NYT10 tiers json (Macro-F1 per training-frequency tier)")
    ap.add_argument("--c-max", type=float, default=0.25, help="c_max that defines the gated subset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ours = json.load(open(a.rel2id)); m = json.load(open(a.nyt10m_rel2id)); K = len(m); na = m["NA"]
    full = D.load_jsonl(a.test_full)
    idx = {}
    for i, x in enumerate(full):
        idx.setdefault((x["text"], x["h"]["name"], x["t"]["name"]), i)
    test = D.load_jsonl(a.nyt10m_test)
    rows = np.array([idx[(x["text"], x["h"]["name"], x["t"]["name"])] for x in test])
    gold = np.array([m[x["relation"]] for x in test])
    bag_ids = [D.bag_key(x) for x in test]
    keys, bags, _ = E.bag_aggregate(np.zeros((len(test), 1)), bag_ids)
    members = [np.array(bags[k]) for k in keys]
    bag_gold = [set(int(gold[i]) for i in mm) for mm in members]
    nb = len(keys)
    P = build_map(ours, m)

    uid_of_row = [full[i]["uid"] for i in rows]

    def raw(d, fused=True):
        # rows are taken by uid: the eval dir may cover the full test set or only the NYT10m subset
        u = json.load(open(os.path.join(d, "uids.json"))); pos = {x: i for i, x in enumerate(u)}
        return np.load(os.path.join(d, "probs_fused.npy" if fused else "probs.npy"))[[pos[x] for x in uid_of_row]]

    def proj(d, fused=True):
        return raw(d, fused) @ P

    # tiers of NYT10m classes come from the NYT10 relation of the same (or renamed) label; merged classes are not tiered
    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]; inv = {v: k for k, v in RENAME.items()}
        tier_of_class = {j: t[inv.get(r, r)] for r, j in m.items() if inv.get(r, r) in t and r != "/business/location"}

    def extra(d):
        """Per-tier Macro-F1 and the gated subset (instances this model sends to the LLM; accuracy before/after fusion on human labels)."""
        pf, pc = proj(d), proj(d, False); yf, yc = pf.argmax(1), pc.argmax(1)
        o = {}
        if tier_of_class:
            o["tier_macro_f1"] = {k: v[0] for k, v in E.tier_macro_f1(gold, yf, K, tier_of_class, na).items()}
        need = E.needs_llm(raw(d, False), a.c_max); g = gold[need]
        o["gated"] = {"n_called": int(need.sum()), "acc_before": float((yc[need] == g).mean()), "acc_after": float((yf[need] == g).mean()),
                      "wrong_to_right": int(((yc[need] != g) & (yf[need] == g)).sum()), "right_to_wrong": int(((yc[need] == g) & (yf[need] != g)).sum())}
        return o

    def metrics(p, pick):
        bp = np.stack([p[mm].max(axis=0) for mm in members])
        yp = p.argmax(1)
        sc, lab = E._facts(bp[pick], [bag_gold[i] for i in pick], K, na)
        auc = E.pr_curve_auc(sc, lab)[0]; pn = E.precision_at_n(sc, lab)
        sel = np.concatenate([members[i] for i in pick])
        s = E.sentence_level_summary(gold[sel], yp[sel], K, na_id=na)
        return {"auc_pr": auc, "p@100": pn[100], "mean_p@n": float(np.mean([pn[100], pn[200], pn[300]])),
                "micro_f1_nonNA": s["micro_f1_nonNA"], "macro_f1": E.macro_f1(gold[sel], yp[sel], K, na_id=na),
                "macro_f1_support20": s["macro_f1_support20"]}

    allp = np.arange(nb)
    rname, rdir = a.ref.split("=", 1)
    rp = proj(rdir)
    out = {"n_sentences": len(test), "n_bags": nb, "n_positive_bags": sum(1 for g in bag_gold if any(r != na for r in g)),
           "ref": {"name": rname, "dir": rdir, "fused": metrics(rp, allp), "classifier_only": metrics(proj(rdir, False), allp), **extra(rdir)},
           "variants": {}}
    print(rname, {k: round(100 * v, 2) for k, v in out["ref"]["fused"].items()}, flush=True)
    for spec in a.var:
        name, d = spec.split("=", 1)
        vp = proj(d); vm = metrics(vp, allp)
        rng = np.random.RandomState(0); diffs = {k: [] for k in vm}
        for _ in range(a.n_boot):
            pick = rng.randint(0, nb, nb)
            r = metrics(rp, pick); v = metrics(vp, pick)
            for k in diffs:
                diffs[k].append(r[k] - v[k])
        paired = {}
        for k, ds in diffs.items():
            ds = np.array(ds); obs = out["ref"]["fused"][k] - vm[k]
            paired[k] = {"full_minus_var": obs, "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                         "p_two_sided": float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))}
        out["variants"][name] = {"dir": d, "fused": vm, "classifier_only": metrics(proj(d, False), allp), "paired": paired, **extra(d)}
        print(name, {k: round(100 * v, 2) for k, v in vm.items()},
              {k: [round(100 * v["full_minus_var"], 2), round(v["p_two_sided"], 3)] for k, v in paired.items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
