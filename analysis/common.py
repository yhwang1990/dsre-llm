"""Shared helpers for the analysis scripts that rebuild the tables and figures of the paper
from saved per-sentence probabilities and cached LLM votes (CPU only, no retraining).

All paths are relative to DSRE_ROOT (default: ./workspace), laid out as
  data/ontology/, data_prepared/<dataset>/, data_ext/nyt10m/, cache/<dataset>/, runs/<run>/, results/
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D
from dsre import evaluate as E
from dsre.ontology import Ontology

R = os.environ.get("DSRE_ROOT", "workspace")
RUNS = R + "/runs"; P = R + "/data_prepared"; C = R + "/cache"; RES = os.environ.get("DSRE_RESULTS", R + "/results")
ONTO = R + "/data/ontology/relation_ontology_all.json"
os.makedirs(RES, exist_ok=True)

# Configuration of DSRE-LLM (run-name tag) and the validation-selected trust ceilings (Section 4.1.4).
FN = "cwf5l1"
CMAX = {"nyt10": 0.25, "gids": 0.05}
FTLR = "2e-4"           # learning rate of the Qwen3-8B-FT baseline, selected on the NYT10 validation set
SEEDS = (13, 42, 123)
# Baseline configurations selected on the validation sets (see reproduce/baselines.sh for the grids).
BEST = {("nyt10", "pcnn"): "lr1e-3e15", ("nyt10", "focal"): "m3e3", ("nyt10", "mil"): "m3e3", ("nyt10", "nli"): "m3e3",
        ("nyt10", "cgre"): "lr0.5e12", ("nyt10", "llmcg"): "lr0.5e12",
        ("gids", "pcnn"): "lr2e-3", ("gids", "focal"): "m3", ("gids", "mil"): "m2", ("gids", "nli"): "m5",
        ("gids", "cgre"): "lr0.2", ("gids", "llmcg"): "lr0.2"}


def load_cache(path):
    """LLM cache files are jsonl with one {"k": uid_or_name, "v": value} record per line."""
    d = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line); d[r["k"]] = r["v"]
                except Exception:
                    pass
    return d


class DS:
    """A test set together with its label space, ontology and entity types; fuses and evaluates probability matrices."""

    def __init__(self, name, test, types):
        self.rel2id = json.load(open(f"{P}/{name}/rel2id.json")); self.K = len(self.rel2id); self.na = self.rel2id.get("NA", -1)
        self.insts = D.load_jsonl(test); self.uids = [x["uid"] for x in self.insts]; self.pos = {u: i for i, u in enumerate(self.uids)}
        self.gold = np.array([self.rel2id.get(x["relation"], self.na) for x in self.insts]); self.bags = [D.bag_key(x) for x in self.insts]
        rels = [r for r in sorted(self.rel2id, key=lambda r: self.rel2id[r]) if r != "NA"]
        self.onto = Ontology(ONTO, relations=rels, has_na=self.na >= 0); self.types = load_cache(types)

    def probs(self, d, name="probs.npy"):
        p = np.load(os.path.join(d, name)); f = os.path.join(d, "uids.json")
        u = json.load(open(f)) if os.path.exists(f) else self.uids
        assert len(u) == len(p)
        return p, u

    def fuse(self, p, u, votes, c):
        """Reliability-aware fusion (eq:onto_gate, eq:fusion) applied to the instances selected by the on-demand criterion."""
        f = p.copy()
        if c <= 0:
            return f
        for i in np.flatnonzero(E.needs_llm(p, c)):
            x = self.insts[self.pos[u[i]]]; v = votes[u[i]]
            ok = bool(self.onto.consistent(self.types.get(x["h"]["name"], "OTHER"), self.types.get(x["t"]["name"], "OTHER"), v["rel"]))
            f[i] = E.fuse(p[i], v["rel"], v["conf"], self.rel2id, ok, c)
        return f

    def ev(self, p, rows=None, boot=0):
        """Evaluate probabilities whose i-th row is test line rows[i] (default: all lines in file order).
        Rows are aligned by position, not by uid: NYT10 lists a sentence once per distant label, so a uid can occur on several lines."""
        idx = np.arange(len(self.insts)) if rows is None else np.asarray(rows)
        assert len(idx) == len(p)
        g = self.gold[idx]; b = [self.bags[i] for i in idx]
        r = E.evaluate(p, g, b, self.K, na_id=self.na)
        if boot:
            r["boot"] = E.bootstrap_ci(p, g, b, self.K, na_id=self.na, n_boot=boot, metrics=("auc_pr", "macro_f1"))
        return r


def subset_rows(full, sub):
    """Line indices of `full` that form the bag-level subset `sub` (e.g. the 20% NYT10 test subset), in file order."""
    keep = set(D.bag_key(x) for x in sub.insts)
    rows = np.array([i for i, b in enumerate(full.bags) if b in keep])
    assert [full.uids[i] for i in rows] == sub.uids
    return rows


def pick(r, keys=("auc_pr", "p@100", "micro_f1_nonNA", "macro_f1")):
    o = {k: 100 * r[k] for k in keys}
    if "boot" in r:
        o.update({"std_" + k: 100 * v["std"] for k, v in r["boot"].items()})
    return o


def mean_std(rows):
    A = 100 * np.array(rows)
    return {"n": len(rows), "mean": A.mean(0).tolist(), "std": (A.std(0, ddof=1) if len(rows) > 1 else 0 * A[0]).tolist()}
