"""Evaluation: bag-level AUC-PR / P@N (held-out protocol), sentence-level Macro-F1, per-tier Macro-F1,
bootstrap confidence intervals, plus the reliability-aware fusion (ontology gate + fusion equation)
and the on-demand LLM-call criterion."""
import numpy as np


# ------------------------------------------------------------------ fusion
def fusion_margin(c_max):
    """If the classifier top-1/top-2 probability margin is >= c_max/(1-c_max), no admissible LLM vote can change the decision."""
    return c_max / (1.0 - c_max)


def needs_llm(prob_cls, c_max):
    """On-demand call criterion; returns a boolean array. prob_cls: [N, K]."""
    s = np.sort(prob_cls, axis=1)
    margin = s[:, -1] - s[:, -2]
    return margin < fusion_margin(c_max)


def fuse(prob_cls, llm_rel, llm_conf, rel2id, onto_ok, c_max):
    """Ontology gate + reliability-aware fusion. beta = 1 when llm_rel is 'NA' (NA has no domain/range to violate)."""
    beta = 1.0 if (llm_rel == "NA" or onto_ok) else 0.0
    omega = beta * min(float(llm_conf), c_max)
    if omega <= 0.0 or llm_rel not in rel2id:
        return prob_cls
    onehot = np.zeros_like(prob_cls)
    onehot[rel2id[llm_rel]] = 1.0
    return omega * onehot + (1.0 - omega) * prob_cls


# ------------------------------------------------------------------ ranking metrics
def pr_curve_auc(scores, labels):
    """Held-out protocol: all (bag, non-NA relation) candidate facts are ranked by score and precision is
    integrated over recall. Tied scores form a single threshold step (otherwise the result would depend
    on row order, since LLM confidences take only a dozen discrete values)."""
    order = np.argsort(-scores, kind="mergesort")
    s, lab = scores[order], labels[order]
    total_pos = lab.sum()
    if total_pos == 0:
        return 0.0, np.array([]), np.array([])
    boundaries = np.flatnonzero(np.diff(s)) if len(s) > 1 else np.array([], dtype=int)
    idx = np.concatenate([boundaries, [len(s) - 1]])
    tp = np.cumsum(lab)[idx]; fp = np.cumsum(1 - lab)[idx]
    precision = tp / np.maximum(tp + fp, 1e-12); recall = tp / total_pos
    auc = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return auc, precision, recall


def precision_at_n(scores, labels, ns=(100, 200, 300)):
    order = np.argsort(-scores, kind="mergesort")
    s, lab = scores[order], labels[order]
    out = {}
    for n in ns:
        if len(lab) < n:
            out[n] = float("nan"); continue
        cut = s[n - 1]
        above = int(np.count_nonzero(s > cut)); tied = int(np.count_nonzero(s == cut))
        hits_above = float(lab[:above].sum()); tied_hits = float(lab[above:above + tied].sum())
        need = n - above
        out[n] = float((hits_above + tied_hits * need / tied if tied else hits_above) / n)
    return out


# ------------------------------------------------------------------ classification metrics
def per_class_f1(y_true, y_pred, n_classes):
    """Returns {k: (f1, support)} for the classes that occur in the gold labels."""
    out = {}
    for k in range(n_classes):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        out[k] = (2 * p * r / (p + r) if p + r else 0.0, tp + fn)
    return out


def macro_f1(y_true, y_pred, n_classes, na_id=0, exclude_na=True):
    """Sentence-level Macro-F1: unweighted mean of per-class F1 over the non-NA relations present in the test set.
    NA is excluded from the average (but counts as FP/FN for the other classes); absent classes are skipped."""
    f = per_class_f1(y_true, y_pred, n_classes)
    vals = [v for k, (v, _) in f.items() if not (exclude_na and k == na_id)]
    return float(np.mean(vals)) if vals else 0.0


def sentence_level_summary(y_true, y_pred, n_classes, na_id=0, min_support=20):
    """Additional views: sentence-level non-NA Micro-P/R/F1, and Macro-F1 over relations with test support >= min_support."""
    f = per_class_f1(y_true, y_pred, n_classes)
    rel = [k for k in f if k != na_id]
    tp = sum(int(((y_pred == k) & (y_true == k)).sum()) for k in rel)
    fp = sum(int(((y_pred == k) & (y_true != k)).sum()) for k in rel)
    fn = sum(int(((y_pred != k) & (y_true == k)).sum()) for k in rel)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    big = [f[k][0] for k in rel if f[k][1] >= min_support]
    return {"micro_precision_nonNA": p, "micro_recall_nonNA": r, "micro_f1_nonNA": (2 * p * r / (p + r) if p + r else 0.0),
            "macro_f1_support%d" % min_support: (float(np.mean(big)) if big else 0.0), "n_relations_support%d" % min_support: len(big)}


def tier_macro_f1(y_true, y_pred, n_classes, tier_of_class, na_id=0):
    """Macro-F1 per training-frequency tier: {tier: (macro_f1, n_relations_scored)}."""
    f = per_class_f1(y_true, y_pred, n_classes)
    by = {}
    for k, (v, _) in f.items():
        if k == na_id or k not in tier_of_class:
            continue
        by.setdefault(tier_of_class[k], []).append(v)
    return {t: (float(np.mean(v)), len(v)) for t, v in by.items()}


# ------------------------------------------------------------------ bag aggregation
def bag_aggregate(probs, bag_ids):
    bags = {}
    for i, b in enumerate(bag_ids):
        bags.setdefault(b, []).append(i)
    keys = list(bags)
    out = np.stack([probs[bags[k]].max(axis=0) for k in keys])
    return keys, bags, out


def _facts(bag_probs, bag_gold_sets, n_classes, na_id):
    K = n_classes
    rel_idx = np.array([r for r in range(K) if r != na_id])
    scores = bag_probs[:, rel_idx].reshape(-1)
    labels = np.zeros((bag_probs.shape[0], len(rel_idx)), dtype=np.int8)
    for bi, g in enumerate(bag_gold_sets):
        for r in g:
            if r != na_id:
                labels[bi, np.searchsorted(rel_idx, r)] = 1
    return scores, labels.reshape(-1)


def evaluate(probs, gold, bag_ids, n_classes, na_id=0, tier_of_class=None):
    keys, bags, bag_probs = bag_aggregate(probs, bag_ids)
    gold = np.asarray(gold)
    bag_gold = [set(int(gold[i]) for i in bags[k]) for k in keys]
    scores, labels = _facts(bag_probs, bag_gold, n_classes, na_id)
    auc, _, _ = pr_curve_auc(scores, labels)
    pn = precision_at_n(scores, labels)
    y_pred = probs.argmax(axis=1)
    res = {"auc_pr": auc, "p@100": pn[100], "p@200": pn[200], "p@300": pn[300],
           "mean_p@n": float(np.mean([pn[100], pn[200], pn[300]])),
           "macro_f1": macro_f1(gold, y_pred, n_classes, na_id=na_id),
           "n_instances": int(len(gold)), "n_bags": len(keys),
           "n_positive_bags": int(sum(1 for g in bag_gold if any(r != na_id for r in g))),
           "n_relations_scored": len([k for k in per_class_f1(gold, y_pred, n_classes) if k != na_id])}
    res.update(sentence_level_summary(gold, y_pred, n_classes, na_id=na_id))
    if tier_of_class:
        res["tier_macro_f1"] = {t: {"macro_f1": v, "n_relations": n}
                                for t, (v, n) in tier_macro_f1(gold, y_pred, n_classes, tier_of_class, na_id).items()}
    return res


def bootstrap_ci(probs, gold, bag_ids, n_classes, na_id=0, n_boot=200, seed=0, metrics=("auc_pr", "macro_f1", "mean_p@n")):
    """95% confidence intervals by bag-level resampling (sentences follow their bag)."""
    rng = np.random.RandomState(seed)
    keys, bags, bag_probs = bag_aggregate(probs, bag_ids)
    gold = np.asarray(gold); y_pred = probs.argmax(axis=1)
    bag_gold = [set(int(gold[i]) for i in bags[k]) for k in keys]
    members = [np.array(bags[k]) for k in keys]
    nb = len(keys)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.randint(0, nb, nb)
        scores, labels = _facts(bag_probs[pick], [bag_gold[i] for i in pick], n_classes, na_id)
        if "auc_pr" in metrics:
            vals["auc_pr"].append(pr_curve_auc(scores, labels)[0])
        if "mean_p@n" in metrics:
            pn = precision_at_n(scores, labels)
            vals["mean_p@n"].append(float(np.mean([pn[100], pn[200], pn[300]])))
        if "macro_f1" in metrics:
            sel = np.concatenate([members[i] for i in pick])
            vals["macro_f1"].append(macro_f1(gold[sel], y_pred[sel], n_classes, na_id=na_id))
    return {m: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for m, v in vals.items() if v}
