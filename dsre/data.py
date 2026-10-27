"""Data loading and splitting. NYT10, GDS and Wiki80 are converted to one record format;
helpers for entity-pair-level splits, bag-level sampling, stratified subsampling,
controlled label-noise injection and long-tail frequency tiers.

Record: {"text", "relation", "h": {"name", "id", "pos": [s, e]}, "t": {...},
         optional "relation_clean", "is_noisy", "uid"}
"""
import hashlib, json, os, random
from collections import Counter, defaultdict


class Instance(dict):
    pass


def uid_of(x):
    """Position-independent instance key: same sentence + same entity pair -> same key (LLM caches are indexed by it)."""
    s = "%s|%s|%s" % (x["text"], x["h"]["name"], x["t"]["name"])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _finish(x):
    x.setdefault("uid", uid_of(x))
    return x


# ------------------------------------------------------------------ loaders
def load_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if line:
                out.append(_finish(Instance(json.loads(line))))
    return out


def save_jsonl(insts, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in insts:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def load_nyt10(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(_finish(Instance(text=d["text"], relation=d["relation"], h=d["h"], t=d["t"])))
    return out


def load_gids(path, limit=None):
    """GDS tsv: head_mid, tail_mid, head_name, tail_name, relation, sentence (entity tokens joined by underscores)."""
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            hid, tid, hname, tname, rel, sent = parts[:6]
            hs, ts = sent.find(hname), sent.find(tname)
            if hs < 0 or ts < 0:
                continue
            out.append(_finish(Instance(
                text=sent, relation=rel,
                h={"name": hname, "id": hid, "pos": [hs, hs + len(hname)]},
                t={"name": tname, "id": tid, "pos": [ts, ts + len(tname)]})))
    return out


def load_wiki80(path, limit=None):
    """OpenNRE wiki80: {"token": [...], "h": {"name","id","pos":[tok_s, tok_e]}, "t": ..., "relation"}"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            toks = [t.replace("\n", " ").strip() or "," for t in d["token"]]
            starts, text = [], ""
            for t in toks:
                if text:
                    text += " "
                starts.append(len(text)); text += t
            ends = [s + len(t) for s, t in zip(starts, toks)]

            def span(e):
                a, b = e["pos"][0], e["pos"][1]
                b = min(b, len(toks))
                return [starts[a], ends[b - 1]]
            h = {"name": text[span(d["h"])[0]:span(d["h"])[1]], "id": d["h"].get("id", ""), "pos": span(d["h"])}
            t = {"name": text[span(d["t"])[0]:span(d["t"])[1]], "id": d["t"].get("id", ""), "pos": span(d["t"])}
            out.append(_finish(Instance(text=text, relation=d["relation"], h=h, t=t)))
    return out


def load_any(dataset, path):
    return {"nyt10": load_nyt10, "gids": load_gids, "wiki80": load_wiki80, "jsonl": load_jsonl}[dataset](path)


# ------------------------------------------------------------------ bags
def bag_key(inst):
    return (inst["h"]["id"] or inst["h"]["name"], inst["t"]["id"] or inst["t"]["name"])


def group_into_bags(insts):
    bags = defaultdict(list)
    for i, x in enumerate(insts):
        bags[bag_key(x)].append(i)
    return bags


def bag_label(insts, idxs):
    rels = sorted({insts[i]["relation"] for i in idxs} - {"NA"})
    return rels[0] if rels else "NA"


def split_by_bag(insts, frac, seed=42):
    """Entity-pair-level split: all sentences of an entity pair fall on the same side
    (no entity pair shared between training and held-out data), stratified by bag label.
    Returns (train_idx, held_idx, stats)."""
    rng = random.Random(seed)
    bags = group_into_bags(insts)
    by_label = defaultdict(list)
    for k, idxs in bags.items():
        by_label[bag_label(insts, idxs)].append(k)
    held = set()
    for lab, keys in by_label.items():
        keys = sorted(keys); rng.shuffle(keys)
        n = int(round(len(keys) * frac))
        held.update(keys[:n])
    tr, hd = [], []
    for k, idxs in bags.items():
        (hd if k in held else tr).extend(idxs)
    tr.sort(); hd.sort()
    stats = {"train_sents": len(tr), "held_sents": len(hd), "train_bags": len(bags) - len(held),
             "held_bags": len(held), "entity_pair_overlap": 0}
    return tr, hd, stats


def sentence_split_overlap(insts, frac, seed=42):
    """Entity-pair overlap that a sentence-level stratified split would produce (for comparison with split_by_bag)."""
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for i, x in enumerate(insts):
        by_rel[x["relation"]].append(i)
    held = set()
    for r, idxs in by_rel.items():
        idxs = list(idxs); rng.shuffle(idxs)
        held.update(idxs[:int(round(len(idxs) * frac))])
    tr_bags = {bag_key(insts[i]) for i in range(len(insts)) if i not in held}
    hd_bags = {bag_key(insts[i]) for i in held}
    inter = tr_bags & hd_bags
    hd_sents_overlap = sum(1 for i in held if bag_key(insts[i]) in tr_bags)
    return {"held_bags": len(hd_bags), "overlapping_bags": len(inter),
            "overlap_bag_frac": len(inter) / max(len(hd_bags), 1),
            "held_sents": len(held), "held_sents_with_pair_in_train": hd_sents_overlap,
            "overlap_sent_frac": hd_sents_overlap / max(len(held), 1)}


def subsample_bags(insts, frac, seed=42):
    """Uniform bag-level random sampling (keeps the NA ratio); used for the 20% test subset."""
    rng = random.Random(seed)
    bags = group_into_bags(insts)
    keys = sorted(bags); rng.shuffle(keys)
    keep = set(keys[:int(round(len(keys) * frac))])
    idx = sorted(i for k in keep for i in bags[k])
    return idx


def stratified_subsample(insts, n, seed=42, min_per_rel=1):
    """Stratified subsampling by relation (NA included); keeps at least min_per_rel instances per relation if available."""
    if n is None or n >= len(insts):
        return list(range(len(insts)))
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for i, x in enumerate(insts):
        by_rel[x["relation"]].append(i)
    total = len(insts)
    out = []
    for rel, xs in sorted(by_rel.items()):
        k = max(min_per_rel, int(round(len(xs) * n / total)))
        xs = list(xs); rng.shuffle(xs)
        out.extend(xs[:k])
    rng.shuffle(out)
    return sorted(out[:n]) if len(out) > n else sorted(out)


def relation_counts(insts):
    return Counter(x["relation"] for x in insts)


def class_coverage(insts, rel2id):
    c = relation_counts(insts)
    return {"relations_present_nonNA": sum(1 for r in c if r != "NA"),
            "relations_total_nonNA": sum(1 for r in rel2id if r != "NA"),
            "NA_frac": c.get("NA", 0) / max(len(insts), 1)}


# ------------------------------------------------------------------ tiers and noise
TIERS = [(1, 10), (11, 30), (31, 100), (101, 300), (301, 10 ** 9)]
TIER_NAMES = ["1-10", "11-30", "31-100", "101-300", "300+"]


def relation_tiers(train_counts):
    """Assign non-NA relations to five tiers by training frequency. Returns {relation: tier_name}."""
    out = {}
    for r, n in train_counts.items():
        if r == "NA":
            continue
        for (lo, hi), name in zip(TIERS, TIER_NAMES):
            if lo <= n <= hi:
                out[r] = name; break
    return out


def inject_noise(insts, rate, mode, seed=42, transition=None, relations=None):
    """Controlled label noise (used for Wiki80).
    symmetric : with probability `rate`, replace the label by any other class uniformly
    asymmetric: with probability `rate`, replace the label by a semantically close class drawn from transition[k]
    Every record keeps relation_clean and is_noisy."""
    rng = random.Random(seed)
    rels = relations or sorted({x["relation"] for x in insts} - {"NA"})
    out = []
    for x in insts:
        y = dict(x)
        y["relation_clean"] = x["relation"]; y["is_noisy"] = 0
        if x["relation"] != "NA" and rate > 0 and rng.random() < rate:
            if mode == "symmetric":
                cand = [r for r in rels if r != x["relation"]]
                y["relation"] = rng.choice(cand)
            else:
                t = transition[x["relation"]]
                ks, ps = zip(*sorted(t.items()))
                y["relation"] = rng.choices(ks, weights=ps, k=1)[0]
            y["is_noisy"] = int(y["relation"] != x["relation"])
        out.append(Instance(y))
    return out


def insert_entity_markers(text, h, t):
    spans = sorted([(h["pos"][0], h["pos"][1], "E1"), (t["pos"][0], t["pos"][1], "E2")],
                   key=lambda s: s[0], reverse=True)
    s = text
    for a, b, tag in spans:
        if a < 0 or b > len(s) or a >= b:
            continue
        s = s[:a] + "[%s] " % tag + s[a:b] + " [/%s]" % tag + s[b:]
    return s
