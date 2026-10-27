"""Generate all data splits once (fixed seeds, indices written to disk) and the statistics of Table 1.

Output layout (--out):
  nyt10/   rel2id.json  val.jsonl  train_main.jsonl  lowres_1pct.jsonl  lowres_5pct.jsonl
           test_full.jsonl  test_sub20.jsonl  tiers_main.json  tiers_pool.json  split_stats.json
  gids/    rel2id.json  train.jsonl  dev.jsonl  test.jsonl
  wiki80/  rel2id.json  train_main.jsonl  test.jsonl (= official val)  train_main_{symmetric,asymmetric}_{10,20,30,40}.jsonl
  stats.json  (sentences / entity pairs / positive pairs / relations / NA share / md5 of the source files, per file)
"""
import argparse, hashlib, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(insts, rel2id):
    bags = D.group_into_bags(insts)
    c = Counter(x["relation"] for x in insts)
    pos_bags = sum(1 for idxs in bags.values() if any(insts[i]["relation"] != "NA" for i in idxs))
    return {"sentences": len(insts), "bags": len(bags), "positive_bags": pos_bags,
            "nonNA_sentences": sum(v for k, v in c.items() if k != "NA"),
            "NA_frac": round(c.get("NA", 0) / max(len(insts), 1), 4),
            "relations_present_nonNA": sum(1 for k in c if k != "NA"),
            "relations_in_label_set": len(rel2id), "label_set_has_NA": "NA" in rel2id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="raw data directory containing processed/ wiki80/ gds/")
    ap.add_argument("--asym", required=True, help="wiki80_asym_T.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nyt-main-frac", type=float, default=0.10, help="fraction of the training pool used as the main training subset")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-sub-frac", type=float, default=0.20)
    ap.add_argument("--wiki80-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    R = a.raw_root; O = a.out
    stats = {"seed": a.seed, "sources": {}, "splits": {}}

    # ------------------------------------------------------------- NYT10
    def pick(*cands):
        for c in cands:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(cands)
    src_tr = pick(os.path.join(R, "nyt10", "nyt10_train.txt"), os.path.join(R, "processed", "nyt10_train.jsonl"))
    src_te = pick(os.path.join(R, "nyt10", "nyt10_test.txt"), os.path.join(R, "processed", "nyt10_test.jsonl"))
    rel2id = json.load(open(pick(os.path.join(R, "nyt10", "nyt10_rel2id.json"), os.path.join(R, "processed", "nyt10_rel2id.json"))))
    os.makedirs(os.path.join(O, "nyt10"), exist_ok=True)
    json.dump(rel2id, open(os.path.join(O, "nyt10", "rel2id.json"), "w"), indent=1)
    pool = D.load_nyt10(src_tr); test = D.load_nyt10(src_te)
    stats["sources"]["nyt10_train"] = {"path": src_tr, "md5": md5(src_tr)}
    stats["sources"]["nyt10_test"] = {"path": src_te, "md5": md5(src_te)}
    print("[nyt10] pool=%d test=%d" % (len(pool), len(test)), flush=True)

    tr_idx, val_idx, sp = D.split_by_bag(pool, a.val_frac, seed=a.seed)
    val = [pool[i] for i in val_idx]; rest = [pool[i] for i in tr_idx]
    D.save_jsonl(val, os.path.join(O, "nyt10", "val.jsonl"))
    sp["sentence_stratified_split_overlap"] = D.sentence_split_overlap(pool, a.val_frac, seed=a.seed)
    print("[nyt10] bag-disjoint val: %s" % json.dumps(sp), flush=True)

    n_main = int(round(len(pool) * a.nyt_main_frac))
    main_idx = D.stratified_subsample(rest, n_main, seed=a.seed)
    main = [rest[i] for i in main_idx]
    D.save_jsonl(main, os.path.join(O, "nyt10", "train_main.jsonl"))
    for frac, name in ((0.01, "lowres_1pct"), (0.05, "lowres_5pct")):
        idx = D.stratified_subsample(main, int(round(len(pool) * frac)), seed=a.seed)
        sub = [main[i] for i in idx]                       # low-resource subsets are nested in the main subset
        D.save_jsonl(sub, os.path.join(O, "nyt10", name + ".jsonl"))
        stats["splits"]["nyt10/" + name] = dict(describe(sub, rel2id), coverage=D.class_coverage(sub, rel2id))
    D.save_jsonl(test, os.path.join(O, "nyt10", "test_full.jsonl"))
    sub_idx = D.subsample_bags(test, a.test_sub_frac, seed=a.seed)
    D.save_jsonl([test[i] for i in sub_idx], os.path.join(O, "nyt10", "test_sub20.jsonl"))

    cnt_main = D.relation_counts(main); cnt_pool = D.relation_counts(pool)
    json.dump({"tier_of_relation": D.relation_tiers(cnt_main), "counts": cnt_main},
              open(os.path.join(O, "nyt10", "tiers_main.json"), "w"), indent=1)
    json.dump({"tier_of_relation": D.relation_tiers(cnt_pool), "counts": cnt_pool},
              open(os.path.join(O, "nyt10", "tiers_pool.json"), "w"), indent=1)
    json.dump(sp, open(os.path.join(O, "nyt10", "split_stats.json"), "w"), indent=1)
    for name, ins in (("pool", pool), ("val", val), ("train_main", main), ("test_full", test),
                      ("test_sub20", [test[i] for i in sub_idx])):
        stats["splits"]["nyt10/" + name] = dict(describe(ins, rel2id), coverage=D.class_coverage(ins, rel2id))
    stats["splits"]["nyt10/val"]["entity_pair_overlap_with_train_main"] = len(
        {D.bag_key(x) for x in val} & {D.bag_key(x) for x in main})

    # ------------------------------------------------------------- GIDS
    os.makedirs(os.path.join(O, "gids"), exist_ok=True)
    g_rel2id = {"NA": 0, "/people/person/education./education/education/institution": 1,
                "/people/person/education./education/education/degrees": 2,
                "/people/person/place_of_birth": 3, "/people/deceased_person/place_of_death": 4}
    json.dump(g_rel2id, open(os.path.join(O, "gids", "rel2id.json"), "w"), indent=1)
    for split in ("train", "dev", "test"):
        src = os.path.join(R, "gds", "%s.tsv" % split)
        if os.path.exists(src):
            ins = D.load_gids(src)
        else:
            src = os.path.join(R, "processed", "gids_%s.jsonl" % split); ins = D.load_jsonl(src)
        D.save_jsonl(ins, os.path.join(O, "gids", "%s.jsonl" % split))
        stats["sources"]["gids_" + split] = {"path": src, "md5": md5(src)}
        stats["splits"]["gids/" + split] = describe(ins, g_rel2id)
    print("[gids] done", flush=True)

    # ------------------------------------------------------------- Wiki80
    os.makedirs(os.path.join(O, "wiki80"), exist_ok=True)
    w_rel2id = json.load(open(os.path.join(R, "wiki80", "wiki80_rel2id.json")))
    json.dump(w_rel2id, open(os.path.join(O, "wiki80", "rel2id.json"), "w"), indent=1)
    src_wtr = os.path.join(R, "wiki80", "wiki80_train.txt"); src_wva = os.path.join(R, "wiki80", "wiki80_val.txt")
    wtr = D.load_wiki80(src_wtr); wva = D.load_wiki80(src_wva)
    stats["sources"]["wiki80_train"] = {"path": src_wtr, "md5": md5(src_wtr)}
    stats["sources"]["wiki80_val"] = {"path": src_wva, "md5": md5(src_wva)}
    idx = D.stratified_subsample(wtr, a.wiki80_per_class * len(w_rel2id), seed=a.seed)
    wmain = [wtr[i] for i in idx]
    D.save_jsonl(wmain, os.path.join(O, "wiki80", "train_main.jsonl"))
    D.save_jsonl(wva, os.path.join(O, "wiki80", "test.jsonl"))
    T = json.load(open(a.asym))["transition"]
    rels = sorted(w_rel2id)
    for rate in (0.1, 0.2, 0.3, 0.4):
        for mode in ("symmetric", "asymmetric"):
            noisy = D.inject_noise(wmain, rate, mode, seed=a.seed, transition=T, relations=rels)
            name = "train_main_%s_%d" % (mode, int(rate * 100))
            D.save_jsonl(noisy, os.path.join(O, "wiki80", name + ".jsonl"))
            stats["splits"]["wiki80/" + name] = {"sentences": len(noisy),
                                                 "actual_noise_rate": round(sum(x["is_noisy"] for x in noisy) / len(noisy), 4)}
    stats["splits"]["wiki80/train_full"] = describe(wtr, w_rel2id)
    stats["splits"]["wiki80/train_main"] = describe(wmain, w_rel2id)
    stats["splits"]["wiki80/test(official val)"] = describe(wva, w_rel2id)
    print("[wiki80] done", flush=True)

    json.dump(stats, open(os.path.join(O, "stats.json"), "w"), indent=1)
    print(json.dumps(stats["splits"], indent=1))
    print("[done] wrote", O)


if __name__ == "__main__":
    main()
