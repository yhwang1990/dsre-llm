"""Inference + evaluation: classifier probabilities -> on-demand call criterion -> reliability-aware fusion (eq:onto_gate, eq:fusion) ->
bag-level AUC-PR / P@N, sentence-level Macro-F1, per-tier Macro-F1, bootstrap CI, c_max sensitivity, inference cost.

Two-stage usage:
  1) python infer.py ... --votes <votes.jsonl>      # encode the test set, write probs.npy and need_uids.json
     exits with code 3 if LLM votes are missing for required instances; fill them with precompute.py --stages votes --uids need_uids.json
  2) python infer.py ... --votes <votes.jsonl> --reuse-probs   # fuse and evaluate directly
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre.model import DSRELLM
from dsre.ontology import Ontology
from dsre import evaluate as E
from train import REDataset, make_collate, build_base, load_cache, TEMPLATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--votes", default=None, help="LLM vote cache jsonl (keyed by uid); default = no fusion")
    ap.add_argument("--c-max", type=float, default=0.3)
    ap.add_argument("--no-gate", action="store_true", help="ablation: remove the ontology gate")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--tiers", default=None, help="tiers json (relation -> tier)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--sweep-cmax", action="store_true", help="sweep 0.1..0.9 (requires votes for all instances)")
    ap.add_argument("--reuse-probs", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="instances without a vote keep the classifier output")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rel2id = json.load(open(a.rel2id)); K = len(rel2id)
    has_na = "NA" in rel2id; na_id = rel2id.get("NA", -1)
    rel_rows = [r for r in sorted(rel2id, key=lambda r: rel2id[r]) if r != "NA"]
    insts = D.load_jsonl(a.test_file)
    tag = a.tag or os.path.splitext(os.path.basename(a.test_file))[0]
    out_dir = os.path.join(a.run, "eval_" + tag); os.makedirs(out_dir, exist_ok=True)
    gold = np.array([rel2id.get(x["relation"], na_id) for x in insts])
    bag_ids = [D.bag_key(x) for x in insts]
    uids = [x["uid"] for x in insts]
    print("[data] %s: %d instances, K=%d" % (a.test_file, len(insts), K), flush=True)

    # ------------------------------------------------------------- classifier probabilities
    probs_path = os.path.join(out_dir, "probs.npy")
    cost = {}
    if a.reuse_probs and os.path.exists(probs_path):
        probs = np.load(probs_path)
        cost = json.load(open(os.path.join(out_dir, "encode_cost.json"))) if os.path.exists(os.path.join(out_dir, "encode_cost.json")) else {}
    else:
        from peft import PeftModel
        base, tok, d_model = build_base(a.model_path, a.quant)
        backbone = PeftModel.from_pretrained(base, os.path.join(a.run, "lora"))
        pool = json.load(open(os.path.join(a.run, "config.json"))).get("pool", "entity")
        TEMPLATE["on"] = pool in ("last", "both", "ht_last")
        model = DSRELLM(backbone, K, d_model, pool=pool).cuda()
        ck = torch.load(os.path.join(a.run, "heads.pt"), map_location="cpu")
        model.proj.load_state_dict(ck["proj"]); model.classifier.load_state_dict(ck["classifier"])
        model.proj.float(); model.classifier.float(); model.eval()
        ds = REDataset(insts, rel2id, tok, a.max_len, aug=None, use_hardneg=False, na_id=na_id)
        # batches sorted by length to reduce padding (each sentence is still encoded independently; probabilities are written back by index)
        order = sorted(range(len(insts)), key=lambda i: len(insts[i]["text"]))
        batches = [order[j:j + a.batch_size] for j in range(0, len(order), a.batch_size)]
        dl = DataLoader(ds, batch_sampler=batches, num_workers=10, pin_memory=True, prefetch_factor=4, collate_fn=make_collate(tok.pad_token_id))
        probs = np.zeros((len(insts), K), dtype=np.float32)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); off = 0
        with torch.no_grad():
            for b in dl:
                _, _, logits = model(b["ids"].cuda(), b["am"].cuda(), b["em"].cuda())
                p = F.softmax(logits.float(), dim=-1).cpu().numpy()
                probs[b["idx"].numpy()] = p; off += len(p)
                if off % (a.batch_size * 50) == 0:
                    print("  encoded %d/%d  %.1f inst/s" % (off, len(insts), off / (time.time() - t0)), flush=True)
        torch.cuda.synchronize()
        cost = {"encode_seconds": time.time() - t0, "inst_per_sec": len(insts) / (time.time() - t0),
                "ms_per_inst": 1000 * (time.time() - t0) / len(insts), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        np.save(probs_path, probs); json.dump(cost, open(os.path.join(out_dir, "encode_cost.json"), "w"), indent=1)
        json.dump(uids, open(os.path.join(out_dir, "uids.json"), "w"))
        del model, backbone; torch.cuda.empty_cache()

    tier_of_class = None
    if a.tiers:
        t = json.load(open(a.tiers))["tier_of_relation"]
        tier_of_class = {rel2id[r]: tier for r, tier in t.items() if r in rel2id}

    res = {"n": len(insts), "c_max": a.c_max, "encode_cost": cost}
    res["classifier_only"] = E.evaluate(probs, gold, bag_ids, K, na_id=na_id, tier_of_class=tier_of_class)
    print("[classifier only]", json.dumps({k: v for k, v in res["classifier_only"].items() if not isinstance(v, dict)}), flush=True)

    # ------------------------------------------------------------- on-demand calls + fusion
    need = E.needs_llm(probs, a.c_max)
    need_uids = [u for u, n in zip(uids, need) if n]
    json.dump(need_uids, open(os.path.join(out_dir, "need_uids.json"), "w"))
    res["llm_call_fraction"] = float(need.mean()); res["llm_calls_needed"] = int(need.sum())
    print("[on-demand] margin<%.3f: %d/%d (%.1f%%) instances need an LLM vote"
          % (E.fusion_margin(a.c_max), need.sum(), len(need), 100 * need.mean()), flush=True)

    if a.votes:
        votes = load_cache(a.votes)
        have = np.array([u in votes for u in uids])
        missing = int((need & ~have).sum())
        res["votes_available"] = int(have.sum()); res["votes_missing_for_needed"] = missing
        if missing and not a.allow_partial and not a.sweep_cmax:
            json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
            print("[fusion] %d needed instances lack votes -> run precompute votes with --uids %s"
                  % (missing, os.path.join(out_dir, "need_uids.json")), flush=True)
            sys.exit(3)
        onto = Ontology(a.ontology, relations=rel_rows, has_na=has_na)
        types = load_cache(os.path.join(a.cache_root, "entity_types.jsonl"))

        def fuse_all(c_max):
            nd = E.needs_llm(probs, c_max)
            fused = probs.copy(); n_used = n_gated = n_na = 0
            for i, x in enumerate(insts):
                if not nd[i]:
                    continue
                v = votes.get(uids[i])
                if not v:
                    continue
                ok = True if a.no_gate else bool(onto.consistent(types.get(x["h"]["name"], "OTHER"),
                                                                  types.get(x["t"]["name"], "OTHER"), v["rel"]))
                if v["rel"] == "NA":
                    n_na += 1
                if not ok:
                    n_gated += 1
                else:
                    n_used += 1
                fused[i] = E.fuse(probs[i], v["rel"], v["conf"], rel2id, ok, c_max)
            return fused, {"votes_used": n_used, "votes_gated_out": n_gated, "votes_NA": n_na, "call_fraction": float(nd.mean())}

        fused, info = fuse_all(a.c_max)
        res["fused"] = E.evaluate(fused, gold, bag_ids, K, na_id=na_id, tier_of_class=tier_of_class)
        res["fused"].update(info)
        np.save(os.path.join(out_dir, "probs_fused.npy"), fused)
        print("[fused]", json.dumps({k: v for k, v in res["fused"].items() if not isinstance(v, dict)}), flush=True)
        if a.bootstrap:
            res["fused"]["bootstrap"] = E.bootstrap_ci(fused, gold, bag_ids, K, na_id=na_id, n_boot=a.bootstrap)
        if a.sweep_cmax:
            sweep = {}
            for c in (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9):
                f, inf = fuse_all(c)
                r = E.evaluate(f, gold, bag_ids, K, na_id=na_id)
                agree = float(np.mean([f[i].argmax() == rel2id.get(votes[uids[i]]["rel"], -2)
                                       for i in range(len(insts)) if uids[i] in votes])) if votes else None
                sweep[str(c)] = {"auc_pr": r["auc_pr"], "macro_f1": r["macro_f1"], "mean_p@n": r["mean_p@n"],
                                 "call_fraction": inf["call_fraction"], "final_equals_llm_frac": agree}
            res["cmax_sweep"] = sweep
    if a.bootstrap:
        res["classifier_only"]["bootstrap"] = E.bootstrap_ci(probs, gold, bag_ids, K, na_id=na_id, n_boot=a.bootstrap)
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    print("[saved]", os.path.join(out_dir, "result.json"))


if __name__ == "__main__":
    main()
