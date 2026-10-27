"""DSRE-LLM training.

Joint objective (eq:joint_loss):  L = lam1*L_weight + lam2*L_contrast + lam3*L_reg
  L_weight   = confidence-weighted CE (eq:weighted_ce), W_final = a_i * sigma(entropy term) * rho_y (eq:weight, eq:final_weight)
  L_contrast = alpha * NT-Xent (with hard negatives, non-NA anchors only) + (1 - alpha) * reliability-weighted prototype loss (non-NA only) (eq:dcd_total)
Centroids: mean initialisation from one pass before training (label + ontology filter); relations with fewer than N_min sentences use LLM-generated anchor sentences;
      afterwards EMA update with clean set = label match AND ontology-consistent AND (warm-up OR NN test with delta) (eq:clean_set, eq:nn_test).
Centroids and weights are not back-propagated.
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsre import data as D
from dsre.model import DSRELLM, nt_xent, prototype_loss, confidence, nn_test, CentroidBank
from dsre.ontology import Ontology


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_cache(path):
    d = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line); d[r["k"]] = r["v"]
                except Exception:
                    pass
    return d


# ------------------------------------------------------------------ data
TEMPLATE = {"on": False}


def encode_text(tok, text, spans, max_len, h=None, t=None):
    """Returns (ids, entity_mask); spans = [(s, e), ...] character spans.
    With the template enabled a prompt is appended after the sentence; the sentence is truncated if needed so that the template (last token) is always kept."""
    if TEMPLATE["on"]:
        suffix = (" In this sentence, the relation between %s and %s is" % (h, t)) if h else \
                 " In this sentence, the relation between the two entities is"
        sids = tok(suffix, add_special_tokens=False)["input_ids"]
        e = tok(text, truncation=True, max_length=max(8, max_len - len(sids)), return_offsets_mapping=True)
        offs = e["offset_mapping"]; ent = [0] * len(offs)
        for k, (a, b) in enumerate(spans):
            for j, (s0, s1) in enumerate(offs):
                if s1 > s0 and s0 < b and s1 > a:
                    ent[j] = k + 1
        return e["input_ids"] + sids, ent + [0] * len(sids)
    e = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    offs = e["offset_mapping"]
    ent = [0] * len(offs)
    for k, (a, b) in enumerate(spans):
        for j, (s0, s1) in enumerate(offs):
            if s1 > s0 and s0 < b and s1 > a:
                ent[j] = k + 1
    return e["input_ids"], ent


class REDataset(Dataset):
    def __init__(self, insts, rel2id, tok, max_len, aug=None, use_hardneg=True, na_id=-1):
        self.x, self.rel2id, self.tok, self.max_len = insts, rel2id, tok, max_len
        self.aug = aug or {}; self.use_hardneg = use_hardneg; self.na_id = na_id

    def __len__(self):
        return len(self.x)

    def _spans_in(self, text, x):
        hs, ts = text.find(x["h"]["name"]), text.find(x["t"]["name"])
        if hs < 0 or ts < 0:
            return None
        return [(hs, hs + len(x["h"]["name"])), (ts, ts + len(x["t"]["name"]))]

    def __getitem__(self, i):
        x = self.x[i]
        ids, ent = encode_text(self.tok, x["text"], [tuple(x["h"]["pos"]), tuple(x["t"]["pos"])], self.max_len, x["h"]["name"], x["t"]["name"])
        y = self.rel2id.get(x["relation"], self.na_id)
        a = self.aug.get(x["uid"]) or {}
        ids2, ent2, is_anchor = ids, ent, 0
        pos = a.get("pos") or []
        if y != self.na_id and pos:
            p = random.choice(pos)                       # sample one positive view per step
            sp = self._spans_in(p, x)
            if sp:
                ids2, ent2 = encode_text(self.tok, p, sp, self.max_len, x["h"]["name"], x["t"]["name"]); is_anchor = 1
        idsn, entn, has_neg = None, None, 0
        neg = a.get("neg")
        if self.use_hardneg and y != self.na_id and neg:
            sp = self._spans_in(neg, x)
            if sp:
                idsn, entn = encode_text(self.tok, neg, sp, self.max_len, x["h"]["name"], x["t"]["name"]); has_neg = 1
        return {"ids": ids, "ent": ent, "ids2": ids2, "ent2": ent2, "idsn": idsn, "entn": entn,
                "y": y, "idx": i, "anchor": is_anchor, "has_neg": has_neg,
                "noisy": int(x.get("is_noisy", 0))}


def pad_batch(seqs, ents, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    em = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (s, e) in enumerate(zip(seqs, ents)):
        ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1; em[i, :len(e)] = torch.tensor(e)
    return ids, am, em


def make_collate(pad_id):
    def collate(items):
        ids, am, em = pad_batch([it["ids"] for it in items], [it["ent"] for it in items], pad_id)
        ids2, am2, em2 = pad_batch([it["ids2"] for it in items], [it["ent2"] for it in items], pad_id)
        negs = [(it["idsn"], it["entn"]) for it in items if it["has_neg"]]
        if negs:
            idsn, amn, emn = pad_batch([n[0] for n in negs], [n[1] for n in negs], pad_id)
        else:
            idsn = amn = emn = None
        return {"ids": ids, "am": am, "em": em, "ids2": ids2, "am2": am2, "em2": em2,
                "idsn": idsn, "amn": amn, "emn": emn,
                "y": torch.tensor([it["y"] for it in items]), "idx": torch.tensor([it["idx"] for it in items]),
                "anchor": torch.tensor([it["anchor"] for it in items], dtype=torch.bool),
                "noisy": torch.tensor([it["noisy"] for it in items])}
    return collate


# ------------------------------------------------------------------ model
def build_base(model_path, quant="none", grad_ckpt=False):
    """Frozen backbone (bf16, optional 4-bit), without LoRA."""
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_path)
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size")
    kw = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    backbone = AutoModel.from_pretrained(model_path, device_map={"": 0}, **kw)
    for p in backbone.parameters():
        p.requires_grad = False
    if grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    return backbone, tok, d_model


def build_backbone(model_path, lora_r, lora_alpha, lora_dropout, targets, quant="none", grad_ckpt=False):
    from peft import LoraConfig, get_peft_model
    backbone, tok, d_model = build_base(model_path, quant, grad_ckpt)
    present = {n.split(".")[-1] for n, m in backbone.named_modules() if isinstance(m, torch.nn.Linear)
               or m.__class__.__name__.endswith("Linear4bit")}
    tg = [t for t in targets if t in present]
    if not tg:
        raise RuntimeError("no LoRA targets among %s; present=%s" % (targets, sorted(present)))
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                      task_type="FEATURE_EXTRACTION", target_modules=tg)
    backbone = get_peft_model(backbone, lcfg)
    backbone.print_trainable_parameters()
    print("[lora] targets=%s quant=%s d_model=%d" % (tg, quant, d_model), flush=True)
    return backbone, tok, d_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--aug-tag", default=None, help="tag of <tag>_aug.jsonl / <tag>_anchors.jsonl; defaults to the training file name")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # hyperparameters of the paper
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--tau-p", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.995)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--lam1", type=float, default=0.6)
    ap.add_argument("--lam2", type=float, default=0.3)
    ap.add_argument("--lam3", type=float, default=0.1)
    ap.add_argument("--pool", choices=["entity", "last", "both", "ht", "ht_last"], default="entity")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--t-warm", type=int, default=500)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--rho-max", type=float, default=10.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--grad-ckpt", action="store_true")
    # ablation switches
    ap.add_argument("--no-onto", action="store_true", help="do not use the ontology filter for the clean set")
    ap.add_argument("--no-cw", action="store_true", help="W_final = 1 (no confidence weighting)")
    ap.add_argument("--cw-norm", action="store_true", help="normalise W_final of the non-NA samples in a batch to mean 1 (relative weights)")
    ap.add_argument("--no-agree", action="store_true", help="disable the label-agreement factor a_i")
    ap.add_argument("--cw-classnorm", action="store_true", help="normalise confidence weights per relation by their EMA mean before multiplying by rho")
    ap.add_argument("--la-tau", type=float, default=0.0, help="logit-adjusted loss (Menon et al. 2021): CE on logits + tau*log prior during training, raw logits at inference")
    ap.add_argument("--agree-co", action="store_true", help="relations co-occurring with the same entity pair in training are not competitors in a_i and the nearest-centroid test")
    ap.add_argument("--ce-comask", action="store_true", help="mask the other distant labels of the same entity pair in the classification CE")
    ap.add_argument("--no-freq", action="store_true", help="rho = 1 (no frequency compensation)")
    ap.add_argument("--lam2-sched", choices=["const", "linear"], default="const",
                    help="schedule of the contrastive weight: linear = decays linearly from lam2 to 0")
    ap.add_argument("--cw-floor", type=float, default=0.0, help="soft confidence weight W <- f + (1 - f) * W, i.e. a floor f for every sample")
    ap.add_argument("--cw-floor-sched", choices=["const", "linear"], default="const", help="linear = the weight floor rises linearly from --cw-floor to 1")
    ap.add_argument("--rho-sched", choices=["const", "linear"], default="const", help="linear = frequency compensation decays linearly from rho_k to 1")
    ap.add_argument("--cw-warm", action="store_true", help="W_final = 1 before t_warm steps")
    ap.add_argument("--no-anchor", action="store_true", help="do not initialise long-tail centroids with LLM anchors")
    ap.add_argument("--no-hardneg", action="store_true")
    ap.add_argument("--dump-weights", action="store_true", help="dump per-sample W/a/H/clean in the last epoch (noise analysis)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N instances")
    ap.add_argument("--init-frac", type=float, default=0.5, help="fraction of training samples (random subset) used for centroid initialisation")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    TEMPLATE["on"] = args.pool in ("last", "both", "ht_last")

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    dev = "cuda"

    # ------------------------------------------------------------- data
    rel2id = json.load(open(args.rel2id)); K = len(rel2id)
    has_na = "NA" in rel2id; na_id = rel2id.get("NA", -1)
    rel_rows = [r for r in sorted(rel2id, key=lambda r: rel2id[r]) if r != "NA"]
    relpos = torch.full((K,), -1, dtype=torch.long)          # class id -> centroid row (NA = -1)
    for row, r in enumerate(rel_rows):
        relpos[rel2id[r]] = row
    n_rel = len(rel_rows)
    train = D.load_jsonl(args.train_file)
    if args.limit:
        train = train[:args.limit]
    tag = args.aug_tag or os.path.splitext(os.path.basename(args.train_file))[0]
    aug = load_cache(os.path.join(args.cache_root, "%s_aug.jsonl" % tag))
    anchors = load_cache(os.path.join(args.cache_root, "%s_anchors.jsonl" % tag))
    types = load_cache(os.path.join(args.cache_root, "entity_types.jsonl"))
    onto = Ontology(args.ontology, relations=rel_rows, has_na=has_na)
    co_mask = None
    if args.agree_co:                                         # non-NA relations carried by the same entity pair in the training set
        pair_rels = {}
        for x in train:
            if x["relation"] != "NA":
                pair_rels.setdefault(D.bag_key(x), set()).add(x["relation"])
        co_mask = torch.zeros(n_rel, n_rel, dtype=torch.bool)
        for rs in pair_rels.values():
            for r in rs:
                for q in rs:
                    if q != r:
                        co_mask[rel_rows.index(r), rel_rows.index(q)] = True
        print("[agree-co] co-labelled relation pairs: %d" % int(co_mask.sum()), flush=True)
    ce_mask = None
    if args.ce_comask:                                        # per training instance: the other non-NA labels of the same entity pair (K-dim bool)
        pair_ids = {}
        for x in train:
            if x["relation"] != "NA" and x["relation"] in rel2id:
                pair_ids.setdefault(D.bag_key(x), set()).add(rel2id[x["relation"]])
        ce_mask = torch.zeros(len(train), K, dtype=torch.bool)
        for i, x in enumerate(train):
            yi = rel2id.get(x["relation"], na_id)
            for k in pair_ids.get(D.bag_key(x), ()):
                if k != yi:
                    ce_mask[i, k] = True
        print("[ce-comask] instances with masked co-labels: %d" % int(ce_mask.any(1).sum()), flush=True)
    n_aug = sum(1 for x in train if x["uid"] in aug and (aug[x["uid"]].get("pos")))
    print("[data] train=%d K=%d (NA=%s) aug_available=%d anchors=%d types=%d"
          % (len(train), K, has_na, n_aug, len(anchors), len(types)), flush=True)

    # frequency compensation rho_k = min(rho_bar^{-1} sqrt(N_tot / n_k), rho_max), normalised to mean 1 (eq:final_weight)
    counts = np.ones(K)
    for x in train:
        counts[rel2id.get(x["relation"], na_id)] += 1
    N_tot = float(len(train))
    raw = np.sqrt(N_tot / counts)
    ys = np.array([rel2id.get(x["relation"], na_id) for x in train])
    rho_bar = raw[ys[ys != na_id]].mean() if (ys != na_id).any() else 1.0
    rho = np.minimum(raw / rho_bar, args.rho_max)
    if has_na:
        rho[na_id] = 1.0
    if args.no_freq:
        rho[:] = 1.0
    rho_t = torch.tensor(rho, dtype=torch.float32, device=dev)
    log_prior = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=dev)
    cls_wbar = torch.full((K,), -1.0, device=dev)            # --cw-classnorm: EMA mean of the confidence weight per relation

    # ontology consistency of each training label with the entity types, computed once
    onto_ok_all = torch.tensor([1 if args.no_onto else onto.consistent(types.get(x["h"]["name"], "OTHER"),
                                                                      types.get(x["t"]["name"], "OTHER"), x["relation"])
                                for x in train], dtype=torch.bool, device=dev)

    # ------------------------------------------------------------- model
    backbone, tok, d_model = build_backbone(args.model_path, args.lora_r, args.lora_alpha, args.lora_dropout,
                                            args.lora_targets.split(","), args.quant, args.grad_ckpt)
    model = DSRELLM(backbone, K, d_model, pool=args.pool).to(dev)
    d_model = model.d_rep
    model.proj.float(); model.classifier.float()
    ds = REDataset(train, rel2id, tok, args.max_len, aug=aug, use_hardneg=not args.no_hardneg, na_id=na_id)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True,
                    collate_fn=make_collate(tok.pad_token_id), persistent_workers=True)
    head_params = list(model.proj.parameters()) + list(model.classifier.parameters())
    hid = {id(p) for p in head_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            weight_decay=0.01)
    total_steps = max(1, len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    relpos = relpos.to(dev)
    if co_mask is not None:
        co_mask = co_mask.to(dev)
    if ce_mask is not None:
        ce_mask = ce_mask.to(dev)

    # ------------------------------------------------------------- centroid initialisation (one pass before training)
    bank = CentroidBank(n_rel, d_model, momentum=args.momentum, device=dev)
    t_init = time.time()
    sums = torch.zeros(n_rel, d_model, device=dev); cnts = torch.zeros(n_rel, device=dev)
    model.eval()
    rng_init = random.Random(args.seed)
    init_idx = [i for i in range(len(train)) if train[i]["relation"] != "NA" and rng_init.random() < max(args.init_frac, 0.0)] \
        if args.init_frac < 1 else list(range(len(train)))
    init_ds = torch.utils.data.Subset(REDataset(train, rel2id, tok, args.max_len, aug=None, use_hardneg=False, na_id=na_id), init_idx)
    init_dl = DataLoader(init_ds,
                         batch_size=args.batch_size * 6, shuffle=False, num_workers=10, pin_memory=True, prefetch_factor=4,
                         collate_fn=make_collate(tok.pad_token_id))
    with torch.no_grad():
        for b in init_dl:
            h = model.encode(b["ids"].to(dev), b["am"].to(dev), b["em"].to(dev)).float()
            y = b["y"].to(dev); yp = relpos[y]
            ok = (yp >= 0) & onto_ok_all[b["idx"].to(dev)]
            if ok.any():
                sums.index_add_(0, yp[ok], h[ok]); cnts.index_add_(0, yp[ok], torch.ones(int(ok.sum()), device=dev))
    bank.init_mean(sums, cnts)
    n_anchor = 0
    if not args.no_anchor:
        for row, r in enumerate(rel_rows):
            if cnts[row] < args.n_min and anchors.get(r):
                items = [encode_text(tok, s, [], args.max_len) for s in anchors[r]]
                ids, am, em = pad_batch([i for i, _ in items], [e for _, e in items], tok.pad_token_id)
                with torch.no_grad():
                    hv = model.encode(ids.to(dev), am.to(dev), em.to(dev)).float().mean(0)
                if cnts[row] > 0:        # few real samples: real mean and anchors weighted equally
                    hv = 0.5 * hv + 0.5 * bank.c[row]
                bank.init_anchor(row, hv); n_anchor += 1
    print("[init] centroids: %d/%d from data, %d anchored, %d empty  (%.1fs)"
          % (int((cnts > 0).sum()), n_rel, n_anchor, int((~bank.initialised).sum()), time.time() - t_init), flush=True)

    # ------------------------------------------------------------- train
    step, t0 = 0, time.time()
    tokens_seen = 0
    log = open(os.path.join(args.out, "train.log"), "a")
    dump = open(os.path.join(args.out, "weights_dump.jsonl"), "w") if args.dump_weights else None
    torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        model.train()
        for b in dl:
            ids, am, em = b["ids"].to(dev), b["am"].to(dev), b["em"].to(dev)
            y = b["y"].to(dev); yp = relpos[y]
            h, z, logits = model(ids, am, em)
            tokens_seen += int(am.sum())
            anchor = b["anchor"].to(dev); a_idx = anchor.nonzero(as_tuple=True)[0]
            z2, zneg = z.new_zeros((0, z.size(1))), None
            if args.alpha > 0 and args.lam2 > 0 and a_idx.numel() > 0:
                # forward the positive views of the anchors only (non-NA with a paraphrase); original views of NA samples still serve as negatives
                ids2, am2, em2 = b["ids2"][a_idx.cpu()].to(dev), b["am2"][a_idx.cpu()].to(dev), b["em2"][a_idx.cpu()].to(dev)
                L2 = int(am2.sum(1).max()); ids2, am2, em2 = ids2[:, :L2], am2[:, :L2], em2[:, :L2]
                _, z2, _ = model(ids2, am2, em2)
                tokens_seen += int(am2.sum())
                if b["idsn"] is not None:
                    _, zneg, _ = model(b["idsn"].to(dev), b["amn"].to(dev), b["emn"].to(dev))
                    tokens_seen += int(b["amn"].sum())

            # confidence weights (eq:weight, eq:final_weight), all detached
            W, P, H, a = confidence(h.detach(), bank.c, yp, args.tau_p, args.gamma,
                                    use_agree=(not args.no_agree) and step >= args.t_warm, co_mask=co_mask)
            f_t = args.cw_floor + (1 - args.cw_floor) * step / max(total_steps, 1) if args.cw_floor_sched == "linear" else args.cw_floor
            if f_t > 0:
                W = f_t + (1 - f_t) * W
            if args.cw_classnorm and not args.no_cw and (yp >= 0).any():
                rel_m = yp >= 0
                for k in y[rel_m].unique():
                    mk = W[y == k].mean()
                    cls_wbar[k] = mk if cls_wbar[k] < 0 else 0.95 * cls_wbar[k] + 0.05 * mk
                W = W.clone(); W[rel_m] = W[rel_m] / cls_wbar[y[rel_m]].clamp(min=1e-12)
            rho_y = 1 + (rho_t[y] - 1) * (1 - step / max(total_steps, 1)) if args.rho_sched == "linear" else rho_t[y]
            W_final = torch.ones_like(W) if (args.no_cw or (args.cw_warm and step < args.t_warm)) else W * rho_y
            if args.cw_norm and not args.no_cw and (yp >= 0).any():
                rel_m = yp >= 0
                W_final = W_final.clone(); W_final[rel_m] = W_final[rel_m] / W_final[rel_m].mean().clamp(min=1e-6)

            lg = logits.float() + args.la_tau * log_prior
            if ce_mask is not None:
                lg = lg.masked_fill(ce_mask[b["idx"].to(dev)], -1e4)
            ce = F.cross_entropy(lg, y, reduction="none")
            L_weight = (W_final * ce).mean()
            L_inst = nt_xent(z.float(), z2.float(), a_idx, zneg.float() if zneg is not None else None, args.tau) \
                if (args.alpha > 0 and args.lam2 > 0) else torch.zeros((), device=dev)
            sel = yp >= 0
            if args.alpha < 1 and args.lam2 > 0 and sel.any():
                lp = prototype_loss(h[sel], yp[sel], bank.c, args.tau_p)
                L_proto = (W_final[sel] * lp).sum() / W_final[sel].sum().clamp(min=1e-6)
            else:
                L_proto = torch.zeros((), device=dev)
            L_contrast = args.alpha * L_inst + (1 - args.alpha) * L_proto
            L_reg = (model.classifier.weight ** 2).sum()
            lam2_t = args.lam2 * (1.0 - step / max(total_steps, 1)) if args.lam2_sched == "linear" else args.lam2
            loss = args.lam1 * L_weight + lam2_t * L_contrast + args.lam3 * L_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            # clean set and centroid EMA (eq:clean_set, eq:nn_test, eq:centroid_update)
            with torch.no_grad():
                onto_ok = onto_ok_all[b["idx"].to(dev)]
                clean = sel & onto_ok
                if step >= args.t_warm:
                    clean &= nn_test(P, yp, args.delta, co_mask=co_mask)
                bank.update(h.detach(), yp, W_final, clean)

            if dump is not None and ep == args.epochs - 1:
                for j in range(len(y)):
                    dump.write(json.dumps({"uid": train[int(b["idx"][j])]["uid"], "y": int(y[j]),
                                           "noisy": int(b["noisy"][j]), "W": round(float(W[j]), 5),
                                           "a": round(float(a[j]), 5), "H": round(float(H[j]), 5),
                                           "W_final": round(float(W_final[j]), 5), "clean": int(clean[j]),
                                           "p_label": round(float(P[j, yp[j]]) if yp[j] >= 0 else -1, 5),
                                           "p_max": round(float(P[j].max()), 5)}) + "\n")
            if step % args.log_every == 0:
                el = time.time() - t0
                msg = ("ep %d step %d/%d loss %.4f (Lw %.4f Li %.4f Lp %.4f) W %.3f a %.3f clean %.2f "
                       "anchors %d/%d  %.2fs/step  peak %.1fGB"
                       % (ep, step, total_steps, loss.item(), L_weight.item(), L_inst.item(), L_proto.item(),
                          W_final.mean().item(), a.mean().item(), clean.float().mean().item(),
                          int(anchor.sum()), len(y), el / max(step, 1),
                          torch.cuda.max_memory_allocated() / 1e9))
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
            step += 1
        print("[epoch %d] done  %.1f min" % (ep, (time.time() - t0) / 60), flush=True)

    # ------------------------------------------------------------- save
    model.backbone.save_pretrained(os.path.join(args.out, "lora"))
    torch.save({"proj": model.proj.state_dict(), "classifier": model.classifier.state_dict(),
                "centroids": bank.c.cpu(), "relpos": relpos.cpu(), "rel_rows": rel_rows}, os.path.join(args.out, "heads.pt"))
    stats = {"train_seconds": time.time() - t0, "init_seconds": time.time() - t_init - (time.time() - t0),
             "steps": step, "epochs": args.epochs, "n_train": len(train), "tokens_seen": tokens_seen,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
             "sec_per_step": (time.time() - t0) / max(step, 1), "n_anchored": n_anchor,
             "n_aug_available": n_aug, "rho_bar": float(rho_bar)}
    json.dump(stats, open(os.path.join(args.out, "train_stats.json"), "w"), indent=2)
    if dump is not None:
        dump.close()
    print("[done] %s" % json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
