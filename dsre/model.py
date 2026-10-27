"""DSRE-LLM encoder, projection head, classifier head and training objectives (one-to-one with the equations in the paper).

h_i  : mean pooling of the last backbone layer over the head/tail entity tokens   (Sec. 3.4.1)
z_i  : projection head g(.) followed by L2 normalisation                          (eq:proj)
y^_i : Softmax(W_c h_i + b_c)                                 (eq:classifier)
instance level: NT-Xent with hard negatives                                      (eq:ntxent)
prototype level: non-NA relation centroids, prototype loss / entropy confidence / label-agreement factor (eq:proto_loss, eq:prob_dist, eq:entropy, eq:agree, eq:weight)
centroid bank: EMA update; an empty clean set keeps the previous centroid         (eq:clean_set, eq:nn_test, eq:centroid_update)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_hidden=1024, d_out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))

    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)


class DSRELLM(nn.Module):
    """pool: entity = mean over the entity spans; last = last token of the sentence-final prompt template (it sees the whole sentence in a causal model);
    both = concatenation of the two (representation size 2d)."""
    def __init__(self, backbone, n_classes, d_model, proj_dim=256, dropout=0.1, pool="entity"):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.pool = pool
        self.d_rep = d_model * {"entity": 1, "last": 1, "both": 2, "ht": 2, "ht_last": 3}[pool]
        self.proj = ProjectionHead(self.d_rep, d_out=proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.d_rep, n_classes)

    def encode(self, input_ids, attention_mask, entity_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            hs = out.hidden_states[-1]
        am = attention_mask.to(hs.device)
        last = hs[torch.arange(hs.size(0), device=hs.device), am.sum(1) - 1]      # right padding
        if self.pool == "last":
            return last
        em = entity_mask.to(hs.device)

        def span_mean(m):
            m = m.float(); empty = m.sum(dim=1) == 0
            if empty.any():
                m = m.clone(); m[empty] = am[empty].float()
            w = (m * am.float()).unsqueeze(-1)
            return (hs * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        if self.pool in ("ht", "ht_last"):                 # pool head and tail separately and concatenate: keeps direction
            head, tail = span_mean(em == 1), span_mean(em == 2)
            return torch.cat([head, tail] + ([last] if self.pool == "ht_last" else []), dim=-1)
        ent = span_mean(em > 0)
        return ent if self.pool == "entity" else torch.cat([ent, last], dim=-1)

    def forward(self, input_ids, attention_mask, entity_mask):
        h = self.encode(input_ids, attention_mask, entity_mask).float()
        if self.classifier.weight.device != h.device:
            self.proj.to(h.device); self.classifier.to(h.device)
        z = self.proj(h)
        logits = self.classifier(self.dropout(h))
        return h, z, logits


# ------------------------------------------------------------------ instance level
def nt_xent(z, z_aug, anchor_idx, z_neg=None, temperature=0.07):
    """eq:ntxent. z: [N,d] all original views; z_aug: [A,d] positive views of the anchors only (non-NA with a paraphrase), anchor_idx: [A] their rows in z;
    z_neg: [M,d] hard negatives, used only in the denominator. Keys = all originals + anchor positives + hard negatives; loss averaged over the 2A anchor views."""
    A = z_aug.size(0)
    if A == 0:
        return z.new_zeros(())
    n = z.size(0)
    keys = torch.cat([z, z_aug] + ([z_neg] if z_neg is not None and z_neg.numel() > 0 else []), dim=0)
    queries = torch.cat([z[anchor_idx], z_aug], dim=0)                 # [2A,d]
    sim = queries @ keys.t() / temperature                             # [2A, N+A(+M)]
    rows = torch.arange(2 * A, device=z.device)
    self_cols = torch.cat([anchor_idx, n + torch.arange(A, device=z.device)])
    sim[rows, self_cols] = float("-inf")                               # I[j != i]
    targets = torch.cat([n + torch.arange(A, device=z.device), anchor_idx])
    return F.cross_entropy(sim, targets)


# ------------------------------------------------------------------ prototype level
def prototype_logits(h, centroids, tau_p):
    hn = F.normalize(h, dim=-1); cn = F.normalize(centroids, dim=-1)
    return hn @ cn.t() / tau_p                                  # [B,K]


def prototype_loss(h, ypos, centroids, tau_p=0.1):
    """eq:proto_loss, per sample; ypos is the row of the non-NA relation in `centroids`."""
    return F.cross_entropy(prototype_logits(h, centroids, tau_p), ypos, reduction="none")


@torch.no_grad()
def confidence(h, centroids, ypos, tau_p=0.1, gamma=3.0, use_agree=True, co_mask=None):
    """eq:prob_dist / eq:entropy / eq:agree / eq:weight.
    ypos: [B], centroid row for non-NA samples, -1 for NA samples (W=1, a=1).
    co_mask: optional [n_rel,n_rel] bool; (label, k) pairs marked True are co-occurring relations of the same entity pair and are not competitors in the denominator of a_i.
    Returns W, P, H, a (all detached)."""
    P = F.softmax(prototype_logits(h, centroids, tau_p), dim=-1)   # [B,K]
    H = -(P * (P + 1e-12).log()).sum(dim=-1)
    h_max = torch.log(torch.tensor(float(centroids.size(0)), device=h.device))
    W_ent = torch.sigmoid(gamma * (h_max - 2.0 * H) / h_max)
    is_rel = ypos >= 0
    a = torch.ones_like(W_ent)
    if use_agree and is_rel.any():
        Pr = P[is_rel]
        pk = Pr.gather(1, ypos[is_rel].unsqueeze(1)).squeeze(1)
        if co_mask is not None:
            Pr = Pr.masked_fill(co_mask[ypos[is_rel]], 0.0)
        a[is_rel] = pk / Pr.max(dim=1).values.clamp(min=1e-12)
    W = a * W_ent
    W[~is_rel] = 1.0
    return W, P, H, a


@torch.no_grad()
def nn_test(P, ypos, delta, co_mask=None):
    """eq:nn_test: nearest centroid == label centroid and membership probability >= delta. Always False for NA samples.
    co_mask as in confidence(): co-occurring relations do not compete for the nearest centroid."""
    is_rel = ypos >= 0
    ok = torch.zeros_like(is_rel)
    if is_rel.any():
        Pr = P[is_rel]; yr = ypos[is_rel]
        if co_mask is not None:
            Pr = Pr.masked_fill(co_mask[yr], 0.0)
        ok[is_rel] = (Pr.argmax(dim=1) == yr) & (Pr.gather(1, yr.unsqueeze(1)).squeeze(1) >= delta)
    return ok


class CentroidBank:
    """EMA centroids of the non-NA relations (eq:centroid_update). Row = position index of the non-NA relation."""

    def __init__(self, n_rel, dim, momentum=0.995, device="cuda"):
        self.c = torch.zeros(n_rel, dim, device=device, dtype=torch.float32)
        self.initialised = torch.zeros(n_rel, dtype=torch.bool, device=device)
        self.m = momentum

    @torch.no_grad()
    def init_mean(self, sums, counts):
        """Mean initialisation after one pass before training; rows with counts == 0 are left for the anchors."""
        has = counts > 0
        self.c[has] = sums[has] / counts[has].unsqueeze(1)
        self.initialised |= has

    @torch.no_grad()
    def init_anchor(self, k, vector):
        self.c[k] = vector.to(self.c.device).float(); self.initialised[k] = True

    @torch.no_grad()
    def update(self, h, ypos, weights, clean_mask):
        """Confidence-weighted mean + momentum; a relation with an empty clean set keeps its previous centroid."""
        for k in ypos[clean_mask].unique():
            sel = (ypos == k) & clean_mask
            w = weights[sel].unsqueeze(-1)
            if w.sum() <= 0:
                continue
            new = (w * h[sel]).sum(0) / w.sum()
            if self.initialised[k]:
                self.c[k] = self.m * self.c[k] + (1 - self.m) * new
            else:
                self.c[k] = new; self.initialised[k] = True
