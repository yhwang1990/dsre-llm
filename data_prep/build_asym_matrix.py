"""Transition matrix of the asymmetric noise: labels flip according to the semantic similarity of relation names.

Similarity = cosine similarity of all-MiniLM-L6-v2 sentence embeddings of the relation names (Wikidata property labels for Wiki80).
Each relation k may only flip to its top-m most similar relations, with flip probability proportional to softmax(sim / T).
Output JSON: {"relations": [...], "top_m": m, "temperature": T,
            "transition": {k: {k': prob, ...}}, "similarity": [[...]]}
Usage: python3 build_asym_matrix.py --rel2id wiki80_rel2id.json --out wiki80_asym_T.json
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def embed(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    with torch.no_grad():
        e = tok(texts, padding=True, truncation=True, return_tensors="pt")
        hs = mdl(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        v = (hs * m).sum(1) / m.sum(1)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-m", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    a = ap.parse_args()
    rels = [r for r in sorted(json.load(open(a.rel2id)), key=lambda x: json.load(open(a.rel2id))[x]) if r != "NA"]
    V = embed(rels)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    T = {}
    for i, r in enumerate(rels):
        top = np.argsort(-S[i])[: a.top_m]
        w = np.exp(S[i, top] / a.temperature); w = w / w.sum()
        T[r] = {rels[j]: float(p) for j, p in zip(top, w)}
    json.dump({"relations": rels, "top_m": a.top_m, "temperature": a.temperature,
               "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
               "transition": T, "similarity": np.round(S, 4).tolist()},
              open(a.out, "w"), indent=1)
    for r in rels[:8]:
        print("%-45s -> %s" % (r, ", ".join("%s(%.2f)" % kv for kv in T[r].items())))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
