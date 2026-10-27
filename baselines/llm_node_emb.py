"""LLM-CG (Liu & Qi, Neurocomputing 2025), step 1: initial semantic representations of the constraint-graph nodes (entity types, relations) from an LLM.
From the original abstract: "LLMs provide initial node representations, which are then refined by a graph neural network encoder".
The original code and LLM details are not public; for a controlled comparison the same backbone (Qwen3-8B) is used and each node description is encoded by the mean of the last hidden layer.
There are few nodes (types + relations < 80), so this runs on CPU.
Usage: python llm_node_emb.py --model Qwen3-8B --rel2id rel2id.json --types entity_types.jsonl --out node_emb.npz
"""
import argparse, json
import numpy as np
import torch


def rel_text(r):
    if r == "NA":
        return "No relation: the two entities are not related."
    parts = [p for p in r.strip("/").replace(".", "/").split("/") if p]
    return "Relation: %s (from %s)." % (parts[-1].replace("_", " "), " / ".join(p.replace("_", " ") for p in parts[:-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--rel2id", required=True)
    ap.add_argument("--types", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); rels = sorted(rel2id, key=rel2id.get)
    types = sorted({json.loads(l)["v"] for l in open(a.types)} | {"OTHER"})
    texts = ["Entity type: %s." % t.lower() for t in types] + [rel_text(r) for r in rels]
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).eval()
    torch.set_num_threads(8)
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt")
            h = model(**enc).last_hidden_state[0].float()
            embs.append(h.mean(0).numpy())
    np.savez(a.out, emb=np.stack(embs), types=np.array(types), rels=np.array(rels), texts=np.array(texts))
    print("saved", a.out, len(types), "types", len(rels), "relations", embs[0].shape)


if __name__ == "__main__":
    main()
