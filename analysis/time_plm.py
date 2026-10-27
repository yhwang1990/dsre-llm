"""Table 4: measured inference throughput of the RoBERTa-MIL encoder (roberta-base, max_len 128, bf16, tokenization included),
timed on the first 20,000 sentences of the NYT10 test set. Requires a GPU. Output: results/roberta_infer_timing.json"""
import json, time
import torch
from transformers import AutoTokenizer, AutoModel
from common import D, R, P, RES

X = D.load_jsonl(P + "/nyt10/test_full.jsonl", limit=20000)
tok = AutoTokenizer.from_pretrained(R + "/models/roberta-base"); m = AutoModel.from_pretrained(R + "/models/roberta-base").cuda().eval()
head = torch.nn.Linear(768 * 2, 58).cuda()
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for s in range(0, len(X), 256):
        b = tok([x["text"] for x in X[s:s + 256]], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
        h = m(**b).last_hidden_state; head(torch.cat([h[:, 0], h.mean(1)], -1)).softmax(-1).cpu()
torch.cuda.synchronize(); dt = time.time() - t0
r = {"n": len(X), "seconds": dt, "ms_per_inst": 1000 * dt / len(X), "inst_per_sec": len(X) / dt, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
print(json.dumps(r)); json.dump(r, open(RES + "/roberta_infer_timing.json", "w"))
