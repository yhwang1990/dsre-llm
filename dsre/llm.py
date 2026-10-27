"""vLLM (OpenAI-compatible API) client: entity typing, ontology-guided six-step CoT, paraphrase / hard-negative generation,
long-tail relation descriptions (centroid anchors) and the zero-shot baseline. All outputs use guided-JSON decoding with the relation field restricted to the candidate enum.
Qwen3 runs in non-thinking mode (enable_thinking=False) with greedy decoding."""
import json, threading, time
import urllib.request

COARSE_TYPES = ["PERSON", "LOCATION", "ORGANIZATION", "WORK", "EVENT",
                "PROFESSION", "ETHNICITY", "LANGUAGE", "RELIGION", "INDUSTRY",
                "DEGREE", "OTHER"]


class LLMClient:
    def __init__(self, base_url, model, timeout=600, max_retries=3, enable_thinking=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self._lock = threading.Lock()
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(self, prompt, max_tokens=256, temperature=0.0, schema=None):
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if schema is not None:
            body["guided_json"] = schema
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                u = d.get("usage") or {}
                with self._lock:
                    self.n_calls += 1
                    self.prompt_tokens += int(u.get("prompt_tokens", 0))
                    self.completion_tokens += int(u.get("completion_tokens", 0))
                return d["choices"][0]["message"].get("content") or ""
            except Exception as e:
                last = e; time.sleep(2 * (attempt + 1))
        raise RuntimeError("LLM call failed after %d retries: %r" % (self.max_retries, last))

    def usage(self):
        return {"calls": self.n_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


def _json(s):
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                return {}
        return {}


# ------------------------------------------------------------------ entity typing (CoT step 1)
TYPE_PROMPT = """Classify the entity into exactly one coarse type.

Types: {types}

Entity: "{name}"
Context: "{ctx}"

Answer with JSON: {{"type": "<one type from the list>"}}"""
TYPE_SCHEMA = {"type": "object", "properties": {"type": {"type": "string", "enum": COARSE_TYPES}},
               "required": ["type"]}


def type_entity(client, name, context):
    out = client.chat(TYPE_PROMPT.format(types=", ".join(COARSE_TYPES), name=name, ctx=context[:300]),
                      max_tokens=32, schema=TYPE_SCHEMA)
    t = str(_json(out).get("type", "")).strip().upper()
    return t if t in COARSE_TYPES else "OTHER"


# ------------------------------------------------------------------ ontology-guided six-step CoT (Sec. 3.1)
RE_PROMPT = """You are a relation extraction expert. Follow this six-step protocol.

Step 1. Entity type identification: confirm the head and tail entity types.
Step 2. Ontology constraint retrieval: consider only the candidate relations listed below.
Step 3. Syntactic-semantic matching: find the trigger words in the sentence that match a candidate relation.
Step 4. Bidirectional consistency verification: check the relation reads sensibly as (head, r, tail) and (tail, r-inverse, head).
Step 5. Mutually exclusive relation filtering: drop candidates that logically conflict with your choice.
Step 6. Answer with JSON: a one-sentence summary of your reasoning (at most 40 words), the chosen relation, and a confidence in [0,1].

Note: many entity pairs in this data merely co-occur in a sentence. Answer "NA" when the sentence does not express any candidate relation between these two entities; co-occurrence alone is not evidence. If the sentence does express a candidate relation, choose it.

Candidate relations for a ({ht}, {tt}) entity pair (type-compatible; you MUST choose from this list, or "NA"):
{cands}

Sentence: "{sent}"
Head entity: "{h}" (type: {ht})
Tail entity: "{t}" (type: {tt})"""


def re_schema(candidates):
    return {"type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "relation": {"type": "string", "enum": list(candidates)},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["reasoning", "relation", "confidence"]}


def predict_relation(client, sent, h, t, ht, tt, candidates):
    cands = "\n".join("  - %s" % c for c in candidates)
    out = client.chat(RE_PROMPT.format(sent=sent[:1200], h=h, t=t, ht=ht, tt=tt, cands=cands),
                      max_tokens=200, schema=re_schema(candidates))
    d = _json(out)
    rel = d.get("relation")
    if rel not in candidates:
        return "NA", 0.0, d.get("reasoning", "")
    try:
        conf = min(max(float(d.get("confidence", 0.0)), 0.0), 1.0)
    except Exception:
        conf = 0.0
    return rel, conf, d.get("reasoning", "")


# ------------------------------------------------------------------ paraphrases / hard negatives (Sec. 3.2.1, non-NA only)
AUG_PROMPT = """Rewrite the sentence below for contrastive learning.

Sentence: "{sent}"
Head entity: "{h}"
Tail entity: "{t}"
Stated relation between them: {rel}

Produce:
- 2 POSITIVE paraphrases: keep the relation between "{h}" and "{t}" exactly the same, but vary
  vocabulary, syntactic structure (e.g. active/passive) and information order. Both entity
  mentions must appear verbatim.
- 1 HARD NEGATIVE: keep BOTH entity mentions verbatim but describe a clearly DIFFERENT relation
  between them (the stated relation must no longer hold).

Answer with JSON: {{"positives": ["<s1>", "<s2>"], "hard_negative": "<s3>"}}"""
AUG_SCHEMA = {"type": "object",
              "properties": {"positives": {"type": "array", "items": {"type": "string"}},
                             "hard_negative": {"type": "string"}},
              "required": ["positives", "hard_negative"]}


def augment(client, sent, h, t, rel):
    out = client.chat(AUG_PROMPT.format(sent=sent[:1200], h=h, t=t, rel=rel), max_tokens=400, schema=AUG_SCHEMA)
    d = _json(out)
    pos = [p for p in (d.get("positives") or []) if isinstance(p, str) and h in p and t in p][:2]
    neg = d.get("hard_negative")
    if not (isinstance(neg, str) and h in neg and t in neg):
        neg = None
    return pos, neg


# ------------------------------------------------------------------ long-tail relation descriptions (centroid anchors)
ANCHOR_PROMPT = """Write {n} short, diverse English sentences, each expressing the relation "{rel}" between a
head entity and a tail entity. Use concrete but fictional entity names, vary the wording, and make the relation
unambiguous. Answer with JSON: {{"sentences": ["...", ...]}}"""
ANCHOR_SCHEMA = {"type": "object", "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
                 "required": ["sentences"]}


def anchor_sentences(client, rel, n=8):
    out = client.chat(ANCHOR_PROMPT.format(n=n, rel=rel), max_tokens=600, schema=ANCHOR_SCHEMA)
    return [s for s in (_json(out).get("sentences") or []) if isinstance(s, str) and s.strip()][:n]


# ------------------------------------------------------------------ zero-shot baseline (no ontology, no CoT)
ZS_PROMPT = """Identify the relation between the two entities in the sentence.
Choose exactly one relation from this list (answer "NA" if none holds):
{cands}

Note: many entity pairs in this data merely co-occur in a sentence. Answer "NA" when the sentence does not express any listed relation between these two entities; co-occurrence alone is not evidence. If the sentence does express a listed relation, choose it.

Sentence: "{sent}"
Head entity: "{h}"
Tail entity: "{t}"

Answer with JSON: {{"relation": "<one item from the list>", "confidence": <float 0-1>}}"""


def zs_schema(all_relations):
    return {"type": "object", "properties": {"relation": {"type": "string", "enum": list(all_relations)},
                                             "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["relation", "confidence"]}


def predict_relation_zeroshot(client, sent, h, t, all_relations):
    cands = "\n".join("  - %s" % c for c in all_relations)
    out = client.chat(ZS_PROMPT.format(sent=sent[:1200], h=h, t=t, cands=cands), max_tokens=64,
                      schema=zs_schema(all_relations))
    d = _json(out)
    rel = d.get("relation")
    if rel not in all_relations:
        return "NA", 0.0
    try:
        return rel, min(max(float(d.get("confidence", 0.0)), 0.0), 1.0)
    except Exception:
        return rel, 0.0
