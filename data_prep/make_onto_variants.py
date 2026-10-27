"""Two alternative ontologies for the ontology-isolating ablation variants (only domain/range change, entity types are unchanged):
coarse: the 12 types are folded into 4 super-types (PERSON / PLACE=LOCATION / GROUP=ORGANIZATION,ETHNICITY,RELIGION,INDUSTRY /
        ABSTRACT=WORK,EVENT,LANGUAGE,PROFESSION,DEGREE,OTHER), and each domain/range is expanded to all types of its super-type;
noisy: 20% of the relations (fixed seed) get their domain or range (50% each) replaced by 1-2 randomly drawn other types.
Usage: python make_onto_variants.py --ontology relation_ontology_all.json --out-dir data/ontology"""
import argparse, json, os, random

GROUPS = {"PERSON": ["PERSON"], "PLACE": ["LOCATION"], "GROUP": ["ORGANIZATION", "ETHNICITY", "RELIGION", "INDUSTRY"],
          "ABSTRACT": ["WORK", "EVENT", "LANGUAGE", "PROFESSION", "DEGREE", "OTHER"]}
SUPER = {t: g for g, ts in GROUPS.items() for t in ts}
TYPES = sorted(SUPER)

ap = argparse.ArgumentParser(); ap.add_argument("--ontology", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--noise", type=float, default=0.2); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
onto = json.load(open(a.ontology))
expand = lambda ts: sorted({u for t in ts for u in GROUPS[SUPER.get(t, "ABSTRACT")]})
coarse = {r: dict(v, domain=expand(v["domain"]), range=expand(v["range"])) for r, v in onto.items()}
rng = random.Random(a.seed); rels = sorted(onto); hit = set(rng.sample(rels, round(a.noise * len(rels))))
noisy = {}
for r in rels:
    v = dict(onto[r])
    if r in hit:
        side = "domain" if rng.random() < 0.5 else "range"
        pool = [t for t in TYPES if t not in onto[r][side]] or [t for t in TYPES if t != "OTHER"]
        v[side] = sorted(rng.sample(pool, min(rng.choice([1, 2]), len(pool))))
    noisy[r] = v
json.dump(coarse, open(os.path.join(a.out_dir, "onto_coarse.json"), "w"), indent=1)
json.dump(noisy, open(os.path.join(a.out_dir, "onto_noisy.json"), "w"), indent=1)
json.dump({"noisy_relations": sorted(hit), "noise": a.noise, "seed": a.seed}, open(os.path.join(a.out_dir, "onto_noisy_meta.json"), "w"), indent=1)
print("relations %d, corrupted %d" % (len(rels), len(hit)))
