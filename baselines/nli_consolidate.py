"""BERT+NLI (DSRE-NLI, Zhou et al., AAAI 2023), step 1: denoise the distantly supervised training labels with an off-the-shelf MNLI model.
Following the official implementation (github.com/kangISU/DSRE-NLI, nli/nli_base.py + mcre/consolidate.py):
  - each relation turns (head, tail) into hypothesis sentences via templates, the premise is the original sentence; only relations whose domain/range admit the entity types are scored
  - relation score = max entailment probability over its templates; if the best score is below the negative threshold (0.95) the instance is NA
  - NPIN consolidation: instances assigned a relation by NLI take the NLI label; instances that are NA for both NLI and the distant label are kept; the rest are dropped
The official code provides templates for 10 NYT relations only (the first 10 entries below are copied from its genr_patt); the remaining relations are written in the same style.
Step 2 trains a BERT multi-class model on the consolidated training set (plm_re.py --method bert-ce), scored by the same evaluation code as all other methods.
Usage: python nli_consolidate.py --train-file train_main.jsonl --rel2id rel2id.json --ontology onto.json --types entity_types.jsonl --out train_main_nli.jsonl"""
import argparse, collections, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsre import data as D

T = {  # {subj} = head entity, {obj} = tail entity
    "/location/location/contains": ["{obj} is located in {subj}", "{subj} , including {obj}"],
    "/people/person/nationality": ["{obj} is the nationality of {subj}", "{subj} , president of {obj}", "{obj} , Prime Minister {subj}",
                                   "{obj} 's foreign minister , {subj}", "{obj} 's former prime minister , {subj}", "{obj} President {subj}",
                                   "{obj} 's acting prime minister , {subj}"],
    "/location/country/capital": ["{obj} is the capital of {subj}"],
    "/people/person/place_lived": ["{subj} lives in {obj}", "{subj} , Republican from {obj}", "{subj} , Democrat of {obj}", "{obj} Gov. {subj}",
                                   "{obj} mayor , {subj}", "{subj} , the former senator from {obj}", "{subj} , who was mayor when {obj}",
                                   "{obj} Coach {subj}", "{obj} State Coach {subj}"],
    "/business/person/company": ["{subj} works in {obj}", "{subj} , the head of {obj}", "{obj} 's chairman , {subj}", "{obj} 's president , {subj}",
                                 "{subj} chief executive , {obj}", "{obj} secretary general , {subj}", "{subj} , a professor at {obj}",
                                 "{subj} , founder of {obj}", "{obj} newsman {subj}", "{subj} , the co-founder of {obj}"],
    "/location/neighborhood/neighborhood_of": ["{subj} is in the neighborhood of {obj}"],
    "/people/person/place_of_birth": ["{subj} was born in {obj}"],
    "/people/deceased_person/place_of_death": ["{subj} died in {obj}"],
    "/business/company/founders": ["{subj} was founded by {obj}", "{subj} co-founder {obj}"],
    "/people/person/children": ["{subj} is the parent of {obj}"],
    # ---- templates added for relations not covered by the official code
    "/location/administrative_division/country": ["{subj} is in the country {obj}"],
    "/location/country/administrative_divisions": ["{obj} is an administrative division of {subj}"],
    "/location/us_state/capital": ["{obj} is the capital of {subj}"],
    "/location/fr_region/capital": ["{obj} is the capital of {subj}"],
    "/location/cn_province/capital": ["{obj} is the capital of {subj}"],
    "/location/it_region/capital": ["{obj} is the capital of {subj}"],
    "/location/br_state/capital": ["{obj} is the capital of {subj}"],
    "/location/de_state/capital": ["{obj} is the capital of {subj}"],
    "/location/mx_state/capital": ["{obj} is the capital of {subj}"],
    "/location/jp_prefecture/capital": ["{obj} is the capital of {subj}"],
    "/location/province/capital": ["{obj} is the capital of {subj}"],
    "/location/in_state/administrative_capital": ["{obj} is the administrative capital of {subj}"],
    "/location/in_state/legislative_capital": ["{obj} is the legislative capital of {subj}"],
    "/location/in_state/judicial_capital": ["{obj} is the judicial capital of {subj}"],
    "/location/us_county/county_seat": ["{obj} is the county seat of {subj}"],
    "/base/locations/countries/states_provinces_within": ["{obj} is a state or province of {subj}"],
    "/location/country/languages_spoken": ["{obj} is spoken in {subj}"],
    "/people/person/religion": ["{subj} follows {obj}"],
    "/people/person/ethnicity": ["{subj} is {obj}"],
    "/people/person/profession": ["{subj} is a {obj}"],
    "/people/profession/people_with_this_profession": ["{obj} is a {subj}"],
    "/people/ethnicity/people": ["{obj} is {subj}"],
    "/people/ethnicity/geographic_distribution": ["{subj} people live in {obj}"],
    "/people/ethnicity/included_in_group": ["{subj} are part of {obj}"],
    "/people/ethnicity/includes_groups": ["{obj} are part of {subj}"],
    "/people/family/members": ["{obj} is a member of the {subj} family"],
    "/people/family/country": ["the {subj} family is from {obj}"],
    "/people/deceased_person/place_of_burial": ["{subj} is buried in {obj}"],
    "/people/place_of_interment/interred_here": ["{obj} is buried in {subj}"],
    "/business/company/place_founded": ["{subj} was founded in {obj}"],
    "/business/company/advisors": ["{obj} is an advisor to {subj}"],
    "/business/company_advisor/companies_advised": ["{subj} is an advisor to {obj}"],
    "/business/company/major_shareholders": ["{obj} is a major shareholder of {subj}"],
    "/business/company_shareholder/major_shareholder_of": ["{subj} is a major shareholder of {obj}"],
    "/business/company/industry": ["{subj} is in the {obj} industry"],
    "/business/company/locations": ["{subj} has a location in {obj}"],
    "/business/business_location/parent_company": ["{subj} is owned by {obj}"],
    "/business/shopping_center/owner": ["{subj} is owned by {obj}"],
    "/business/shopping_center_owner/shopping_centers_owned": ["{subj} owns {obj}"],
    "/sports/sports_team/location": ["{subj} is based in {obj}"],
    "/sports/sports_team_location/teams": ["{obj} is based in {subj}"],
    "/film/film/featured_film_locations": ["{subj} was filmed in {obj}"],
    "/film/film_location/featured_in_films": ["{obj} was filmed in {subj}"],
    "/film/film_festival/location": ["{subj} takes place in {obj}"],
    "/time/event/locations": ["{subj} took place in {obj}"],
    "/broadcast/content/location": ["{subj} is broadcast from {obj}"],
    "/broadcast/producer/location": ["{subj} is located in {obj}"],
    # GIDS
    "/people/person/education./education/education/institution": ["{subj} studied at {obj}", "{subj} graduated from {obj}"],
    "/people/person/education./education/education/degrees": ["{subj} has a {obj} degree", "{subj} received a {obj}"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True); ap.add_argument("--rel2id", required=True)
    ap.add_argument("--ontology", required=True); ap.add_argument("--types", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v2-xlarge-mnli")
    ap.add_argument("--negative-threshold", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel2id = json.load(open(a.rel2id)); rels = [r for r in rel2id if r != "NA"]
    missing = [r for r in rels if r not in T]
    assert not missing, missing
    onto = json.load(open(a.ontology))
    types = {}
    for l in open(a.types):
        x = json.loads(l); types[x["k"]] = x["v"]
    train = D.load_jsonl(a.train_file)

    def ok(r, th, tt):   # valid conditions: entity types must lie in the domain/range of the relation; unknown types (OTHER) are not filtered
        o = onto.get(r)
        if o is None:
            return True
        return (th == "OTHER" or th in o["domain"]) and (tt == "OTHER" or tt in o["range"])

    # the same sentence/entity pair appears once per label in the training set; score it only once
    uniq = {}
    for x in train:
        uniq.setdefault((x["text"], x["h"]["name"], x["t"]["name"]), None)
    keys = list(uniq)
    pairs = []          # (key_idx, rel_idx, premise, hypothesis)
    for ki, (text, h, t) in enumerate(keys):
        th, tt = types.get(h, "OTHER"), types.get(t, "OTHER")
        for ri, r in enumerate(rels):
            if ok(r, th, tt):
                for tp in T[r]:
                    pairs.append((ki, ri, text, tp.format(subj=h, obj=t)))
    print("[nli] %d train sentences, %d unique (sentence, pair), %d NLI pairs" % (len(train), len(keys), len(pairs)), flush=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    ent = {k.lower(): v for k, v in model.config.label2id.items()}["entailment"]
    score = np.zeros((len(keys), len(rels)), dtype=np.float32)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][2]))   # batches sorted by length
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), a.batch_size):
            bi = order[s:s + a.batch_size]
            enc = tok([pairs[i][2] for i in bi], [pairs[i][3] for i in bi], truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
            p = model(**enc).logits.float().softmax(-1)[:, ent].cpu().numpy()
            for j, i in enumerate(bi):
                ki, ri = pairs[i][0], pairs[i][1]
                if p[j] > score[ki, ri]:
                    score[ki, ri] = p[j]
            if (s // a.batch_size) % 200 == 0:
                print("  %d/%d  %.0f pairs/s" % (s, len(pairs), (s + len(bi)) / (time.time() - t0)), flush=True)
    best = score.argmax(1); top = score.max(1)
    pred = {k: (rels[best[i]] if top[i] >= a.negative_threshold else "NA") for i, k in enumerate(keys)}
    # NPIN consolidation
    out, stat = [], collections.Counter()
    for x in train:
        p = pred[(x["text"], x["h"]["name"], x["t"]["name"])]
        if p != "NA":
            y = dict(x); y["relation"] = p; y["ds_relation"] = x["relation"]; out.append(y)
            stat["pos_agree" if p == x["relation"] else "pos_revised"] += 1
        elif x["relation"] == "NA":
            out.append(x); stat["neg"] += 1
        else:
            stat["dropped"] += 1
    # a sentence repeated for several labels can become identical after NLI relabelling; deduplicate
    seen, final = set(), []
    for y in out:
        k = (y["text"], y["h"]["name"], y["t"]["name"], y["relation"])
        if k not in seen:
            seen.add(k); final.append(y)
    with open(a.out, "w") as f:
        for y in final:
            f.write(json.dumps(y, ensure_ascii=False) + "\n")
    json.dump({"stat": dict(stat), "n_out": len(final), "n_in": len(train), "nli_seconds": time.time() - t0,
               "model": a.model, "negative_threshold": a.negative_threshold}, open(a.out + ".stat.json", "w"), indent=1)
    print("[done]", dict(stat), "kept", len(final), flush=True)


if __name__ == "__main__":
    main()
