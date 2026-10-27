"""Build type-level domain/range ontology for NYT10 + GIDS relations from Wikidata.

Implements the constraint source of DSRE-LLM Eq.(3):
  L_onto(e_h, e_t, r_k) = 1[type(e_h) in D_rk] * 1[type(e_t) in R_rk]

Constraints come from Wikidata property constraints (P2302):
  Q21503250 = type constraint        -> subject/domain classes (qualifier P2308)
  Q21510865 = value-type constraint  -> object/range classes  (qualifier P2308)
Wikidata classes are folded into a coarse type inventory.
"""
import json, os, sys, time, urllib.request

UA = os.environ.get("WIKIDATA_UA", "DSRE-ontology-builder/1.0 (research; set WIKIDATA_UA to include your contact)")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ontology")

# ---------------------------------------------------------------- coarse types
COARSE = ["PERSON", "LOCATION", "ORGANIZATION", "WORK", "EVENT",
          "PROFESSION", "ETHNICITY", "LANGUAGE", "RELIGION", "INDUSTRY", "DEGREE", "OTHER"]

# Wikidata class -> coarse type anchors (checked directly, then via P279* ancestors)
ANCHORS = {
    "Q5": "PERSON", "Q215627": "PERSON", "Q95074": "PERSON", "Q729": "OTHER",
    "Q2221906": "LOCATION", "Q17334923": "LOCATION", "Q56061": "LOCATION",
    "Q515": "LOCATION", "Q6256": "LOCATION", "Q486972": "LOCATION",
    "Q82794": "LOCATION", "Q618123": "LOCATION", "Q27096213": "LOCATION",
    "Q3895768": "LOCATION", "Q35145263": "LOCATION", "Q13226383": "LOCATION",
    "Q43229": "ORGANIZATION", "Q4830453": "ORGANIZATION", "Q783794": "ORGANIZATION",
    "Q3918": "ORGANIZATION", "Q875538": "ORGANIZATION", "Q2385804": "ORGANIZATION",
    "Q4438121": "ORGANIZATION", "Q79913": "ORGANIZATION", "Q17334831": "ORGANIZATION",
    "Q11424": "WORK", "Q7725634": "WORK", "Q386724": "WORK", "Q2431196": "WORK",
    "Q15416": "WORK", "Q1656682": "EVENT", "Q1190554": "EVENT",
    "Q28640": "PROFESSION", "Q12737077": "PROFESSION",
    "Q41710": "ETHNICITY", "Q17573152": "ETHNICITY",
    "Q34770": "LANGUAGE", "Q9174": "RELIGION", "Q879146": "RELIGION",
    "Q8148": "INDUSTRY", "Q268592": "INDUSTRY",
    "Q189533": "DEGREE", "Q6379050": "DEGREE",
}

# ------------------------------------------- Freebase relation -> Wikidata prop
# inv=True means the Freebase relation is the inverse direction of the WD property,
# so its domain/range are the WD range/domain.
FB2WD = {
    "/people/person/place_of_birth":                    ("P19",  False),
    "/people/deceased_person/place_of_death":           ("P20",  False),
    "/people/deceased_person/place_of_burial":          ("P119", False),
    "/people/place_of_interment/interred_here":         ("P119", True),
    "/people/person/nationality":                       ("P27",  False),
    "/people/person/religion":                          ("P140", False),
    "/people/person/profession":                        ("P106", False),
    "/people/profession/people_with_this_profession":   ("P106", True),
    "/people/person/children":                          ("P40",  False),
    "/people/person/ethnicity":                         ("P172", False),
    "/people/person/place_lived":                       ("P551", False),
    "/people/family/members":                           ("P527", False),
    "/people/family/country":                           ("P17",  False),
    "/people/ethnicity/included_in_group":              ("P361", False),
    "/people/ethnicity/includes_groups":                ("P527", False),
    "/business/company/founders":                       ("P112", False),
    "/business/person/company":                         ("P108", False),
    "/business/company/place_founded":                  ("P740", False),
    "/business/company/industry":                       ("P452", False),
    "/business/company/locations":                      ("P159", False),
    "/business/shopping_center/owner":                  ("P127", False),
    "/business/shopping_center_owner/shopping_centers_owned": ("P127", True),
    "/business/company/major_shareholders":             ("P127", False),
    "/business/company_shareholder/major_shareholder_of": ("P127", True),
    "/business/business_location/parent_company":       ("P749", False),
    "/location/country/capital":                        ("P36",  False),
    "/location/fr_region/capital":                      ("P36",  False),
    "/location/cn_province/capital":                    ("P36",  False),
    "/location/it_region/capital":                      ("P36",  False),
    "/location/us_state/capital":                       ("P36",  False),
    "/location/br_state/capital":                       ("P36",  False),
    "/location/mx_state/capital":                       ("P36",  False),
    "/location/de_state/capital":                       ("P36",  False),
    "/location/jp_prefecture/capital":                  ("P36",  False),
    "/location/province/capital":                       ("P36",  False),
    "/location/us_county/county_seat":                  ("P36",  False),
    "/location/in_state/administrative_capital":        ("P36",  False),
    "/location/in_state/legislative_capital":           ("P36",  False),
    "/location/in_state/judicial_capital":              ("P36",  False),
    "/location/location/contains":                      ("P150", False),
    "/location/country/administrative_divisions":       ("P150", False),
    "/base/locations/countries/states_provinces_within": ("P150", False),
    "/location/administrative_division/country":        ("P17",  False),
    "/location/neighborhood/neighborhood_of":           ("P131", False),
    "/location/country/languages_spoken":               ("P37",  False),
    "/sports/sports_team/location":                     ("P159", False),
    "/sports/sports_team_location/teams":               ("P159", True),
    "/broadcast/producer/location":                     ("P159", False),
    "/broadcast/content/location":                      ("P495", False),
    "/film/film/featured_film_locations":               ("P915", False),
    "/film/film_location/featured_in_films":            ("P915", True),
    "/film/film_festival/location":                     ("P276", False),
    "/time/event/locations":                            ("P276", False),
    # GIDS
    "/people/person/education./education/education/institution": ("P69",  False),
    "/people/person/education./education/education/degrees":     ("P512", False),
}

# Relations with no clean Wikidata counterpart -> types read off the Freebase schema path
MANUAL = {
    "/business/company_advisor/companies_advised": (["ORGANIZATION", "PERSON"], ["ORGANIZATION"]),
    "/business/company/advisors":                  (["ORGANIZATION"], ["ORGANIZATION", "PERSON"]),
    "/people/ethnicity/geographic_distribution":   (["ETHNICITY"], ["LOCATION"]),
    "/people/ethnicity/people":                    (["ETHNICITY"], ["PERSON"]),
}

_cache = {}


def wd_entity(qid):
    if qid in _cache:
        return _cache[qid]
    url = "https://www.wikidata.org/wiki/Special:EntityData/%s.json" % qid
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            ent = d["entities"][qid]
            _cache[qid] = ent
            time.sleep(0.4)
            return ent
        except Exception as e:
            sys.stderr.write("  retry %s (%s)\n" % (qid, type(e).__name__))
            time.sleep(3 * (attempt + 1))
    _cache[qid] = None
    return None


def constraint_classes(prop):
    """Return (domain_qids, range_qids) from the property's P2302 constraints."""
    ent = wd_entity(prop)
    if ent is None:
        return [], []
    dom, rng = [], []
    for c in ent["claims"].get("P2302", []):
        try:
            ctype = c["mainsnak"]["datavalue"]["value"]["id"]
        except Exception:
            continue
        if ctype not in ("Q21503250", "Q21510865"):
            continue
        qs = [q["datavalue"]["value"]["id"]
              for q in c.get("qualifiers", {}).get("P2308", []) if q.get("datavalue")]
        (dom if ctype == "Q21503250" else rng).extend(qs)
    return dom, rng


def coarse_of(qid, depth=0):
    """Fold a Wikidata class into the coarse inventory via anchors, then P279 ancestors."""
    if qid in ANCHORS:
        return ANCHORS[qid]
    if depth >= 2:
        return None
    ent = wd_entity(qid)
    if ent is None:
        return None
    for c in ent["claims"].get("P279", []):
        try:
            parent = c["mainsnak"]["datavalue"]["value"]["id"]
        except Exception:
            continue
        got = coarse_of(parent, depth + 1)
        if got:
            return got
    return None


def fold(qids):
    out = []
    for q in qids:
        t = coarse_of(q)
        if t and t not in out:
            out.append(t)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rel2id = json.load(open(os.path.join(OUT, "..", "nyt10", "nyt10_rel2id.json")))
    gids_rels = ["/people/deceased_person/place_of_death",
                 "/people/person/education./education/education/degrees",
                 "/people/person/education./education/education/institution",
                 "/people/person/place_of_birth", "NA"]
    all_rels = sorted(set(list(rel2id) + gids_rels))

    ontology, unresolved = {}, []
    for r in all_rels:
        if r == "NA":
            ontology[r] = {"domain": COARSE, "range": COARSE, "source": "NA-wildcard"}
            continue
        if r in MANUAL:
            d, g = MANUAL[r]
            ontology[r] = {"domain": d, "range": g, "source": "freebase-schema-path"}
            continue
        if r not in FB2WD:
            unresolved.append(r)
            ontology[r] = {"domain": COARSE, "range": COARSE, "source": "unmapped-permissive"}
            continue
        prop, inv = FB2WD[r]
        dq, rq = constraint_classes(prop)
        d, g = fold(dq), fold(rq)
        if inv:
            d, g = g, d
        if not d:
            d = COARSE
        if not g:
            g = COARSE
        ontology[r] = {"domain": d, "range": g,
                       "source": "wikidata:%s%s" % (prop, "(inverse)" if inv else ""),
                       "wd_domain_classes": rq if inv else dq,
                       "wd_range_classes": dq if inv else rq}
        print("%-58s %-22s D=%s R=%s" % (r, ontology[r]["source"], d, g))

    path = os.path.join(OUT, "relation_ontology.json")
    json.dump(ontology, open(path, "w"), indent=2, ensure_ascii=False)
    print("\nwrote %s  (%d relations, %d unmapped)" % (path, len(ontology), len(unresolved)))
    if unresolved:
        print("unmapped:", unresolved)


if __name__ == "__main__":
    main()
