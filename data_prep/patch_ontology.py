"""Back off to the Freebase schema path where Wikidata declares no type constraint.

Wikidata leaves subject-type / value-type constraints undeclared for several
properties (P17, P131, P159, P276, P361, P527, P740). Treating "no constraint"
as "any type admissible" is logically right but leaves the Eq.(3) gate inert for
those relations. The Freebase relation path is itself a type-level schema
(/domain/type/property), so it supplies the missing side.
"""
import json, os, sys

BACKOFF = {
    "/broadcast/producer/location":              (["ORGANIZATION", "PERSON"], None),
    "/business/company/locations":               (["ORGANIZATION"], None),
    "/business/company/place_founded":           (None, ["LOCATION"]),
    "/film/film_festival/location":              (["EVENT", "ORGANIZATION"], None),
    "/location/administrative_division/country": (["LOCATION"], None),
    "/location/neighborhood/neighborhood_of":    (["LOCATION"], None),
    "/people/ethnicity/included_in_group":       (["ETHNICITY"], ["ETHNICITY"]),
    "/people/ethnicity/includes_groups":         (["ETHNICITY"], ["ETHNICITY"]),
    "/people/family/country":                    (["PERSON", "ETHNICITY"], None),
    "/people/family/members":                    (["PERSON", "ETHNICITY"], ["PERSON"]),
    "/sports/sports_team/location":              (["ORGANIZATION"], None),
    "/sports/sports_team_location/teams":        (None, ["ORGANIZATION"]),
    "/time/event/locations":                     (["EVENT"], None),
}

path = sys.argv[1] if len(sys.argv) > 1 else "data/ontology/relation_ontology.json"
o = json.load(open(path, encoding="utf-8"))
n = 0
for rel, (d, r) in BACKOFF.items():
    if rel not in o:
        print("  ! missing:", rel); continue
    spec = o[rel]
    changed = []
    if d is not None and len(spec["domain"]) >= 12:
        spec["domain"] = d; changed.append("domain")
    if r is not None and len(spec["range"]) >= 12:
        spec["range"] = r; changed.append("range")
    if changed:
        spec["source"] = spec["source"] + " + freebase-path-backoff(%s)" % ",".join(changed)
        n += 1
        print("  %-58s D=%s R=%s" % (rel, spec["domain"], spec["range"]))
json.dump(o, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\npatched %d relations -> %s" % (n, path))

still = [r for r, v in o.items() if r != "NA" and (len(v["domain"]) >= 12 or len(v["range"]) >= 12)]
print("still fully permissive on some side: %d  %s" % (len(still), still))
