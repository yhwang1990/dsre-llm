"""Wiki80 relation -> Wikidata property -> type-level domain/range constraints.

Wiki80 relation names are the English labels of Wikidata properties; they are resolved to PIDs with wbsearchentities(type=property),
then the P2302 constraint extraction and coarse-type folding of build_ontology.py are reused. The output has the same format as
relation_ontology.json, and both files are merged into relation_ontology_all.json (NYT10 + GDS + Wiki80).

Usage: python3 build_wiki80_ontology.py --rel2id <wiki80_rel2id.json> --base <relation_ontology.json> --out <dir>
"""
import argparse, json, os, sys, time, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_ontology as B

# relations whose label is ambiguous or whose search result is not unique: PID fixed manually
PINNED = {
    "country": "P17", "location": "P276", "part of": "P361", "has part": "P527",
    "instance of": "P31", "follows": "P155", "followed by": "P156",
    "participant": "P710", "participant of": "P1344", "winner": "P1346",
    "operator": "P137", "owned by": "P127", "occupant": "P466", "crosses": "P177",
    "tributary": "P974", "mouth of the watercourse": "P403", "sport": "P641",
    "platform": "P400", "developer": "P178", "publisher": "P123", "performer": "P175",
    "characters": "P674", "movement": "P135", "genre": "P136", "religion": "P140",
    "residence": "P551", "spouse": "P26", "child": "P40", "father": "P22", "mother": "P25",
    "sibling": "P3373", "occupation": "P106", "position held": "P39", "member of": "P463",
    "league": "P118", "manufacturer": "P176", "distributor": "P750", "director": "P57",
    "composer": "P86", "screenwriter": "P58", "architect": "P84", "instrument": "P1303",
    "voice type": "P412", "record label": "P264", "notable work": "P800",
    "nominated for": "P1411", "field of work": "P101", "work location": "P937",
    "military branch": "P241", "military rank": "P410", "head of government": "P6",
    "headquarters location": "P159", "country of citizenship": "P27",
    "country of origin": "P495", "original network": "P449", "original language of film or TV show": "P364",
    "language of work or name": "P407", "main subject": "P921", "after a work by": "P1877",
    "applies to jurisdiction": "P1001", "competition class": "P2094", "constellation": "P59",
    "contains administrative territorial entity": "P150", "heritage designation": "P1435",
    "licensed to broadcast to": "P1408", "located in or next to body of water": "P206",
    "located in the administrative territorial entity": "P131", "located on terrain feature": "P706",
    "location of formation": "P740", "member of political party": "P102", "mountain range": "P4552",
    "operating system": "P306", "participating team": "P1923", "place served by transport hub": "P931",
    "position played on team / speciality": "P413", "said to be the same as": "P460",
    "sports season of league or competition": "P3450", "subsidiary": "P355",
    "successful candidate": "P991", "taxon rank": "P105",
}


def search_property(label):
    q = urllib.parse.urlencode({"action": "wbsearchentities", "search": label, "language": "en",
                                "type": "property", "format": "json", "limit": 5})
    req = urllib.request.Request("https://www.wikidata.org/w/api.php?" + q, headers={"User-Agent": B.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    for hit in d.get("search", []):
        if hit.get("label", "").lower() == label.lower():
            return hit["id"]
    return d["search"][0]["id"] if d.get("search") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel2id", required=True)
    ap.add_argument("--base", required=True, help="existing NYT10 + GDS ontology json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rels = sorted(json.load(open(a.rel2id)))
    onto, mapping = {}, {}
    for r in rels:
        pid = PINNED.get(r)
        if pid is None:
            try:
                pid = search_property(r)
            except Exception as e:
                sys.stderr.write("search failed %s: %r\n" % (r, e)); pid = None
            time.sleep(0.3)
        mapping[r] = pid
        if pid is None:
            onto[r] = {"domain": B.COARSE, "range": B.COARSE, "source": "unmapped-permissive"}
            print("%-50s UNMAPPED" % r, flush=True); continue
        dq, rq = B.constraint_classes(pid)
        d, g = B.fold(dq), B.fold(rq)
        src = "wikidata:%s" % pid
        if not d:
            d = B.COARSE; src += "+permissive-domain"
        if not g:
            g = B.COARSE; src += "+permissive-range"
        onto[r] = {"domain": d, "range": g, "source": src,
                   "wd_domain_classes": dq, "wd_range_classes": rq}
        print("%-50s %-6s D=%s R=%s" % (r, pid, d, g), flush=True)
    json.dump(mapping, open(os.path.join(a.out, "wiki80_rel2pid.json"), "w"), indent=2)
    json.dump(onto, open(os.path.join(a.out, "wiki80_ontology.json"), "w"), indent=2, ensure_ascii=False)
    base = json.load(open(a.base, encoding="utf-8"))
    merged = dict(base); merged.update({k: v for k, v in onto.items() if k not in merged})
    merged.setdefault("NA", {"domain": B.COARSE, "range": B.COARSE, "source": "NA-wildcard"})
    json.dump(merged, open(os.path.join(a.out, "relation_ontology_all.json"), "w"), indent=2, ensure_ascii=False)
    n_perm = sum(1 for v in onto.values() if "permissive" in v["source"])
    print("\nwrote %d wiki80 relations (%d with a permissive side); merged=%d" % (len(onto), n_perm, len(merged)))


if __name__ == "__main__":
    main()
