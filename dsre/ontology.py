"""Ontology gate (onto_filter) and constrained predicate space (constrained_space).
Unknown types (OTHER/UNK/None) pass: an unknown type cannot create a violation. NA always passes (no domain/range to violate)."""
import json


class Ontology:
    def __init__(self, path, relations=None, has_na=True):
        self.spec = json.load(open(path, encoding="utf-8"))
        self.has_na = has_na
        self.relations = relations or [r for r in self.spec if r != "NA"]

    def consistent(self, head_type, tail_type, relation):
        if relation == "NA":
            return 1
        s = self.spec.get(relation)
        if s is None:
            return 1
        if head_type in (None, "UNK", "OTHER") or tail_type in (None, "UNK", "OTHER"):
            return 1
        return int(head_type in s["domain"] and tail_type in s["range"])

    def constrained_space(self, head_type, tail_type):
        cands = [r for r in self.relations if self.consistent(head_type, tail_type, r)]
        if self.has_na:
            cands = cands + ["NA"]
        return cands if cands else (["NA"] if self.has_na else list(self.relations))

    def stats(self, typed_pairs):
        """Leakage probe: mean number of non-NA candidates, share with exactly one non-NA candidate, share with none (only NA left)."""
        sizes = []
        for a, b in typed_pairs:
            sizes.append(sum(1 for r in self.relations if self.consistent(a, b, r)))
        n = max(len(sizes), 1)
        return {"n": len(sizes), "n_relations": len(self.relations),
                "mean_size": sum(sizes) / n,
                "frac_single_relation": sum(1 for s in sizes if s == 1) / n,
                "frac_none": sum(1 for s in sizes if s == 0) / n,
                "frac_unconstrained": sum(1 for s in sizes if s == len(self.relations)) / n}
