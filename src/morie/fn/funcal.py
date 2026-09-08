# morie.fn -- function file (rootcoder007/morie)
r"""eggNOG-mapper v2: functional annotation through orthology.

Cantalapiedra, C. P., Hernandez-Plaza, A., Letunic, I., Bork, P., &
Huerta-Cepas, J. (2021) "eggNOG-mapper v2: Functional Annotation,
Orthology Assignments, and Domain Prediction at the Metagenomic Scale",
*Molecular Biology and Evolution* 38(12), 5825-5829.
doi:10.1093/molbev/msab293

The premise is the first sentence of the paper: infer function "via
orthology, rather than by homology" -- a best BLAST hit is a homologue,
which may be a paralogue that has drifted in function, whereas an
orthologue is the gene that speciation separated and is far more likely
to have kept it. So the pipeline never transfers a term from a hit
directly. It runs in four stages (Figure 1):

1. **Seed orthologs.** Search each query against the reference and keep
   the best hit that clears the e-value, bit-score and coverage cut-offs.
   The search tool (DIAMOND, MMseqs2, HMMER) changes speed and
   sensitivity, not the logic downstream, so it is a label here.
2. **Orthology assignment.** The seed hit places the query in a
   precomputed orthologous group, and the members of that group are the
   query's orthologues. Each relationship is typed by how many genes sit
   on each side: ``one2one``, ``one2many``, ``many2one``, ``many2many``.
3. **Taxonomic scope.** Terms may only be transferred from orthologues
   inside the requested lineage, "preventing transferring functional
   terms from orthologs of unwanted lineages".
4. **Transfer.** Terms held by the surviving orthologues become the
   query's annotation, per source: protein name, KEGG pathways and
   modules, GO labels, EC numbers, BiGG reactions, CAZy, COG category,
   the OG itself, and free-text description.

**What is and is not here.** The eggNOG v5 database (5,090 organisms,
4.4 M OGs) is not vendored, and no sequence search is run: the alignment
tools are external programs. Everything that decides *what gets
annotated and with what* is here and is exact -- the hit filters, the
orthology typing, the scope restriction, and the support rule for
accepting a term. Feed it hits and an orthologous-group table (from a
real eggNOG download, or from anything else with the same shape).

``min_support`` has no counterpart in the paper's defaults, where a term
held by any in-scope orthologue is transferred; it is exposed because a
single divergent orthologue is exactly how a wrong term propagates, and
``min_support=1`` reproduces the paper's behaviour.
"""

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "funcal",
    "functional_annotation",
    "seed_orthologs",
    "assign_orthologs",
    "transfer_terms",
    "ORTHOLOGY_TYPES",
    "ANNOTATION_SOURCES",
]

#: The four ways an orthology relationship can be shaped.
ORTHOLOGY_TYPES = ("one2one", "one2many", "many2one", "many2many")

#: The annotation sources the paper lists.
ANNOTATION_SOURCES = ("name", "kegg_pathway", "kegg_module", "go", "ec",
                      "bigg", "cazy", "cog_category", "og", "description")

_SEARCHERS = ("diamond", "mmseqs", "hmmer")


def _hit(h):
    try:
        q, t = h["query"], h["target"]
    except (KeyError, TypeError):
        raise ValueError("funcal: a hit needs 'query' and 'target'")
    out = {"query": q, "target": t,
           "evalue": float(h.get("evalue", 0.0)),
           "score": float(h.get("score", 0.0)),
           "query_cov": float(h.get("query_cov", 1.0)),
           "target_cov": float(h.get("target_cov", 1.0))}
    if out["evalue"] < 0:
        raise ValueError("funcal: an e-value cannot be negative")
    for k in ("query_cov", "target_cov"):
        if not 0.0 <= out[k] <= 1.0:
            raise ValueError("funcal: %s must be a fraction in [0, 1]" % k)
    return out


def seed_orthologs(hits, evalue=1e-3, score=60.0, query_cov=0.2,
                   target_cov=0.2, searcher="diamond"):
    """Stage 1: the best surviving hit per query.

    A hit must clear every cut-off; among those, the one with the lowest
    e-value wins, ties broken by the higher bit-score.
    """
    if searcher not in _SEARCHERS:
        raise ValueError("funcal: searcher must be one of %s"
                         % (_SEARCHERS,))
    if evalue <= 0 or score < 0:
        raise ValueError("funcal: evalue must be positive and score "
                         "non-negative")
    for c in (query_cov, target_cov):
        if not 0.0 <= c <= 1.0:
            raise ValueError("funcal: coverage cut-offs are fractions")
    kept = {}
    for raw in hits:
        h = _hit(raw)
        if h["evalue"] > evalue or h["score"] < score:
            continue
        if h["query_cov"] < query_cov or h["target_cov"] < target_cov:
            continue
        best = kept.get(h["query"])
        if best is None or (h["evalue"], -h["score"]) < \
                (best["evalue"], -best["score"]):
            kept[h["query"]] = h
    return dict((q, dict(h, searcher=searcher)) for q, h in kept.items())


def _type_of(n_query_side, n_target_side):
    left = "one" if n_query_side <= 1 else "many"
    right = "one" if n_target_side <= 1 else "many"
    return "%s2%s" % (left, right)


def assign_orthologs(seeds, groups, taxa=None, target_taxa=None,
                     target_types=None):
    """Stages 2 and 3: group members, typed, then restricted by lineage.

    ``groups`` maps a seed target to ``{"og": name, "members": [...]}``.
    ``taxa`` maps a member to its lineage as a list from root to tip, so
    ``target_taxa`` can name any level of it.
    """
    if target_types is not None:
        bad = [t for t in target_types if t not in ORTHOLOGY_TYPES]
        if bad:
            raise ValueError("funcal: unknown orthology type %r" % bad[0])
    taxa = taxa or {}
    out = {}
    for q, seed in seeds.items():
        g = groups.get(seed["target"])
        if not g:
            out[q] = {"og": None, "orthologs": [], "seed": seed["target"],
                      "dropped_by_scope": 0}
            continue
        members = [m for m in g.get("members", []) if m != seed["target"]]
        # how many query-side genes share this group: the co-orthologues
        same_group = [p for p, s in seeds.items()
                      if groups.get(s["target"], {}).get("og") ==
                      g.get("og")]
        rels, dropped = [], 0
        for m in members:
            lineage = taxa.get(m, [])
            if target_taxa is not None and not any(t in lineage
                                                   for t in target_taxa):
                dropped += 1
                continue
            n_target = sum(1 for x in members if taxa.get(x, [None])[:1] ==
                           lineage[:1]) if lineage else 1
            kind = _type_of(len(same_group), n_target)
            if target_types is not None and kind not in target_types:
                continue
            rels.append({"ortholog": m, "type": kind, "lineage": lineage})
        out[q] = {"og": g.get("og"), "orthologs": rels,
                  "seed": seed["target"], "dropped_by_scope": dropped}
    return out


def transfer_terms(assignments, annotations, sources=None, min_support=1):
    """Stage 4: terms held by the surviving orthologues become the query's.

    ``annotations`` maps a gene to ``{source: [terms]}``. A term is kept
    once ``min_support`` orthologues carry it.
    """
    if min_support < 1:
        raise ValueError("funcal: min_support must be at least 1")
    srcs = tuple(sources) if sources is not None else ANNOTATION_SOURCES
    bad = [s for s in srcs if s not in ANNOTATION_SOURCES]
    if bad:
        raise ValueError("funcal: unknown annotation source %r" % bad[0])
    out = {}
    for q, a in assignments.items():
        counts = {}
        for rel in a["orthologs"]:
            ann = annotations.get(rel["ortholog"], {})
            for s in srcs:
                for term in ann.get(s, []):
                    counts.setdefault(s, {})
                    counts[s][term] = counts[s].get(term, 0) + 1
        kept = {}
        for s in srcs:
            kept[s] = sorted(t for t, c in counts.get(s, {}).items()
                             if c >= min_support)
        out[q] = {"og": a["og"], "seed": a["seed"],
                  "n_orthologs": len(a["orthologs"]),
                  "terms": kept, "support": counts}
    return out


def funcal(hits, groups, annotations, taxa=None, target_taxa=None,
           target_types=None, sources=None, evalue=1e-3, score=60.0,
           query_cov=0.2, target_cov=0.2, min_support=1,
           searcher="diamond"):
    """Annotate queries by orthology (Cantalapiedra et al. 2021)."""
    seeds = seed_orthologs(hits, evalue, score, query_cov, target_cov,
                           searcher)
    assigned = assign_orthologs(seeds, groups, taxa, target_taxa,
                                target_types)
    annotated = transfer_terms(assigned, annotations, sources,
                               min_support)
    queries = sorted(set(_hit(h)["query"] for h in hits))
    n_ann = sum(1 for q in annotated
                if any(annotated[q]["terms"][s] for s in
                       annotated[q]["terms"]))
    return RichResult(payload={
        "estimate": annotated,
        "annotations": annotated,
        "seeds": seeds,
        "orthologs": assigned,
        "n_queries": len(queries),
        "n_with_seed": len(seeds),
        "n_annotated": n_ann,
        "searcher": searcher,
        "target_taxa": list(target_taxa) if target_taxa else None,
        "target_types": list(target_types) if target_types else None,
        "min_support": int(min_support),
        "method": ("eggNOG-mapper v2 (Cantalapiedra et al. 2021): seed "
                   "orthologs, orthologous-group assignment, taxonomic "
                   "scoping, then functional transfer from orthologs"),
        "note": ("no sequence search and no eggNOG v5 database are "
                 "bundled -- hits and the orthologous-group table are "
                 "supplied; everything that decides what is annotated "
                 "and with what is computed here. min_support=1 is the "
                 "paper's behaviour"),
    })


functional_annotation = funcal


def cheatsheet():
    return ("funcal: eggNOG-mapper v2 (Cantalapiedra et al. 2021). "
            "Function is transferred from ORTHOLOGS, not from the best "
            "hit: filter hits to a seed ortholog per query, take the "
            "members of its orthologous group, type each relationship "
            "one2one/one2many/many2one/many2many, drop orthologs outside "
            "the requested taxonomic scope, then transfer terms (name, "
            "KEGG, GO, EC, BiGG, CAZy, COG category, OG, description) "
            "held by the survivors.")
