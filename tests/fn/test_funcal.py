"""Tests for funcal (Cantalapiedra et al. 2021, eggNOG-mapper v2)."""

from morie.fn.funcal import (ANNOTATION_SOURCES, ORTHOLOGY_TYPES,
                             assign_orthologs, funcal,
                             functional_annotation, seed_orthologs,
                             transfer_terms)

HITS = [
    {"query": "q1", "target": "tA", "evalue": 1e-40, "score": 300.0,
     "query_cov": 0.9, "target_cov": 0.9},
    {"query": "q1", "target": "tB", "evalue": 1e-10, "score": 120.0,
     "query_cov": 0.9, "target_cov": 0.9},
    {"query": "q2", "target": "tC", "evalue": 1e-2, "score": 300.0,
     "query_cov": 0.9, "target_cov": 0.9},
    {"query": "q3", "target": "tD", "evalue": 1e-40, "score": 10.0,
     "query_cov": 0.9, "target_cov": 0.9},
    {"query": "q4", "target": "tE", "evalue": 1e-40, "score": 300.0,
     "query_cov": 0.05, "target_cov": 0.9},
    {"query": "q5", "target": "tF", "evalue": 1e-40, "score": 300.0,
     "query_cov": 0.9, "target_cov": 0.05},
]
GROUPS = {"tA": {"og": "OG1", "members": ["tA", "m_bact", "m_arch",
                                          "m_euk1", "m_euk2"]}}
TAXA = {"m_bact": ["cellular", "Bacteria", "Firmicutes"],
        "m_arch": ["cellular", "Archaea", "Euryarchaeota"],
        "m_euk1": ["cellular", "Eukaryota", "Fungi"],
        "m_euk2": ["cellular", "Eukaryota", "Metazoa"]}
ANN = {
    "tA": {"go": ["GO:SEED_ONLY"], "name": ["seedname"]},
    "m_bact": {"go": ["GO:0006096", "GO:SHARED"], "cog_category": ["G"],
               "name": ["pfkA"]},
    "m_arch": {"go": ["GO:SHARED"], "cog_category": ["G"]},
    "m_euk1": {"go": ["GO:SHARED"], "kegg_pathway": ["ko00010"]},
    "m_euk2": {"go": ["GO:SHARED"]},
}


def test_each_cutoff_rejects_its_own_case():
    assert sorted(seed_orthologs(HITS)) == ["q1"]
    assert sorted(seed_orthologs(HITS, evalue=1.0)) == ["q1", "q2"]
    assert sorted(seed_orthologs(HITS, score=5.0)) == ["q1", "q3"]
    assert sorted(seed_orthologs(HITS, query_cov=0.0)) == ["q1", "q4"]
    assert sorted(seed_orthologs(HITS, target_cov=0.0)) == ["q1", "q5"]


def test_the_best_hit_wins_and_ties_break_on_score():
    assert seed_orthologs(HITS)["q1"]["target"] == "tA"
    tie = seed_orthologs([
        {"query": "z", "target": "a", "evalue": 1e-20, "score": 100.0},
        {"query": "z", "target": "b", "evalue": 1e-20, "score": 200.0}])
    assert tie["z"]["target"] == "b"


def test_group_membership_and_typing():
    seeds = seed_orthologs(HITS)
    a = assign_orthologs(seeds, GROUPS, TAXA)
    assert a["q1"]["og"] == "OG1"
    assert sorted(r["ortholog"] for r in a["q1"]["orthologs"]) == \
        ["m_arch", "m_bact", "m_euk1", "m_euk2"]
    assert "tA" not in [r["ortholog"] for r in a["q1"]["orthologs"]]
    assert all(r["type"] in ORTHOLOGY_TYPES for r in a["q1"]["orthologs"])


def test_taxonomic_scope():
    seeds = seed_orthologs(HITS)
    b = assign_orthologs(seeds, GROUPS, TAXA, target_taxa=["Bacteria"])
    assert [r["ortholog"] for r in b["q1"]["orthologs"]] == ["m_bact"]
    assert b["q1"]["dropped_by_scope"] == 3
    d = assign_orthologs(seeds, GROUPS, TAXA, target_taxa=["Metazoa"])
    assert [r["ortholog"] for r in d["q1"]["orthologs"]] == ["m_euk2"]


def test_a_seed_without_a_group_is_not_an_error():
    a = assign_orthologs({"qX": {"target": "nope", "evalue": 0.0,
                                 "score": 0.0}}, GROUPS, TAXA)
    assert a["qX"]["og"] is None and a["qX"]["orthologs"] == []


def test_terms_come_from_orthologs_not_from_the_hit():
    res = funcal(HITS, GROUPS, ANN, TAXA)
    t = res["annotations"]["q1"]["terms"]
    assert "GO:SEED_ONLY" not in t["go"]
    assert "seedname" not in t["name"]
    assert "GO:SHARED" in t["go"] and "GO:0006096" in t["go"]
    assert "pfkA" in t["name"]


def test_support_counts_and_min_support():
    res = funcal(HITS, GROUPS, ANN, TAXA)
    sup = res["annotations"]["q1"]["support"]["go"]
    assert sup["GO:SHARED"] == 4 and sup["GO:0006096"] == 1
    two = funcal(HITS, GROUPS, ANN, TAXA, min_support=2)
    assert "GO:0006096" not in two["annotations"]["q1"]["terms"]["go"]
    assert "GO:SHARED" in two["annotations"]["q1"]["terms"]["go"]
    five = funcal(HITS, GROUPS, ANN, TAXA, min_support=5)
    assert five["annotations"]["q1"]["terms"]["go"] == []


def test_scope_and_sources_change_the_annotation():
    b = funcal(HITS, GROUPS, ANN, TAXA, target_taxa=["Bacteria"])
    assert b["annotations"]["q1"]["terms"]["go"] == ["GO:0006096",
                                                     "GO:SHARED"]
    assert b["annotations"]["q1"]["terms"]["kegg_pathway"] == []
    one = funcal(HITS, GROUPS, ANN, TAXA, sources=["cog_category"])
    assert sorted(one["annotations"]["q1"]["terms"]) == ["cog_category"]
    assert one["annotations"]["q1"]["terms"]["cog_category"] == ["G"]


def test_counts_and_sources():
    res = funcal(HITS, GROUPS, ANN, TAXA)
    assert res["n_queries"] == 5
    assert res["n_with_seed"] == 1
    assert res["n_annotated"] == 1
    assert set(ANNOTATION_SOURCES) >= {"name", "kegg_pathway", "go", "ec",
                                       "bigg", "cazy", "cog_category",
                                       "og", "description",
                                       "kegg_module"}


def test_validation():
    seeds = seed_orthologs(HITS)
    a = assign_orthologs(seeds, GROUPS, TAXA)
    for call in (lambda: seed_orthologs([{"query": "a"}]),
                 lambda: seed_orthologs(HITS, searcher="blast"),
                 lambda: seed_orthologs(HITS, evalue=0.0),
                 lambda: seed_orthologs(HITS, score=-1.0),
                 lambda: seed_orthologs(HITS, query_cov=1.5),
                 lambda: seed_orthologs([{"query": "a", "target": "b",
                                          "evalue": -1.0}]),
                 lambda: seed_orthologs([{"query": "a", "target": "b",
                                          "query_cov": 2.0}]),
                 lambda: assign_orthologs(seeds, GROUPS, TAXA,
                                          target_types=["one2three"]),
                 lambda: transfer_terms(a, ANN, min_support=0),
                 lambda: transfer_terms(a, ANN, sources=["smiles"]),
                 lambda: funcal(HITS, GROUPS, ANN, TAXA, min_support=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert functional_annotation is funcal
