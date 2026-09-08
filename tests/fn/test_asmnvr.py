"""Tests for asmnvr (Pevzner, Tang & Waterman 2001, de Bruijn assembly).

The file this replaces was the generated stub: it imported a function
called ``genome_assembly`` that does not exist, fed 100 Gaussian random
numbers in as "reads", and asserted only that the result was a dict. It
had never been updated when the module was actually implemented.
"""

from morie.fn.asmnvr import (asmnvr, de_bruijn_graph, de_novo_assembly,
                             eulerian_path)


def _tile(seq, read_len):
    return [seq[i:i + read_len] for i in range(len(seq) - read_len + 1)]


def test_reads_tiling_a_sequence_reassemble_it_exactly():
    seq = "ATGGCGTGCAAGCTTAC"
    res = asmnvr(_tile(seq, 6), k=5)
    assert res["sequence"] == seq
    assert res["contigs"] == [seq]
    assert res["unambiguous"] is True


def test_de_bruijn_vertices_are_k_minus_one_mers():
    seq = "ATGGCGTGCA"
    reads = _tile(seq, 5)
    edges, indeg, outdeg = de_bruijn_graph(reads, 4)
    verts = set(list(indeg) + list(outdeg))
    assert all(len(v) == 3 for v in verts)
    # every 4-mer of the sequence is an edge from its first 3 to its last 3
    for i in range(len(seq) - 3):
        mer = seq[i:i + 4]
        assert mer[1:] in edges.get(mer[:-1], {})


def test_edge_count_matches_the_kmers():
    seq = "ATGGCGTGCAAGCTTAC"
    res = asmnvr(_tile(seq, 6), k=5)
    # a sequence of length L has L - k + 1 distinct k-mers when it has no
    # repeated ones, and each is one edge under multiplicity="set"
    assert res["n_kmers"] == len(seq) - 5 + 1


def test_multiplicity_set_versus_count():
    seq = "ATGCATGC"
    reads = _tile(seq, 4)
    as_set = asmnvr(reads, k=3, multiplicity="set")
    as_count = asmnvr(reads, k=3, multiplicity="count")
    # "set" collapses the repeat: the assembly is shorter than the truth,
    # and the module says the length is only a lower bound
    assert as_set["sequence"] == "ATGCAT"
    assert len(as_set["sequence"]) < len(seq)
    assert as_set["length_is_lower_bound"] is True
    # "count" keeps every occurrence, so the repeat survives
    assert as_count["n_kmers"] > as_set["n_kmers"]
    assert as_count["length_is_lower_bound"] is False


def test_a_branching_graph_is_reported_not_resolved():
    res = asmnvr(["ATGC", "ATGA", "TGCC", "TGAA"], k=3)
    assert res["sequence"] is None          # no Eulerian path
    assert res["unambiguous"] is False
    assert res["branching"] == ["TG"]
    # the unitigs are still available
    assert sorted(res["contigs"]) == ["ATG", "TGAA", "TGCC"]


def test_eulerian_path_existence_condition():
    # a simple chain A -> B -> C has a path
    edges = {"A": {"B": 1}, "B": {"C": 1}}
    indeg = {"B": 1, "C": 1, "A": 0}
    outdeg = {"A": 1, "B": 1, "C": 0}
    assert eulerian_path(edges, indeg, outdeg) == ["A", "B", "C"]
    # two disconnected edges do not
    edges2 = {"A": {"B": 1}, "C": {"D": 1}}
    indeg2 = {"B": 1, "D": 1, "A": 0, "C": 0}
    outdeg2 = {"A": 1, "C": 1, "B": 0, "D": 0}
    assert eulerian_path(edges2, indeg2, outdeg2) is None


def test_a_cycle_has_an_eulerian_circuit():
    seq = "ATGCA"          # ATG -> TGC -> GCA, plus the wrap
    res = asmnvr(_tile(seq, 4), k=3)
    assert res["sequence"] == seq


def test_k_defaults_to_the_shortest_read():
    reads = ["ATGGC", "TGGCG", "GGCGT"]
    assert asmnvr(reads)["k"] == 5


def test_reads_of_unequal_length_are_allowed():
    res = asmnvr(["ATGGCG", "TGGCGT", "GGCGTG", "GCGTGCA"], k=5)
    assert res["sequence"] == "ATGGCGTGCA"


def test_validation():
    try:
        asmnvr([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_alias_and_reported_boundary():
    assert de_novo_assembly is asmnvr
    res = asmnvr(_tile("ATGGCGTGCA", 5), k=4)
    assert "NOT implemented" in res["note"]
    assert "Pevzner" in res["method"]
