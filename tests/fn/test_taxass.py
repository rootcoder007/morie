"""Anchored tests for taxass (Wood-Salzberg 2014 RTL classification).

Hand taxonomy: 1 = root; 2, 3 children of 1; 4, 5 children of 2.
All RTL path scores computed by hand in the assertions.
"""

from morie.fn.taxass import taxass, taxonomic_assignment

PARENT = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2}


def test_taxass_hand_rtl_scores():
    # hits: 4,4,5,2,3 -> weights 4:2, 5:1, 2:1, 3:1
    # leaves: 3, 4, 5. RTL(4) = w4 + w2 + w1 = 2 + 1 + 0 = 3,
    # RTL(5) = 1 + 1 + 0 = 2, RTL(3) = 1.
    res = taxass([4, 4, 5, 2, 3], PARENT)
    assert res["taxon"] == 4
    assert res["leaf_scores"] == {3: 1, 4: 3, 5: 2}
    assert res["weights"] == {4: 2, 5: 1, 2: 1, 3: 1}
    assert res["n_hit"] == 5


def test_taxass_tie_resolves_to_lca():
    # hits 4 and 5 tie at RTL = 1 each -> LCA(4, 5) = 2.
    res = taxass([4, 5], PARENT)
    assert res["taxon"] == 2
    # three-way tie across both sides of the root -> root
    res2 = taxass([4, 5, 3], PARENT)
    assert res2["taxon"] == 1


def test_taxass_unclassified_and_ignored_zeros():
    res = taxass([0, 0], PARENT)
    assert res["taxon"] == 0
    assert res["n_hit"] == 0
    # zeros ignored alongside real hits
    res2 = taxass([0, 4, 0], PARENT)
    assert res2["taxon"] == 4
    assert res2["n_kmers"] == 3
    assert res2["n_hit"] == 1


def test_taxass_interior_dominance():
    # heavy interior node: hits 2,2,2,4 -> leaves 4 only on that side
    # (2 is interior once 4 present): RTL(4) = 3 + 1 = 4, RTL leaf 5
    # absent, leaf 3 absent.
    res = taxass([2, 2, 2, 4], PARENT)
    assert res["taxon"] == 4
    assert res["leaf_scores"] == {4: 4}
    assert taxonomic_assignment is taxass
