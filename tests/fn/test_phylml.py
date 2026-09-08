"""Tests for phylml (Felsenstein 1981 pruning likelihood, F81).

Replaces the generated stub, which imported a name the module never had.
"""

import math

from morie.fn.phylml import (optimise_branch, phylml, site_likelihood,
                             substitution_matrix)


def test_the_f81_matrix_is_the_printed_formula():
    pi = [0.1, 0.2, 0.3, 0.4]
    t, u = 0.3, 1.0
    P = substitution_matrix(t, pi, u)
    e = math.exp(-u * t)
    for i in range(4):
        for j in range(4):
            want = (e if i == j else 0.0) + (1.0 - e) * pi[j]
            assert abs(P[i][j] - want) < 1e-12
        assert abs(sum(P[i]) - 1.0) < 1e-12


def test_zero_branch_length_is_the_identity():
    P = substitution_matrix(0.0)
    for i in range(4):
        for j in range(4):
            assert abs(P[i][j] - (1.0 if i == j else 0.0)) < 1e-12


def test_an_infinite_branch_forgets_the_start():
    pi = [0.1, 0.2, 0.3, 0.4]
    P = substitution_matrix(50.0, pi)
    for i in range(4):
        for j in range(4):
            assert abs(P[i][j] - pi[j]) < 1e-9


def test_the_documented_two_taxon_example_runs():
    res = phylml(("a", 0.1, "b", 0.1), {"a": "ACGT", "b": "ACGT"})
    assert res["n_taxa"] == 2
    assert res["n_sites"] == 4
    assert res["log_likelihood"] < 0.0
    assert len(res["site_log_likelihoods"]) == 4


def test_identical_sequences_are_likelier_than_different_ones():
    same = phylml(("a", 0.1, "b", 0.1),
                  {"a": "ACGT", "b": "ACGT"})["log_likelihood"]
    diff = phylml(("a", 0.1, "b", 0.1),
                  {"a": "ACGT", "b": "TGCA"})["log_likelihood"]
    assert same > diff


def test_the_total_is_the_sum_over_sites():
    res = phylml(("a", 0.2, "b", 0.3), {"a": "ACGT", "b": "ACGA"})
    assert abs(res["log_likelihood"] -
               sum(res["site_log_likelihoods"])) < 1e-9
    for i, lk in enumerate(res["site_likelihoods"]):
        assert abs(math.log(lk) - res["site_log_likelihoods"][i]) < 1e-9


def test_a_single_site_matches_the_whole_alignment_of_that_site():
    seqs = {"a": "AAAA", "b": "AAAA"}
    tree = ("a", 0.1, "b", 0.1)
    one = site_likelihood(tree, seqs, 0)
    whole = phylml(tree, seqs)
    assert abs(math.log(one) - whole["site_log_likelihoods"][0]) < 1e-12


def test_optimising_a_branch_finds_a_short_one_for_identical_sequences():
    seqs = {"a": "ACGTACGTAC", "b": "ACGTACGTAC"}
    res = optimise_branch(lambda t: ("a", t, "b", t), seqs)
    assert res["length"] < 0.05                 # nearly no divergence
    assert res["log_likelihood"] > phylml(("a", 1.0, "b", 1.0),
                                          seqs)["log_likelihood"]


def test_optimising_finds_a_longer_branch_for_divergent_sequences():
    near = optimise_branch(lambda t: ("a", t, "b", t),
                           {"a": "ACGTACGTAC", "b": "ACGTACGTAG"})
    far = optimise_branch(lambda t: ("a", t, "b", t),
                          {"a": "ACGTACGTAC", "b": "TGCATGCATG"})
    assert far["length"] > near["length"]


def test_validation():
    seqs = {"a": "ACGT", "b": "ACGT"}
    for call in (lambda: phylml(("a", 0.1, "b", 0.1), seqs, pi=[0.5, 0.5]),
                 lambda: phylml(("a", 0.1, "b", 0.1), seqs,
                                pi=[-0.1, 0.4, 0.4, 0.3]),
                 lambda: phylml(("a", 0.1, "b", 0.1), seqs,
                                pi=[0.5, 0.4, 0.4, 0.3]),
                 lambda: phylml(("a", -0.1, "b", 0.1), seqs)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
