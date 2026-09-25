"""Tests for dpgrf: areal boundary detection with a Dirichlet-process
prior (Li, Banerjee, Hanson and McBean 2015)."""

import math

import numpy as real
import pytest

from morie.fn.dpgrf import (adjacency_pairs, boundary_probabilities,
                            car_precision, coclustering,
                            continuous_prior_tie_probability,
                            dp_grouped_random_field, sample_labels)

# a 4-region path 0 - 1 - 2 - 3
PATH = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


def test_dpgrf_basic():
    """Only adjacent pairs can carry a boundary."""
    a = adjacency_pairs(PATH)
    assert a["pairs"] == [(0, 1), (1, 2), (2, 3)]
    assert a["degrees"] == [1.0, 2.0, 2.0, 1.0]
    assert a["n_regions"] == 4


def test_car_precision_is_tau_times_d_minus_rho_w():
    rho, tau = 0.9, 2.0
    out = car_precision(PATH, rho=rho, tau=tau)
    D = [1.0, 2.0, 2.0, 1.0]
    want = [[tau * ((D[i] if i == j else 0.0) - rho * PATH[i][j])
             for j in range(4)] for i in range(4)]
    assert out["Q"] == want
    ev = real.linalg.eigvalsh(real.asarray(want))
    assert out["min_eigenvalue"] == pytest.approx(ev.min(), abs=1e-10)
    assert not out["singular"] and ev.min() > 0
    # rho = 1: every row of D - W sums to zero, so Q is singular
    icar = car_precision(PATH, rho=1.0)
    assert icar["singular"]
    assert icar["min_eigenvalue"] == pytest.approx(0.0, abs=1e-9)


def test_polya_urn_cluster_count_matches_antoniak():
    # E[#clusters] for n draws from a DP(alpha) urn is
    # sum_{i=1}^{n} alpha / (alpha + i - 1)
    n, alpha, reps = 10, 1.5, 4000
    want = sum(alpha / (alpha + i - 1) for i in range(1, n + 1))
    counts = []
    for s in range(reps):
        draw = sample_labels(n, alpha, seed=s)
        assert len(draw["labels"]) == n
        assert draw["n_clusters"] == len(set(draw["labels"]))
        counts.append(draw["n_clusters"])
    mean = sum(counts) / reps
    var = sum((c - mean) ** 2 for c in counts) / (reps - 1)
    assert abs(mean - want) < 5.0 * math.sqrt(var / reps)


def test_coclustering_and_boundaries_by_hand():
    draws = [[0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1], [2, 2, 1, 1]]
    co = coclustering(draws)
    # P(z_1 = z_2): draws 1, 2, 4 agree -> 3/4
    assert co["matrix"][0][1] == 0.75
    assert co["matrix"][1][2] == 0.5 and co["matrix"][2][3] == 0.75
    assert co["symmetric"] and co["unit_diagonal"]
    b = boundary_probabilities(PATH, draws, threshold=0.4)
    p = {d["pair"]: d["p_difference"] for d in b["ranked"]}
    assert p == {(0, 1): 0.25, (1, 2): 0.5, (2, 3): 0.25}
    assert b["boundaries"] == [(1, 2)] and b["n_boundaries"] == 1
    assert dp_grouped_random_field is boundary_probabilities
    assert continuous_prior_tie_probability()["probability"] == 0.0


def test_dpgrf_edge():
    with pytest.raises(ValueError, match="not square"):
        adjacency_pairs([[0, 1, 0], [1, 0, 1]])
    with pytest.raises(ValueError, match="symmetric"):
        adjacency_pairs([[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="rho"):
        car_precision(PATH, rho=1.5)
    with pytest.raises(ValueError, match="no neighbours"):
        car_precision([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="no label draws"):
        coclustering([])
    with pytest.raises(ValueError, match="differ in length"):
        coclustering([[0, 1], [0]])
