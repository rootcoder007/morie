"""seqsim: conditional-distribution identities of SGS/SIS, transforms, summaries and Markov chains."""

import math

import pytest

from morie.fn._rng import random_normal, random_uniform
from morie.fn.krgsys import kriging_covariance
from morie.fn.seqsim import (
    back_transform,
    markov_chain_simulate,
    normal_score,
    sgs_cross_validation,
    sgs_simulate,
    simulation_summary,
    sis_simulate,
    transition_matrix,
)

M = {"model": "Exp", "psill": 1.0, "range": 2.0}


def test_normal_score_roundtrip():
    z = [5.0, 1.0, 3.0, 3.0, 9.0]
    ns = normal_score(z)
    assert sorted(ns.scores) == ns.table_y and ns.table_z == [1.0, 3.0, 3.0, 5.0, 9.0]
    assert back_transform(ns.scores, ns.table_z, ns.table_y) == pytest.approx(z)
    assert back_transform([-9.0, 9.0], ns.table_z, ns.table_y) == [1.0, 9.0]


def test_sgs_single_node_is_the_kriging_distribution():
    D, z = [(0.0, 0.0), (3.0, 0.0)], [1.0, -0.5]
    r = sgs_simulate(D, z, [(1.0, 0.0)], M, mean=0.2, seed=5)
    C = [[kriging_covariance(abs(a[0] - b[0]), M) for b in D] for a in D]
    c0 = [kriging_covariance(1.0, M), kriging_covariance(2.0, M)]
    det = C[0][0] * C[1][1] - C[0][1] ** 2
    w = [(C[1][1] * c0[0] - C[0][1] * c0[1]) / det, (C[0][0] * c0[1] - C[0][1] * c0[0]) / det]
    mu = 0.2 + w[0] * 0.8 + w[1] * -0.7
    var = 1.0 - w[0] * c0[0] - w[1] * c0[1]
    e = float(random_normal(1, seed=5, stream=1)[0])
    assert r.realizations[0][0] == pytest.approx(mu + math.sqrt(var) * e, abs=1e-12)
    ok = sgs_simulate(D, z, [(1.0, 0.0)], M, kriging="ordinary", seed=5)
    assert ok.realizations[0][0] != r.realizations[0][0]
    # nodes on data reproduce the data exactly
    g = sgs_simulate(D, z, [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)], M, nsim=3, seed=1)
    assert all(rr[0] == 1.0 and rr[2] == -0.5 for rr in g.realizations)
    assert all(sorted(p) == [0, 1, 2] for p in g.paths)


def test_sgs_moments_and_collocated():
    grid = [(float(i), 0.0) for i in range(6)]
    r = sgs_simulate([], [], grid, M, nsim=400, seed=11)
    s = simulation_summary(r.realizations)
    assert all(abs(v) < 0.25 for v in s.etype) and all(abs(v - 1) < 0.25 for v in s.variance)
    # collocated cokriging with rho = 1 and a known secondary reproduces the secondary exactly
    sec = [0.3, -1.0, 0.5, 2.0, 0.0, 1.1]
    c = sgs_simulate([], [], grid, M, secondary=sec, rho=1.0, seed=2)
    assert c.realizations[0] == pytest.approx(sec, abs=1e-9)


def test_sis_and_summaries():
    m = {"model": "Sph", "psill": 0.25, "range": 3.0}
    r = sis_simulate(
        [(0.0, 0.0), (4.0, 0.0)], [0, 1], [(0.0, 0.0), (1.0, 0.0), (4.0, 0.0)], [0, 1], m, categorical=True, nsim=5
    )
    assert all(rr[0] == 0 and rr[2] == 1 for rr in r.realizations)
    cont = sis_simulate(
        [(0, 0), (2, 0), (5, 0)], [1.0, 3.0, 7.0], [(1.0, 0.0), (3.5, 0.0)], [2.0, 5.0], [m, m], nsim=20, seed=3
    )
    for probs in cont.probabilities:
        for p in probs:
            assert all(0 <= a <= b <= 1 for a, b in zip(p, p[1:]))
    s = simulation_summary(
        [[1.0, 4.0, 0.0], [3.0, 2.0, 1.0], [2.0, 9.0, 2.0]],
        probs=(0.5,),
        threshold=2.5,
        data=[1.0, 2.0, 3.0],
        blocks=["a", "a", "b"],
    )
    assert s.etype == [2.0, 5.0, 1.0] and s.variance == pytest.approx([1.0, 13.0, 1.0])
    assert s.percentiles["0.5"] == [2.0, 4.0, 1.0] and s.exceedance == [1 / 3, 2 / 3, 0.0]
    assert s.block_averages == [[2.5, 0.0], [2.5, 1.0], [5.5, 2.0]]
    assert s.ks_distance[0] == pytest.approx(1 / 3)


def test_validation_transitions_markov():
    cv = sgs_cross_validation([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], [0.1, 0.4, 0.2, -0.3, 0.0], M, nsim=50)
    assert len(cv.etype_error) == 5 and cv.rmse >= 0 and 0 <= cv.coverage <= 1
    t = transition_matrix([["a", "a", "b", "b", "b", "a"], ["b", "a"]])
    assert t.counts == [[1.0, 1.0], [2.0, 2.0]] and t.matrix == [[0.5, 0.5], [0.5, 0.5]]
    assert t.proportions == pytest.approx([4 / 8, 4 / 8]) and t.mean_run_length == [2.0, 2.0]
    P = [[0.7, 0.3], [0.4, 0.6]]
    seq = markov_chain_simulate(P, 0, 12, seed=4)
    u = random_uniform(12, seed=4, stream=0)
    want = [0]
    for k in range(1, 12):
        row = P[want[-1]]
        want.append(0 if float(u[k]) < row[0] else 1)
    assert seq == want
    assert markov_chain_simulate(P, 0, 5, conditioning={3: 1})[3] == 1
