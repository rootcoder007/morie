"""Tests for geosim2: geostatistical simulation on the Philox stream."""

import math

from morie.fn._rng import random_normal
from morie.fn.geosim2 import (
    _perm,
    collocated_cosimulate,
    conditional_ensemble,
    lmc_conditional_simulate,
    pfield_simulate,
    sgs_block_simulate,
)

M = {"model": "Exp", "sill": 1.0, "range": 2.0}
DC = [(0.5, 0.5), (4.0, 1.0), (2.0, 3.5)]
DV = [0.8, -0.4, 1.1]
TG = [(i * 1.0, j * 1.0) for j in range(3) for i in range(4)] + [(0.5, 0.5)]


def test_sgs_first_node_is_simple_kriging_draw():
    r = sgs_block_simulate(DC, DV, TG, M, mean=0.1, k=10, seed=3)
    assert r.simulated[-1] == 0.8
    t = _perm(len(TG), 3, 0)[0]
    x = TG[t]
    C = [[math.exp(-math.dist(a, b) / 2) for b in DC] for a in DC]
    c0 = [math.exp(-math.dist(a, x) / 2) for a in DC]

    # solve the 3 x 3 system by Cramer's rule
    def det(A):
        return (
            A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
        )

    d = det(C)
    lam = []
    for q in range(3):
        A = [row[:] for row in C]
        for a in range(3):
            A[a][q] = c0[a]
        lam.append(det(A) / d)
    mu = 0.1 + sum(la * (v - 0.1) for la, v in zip(lam, DV))
    var = 1.0 - sum(la * c for la, c in zip(lam, c0))
    z0 = float(random_normal(len(TG), seed=3, stream=1)[0])
    assert abs(r.simulated[t] - (mu + math.sqrt(var) * z0)) <= 1e-10


def test_ensemble_is_exact_at_data_and_pfield_two_nodes():
    e = conditional_ensemble(DC, DV, [(0.5, 0.5), (1.0, 1.0)], M, n_real=5)
    assert e.etype[0] == 0.8 and e.variance[0] == 0.0
    z = [float(v) for v in random_normal(2, seed=5, stream=0)]
    rho = math.exp(-1 / 2)
    y = [z[0], rho * z[0] + math.sqrt(1 - rho * rho) * z[1]]
    got = pfield_simulate([1.0, 2.0], [0.5, 0.3], [(0.0, 0.0), (1.0, 0.0)], M, seed=5)
    assert abs(got[0] - (1 + 0.5 * y[0])) <= 1e-12 and abs(got[1] - (2 + 0.3 * y[1])) <= 1e-12


def test_lmc_and_collocated_reproduce_data():
    comps = [([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]], {"model": "Exp", "sill": 1.0, "range": 2.0})]
    r = lmc_conditional_simulate([(0.0, 0.0)], [[1.0, None, 0.4]], [(0.0, 0.0), (1.0, 0.0)], comps)
    assert r.simulated[0][0] == 1.0 and r.simulated[0][2] == 0.4
    c = collocated_cosimulate(DC, DV, TG, [0.1] * len(TG), M, 0.5)
    assert c.simulated[-1] == 0.8
