"""Tests for bndsmw: inference on conditional moment inequalities by
the Cramer-von Mises statistic and GMS critical values (Andrews and
Shi 2013). Pieces are recomputed by hand; the inference is checked on
a moment whose identified set is known: E[W - theta] >= 0 gives
theta <= E[W].
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.bndsmw import (S_function, bound_simul_weights,
                             confidence_set, cvm_statistic,
                             hypercube_instruments, weighted_moments)

X = [[0.05], [0.2], [0.3], [0.45], [0.55], [0.7], [0.8], [0.95]]


def test_bndsmw_basic():
    """Nested hypercubes: 1 + 2 + 4 cells, each level a partition."""
    inst = hypercube_instruments(X, n_levels=3)
    assert inst["n_instruments"] == 7
    G = inst["instruments"]
    assert G[0] == [1.0] * 8                      # level 0: everything
    for level in (G[1:3], G[3:7]):
        for i in range(8):
            assert sum(g[i] for g in level) == 1.0   # one cell per point
    assert G[1] == [1.0] * 4 + [0.0] * 4 and G[2] == [0.0] * 4 + [1.0] * 4


def test_weighted_moments_and_S_by_hand():
    m = [[1.0, -2.0], [3.0, 0.5], [-1.0, 1.0], [2.0, 2.5]]
    g = [1.0, 0.0, 1.0, 1.0]
    wm = weighted_moments(m, g)
    v0 = [1.0, 0.0, -1.0, 2.0]
    mu0 = sum(v0) / 4
    assert wm["mean"][0] == pytest.approx(mu0, rel=1e-15)
    assert wm["sd"][0] == pytest.approx(
        math.sqrt(sum((x - mu0) ** 2 for x in v0) / 3), rel=1e-14)
    # S penalises only negative inequality moments; equalities both ways
    assert S_function([1.5, -2.0, 0.0]) == pytest.approx(4.0, rel=1e-15)
    assert S_function([1.5, -2.0, -0.5], form="max") == pytest.approx(4.0, rel=1e-15)
    assert S_function([3.0, 5.0]) == 0.0
    assert S_function([3.0, 2.0], n_equality=1) == pytest.approx(4.0, rel=1e-15)
    assert S_function([3.0, -2.0], n_equality=1) == pytest.approx(4.0, rel=1e-15)


def test_cvm_is_the_q_average_of_S():
    m = [[0.5], [-1.0], [0.2], [-0.3], [0.9], [-0.8], [0.1], [-0.4]]
    inst = hypercube_instruments(X, n_levels=2)["instruments"]
    want = 0.0
    for g in inst:
        wm = weighted_moments(m, g)
        z = math.sqrt(8) * wm["mean"][0] / wm["sd"][0]
        want += S_function([z]) / len(inst)
    assert cvm_statistic(m, inst)["statistic"] == pytest.approx(want, rel=1e-14)
    q = [0.5, 0.25, 0.25]
    got = cvm_statistic(m, inst, weights=q)
    assert got["statistic"] == pytest.approx(
        sum(qi * s for qi, s in zip(q, got["per_instrument"])), rel=1e-14)


def test_confidence_set_recovers_theta_below_the_mean():
    rng = np.random.default_rng(5)
    Xs = [[float(v)] for v in rng.uniform(0.0, 1.0, size=200)]
    W = [2.0 + float(v) for v in rng.normal(0.0, 1.0, size=200)]
    mom = lambda th: [[w - th] for w in W]
    res = confidence_set(mom, [0.5, 1.5, 3.5], Xs, n_levels=2, reps=60)
    assert 0.5 in res["set"] and 1.5 in res["set"]
    assert 3.5 not in res["set"]
    assert bound_simul_weights is confidence_set


def test_bndsmw_edge():
    with pytest.raises(ValueError, match="no observations"):
        hypercube_instruments([])
    with pytest.raises(ValueError, match="non-negative"):
        weighted_moments([[1.0], [2.0]], [1.0, -1.0])
    with pytest.raises(ValueError, match="form must be"):
        S_function([1.0], form="median")
    with pytest.raises(ValueError, match="sum to 1"):
        cvm_statistic([[1.0], [2.0]], [[1.0, 1.0]], weights=[0.7])
