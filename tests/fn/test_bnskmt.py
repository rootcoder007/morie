"""Tests for bnskmt: the Kolmogorov-Smirnov form of the Andrews-Shi
(2013) test -- the supremum of S over the instrument class rather than
its average."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.bndsmw import S_function, hypercube_instruments, weighted_moments
from morie.fn.bnskmt import (bound_kernel_moment, compare_forms,
                             ks_confidence_set, ks_statistic)

X = [[0.05], [0.2], [0.3], [0.45], [0.55], [0.7], [0.8], [0.95]]
M = [[0.5], [-1.0], [0.2], [-0.3], [0.9], [-0.8], [0.1], [-0.4]]


def _S_per_instrument(m, inst):
    out = []
    for g in inst:
        wm = weighted_moments(m, g)
        out.append(S_function([math.sqrt(len(m)) * wm["mean"][0] / wm["sd"][0]]))
    return out


def test_bnskmt_basic():
    """KS is the largest S over the class, and says where it is."""
    inst = hypercube_instruments(X, n_levels=3)["instruments"]
    s = _S_per_instrument(M, inst)
    got = ks_statistic(M, inst)
    assert got["statistic"] == pytest.approx(max(s), rel=1e-14)
    assert got["argmax"] == s.index(max(s))


def test_the_supremum_is_never_below_the_average():
    inst = hypercube_instruments(X, n_levels=3)["instruments"]
    cmp = compare_forms(M, inst)
    s = _S_per_instrument(M, inst)
    assert cmp["ks"] == pytest.approx(max(s), rel=1e-14)
    assert cmp["cvm"] == pytest.approx(sum(s) / len(s), rel=1e-14)
    assert cmp["ks"] >= cmp["cvm"] and cmp["ratio_ks_over_cvm"] >= 1.0


def test_ks_confidence_set_recovers_theta_below_the_mean():
    rng = np.random.default_rng(5)
    Xs = [[float(v)] for v in rng.uniform(0.0, 1.0, size=200)]
    W = [2.0 + float(v) for v in rng.normal(0.0, 1.0, size=200)]
    res = ks_confidence_set(lambda th: [[w - th] for w in W],
                            [0.5, 1.5, 3.5], Xs, n_levels=2, reps=60)
    assert 0.5 in res["set"] and 1.5 in res["set"]
    assert 3.5 not in res["set"]
    assert bound_kernel_moment is ks_confidence_set


def test_bnskmt_edge():
    """Moments that are all non-negative violate nothing."""
    inst = hypercube_instruments(X, n_levels=2)["instruments"]
    ok = [[abs(v[0]) + 0.1] for v in M]
    assert ks_statistic(ok, inst)["statistic"] == 0.0
