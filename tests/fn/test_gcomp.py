"""Tests for morie.fn.gcomp — G-computation time-varying."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gcomp import gcomp


@pytest.fixture()
def data():
    rng = np.random.default_rng(6)
    n = 100
    K = 2
    T_seq = rng.binomial(1, 0.5, (n, K)).astype(float)
    L_seq = rng.standard_normal((n, K, 1))
    Y = 1.5 * T_seq[:, -1] + 0.3 * L_seq[:, -1, 0] + rng.standard_normal(n) * 0.5
    return Y, T_seq, L_seq


def test_keys(data):
    r = gcomp(*data, n_boot=50)
    for k in ("ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_ate_finite(data):
    r = gcomp(*data, n_boot=50)
    assert np.isfinite(r["ate"])


def test_se_positive(data):
    r = gcomp(*data, n_boot=50)
    assert r["se"] >= 0


def test_ci_valid(data):
    r = gcomp(*data, n_boot=50)
    assert r["ci_lower"] <= r["ci_upper"]


def test_mean_treated_returned(data):
    r = gcomp(*data, n_boot=50)
    assert r["mean_treated"] is not None or r["mean_control"] is not None


def test_single_timepoint():
    rng = np.random.default_rng(11)
    n = 80
    T_seq = rng.binomial(1, 0.5, (n, 1)).astype(float)
    L_seq = rng.standard_normal((n, 1, 1))
    Y = 2.0 * T_seq[:, 0] + rng.standard_normal(n)
    r = gcomp(Y, T_seq, L_seq, n_boot=50)
    assert np.isfinite(r["ate"])


def test_cheatsheet():
    from morie.fn.gcomp import cheatsheet

    assert len(cheatsheet()) > 0


def test_feedback_recovers_the_g_formula_effect():
    """A1 -> L2 -> Y feedback: E[Y(1,1)] - E[Y(0,0)] = 1 + 1 + 0.8 = 2.8.
    Regressing Y on the whole history and swapping treatments gives about
    2.0 (it holds L2 fixed); the backward regressions recover 2.8. With
    n = 2000 the standard error of the estimate is about 0.09."""
    import math
    import random

    rnd = random.Random(7)
    Y, T, L = [], [], []
    for _ in range(2000):
        l1 = rnd.gauss(0, 1)
        a1 = 1.0 if rnd.random() < 1 / (1 + math.exp(-l1)) else 0.0
        l2 = 0.8 * a1 + l1 + rnd.gauss(0, 1)
        a2 = 1.0 if rnd.random() < 1 / (1 + math.exp(-l2)) else 0.0
        Y.append(a1 + a2 + l2 + rnd.gauss(0, 1))
        T.append([a1, a2])
        L.append([[l1], [l2]])
    r = gcomp(Y, T, L, n_boot=10)
    assert abs(r["ate"] - 2.8) < 0.35
    assert abs(r["ate"] - 2.0) > 0.45
