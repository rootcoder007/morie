"""Tests for ksr062 (Kosorok shelf)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.ksr062 import kosorok_ch3_pathwise_derivative


def test_ksr062_basic():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(300)
    out = kosorok_ch3_pathwise_derivative(x - x.mean(), x)
    assert out["mean_zero"] is True


def test_ksr062_edge():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(300)
    assert kosorok_ch3_pathwise_derivative(x - x.mean() + 5.0, x)["mean_zero"] is False


def test_derivative_is_the_weighted_inner_product():
    psi = [0.5, -0.2, -0.4, 0.1]
    S = [[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0], [0.2, -0.3]]
    w = [0.1, 0.2, 0.3, 0.4]
    r = kosorok_ch3_pathwise_derivative(psi, S, weights=w)
    want = [sum(w[i] * psi[i] * S[i][j] for i in range(4)) for j in range(2)]
    assert [float(v) for v in r["derivative"]] == pytest.approx(want, rel=1e-13)
    mean = sum(a * b for a, b in zip(w, psi))
    assert r["influence_mean"] == pytest.approx(mean, rel=1e-13)
    assert r["influence_var"] == pytest.approx(sum(a * (b - mean) ** 2 for a, b in zip(w, psi)), rel=1e-13)
