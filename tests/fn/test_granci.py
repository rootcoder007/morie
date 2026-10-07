"""Tests for granci.granger_causality_info."""

import pytest

from morie.fn import _array_core as np
from morie.fn.granci import granger_causality_info


def test_granci_basic():
    rng = np.random.default_rng(42)
    n = 1500
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.normal(size=n)
    ey = rng.normal(size=n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + ey[t]
    out = granger_causality_info(x, y, lag=1)
    assert out["mi"] > 0.05  # measured ~0.11 nats
    assert out["p_value"] < 0.01
    assert granger_causality_info(y, x, lag=1)["mi"] < 0.01


def test_granci_edge():
    with pytest.raises(ValueError):
        granger_causality_info([1.0, 2.0], [1.0, 2.0], lag=1)  # too short
    with pytest.raises(ValueError):
        granger_causality_info([1.0] * 20, [1.0] * 25, lag=1)  # length mismatch


def test_granger_values_equal_rmories_and_an_exact_fit_is_refused():
    # rmorie's morie_granger_test / morie_transfer_entropy_gaussian on the same series
    # (tests/testthat/test-causal_native2.R)
    import math

    from morie.fn.ggrcst import granger_causality

    x = [math.sin(t) for t in range(1, 61)]
    y = [a + 0.3 * math.cos(3 * t) for a, t in zip([0.0] + [0.8 * v for v in x[:-1]], range(1, 61))]
    g = granger_causality(x, y, p=1)
    c = granger_causality_info(x, y, lag=1)
    assert g["statistic"] == pytest.approx(468.71891925114591, rel=1e-10)
    assert c["mi"] == pytest.approx(1.1187555182902371, rel=1e-10)
    assert c["p_value"] == pytest.approx(1.4863128325739663e-30, rel=1e-8)
    # a perfectly predictable response leaves only round-off in the residual sum
    with pytest.raises(ValueError, match="fits exactly"):
        granger_causality_info(list(range(1, 9)), list(range(1, 9)))
    with pytest.raises(ValueError, match="fits exactly"):
        granger_causality([1.0] * 20, [1.0] * 20)
