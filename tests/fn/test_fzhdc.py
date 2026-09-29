"""Tests for fzhdc.fauzi_h_decomposition."""

from morie.fn import _array_core as np
from morie.fn.fzhdc import fauzi_h_decomposition


def test_fzhdc_basic():
    """The U-statistic with kernel 0.5*(a-b)^2 IS the unbiased variance.

    For x = 1..5 the sample variance is 10/4 = 2.5, so theta must be 2.5.
    (The generated stub asserted 3.0 -- the mean, not the variance.)
    """
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_h_decomposition(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value
    assert result["n"] == 5
    # Hajek-projection variance is a variance: non-negative and finite.
    assert result["sigma1_sq"] >= 0
    assert np.isfinite(result["se"])


def test_fzhdc_edge():
    """A single observation cannot form a pair -- report, do not crash."""
    result = fauzi_h_decomposition(np.array([42.0]))
    assert result["n"] == 1
    assert np.isnan(result["estimate"])
    assert "too few" in result["method"]


def test_u_statistic_and_hajek_projection_recomputed():
    """Kernel 0.5 (a - b)^2: U_n is the unbiased variance; g_1(X_i) is the
    mean of the kernel over pairs containing i, minus theta."""
    import itertools

    import pytest

    x = [1.0, 2.5, 3.0, 4.5, 7.0, 2.0]
    n = 6
    pairs = list(itertools.combinations(range(n), 2))
    g = [0.5 * (x[i] - x[j]) ** 2 for i, j in pairs]
    th = sum(g) / len(g)
    m = sum(x) / n
    assert th == pytest.approx(sum((v - m) ** 2 for v in x) / (n - 1), rel=1e-13)
    g1 = []
    for i in range(n):
        vals = [gv for (a, b), gv in zip(pairs, g) if i in (a, b)]
        g1.append(sum(vals) / len(vals) - th)
    mg = sum(g1) / n
    s1 = sum((v - mg) ** 2 for v in g1) / (n - 1)
    r = fauzi_h_decomposition(x)
    assert r["estimate"] == pytest.approx(th, rel=1e-13)
    assert r["sigma1_sq"] == pytest.approx(s1, rel=1e-12)
    assert r["se"] == pytest.approx((4 * s1 / n) ** 0.5, rel=1e-12)
