"""Tests for evgevlm.evt_gev_lmoments."""

from morie.fn import _array_core as np
from morie.fn.evgevlm import evt_gev_lmoments


def test_evgevlm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_lmoments(x)
    assert isinstance(result, dict)
    assert "mu" in result


def test_evgevlm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_gev_lmoments(x)
    assert isinstance(result, dict)


def _hosking_gev(x):
    import math

    xs = sorted(x)
    n = len(xs)

    def b(r):
        tot = 0.0
        for j, v in enumerate(xs, start=1):
            w = 1.0
            for k in range(1, r + 1):
                w *= (j - k) / (n - k)
            tot += w * v
        return tot / n

    b0, b1, b2 = b(0), b(1), b(2)
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2
    c = 2 / (3 + t3) - math.log(2) / math.log(3)
    k = 7.8590 * c + 2.9554 * c * c
    a = l2 * k / ((1 - 2 ** (-k)) * math.gamma(1 + k))
    mu = l1 - a / k * (1 - math.gamma(1 + k))
    return l1, l2, t3, mu, a, k


def test_gev_lmoment_fit_recomputed():
    """Unbiased PWMs -> L-moments -> Hosking's k approximation, alpha, mu."""
    import pytest

    x = [21.0, 34.5, 19.2, 55.1, 28.3, 31.0, 44.2, 25.5, 38.8, 27.1, 60.4, 23.9]
    l1, l2, t3, mu, a, k = _hosking_gev(x)
    r = evt_gev_lmoments(x)
    assert (r["l1"], r["l2"], r["t3"]) == pytest.approx((l1, l2, t3), rel=1e-12)
    assert (r["mu"], r["sigma"], r["k_hosking"]) == pytest.approx((mu, a, k), rel=1e-10)
    assert r["xi"] == pytest.approx(-k, rel=1e-10)
