"""Tests for otdiv.ot_sinkhorn_divergence."""

import math

from morie.fn import _array_core as np
from morie.fn.otdiv import ot_sinkhorn_divergence


def test_otdiv_basic():
    """Test basic functionality."""
    rng_a = np.random.default_rng(44)
    rng_b = np.random.default_rng(42)
    rng_C = np.random.default_rng(43)

    n, m = 10, 12

    # Histograms must be non-negative weights.
    a = rng_a.uniform(0, 1, n)
    b = rng_b.uniform(0, 1, m)

    # Cost matrices with shapes matching the marginals.
    Cab = rng_C.normal(0, 1, (n, m))
    Caa = rng_C.normal(0, 1, (n, n))
    Cbb = rng_C.normal(0, 1, (m, m))

    epsilon = 0.1
    result = ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon)

    assert "S_eps" in result
    assert math.isfinite(result["S_eps"])
    assert "OT_ab" in result
    assert "OT_aa" in result
    assert "OT_bb" in result
    assert "n" in result and result["n"] == n
    assert "m" in result and result["m"] == m


def test_otdiv_edge():
    """Test edge cases with a different shape and larger epsilon."""
    rng = np.random.default_rng(7)

    n, m = 5, 6

    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, m)

    Cab = rng.normal(0, 1, (n, m))
    Caa = rng.normal(0, 1, (n, n))
    Cbb = rng.normal(0, 1, (m, m))

    epsilon = 0.5
    result = ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon)

    assert "S_eps" in result
    assert "OT_ab" in result
    assert "OT_aa" in result
    assert "OT_bb" in result
    assert "method" in result
    assert math.isfinite(result["S_eps"])
    assert math.isfinite(result["OT_ab"])
    assert math.isfinite(result["OT_aa"])
    assert math.isfinite(result["OT_bb"])
    assert result["n"] == n
    assert result["m"] == m
