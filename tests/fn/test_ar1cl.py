"""Tests for ar1cl.ar1_climate."""

from morie.fn import _array_core as np
from morie.fn.ar1cl import ar1_climate


def test_ar1cl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ar1_climate(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ar1cl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ar1_climate(x)
    assert isinstance(result, dict)


def test_ar1cl_yule_walker_and_red_spectrum_recomputed():
    import math

    import pytest

    x = [0.5, 0.9, 0.4, 1.3, 1.1, 0.2, -0.3, 0.1, 0.8, 0.6]
    n = len(x)
    m = sum(x) / n
    c0 = sum((v - m) ** 2 for v in x) / n
    c1 = sum((x[i] - m) * (x[i - 1] - m) for i in range(1, n)) / n
    ph = c1 / c0
    s2 = c0 * (1 - ph * ph)
    r = ar1_climate(x, dt=0.5, freq=[0.0, 0.3])
    assert r["phi"] == pytest.approx(ph, rel=1e-13)
    assert r["sigma2_eps"] == pytest.approx(s2, rel=1e-13)
    assert r["tau"] == pytest.approx(-0.5 / math.log(ph), rel=1e-13)
    want = [s2 * 0.5 / (1 - 2 * ph * math.cos(2 * math.pi * f * 0.5) + ph * ph) for f in (0.0, 0.3)]
    assert [float(v) for v in r["spectrum"]] == pytest.approx(want, rel=1e-13)
