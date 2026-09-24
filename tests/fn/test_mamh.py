"""Tests for mamh.ma_mantel_haenszel."""

import math

from morie.fn import _array_core as np

from morie.fn.mamh import ma_mantel_haenszel


def test_mamh_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    K = 6
    a = rng.integers(1, 25, K).tolist()
    b = rng.integers(10, 60, K).tolist()
    c = rng.integers(1, 25, K).tolist()
    d = rng.integers(10, 60, K).tolist()
    result = ma_mantel_haenszel(a, b, c, d)
    assert result.measure == "OR_MH"
    assert math.isfinite(result.estimate)
    assert math.isfinite(result.ci_lower)
    assert math.isfinite(result.ci_upper)
    assert math.isfinite(result.se)
    assert result.ci_lower <= result.estimate <= result.ci_upper
    assert math.isfinite(result.n) and result.n > 0
    assert result.extra["n_strata"] == K


def test_mamh_edge():
    """Test edge cases."""
    rng = np.random.default_rng(7)
    K = 3
    a = rng.integers(1, 5, K).tolist()
    b = rng.integers(5, 15, K).tolist()
    c = rng.integers(1, 5, K).tolist()
    d = rng.integers(5, 15, K).tolist()
    result = ma_mantel_haenszel(a, b, c, d)
    assert result.measure == "OR_MH"
    assert math.isfinite(result.estimate)
    assert math.isfinite(result.ci_lower)
    assert math.isfinite(result.ci_upper)
    assert result.ci_lower <= result.ci_upper
    assert math.isfinite(result.n) and result.n > 0
    assert result.extra["n_strata"] == K
