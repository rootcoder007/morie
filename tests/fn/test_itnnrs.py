"""Tests for itnnrs.item_nonresponse."""

import math

from morie.fn import _array_core as np

from morie.fn.itnnrs import item_nonresponse


def test_itnnrs_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_R = np.random.default_rng(42)
    n = 100
    y = list(rng_y.normal(0, 1, n))
    R = [float(v) for v in rng_R.integers(0, 2, n)]
    # 3 discrete classes (alternating) so each class has multiple units
    X = [[i % 3] for i in range(n)]
    result = item_nonresponse(y, R, X)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "n_classes" in result
    assert "response_rate" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    assert result["n"] == n


def test_itnnrs_edge():
    """Test edge cases with X=None (single class)."""
    rng_y = np.random.default_rng(43)
    rng_R = np.random.default_rng(42)
    n = 100
    y = list(rng_y.normal(0, 1, n))
    R = [float(v) for v in rng_R.integers(0, 2, n)]
    result = item_nonresponse(y, R, None)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "n_classes" in result
    assert result["n_classes"] == 1
    assert result["n"] == n
