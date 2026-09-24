"""Tests for bndtfm.bound_transform."""

from morie.fn import _array_core as np
import math

from morie.fn.bndtfm import bound_transform


def test_bndtfm_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(41)

    n = 100
    y = rng_y.normal(0, 1, n)
    D = rng_d.integers(0, 2, n)
    X = rng_x.integers(0, 3, n)
    # transform must be non-decreasing in y; the identity satisfies this
    transform = [float(v) for v in y]

    result = bound_transform(y, D, X, transform)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "estimate", "gap", "n_strata", "n"):
        assert key in result
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
    assert math.isfinite(result["width"])
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["gap"])
    assert result["lower"] <= result["upper"]
    assert result["gap"] >= 0
    assert result["n"] == n
    assert result["n_strata"] >= 1


def test_bndtfm_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(41)

    n = 40
    y = rng_y.normal(0, 1, n)
    D = rng_d.integers(0, 2, n)
    X = rng_x.integers(0, 2, n)
    transform = [float(v) for v in y]

    result = bound_transform(y, D, X, transform)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "estimate", "gap", "n_strata", "n"):
        assert key in result
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["gap"])
    assert result["lower"] <= result["upper"]
    assert result["gap"] >= 0
    assert result["n"] == n
    assert result["n_strata"] >= 1
