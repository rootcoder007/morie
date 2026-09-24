"""Tests for mskbnd2.manski_no_assumption_outcome."""

import math

from morie.fn import _array_core as np

from morie.fn.mskbnd2 import manski_no_assumption_outcome


def test_mskbnd2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    y_min = 0.0
    y_max = 10.0
    y = rng.uniform(y_min, y_max, n)
    D = rng.integers(0, 2, n)
    X = rng.integers(0, 5, n)
    result = manski_no_assumption_outcome(y, D, X, y_min, y_max)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "inter_lower", "inter_upper", "n_strata", "n"):
        assert key in result
    assert result["n"] == n
    assert result["n_strata"] >= 1
    assert result["n_strata"] <= 5
    assert result["lower"] <= result["upper"]
    assert result["inter_lower"] <= result["inter_upper"]
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
    assert math.isfinite(result["inter_lower"])
    assert math.isfinite(result["inter_upper"])


def test_mskbnd2_edge():
    """Test edge cases with symmetric support around zero."""
    rng = np.random.default_rng(43)
    n = 40
    y_min = -5.0
    y_max = 5.0
    y = rng.uniform(y_min, y_max, n)
    D = rng.integers(0, 2, n)
    X = rng.integers(0, 3, n)
    result = manski_no_assumption_outcome(y, D, X, y_min, y_max)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "inter_lower", "inter_upper", "n_strata", "n"):
        assert key in result
    assert result["n"] == n
    assert result["n_strata"] >= 1
    assert result["n_strata"] <= 3
    assert math.isfinite(result["width"])
    assert result["width"] >= 0
    assert result["lower"] <= result["upper"]
    assert result["inter_lower"] <= result["inter_upper"]
    assert result["inter_lower"] <= 0
    assert result["inter_upper"] >= 0
