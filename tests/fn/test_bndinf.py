"""Tests for bndinf.bound_inference."""

from morie.fn import _array_core as np

import math
import pytest

from morie.fn.bndinf import bound_inference


def test_bndinf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    raw = rng.normal(0, 1, (n, 2))
    moments = np.array([[min(r), max(r)] for r in raw])
    theta = np.linspace(-3, 3, 50)
    result = bound_inference(theta, moments, 0.05)
    assert isinstance(result, dict)
    for key in ("lower", "upper", "width", "grid_lower", "grid_upper",
                "n_in_set", "cutoff", "criterion_min", "n"):
        assert key in result
    assert result["upper"] >= result["lower"]
    assert math.isclose(result["width"], result["upper"] - result["lower"])
    assert 0 <= result["n_in_set"] <= len(theta)
    assert result["n"] == n
    assert math.isfinite(result["lower"])
    assert math.isfinite(result["upper"])
    assert math.isfinite(result["cutoff"])
    assert math.isfinite(result["criterion_min"])


def test_bndinf_edge():
    """Test that invalid alpha raises ValueError."""
    rng = np.random.default_rng(42)
    n = 10
    raw = rng.normal(0, 1, (n, 2))
    moments = np.array([[min(r), max(r)] for r in raw])
    theta = np.linspace(-1, 1, 20)
    with pytest.raises(ValueError):
        bound_inference(theta, moments, 1.5)
