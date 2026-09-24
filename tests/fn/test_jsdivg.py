"""Tests for jsdivg.jensen_shannon_divergence."""

import math

from morie.fn import _array_core as np

from morie.fn.jsdivg import jensen_shannon_divergence


def test_jsdivg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    p = np.abs(rng.normal(0, 1, 50))
    q = np.abs(rng.normal(0, 1, 50))
    result = jensen_shannon_divergence(p, q)
    assert isinstance(result, dict)
    for key in ("estimate", "distance", "bound", "base", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["distance"])
    assert result["estimate"] >= 0.0
    assert math.isclose(result["distance"], math.sqrt(result["estimate"]))
    assert math.isclose(result["bound"], math.log(2.0) / math.log(2.0))
    assert result["n"] == 50


def test_jsdivg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = np.abs(rng.normal(0, 1, 10))
    q = np.abs(rng.normal(0, 1, 10))
    result = jensen_shannon_divergence(p, q, base=math.e)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    assert result["estimate"] <= result["bound"] + 1e-9
    assert math.isclose(result["base"], math.e)
    assert result["method"] == "Jensen-Shannon divergence"
